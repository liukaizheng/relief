#include "flatten_surface.h"
#include <igl/harmonic.h>
#include <igl/polar_svd.h>

# include <algorithm>

auto write_uv(const std::string& name, const VMat2& uv, const FMat& F) {
    std::ofstream file(name);
    for (Eigen::Index i = 0; i < uv.rows(); ++i) {
        file << "v " << uv(i, 0) << " " << uv(i, 1) << " 0\n";
    }
    for (Eigen::Index i = 0; i < F.rows(); ++i) {
        file << "f " << F(i, 0) + 1 << " " << F(i, 1) + 1 << " " << F(i, 2) + 1 << "\n";
    }
    file.close();
}

template<typename DerivedV1, typename DerivedV2, typename DerivedN>
void cross(const Eigen::MatrixBase<DerivedV1>& V1, const Eigen::MatrixBase<DerivedV2>& V2, Eigen::MatrixBase<DerivedN>& N) {
    N.col(0) = V1.col(1).cwiseProduct(V2.col(2)) - V1.col(2).cwiseProduct(V2.col(1));
    N.col(1) = V1.col(2).cwiseProduct(V2.col(0)) - V1.col(0).cwiseProduct(V2.col(2));
    N.col(2) = V1.col(0).cwiseProduct(V2.col(1)) - V1.col(1).cwiseProduct(V2.col(0));
}

/**
 * Compute per-face differential data for linear triangle basis functions.
 *
 * Input:
 *  - vertices: |V|x3 matrix of vertex positions.
 *  - faces:    |F|x3 matrix of triangle indices (A,B,C ordering).
 *
 * Output tuple:
 *  1) edge_bc:    |F|x3 vector from vertex B to C.
 *  2) grad_phi0:  |F|x3 gradients of barycentric basis phi0 (associated with vertex A).
 *  3) normals:    |F|x3 unit normals for each face.
 *  4) double_area:|F| vector of twice the triangle area.
 *  5) face_vertex_map: sparse map (F x |V|) storing per-face vertex slots (3 per face).
 *  6) grad_rows: stacked gradients for (phi0, phi1, phi2) as three rows per face.
 */
template<typename DerivedV, typename DerivedF>
auto tri_gradients(const Eigen::MatrixBase<DerivedV>& vertices, const Eigen::MatrixBase<DerivedF>& faces) noexcept {
    const auto face_count = faces.rows(); // number of triangles
    const auto vertex_count = vertices.rows(); // number of vertices
    auto edge_bc = (vertices(faces.col(2), Eigen::placeholders::all) - vertices(faces.col(1), Eigen::placeholders::all)).eval(); // edge B->C
    auto edge_ca = (vertices(faces.col(0), Eigen::placeholders::all) - vertices(faces.col(2), Eigen::placeholders::all)).eval(); // edge C->A
    using FaceVectors = Eigen::Matrix<typename DerivedV::Scalar, Eigen::Dynamic, 3>; // reusable per-face 3D rows
    FaceVectors normals(face_count, 3); // face normals (scaled)
    cross(edge_bc, edge_ca, normals);
    auto double_area = normals.rowwise().norm().eval(); // per-face 2*area
    auto area_repeated = double_area.replicate(1, 3).array(); // broadcasted for division
    normals = normals.array() / area_repeated;

    FaceVectors grad_phi0(face_count, 3); // grad of barycentric phi0 (vertex 0)
    cross(normals, edge_bc, grad_phi0);
    grad_phi0 = grad_phi0.array() / area_repeated;

    FaceVectors grad_phi1(face_count, 3); // grad of barycentric phi1 (vertex 1)
    cross(normals, edge_ca, grad_phi1);
    grad_phi1 = grad_phi1.array() / area_repeated;

    Eigen::SparseMatrix<Eigen::Index> face_vertex_map(face_count, vertex_count); // maps face->vertex slot
    std::vector<Eigen::Triplet<Eigen::Index>> fv_entries; // buffer for map entries
    Eigen::Index entry = 0; // running triplet id
    for (Eigen::Index i = 0; i < face_count; i++) {
        fv_entries.emplace_back(i, faces(i, 0), entry++);
        fv_entries.emplace_back(i, faces(i, 1), entry++);
        fv_entries.emplace_back(i, faces(i, 2), entry++);
    }
    face_vertex_map.setFromTriplets(fv_entries.begin(), fv_entries.end());
    face_vertex_map.makeCompressed();

    IVec face_rows = IVec::LinSpaced(face_count, 0, face_count * 3 - 3); // row offsets per face
    FaceVectors grad_rows(face_count * 3, 3); // stacked phi gradients per face
    grad_rows(face_rows, Eigen::placeholders::all) = grad_phi0;
    grad_rows(face_rows.array() + 1, Eigen::placeholders::all) = grad_phi1;
    grad_rows(face_rows.array() + 2, Eigen::placeholders::all) = -grad_phi0 - grad_phi1;

    return std::make_tuple(std::move(edge_bc), std::move(grad_phi0), std::move(normals), std::move(double_area), std::move(face_vertex_map), std::move(grad_rows));
}

/**
 * Build the sparsity pattern for the augmented 4x block system used in SLIM.
 *
 * Input:
 *  - DX: per-face gradient operator (row-major), shape |F|x|V|.
 *  - n_fixed: number of boundary/fixed vertices (front of vertex array).
 *
 * Output:
 *  - RowSpMat A: empty sparse matrix with correct shape/structure for the 4x system
 *    (rows = 4*|F|, cols = 2*|V_free|) ready to be filled with values.
 *  - skip_counts: for each face row in DX, how many leading entries belong to fixed vertices
 *    and must be skipped when forming the free-vertex system.
 */
auto build_system_pattern(const RowSpMat& DX, const Eigen::Index n_fixed) noexcept {
    using IVec = Eigen::Matrix<RowSpMat::StorageIndex, 1, Eigen::Dynamic, Eigen::RowMajor>;
    using Mat4 = Eigen::Matrix<RowSpMat::StorageIndex, 4, Eigen::Dynamic, Eigen::RowMajor>;
    const auto n_free = DX.innerSize() - n_fixed; // number of free vertices
    std::vector<RowSpMat::StorageIndex> pattern_outer(DX.outerSize() * 4 + 1); // row pointer for 4x block rows
    std::vector<RowSpMat::StorageIndex> pattern_inner(DX.nonZeros() * 8); // column indices for duplicated blocks
    pattern_outer[0] = 0;
    const auto dx_outer_ptr = DX.outerIndexPtr(); // source row starts
    const auto dx_inner_ptr = DX.innerIndexPtr(); // source column ids
    std::size_t outer_cursor = 0; // write head for pattern_outer
    RowSpMat::StorageIndex value_cursor = 0; // write head for pattern_inner
    std::vector<Eigen::Index> skip_counts(DX.outerSize()); // leading constrained entries to skip per row
    for (Eigen::Index i = 0; i < DX.outerSize(); i++) {
        auto dx_start = dx_outer_ptr[i];
        const auto dx_end = dx_outer_ptr[i + 1];
        const auto row_inner_ptr = dx_inner_ptr + dx_start; // pointer to this row's inner ids
        skip_counts[i] = std::find_if(row_inner_ptr, dx_inner_ptr + dx_end, [n_fixed](auto x) noexcept { return x >=  n_fixed; }) - row_inner_ptr;
        dx_start += skip_counts[i];
        const auto row_size = dx_end - dx_start; // free entries in this row
        const auto row_double = row_size * 2; // duplicated for u/v
        const auto row_inner_indices = IVec::Map(dx_inner_ptr + dx_start, row_size).array(); // free vertex ids
        auto block_indices = Mat4::Map(pattern_inner.data() + value_cursor, 4, row_double); // 4x duplication block
        block_indices.leftCols(row_size) = (row_inner_indices - n_fixed).replicate<4, 1>();
        block_indices.rightCols(row_size) = (row_inner_indices + (n_free - n_fixed)).replicate<4, 1>();

        IVec::Map(pattern_outer.data() + outer_cursor + 1, 4) = IVec::LinSpaced(4, value_cursor + row_double, value_cursor + row_double * 4);
        value_cursor += row_double * 4;
        outer_cursor += 4;
    }
    pattern_inner.resize(value_cursor);
    RowSpMat A = RowSpMat::Map(DX.rows() * 4, n_free * 2, pattern_inner.size(), pattern_outer.data(), pattern_inner.data(), std::vector<double>(pattern_inner.size()).data());
    return std::make_pair(std::move(A), std::move(skip_counts));
}

/*
 * WJ = | w11 w12 | * | DxU DyU | = |w11 DxU + w12 DxV, w11 DyU + w12 DyV|
 *      | w21 w22 |   | DxV DyV | = |w21 DxU + w22 DxV, w21 DyV + w22 DyV|
 */
/**
 * Populate the 4x system matrix values for SLIM and accumulate boundary RHS terms.
 *
 * Input:
 *  - DX, DY: per-face gradient operators (row-major).
 *  - W: |F|x4 weight matrix derived from polar SVD.
 *  - skip_counts: leading constrained entries to skip per face row.
 *  - uv: current parameterization.
 *
 * Output:
 *  - A: numeric values filled in-place.
 *  - boundary_rhs: contribution from fixed vertices to move to the RHS.
 */
void fill_system_values(const RowSpMat& DX, const RowSpMat& DY, const VMat4& W, const std::vector<Eigen::Index>& skip_counts, const VMat2& uv, RowSpMat& A, Eigen::VectorXd& boundary_rhs) noexcept {
    using RowValues = Eigen::Matrix<RowSpMat::Scalar, 1, Eigen::Dynamic>; // dense row view of gradient values
    using IndexRow = Eigen::Matrix<RowSpMat::StorageIndex, 1, Eigen::Dynamic, Eigen::RowMajor>; // row of vertex ids
    using Block4 = Eigen::Matrix<RowSpMat::Scalar, 4, Eigen::Dynamic, Eigen::RowMajor>; // 4x face block of system
    const auto row_ptr = A.outerIndexPtr(); // offsets into value/inner arrays
    boundary_rhs.setZero();
    for (Eigen::Index face = 0; face < DX.rows(); face++) {
        const auto block_row = face * 4; // starting block row
        const auto block_start = row_ptr[block_row]; // value offset
        const auto block_end = row_ptr[block_row + 1];
        const auto block_size = block_end - block_start; // total cols for this face (u+v)
        const auto half_block = block_size / 2; // cols for free vertices (u or v)
        const auto total_grad_cols = half_block + skip_counts[face]; // includes fixed part
        const auto dx_row = RowValues::Map(DX.valuePtr() + DX.outerIndexPtr()[face], total_grad_cols); // Dx entries (fixed+free)
        const auto dy_row = RowValues::Map(DY.valuePtr() + DY.outerIndexPtr()[face], total_grad_cols); // Dy entries (fixed+free)
        const auto weight_row = W.row(face).transpose().array().replicate(1, total_grad_cols); // polar weights
        auto block_values = Block4::Map(A.valuePtr() + block_start, 4, block_size); // target block in A
        const auto dx_block = (dx_row.replicate<4, 1>().array() * weight_row).eval(); // weighted Dx per block row
        const auto dy_block = (dy_row.replicate<4, 1>().array() * weight_row).eval(); // weighted Dy per block row
        block_values.topRows(2).reshaped<Eigen::RowMajor>(4, half_block) = dx_block.rightCols(half_block);
        block_values.bottomRows(2).reshaped<Eigen::RowMajor>(4, half_block) = dy_block.rightCols(half_block);

        auto fixed_indices = IndexRow::Map(DX.innerIndexPtr() + DX.outerIndexPtr()[face], skip_counts[face]); // constrained vertex ids
        auto fixed_uv = uv(fixed_indices, Eigen::placeholders::all).reshaped(skip_counts[face] * 2, 1); // their uv coords

        auto rhs_slice = boundary_rhs.segment(block_row, 4); // slice for this face

        rhs_slice.segment(0, 2) += dx_block.leftCols(skip_counts[face]).reshaped<Eigen::RowMajor>(2, skip_counts[face] * 2).matrix() * fixed_uv;
        rhs_slice.segment(2, 2) += dy_block.leftCols(skip_counts[face]).reshaped<Eigen::RowMajor>(2, skip_counts[face] * 2).matrix() * fixed_uv;
    }
}

auto transpose_sparse_matrix(const RowSpMat& A) noexcept {
    const auto outer_indices = A.outerIndexPtr();
    const auto inner_indices = A.innerIndexPtr();
    const auto values = A.valuePtr();

    RowSpMat dest(A.cols(), A.rows());
    dest.resizeNonZeros(A.nonZeros());
    auto new_outer_indices = dest.outerIndexPtr();
    auto new_inner_indices = dest.innerIndexPtr();
    auto new_values = dest.valuePtr();

    RowSpMat::IndexVector::Map(new_outer_indices, dest.outerSize()).setZero();
    for (Eigen::Index i = 0; i < A.outerSize(); i++) {
        for (RowSpMat::InnerIterator it(A, i); it; ++it) {
            new_outer_indices[it.index()]++;
        }
    }

    RowSpMat::IndexVector positions(dest.outerSize());
    RowSpMat::IndexVector old_to_new_indices(A.nonZeros());
    RowSpMat::StorageIndex count = 0;
    for (Eigen::Index i = 0; i < dest.outerSize(); i++) {
        auto temp = new_outer_indices[i];
        new_outer_indices[i] = count;
        positions[i] = count;
        count += temp;
    }
    new_outer_indices[dest.outerSize()] = count;

    for (Eigen::Index i = 0; i < A.outerSize(); i++) {
        const auto start = outer_indices[i];
        const auto end = outer_indices[i + 1];
        for (Eigen::Index j = start; j < end; j++) {
            const auto pos = positions[inner_indices[j]]++;
            new_inner_indices[pos] = i;
            new_values[pos] = values[j];
            old_to_new_indices[j] = pos;
        }
    }
    return std::make_tuple(std::move(dest), std::move(old_to_new_indices));
}

/**
 * Cache sparsity/index data to assemble the normal matrix L = At * M * A efficiently.
 *
 * Usage:
 *  - call build(At) once to create an empty L with the correct sparsity and
 *    record index mappings between At entries and L.
 *  - call assign(At, M, L) to fill numeric values given a diagonal weight M.
 *
 * Stored mappings:
 *  - block_entry_offsets: prefix sums over matching entry pairs per L entry.
 *  - left_entry_indices/right_entry_indices: matching (row,col) entry indices in At
 *    that contribute to each upper-triangular entry in L.
 *  - upper_inner_positions: positions in L for the upper triangle entries.
 *  - lower_inner_positions/lower_to_upper_map: mirror positions for the lower triangle.
 */
struct NormalMatrixCache {
    using StorageIndex = RowSpMat::StorageIndex;
    using IndexVector = RowSpMat::IndexVector;
    using ScalarVector = RowSpMat::ScalarVector;

    NormalMatrixCache() noexcept = default;
    auto build(const RowSpMat& At) noexcept;
    void assign(const RowSpMat& At, const Eigen::VectorXd& M, RowSpMat& L) noexcept;

    std::vector<StorageIndex> block_entry_offsets;      // prefix sums delimiting contributing pairs per L entry
    std::vector<StorageIndex> left_entry_indices;       // indices of At entries on the left side (row)
    std::vector<StorageIndex> right_entry_indices;      // matching At entries on the right side (col)
    std::vector<StorageIndex> upper_inner_positions;    // positions in L for upper triangle entries
    std::vector<StorageIndex> lower_inner_positions;    // positions in L for lower triangle entries
    std::vector<StorageIndex> lower_to_upper_map;       // mapping from lower positions to corresponding upper positions
};

auto NormalMatrixCache::build(const RowSpMat& At) noexcept {
    const auto outer_ptr = At.outerIndexPtr();  // row pointers of At
    const auto inner_ptr = At.innerIndexPtr();  // column indices of At
    const auto values = At.valuePtr();
    constexpr RowSpMat::StorageIndex INVALID = -1;

    block_entry_offsets.emplace_back(0);
    std::vector<StorageIndex> normal_outer{0};          // outer pointer for L
    std::vector<StorageIndex> normal_inner;             // inner indices for L
    std::vector<StorageIndex> col_head(At.outerSize(), INVALID); // head of per-column linked list
    std::vector<StorageIndex> col_tail(At.outerSize(), INVALID); // tail of per-column linked list
    std::vector<StorageIndex> col_next;                 // next pointers for linked list
    std::vector<StorageIndex> row_of_inner;             // row index for each L entry (mirrored)
    for (Eigen::Index row = 0; row < At.outerSize(); row++) {
        // setup left part
        auto curr = col_head[row];
        while(curr != INVALID) {
            lower_inner_positions.emplace_back(normal_inner.size()); // position in lower triangle
            lower_to_upper_map.emplace_back(curr); // map to corresponding upper entry
            normal_inner.emplace_back(row_of_inner[curr]); // mirror column
            row_of_inner.emplace_back(INVALID);
            col_next.emplace_back(INVALID);
            curr = col_next[curr];
        }
        for (Eigen::Index col = row; col < At.outerSize(); col++) {
            auto right_it = outer_ptr[col];
            for (StorageIndex left_it = outer_ptr[row]; left_it < outer_ptr[row + 1]; left_it++) {
                const auto left_inner_idx = inner_ptr[left_it];
                while(right_it < outer_ptr[col + 1] && inner_ptr[right_it] < left_inner_idx) {
                    right_it++;
                }

                if (right_it == outer_ptr[col + 1]) {
                    break;
                }

                if (inner_ptr[right_it] == left_inner_idx) {
                    left_entry_indices.emplace_back(left_it);
                    right_entry_indices.emplace_back(right_it);
                    right_it++;
                    if (right_it == outer_ptr[col + 1]) {
                        break;
                    }
                }
            }
            if (left_entry_indices.size() != block_entry_offsets.back()) {
                block_entry_offsets.emplace_back(left_entry_indices.size());
                const StorageIndex new_inner_idx = normal_inner.size();
                col_next.emplace_back(INVALID);
                row_of_inner.emplace_back(row);
                normal_inner.emplace_back(col);
                upper_inner_positions.emplace_back(new_inner_idx);
                if (col_head[col] == INVALID) {
                    col_head[col] = new_inner_idx;
                    col_tail[col] = new_inner_idx;
                } else {
                    col_next[col_tail[col]] = new_inner_idx;
                    col_tail[col] = new_inner_idx;
                }
            }
        }
        normal_outer.emplace_back(normal_inner.size());
    }

    RowSpMat L(At.outerSize(), At.outerSize());
    L.resizeNonZeros(normal_inner.size());
    IndexVector::Map(L.outerIndexPtr(), normal_outer.size()) = IndexVector::Map(normal_outer.data(), normal_outer.size());
    IndexVector::Map(L.innerIndexPtr(), normal_inner.size()) = IndexVector::Map(normal_inner.data(), normal_inner.size());
    return L;
}

void NormalMatrixCache::assign(const RowSpMat& At, const Eigen::VectorXd& M, RowSpMat& L) noexcept {
    const auto at_values = ScalarVector::Map(At.valuePtr(), At.nonZeros()); // values of At
    const auto weighted_products = (at_values(left_entry_indices).array()
        * M(IndexVector::Map(At.innerIndexPtr(), At.nonZeros())(left_entry_indices)).array()
        * at_values(right_entry_indices).array()).eval(); // elementwise At * M * At
    auto normal_values = ScalarVector::Map(L.valuePtr(), L.nonZeros()); // writable values of L
    for (std::size_t block = 0; block < upper_inner_positions.size(); block++) {
        normal_values[upper_inner_positions[block]] = weighted_products.segment(block_entry_offsets[block], block_entry_offsets[block + 1] - block_entry_offsets[block]).sum();
    }

    normal_values(lower_inner_positions) = normal_values(lower_to_upper_map);
}

void assign_rhs(const VMat4& W, const VMat4& R, Eigen::VectorXd& rhs) noexcept {
    Eigen::Index idx = 0;
    for (Eigen::Index i = 0; i < W.rows(); ++i) {
        rhs.segment(idx, 4).reshaped(2, 2) = W.row(i).reshaped<Eigen::RowMajor>(2, 2) * R.row(i).reshaped(2, 2);
        idx += 4;
    }
}

/*
 * Compute y = At * (diag(M) * b)
 */
void multiply_normal_rhs(const RowSpMat& At, const Eigen::VectorXd& M, const Eigen::VectorXd& b, Eigen::VectorXd& rhs) noexcept {
    const auto outer_ptr = At.outerIndexPtr();   // row pointers
    const auto inner_ptr = At.innerIndexPtr();   // column indices
    const auto values_ptr = At.valuePtr();       // nonzero values

    for (Eigen::Index row = 0; row < At.outerSize(); ++row) {
        const auto start = outer_ptr[row];
        const auto end = outer_ptr[row + 1];
        const auto nnz = end - start;
        const auto col_ids = RowSpMat::IndexVector::Map(inner_ptr + start, nnz); // columns in this row
        rhs[row] = (RowSpMat::ScalarVector::Map(values_ptr + start, nnz).array()
            * M(col_ids).array())
            .matrix()
            .transpose()
            * b(col_ids);
    }
}
double get_smallest_pos_quad_zero(const double a,const double b, const double c) noexcept
{
    double t1, t2;
    if(std::abs(a) > 1.0e-10) {
        double delta_in = b * b - 4 * a * c;
        if(delta_in < 0) {
            return INFINITY;
        }

        double delta = std::sqrt(delta_in); // delta >= 0
        if(b >= 0) { // avoid subtracting two similar numbers
            double bd = - b - delta;
            t1 = 2 * c / bd;
            t2 = bd / (2 * a);
        }
        else {
            double bd = - b + delta;
            t1 = bd / (2 * a);
            t2 = (2 * c) / bd;
        }

        if(a < 0) {
            std::swap(t1, t2); // make t1 > t2
        }
        // return the smaller positive root if it exists, otherwise return infinity
        if(t1 > 0) {
            return t2 > 0 ? t2 : t1;
        } else {
            return INFINITY;
        }
    } else {
        if(b == 0) {
            return INFINITY; // just to avoid divide-by-zero
        }
        t1 = -c / b;
        return t1 > 0 ? t1 : INFINITY;
    }
}

double get_min_pos_root_2D(const FMat& F, const VMat2& uv, const VMat2& d) noexcept {
    const Eigen::Vector2i I1{0, 0};
    const Eigen::Vector2i I2{1, 2};

    double max_step = INFINITY;
    for (Eigen::Index i = 0; i < F.rows(); i++) {
        const auto R = F.row(i).eval();
        auto U = uv(R(I2), Eigen::placeholders::all) - uv(R(I1), Eigen::placeholders::all);
        auto V = d(R(I2), Eigen::placeholders::all) - d(R(I1), Eigen::placeholders::all);
        const auto a = V.determinant();
        const auto b = U.col(0).cross(V.col(1)) + V.col(0).cross(U.col(1));
        const auto c = U.determinant();
        max_step = std::min(max_step, get_smallest_pos_quad_zero(a, b, c));
    }
    return std::min(1.0, max_step * 0.8);
}

double line_search(const VMat2& uv, const VMat2& d, VMat2& new_uv, const auto& compute_energy, double old_energy, double t) noexcept {
    if (old_energy == 0.0) {
        old_energy = compute_energy(uv);
    }
    constexpr std::size_t MAX_ITERATIONS = 5;
    double energy = old_energy;
    for (std::size_t i = 0; i < MAX_ITERATIONS; i++) {
        new_uv = uv + t * d;
        energy = compute_energy(new_uv);
        if (energy < old_energy) {
            return energy;
        } else {
            t *= 0.5;
        }
    }
    return energy;
}

double flip_avoiding_line_search(const FMat& F, const VMat2& uv, VMat2& new_uv, const auto& compute_energy, const double energy) {
    const auto d = (new_uv - uv).eval();
    const auto t = get_min_pos_root_2D(F, uv, d);
    return line_search(uv, d, new_uv, compute_energy, energy, t);
}

FlattenSurface::FlattenSurface(VMat&& V, FMat&& F, const std::vector<std::size_t>& segment_offsets) noexcept : V(std::move(V)), F(std::move(F)), n_bnd_points(segment_offsets.back()) {
    IVec I1;
    IVec I2;
    IVec I = IVec::LinSpaced(3, 0 ,2);
    mean_edge_length = 0.0;
    std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> edge_coefficients;
    edge_coefficients.reserve(segment_offsets.size() - 1);
    for (std::size_t i = 0; i + 1 < segment_offsets.size(); ++i) {
        const auto start = segment_offsets[i];
        const auto end = segment_offsets[i + 1];
        I1.resize(end - start);
        I2.resize(I1.size());
        I1.setLinSpaced(start, end - 1);
        I2.segment(0, I2.size() - 1) = I1.segment(1, I1.size() - 1);
        if (i == 3) {
            I2[I2.size() - 1] = 0;
        } else {
            I2[I2.size() - 1] = end;
        }

        auto D = (this->V(I1, I) - this->V(I2, I)).rowwise().norm().eval();
        for (Eigen::Index j = 1; j < D.size(); j++) {
            D[j] = D[j - 1] + D[j];
        }
        double prev_val = 0;
        double edge_length = D[D.size() - 1];
        for (Eigen::Index j = 0; j < D.size(); j++) {
            const auto val = prev_val / edge_length;
            prev_val = D[j];
            D[j] = val;
        }
        mean_edge_length += edge_length * 0.25;
        edge_coefficients.emplace_back(std::move(D));
    }

    // Initialize UV coordinates
    IVec bnd = IVec::LinSpaced(segment_offsets.back(), 0, segment_offsets.back() - 1);
    VMat2 bnd_uv(bnd.size(), 2);
    bnd_uv.row(segment_offsets[1]) << mean_edge_length, 0.0;
    bnd_uv.row(segment_offsets[2]) << mean_edge_length, mean_edge_length;
    bnd_uv.row(segment_offsets[3]) << 0.0, mean_edge_length;
    for (std::size_t i = 0; i + 1 < segment_offsets.size(); ++i) {
        const auto start = segment_offsets[i];
        const auto end = segment_offsets[i + 1];
        Eigen::RowVector2d start_pt = bnd_uv.row(start);
        Eigen::RowVector2d end_pt;
        if (i == 3) {
            end_pt = bnd_uv.row(0);
        } else {
            end_pt = bnd_uv.row(end);
        }

        bnd_uv.middleRows(start, end - start) =
            ((1.0 - edge_coefficients[i].array()).replicate<1, 2>() * start_pt.replicate(end - start, 1).array())
            + edge_coefficients[i].array().replicate<1, 2>() * end_pt.replicate(end - start, 1).array();
    }
    igl::harmonic(this->V, this->F, bnd, bnd_uv, 1, uv);
    // write_uv("uv_init.obj", uv, this->F);
}

void FlattenSurface::slim_solve(const std::size_t n_iterations) {
    Eigen::Index n_free = this->V.rows() - n_bnd_points;
    auto [B1, B2, N, DA, IFV, G] = tri_gradients(this->V, this->F);
    DA /= DA.sum();

    // Compute per-face gradient dot-products for a barycentric basis vector B and
    // write the scalar results into a dense buffer matching the IFV/G ordering.
    auto assemble_gradient_values = [&](const auto& B, double* values) noexcept {
        const auto outer_indices = IFV.outerIndexPtr();
        const auto inner_indices = IFV.innerIndexPtr();
        for (Eigen::Index i = 0; i < IFV.outerSize(); i++) {
            const auto start = outer_indices[i];
            const auto end = outer_indices[i + 1];
            const auto size = end - start;
            const auto I1 = Eigen::Matrix<decltype(IFV)::Scalar, Eigen::Dynamic, 1>::Map(IFV.valuePtr() + start, size);
            const auto I2 = Eigen::Matrix<SpMat::StorageIndex, Eigen::Dynamic, 1>::Map(inner_indices + start, size);
            Eigen::VectorXd::Map(values + start, size) = (G(I1, Eigen::placeholders::all).array() * B(I2, Eigen::placeholders::all).array()).rowwise().sum();
        }
    };

    B1 = B1.array() / B1.rowwise().norm().replicate<1, 3>().array();
    B2 = B2.array() / B2.rowwise().norm().replicate<1, 3>().array();
    RowSpMat DX;
    RowSpMat DY;
    {
        std::vector<double> temp(IFV.nonZeros());
        assemble_gradient_values(B1, temp.data());
        DX = SpMat::Map(IFV.rows(), IFV.cols(), IFV.nonZeros(), IFV.outerIndexPtr(), IFV.innerIndexPtr(), temp.data());
        assemble_gradient_values(B2, temp.data());
        DY = SpMat::Map(IFV.rows(), IFV.cols(), IFV.nonZeros(), IFV.outerIndexPtr(), IFV.innerIndexPtr(), temp.data());
    }

    VMat4 J(B1.rows(), 4);
    VMat4 R(B1.rows(), 4);
    VMat4 W(B1.rows(), 4);

    Eigen::Matrix2d ji, ri, ti, ui, vi;
    Eigen::Vector2d si, swi;
    RowSpMat A;
    RowSpMat At;
    NormalMatrixCache normal_cache;
    RowSpMat L;
    std::vector<Eigen::Index> skip_indices;
    RowSpMat::IndexVector A_to_At_indices;
    Eigen::VectorXd boundary_rhs(B1.rows() * 4);
    Eigen::VectorXd rhs(B1.rows() * 4);
    Eigen::VectorXd real_rhs(n_free * 2);
    const auto DDA = DA.replicate<1, 4>().reshaped<Eigen::RowMajor>().eval();
    VMat2 new_uv = uv;
    double energy = 0.0;

    const auto update_jacobian = [&J, &DX, &DY](const VMat2& uv) noexcept {
        J.col(0) = DX * uv.col(0);
        J.col(1) = DX * uv.col(1);
        J.col(2) = DY * uv.col(0);
        J.col(3) = DY * uv.col(1);
    };

    const auto compute_energy = [&update_jacobian, &DA, &J, &ri, &ti, &ui, &si, &vi](const VMat2& uv) noexcept {
        update_jacobian(uv);
        double energy = 0.0;
        for (Eigen::Index i = 0; i < J.rows(); i++) {
            igl::polar_svd(J.row(i).reshaped(Eigen::fix<2>, Eigen::fix<2>), ri, ti, ui, si, vi);
            const auto si_arr = si.array().cwiseSquare();
            energy +=(si_arr + si_arr.cwiseInverse()).sum() * DA[i];
        }
        return energy;
    };

    for (std::size_t _ = 0; _ < n_iterations; ++_) {
        update_jacobian(uv);

        // update rotation and weight matrices
        for (Eigen::Index i = 0; i < J.rows(); ++i) {
            igl::polar_svd(J.row(i).reshaped(Eigen::fix<2>, Eigen::fix<2>), ri, ti, ui, si, vi);
            const auto si_arr = si.array();
            swi = ((1.0 + si_arr) / si_arr * (1.0 + si_arr.square())).sqrt() / si_arr;
            R.row(i).reshaped(Eigen::fix<2>, Eigen::fix<2>) = ri;
            W.row(i).reshaped<Eigen::RowMajor>(Eigen::fix<2>, Eigen::fix<2>) = ui * swi.asDiagonal() * ui.transpose();
        }
        if (skip_indices.empty()) {
            std::tie(A, skip_indices) = build_system_pattern(DX, n_bnd_points);
        }
        fill_system_values(DX, DY, W, skip_indices, uv, A, boundary_rhs);
        assign_rhs(W, R, rhs);
        rhs -= boundary_rhs;

        if (A_to_At_indices.size() == 0) {
            std::tie(At, A_to_At_indices) = transpose_sparse_matrix(A);
        } else {
            RowSpMat::ScalarVector::Map(At.valuePtr(), A.nonZeros())(A_to_At_indices) = RowSpMat::ScalarVector::Map(A.valuePtr(), A.nonZeros());
        }

        if (L.size() == 0) {
            L = normal_cache.build(At);
        }
        normal_cache.assign(At, DDA, L);
        multiply_normal_rhs(At, DDA, rhs, real_rhs);

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
        new_uv.bottomRows(V.rows() - n_bnd_points).reshaped() = solver.compute(L).solve(real_rhs);
        energy = flip_avoiding_line_search(F, uv, new_uv, compute_energy, energy);
        uv = new_uv;
        std::cout << "Energy: " << energy << std::endl;

        write_uv("uv_new.obj", new_uv, F);
    }
}
