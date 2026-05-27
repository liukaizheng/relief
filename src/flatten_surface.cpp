#include "flatten_surface.h"
#include "eigen_alias.h"
#include <igl/harmonic.h>
#include <igl/polar_svd.h>
#include <fmt/format.h>

#include <algorithm>
#include <ranges>
#include <utility>

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

    Eigen::SparseMatrix<Eigen::Index, Eigen::RowMajor> face_vertex_map(face_count, vertex_count);
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

/**
 * Cache sparsity/index data to assemble the normal matrix L = A^T * M * A.
 *
 * Each row of A is tiny, so building L from local row outer-products avoids
 * intersecting every pair of rows in A^T.
 */
struct NormalMatrixCache {
    using StorageIndex = RowSpMat::StorageIndex;
    using IndexVector = RowSpMat::IndexVector;
    using ScalarVector = RowSpMat::ScalarVector;

    NormalMatrixCache() noexcept = default;
    auto build(const RowSpMat& A) noexcept;
    void assign(const RowSpMat& A, const Eigen::VectorXd& M, RowSpMat& L) noexcept;

    std::vector<StorageIndex> left_entry_indices;    // A value index for the left factor of each contribution
    std::vector<Eigen::Index> contribution_rows;     // A row / M diagonal index for each contribution
    std::vector<StorageIndex> right_entry_indices;   // A value index for the right factor of each contribution
    std::vector<StorageIndex> normal_entry_indices;  // destination value index in L
};

/**
 * Build the sparsity pattern for L = A^T * M * A and cache how each local
 * product maps into L.valuePtr().
 *
 * For one sparse row r of A, the normal matrix contribution is the dense outer
 * product over that row's active columns:
 *
 *   L(i, j) += A(r, i) * M(r) * A(r, j)
 *
 * Therefore every pair of nonzeros in the same row of A defines one possible
 * nonzero coordinate (i, j) in L.  The numeric weights are iteration-dependent,
 * but this coordinate pattern is fixed as long as A's sparsity is fixed.
 *
 * The cached arrays follow the exact same nested loop order.  For contribution
 * k, left_entry_indices[k], contribution_rows[k], and right_entry_indices[k]
 * identify the product factors in A and M; normal_entry_indices[k] identifies
 * the slot in L.valuePtr() that receives the contribution.
 */
auto NormalMatrixCache::build(const RowSpMat& A) noexcept {
    const auto outer_ptr = A.outerIndexPtr();  // row pointers of A
    const auto inner_ptr = A.innerIndexPtr();  // column indices of A

    left_entry_indices.clear();
    contribution_rows.clear();
    right_entry_indices.clear();
    normal_entry_indices.clear();

    // Count row-local outer-product terms.  This is not the final number of
    // nonzeros in L because multiple A rows can contribute to the same L entry;
    // it is the number of cached product destinations assign() will replay.
    std::size_t contribution_count = 0;
    for (Eigen::Index row = 0; row < A.outerSize(); row++) {
        const auto start = outer_ptr[row];
        const auto end = outer_ptr[row + 1];
        const auto row_size = end - start;
        contribution_count += static_cast<std::size_t>(row_size) * static_cast<std::size_t>(row_size);
    }

    // Emit one triplet for every possible local contribution.  setFromTriplets()
    // sorts and coalesces duplicate coordinates, leaving L with the union of all
    // positions touched by A^T * M * A.  The triplet values are placeholders
    // because assign() overwrites L's numeric values every iteration.
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(contribution_count);
    for (Eigen::Index row = 0; row < A.outerSize(); row++) {
        const auto start = outer_ptr[row];
        const auto end = outer_ptr[row + 1];
        for (StorageIndex left_it = start; left_it < end; left_it++) {
            const auto left_col = inner_ptr[left_it];
            for (StorageIndex right_it = start; right_it < end; right_it++) {
                triplets.emplace_back(left_col, inner_ptr[right_it], 0.0);
            }
        }
    }

    RowSpMat L(A.cols(), A.cols());
    L.setFromTriplets(triplets.begin(), triplets.end());

    const auto normal_outer = L.outerIndexPtr();
    const auto normal_inner = L.innerIndexPtr();
    left_entry_indices.reserve(contribution_count);
    contribution_rows.reserve(contribution_count);
    right_entry_indices.reserve(contribution_count);
    normal_entry_indices.reserve(contribution_count);

    // Build the replay map.  For each local pair (left_col, right_col), find the
    // corresponding storage index in row left_col of L.  Both the A row's column
    // list and L's inner-index slice are sorted, so a single advancing cursor is
    // enough for all right_col values for the current left_col.  Store the A/M
    // indices in the same order so assign() can replay without traversing A.
    for (Eigen::Index row = 0; row < A.outerSize(); row++) {
        const auto start = outer_ptr[row];
        const auto end = outer_ptr[row + 1];
        for (StorageIndex left_it = start; left_it < end; left_it++) {
            const auto left_col = inner_ptr[left_it];
            auto normal_it = normal_outer[left_col];
            const auto normal_end = normal_outer[left_col + 1];
            for (StorageIndex right_it = start; right_it < end; right_it++) {
                const auto right_col = inner_ptr[right_it];
                while (normal_it < normal_end && normal_inner[normal_it] < right_col) {
                    normal_it++;
                }
                left_entry_indices.emplace_back(left_it);
                contribution_rows.emplace_back(row);
                right_entry_indices.emplace_back(right_it);
                normal_entry_indices.emplace_back(normal_it);
            }
        }
    }

    return L;
}

void NormalMatrixCache::assign(const RowSpMat& A, const Eigen::VectorXd& M, RowSpMat& L) noexcept {
    const auto a_values = A.valuePtr();
    auto normal_values = ScalarVector::MapAligned(L.valuePtr(), L.nonZeros()); // writable values of L
    normal_values.setZero();
    auto A_values = ScalarVector::MapAligned(A.valuePtr(), A.nonZeros());
    auto left_values = A_values(left_entry_indices).eval();
    auto M_values = M(contribution_rows).eval();
    auto right_values = A_values(right_entry_indices).eval();
    auto product = (left_values.array() * M_values.array() * right_values.array()).eval();

    for (auto&& [idx, val] : std::views::zip(normal_entry_indices, product)) {
        normal_values[idx] += val;
    }
}

void assign_rhs(const VMat4& W, const VMat4& R, Eigen::VectorXd& rhs) noexcept {
    Eigen::Index idx = 0;
    for (Eigen::Index i = 0; i < W.rows(); ++i) {
        rhs.segment(idx, 4).reshaped(2, 2) = W.row(i).reshaped<Eigen::RowMajor>(2, 2) * R.row(i).reshaped(2, 2);
        idx += 4;
    }
}

/*
 * Compute y = A^T * (diag(M) * b)
 */
void multiply_normal_rhs(const RowSpMat& A, const Eigen::VectorXd& M, const Eigen::VectorXd& b, Eigen::VectorXd& rhs)
    noexcept {
    const auto outer_ptr = A.outerIndexPtr();   // row pointers
    const auto inner_ptr = A.innerIndexPtr();   // column indices
    const auto values_ptr = A.valuePtr();       // nonzero values

    rhs.setZero();
    for (Eigen::Index row = 0; row < A.outerSize(); ++row) {
        const auto start = outer_ptr[row];
        const auto end = outer_ptr[row + 1];
        const auto weighted_rhs = M[row] * b[row];
        for (RowSpMat::StorageIndex it = start; it < end; it++) {
            rhs[inner_ptr[it]] += values_ptr[it] * weighted_rhs;
        }
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
FlattenSurface::FlattenSurface(VMat &&V, FMat &&F, VMat2&& uv, const std::size_t n_bnd_points) noexcept : V(std::move(V)), F(std::move(F)), uv(std::move(uv)), n_bnd_points(n_bnd_points) {
}

FlattenSurface::FlattenSurface(VMat &&V, FMat &&F, const std::size_t n_boundary_points) noexcept : V(std::move(V)), F(std::move(F)), n_bnd_points(0) {
    uv.resize(this->V.rows(), 2);
    IVec bnd = IVec::LinSpaced(n_boundary_points, 1, n_boundary_points);
    bnd[n_boundary_points - 1] = 0;
    auto L = (this->V.topRows(n_boundary_points) - this->V(bnd, Eigen::placeholders::all)).rowwise().norm().eval();
    const auto l = L.sum();
    const auto r = l / (2 * M_PI);
    L /= l;
    double theta = 0;
    for (Eigen::Index i = 0; i < n_boundary_points; ++i) {
        const auto temp = L(i);
        L(i) = theta;
        theta += temp;
    }
    L *= 2 * M_PI;

    auto uv_slice = uv.topRows(n_boundary_points);
    uv_slice.col(0) = L.array().cos() * r;
    uv_slice.col(1) = L.array().sin() * r;

    igl::harmonic(this->V, this->F, IVec::LinSpaced(n_boundary_points, 0, n_boundary_points - 1), uv_slice, 1, uv);
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
    this->N = N;
    DA /= DA.sum();

    // Compute per-face gradient dot-products for a barycentric basis vector B and
    // write the scalar results into a dense buffer matching the IFV/G ordering.
    auto assemble_gradient_values = [&](const auto& B, double* values) noexcept {
        const auto outer_indices = IFV.outerIndexPtr();
        for (Eigen::Index i = 0; i < IFV.outerSize(); i++) {
            const auto start = outer_indices[i];
            const auto end = outer_indices[i + 1];
            const auto size = end - start;
            const auto I = Eigen::Matrix<decltype(IFV)::Scalar, Eigen::Dynamic, 1>::Map(IFV.valuePtr() + start, size);
            Eigen::VectorXd::Map(values + start, size) = (G(I, Eigen::placeholders::all) * B.row(i).transpose());
        }
    };

    B1 = B1.array() / B1.rowwise().norm().replicate<1, 3>().array();
    B2 = B2.array() / B2.rowwise().norm().replicate<1, 3>().array();
    RowSpMat DX;
    RowSpMat DY;
    {
        std::vector<double> temp(IFV.nonZeros());
        assemble_gradient_values(B1, temp.data());
        DX = RowSpMat::Map(IFV.rows(), IFV.cols(), IFV.nonZeros(), IFV.outerIndexPtr(), IFV.innerIndexPtr(), temp.data());
        assemble_gradient_values(B2, temp.data());
        DY = RowSpMat::Map(IFV.rows(), IFV.cols(), IFV.nonZeros(), IFV.outerIndexPtr(), IFV.innerIndexPtr(), temp.data());
    }

    VMat4 J(B1.rows(), 4);
    VMat4 R(B1.rows(), 4);
    VMat4 W(B1.rows(), 4);

    Eigen::Matrix2d ji, ri, ti, ui, vi;
    Eigen::Vector2d si, swi;
    RowSpMat A;
    NormalMatrixCache normal_cache;
    RowSpMat L;
    std::vector<Eigen::Index> skip_indices;
    Eigen::VectorXd boundary_rhs(B1.rows() * 4);
    Eigen::VectorXd rhs(B1.rows() * 4);
    Eigen::VectorXd real_rhs(n_free * 2);
    const auto DDA = DA.replicate<1, 4>().reshaped<Eigen::RowMajor>().eval();
    VMat2 new_uv = uv;
    double energy = 0.0;

    const auto update_jacobian = [&J, &DX, &DY](const VMat2& uv) noexcept {
        J.leftCols(2) = DX * uv;
        J.rightCols(2) = DY * uv;
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

        if (L.size() == 0) {
            L = normal_cache.build(A);
        }
        normal_cache.assign(A, DDA, L);
        multiply_normal_rhs(A, DDA, rhs, real_rhs);

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
        new_uv.bottomRows(V.rows() - n_bnd_points).reshaped() = solver.compute(L).solve(real_rhs);
        energy = flip_avoiding_line_search(F, uv, new_uv, compute_energy, energy);
        std::cout << "Energy: " << energy << std::endl;
        uv = new_uv;
    }
    write_uv(fmt::format("uv_new.obj"), new_uv, F);
}
