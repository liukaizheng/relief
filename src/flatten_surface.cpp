#include "flatten_surface.h"
#include <igl/harmonic.h>
#include <igl/grad.h>
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

template<typename DerivedV, typename DerivedF>
auto grad_tri(const Eigen::MatrixBase<DerivedV>& V, const Eigen::MatrixBase<DerivedF>& F) noexcept {
    const auto n_faces = F.rows();
    const auto n_verties = V.rows();
    auto V1 = (V(F.col(2), Eigen::placeholders::all) - V(F.col(1), Eigen::placeholders::all)).eval();
    auto V2 = V(F.col(0), Eigen::placeholders::all) - V(F.col(2), Eigen::placeholders::all);
    using Mat = Eigen::Matrix<typename DerivedV::Scalar, Eigen::Dynamic, 3>;
    Mat N(n_faces, 3);
    cross(V1, V2, N);
    auto DA = N.rowwise().norm().eval(); // double area
    auto DA_M = DA.replicate(1, 3).array();
    N = N.array() / DA_M;

    Mat PV1(n_faces, 3);
    cross(N, V1, PV1);
    PV1 = PV1.array() / DA_M;

    Mat PV2(n_faces, 3);
    cross(N, V2, PV2);
    PV2 = PV2.array() / DA_M;

    // auto V3 = (V(F.col(1), Eigen::placeholders::all) - V(F.col(0), Eigen::placeholders::all)).eval();
    // Mat PV3(n_faces, 3);
    // cross(N, V3, PV3);
    // PV3 = PV3.array() / DA_M;

    Eigen::SparseMatrix<Eigen::Index> IFV(n_faces, n_verties);
    std::vector<Eigen::Triplet<Eigen::Index>> triplets;
    Eigen::Index idx = 0;
    for (Eigen::Index i = 0; i < n_faces; i++) {
        triplets.emplace_back(i, F(i, 0), idx++);
        triplets.emplace_back(i, F(i, 1), idx++);
        triplets.emplace_back(i, F(i, 2), idx++);
    }
    IFV.setFromTriplets(triplets.begin(), triplets.end());
    IFV.makeCompressed();

    IVec I = IVec::LinSpaced(n_faces, 0, n_faces * 3 - 3);
    Mat G(n_faces * 3, 3);
    G(I, Eigen::placeholders::all) = PV1;
    G(I.array() + 1, Eigen::placeholders::all) = PV2;
    G(I.array() + 2, Eigen::placeholders::all) = -PV1 - PV2;


    // Eigen::SparseMatrix<typename DerivedV::Scalar> G(n_faces * 3, n_verties);
    // std::vector<Eigen::Triplet<typename DerivedV::Scalar>> triplets;
    // for (Eigen::Index i = 0; i < n_faces; ++i) {
    //     const auto idx = i * 3;
    //     for (Eigen::Index j = 0; j < 3; j++) {
    //         triplets.emplace_back(idx + j, F(i, 0), PV1(i, j));
    //         triplets.emplace_back(idx + j, F(i, 2), -PV1(i, j));
    //         triplets.emplace_back(idx + j, F(i, 1), PV2(i, j));
    //         triplets.emplace_back(idx + j, F(i, 2), -PV2(i, j));

    //     }

    // }
    // G.setFromTriplets(triplets.begin(), triplets.end());
    return std::make_tuple(std::move(V1), std::move(PV1), std::move(N), std::move(DA), std::move(IFV), std::move(G));
}

auto build_matrix_A(const RowSpMat& DX, const Eigen::Index n_fixed) noexcept {
    using IVec = Eigen::Matrix<RowSpMat::StorageIndex, 1, Eigen::Dynamic, Eigen::RowMajor>;
    using Mat4 = Eigen::Matrix<RowSpMat::StorageIndex, 4, Eigen::Dynamic, Eigen::RowMajor>;
    const auto n_free = DX.innerSize() - n_fixed;
    std::vector<RowSpMat::StorageIndex> outer_indices(DX.outerSize() * 4 + 1);
    std::vector<RowSpMat::StorageIndex> inner_indices(DX.nonZeros() * 8);
    outer_indices[0] = 0;
    const auto dx_outer_indices = DX.outerIndexPtr();
    const auto dx_inner_indices = DX.innerIndexPtr();
    std::size_t idx = 0;
    RowSpMat::StorageIndex pos = 0;
    std::vector<Eigen::Index> skip_indices(DX.outerSize());
    for (Eigen::Index i = 0; i < DX.outerSize(); i++) {
        auto dx_start = dx_outer_indices[i];
        const auto dx_end = dx_outer_indices[i + 1];
        const auto local_dx_inner_indices = dx_inner_indices + dx_start;
        skip_indices[i] = std::find_if(local_dx_inner_indices, dx_inner_indices + dx_end, [n_fixed](auto x) noexcept { return x >=  n_fixed; }) - local_dx_inner_indices;
        dx_start += skip_indices[i];
        const auto size = dx_end - dx_start;
        const auto double_size = size * 2;
        const auto local_inner_indices = IVec::Map(dx_inner_indices + dx_start, size).array();
        auto block = Mat4::Map(inner_indices.data() + pos, 4, double_size);
        block.leftCols(size) = (local_inner_indices - n_fixed).replicate<4, 1>();
        block.rightCols(size) = (local_inner_indices + (n_free - n_fixed)).replicate<4, 1>();

        IVec::Map(outer_indices.data() + idx + 1, 4) = IVec::LinSpaced(4, pos + double_size, pos + double_size * 4);
        pos += double_size * 4;
        idx += 4;
    }
    inner_indices.resize(pos);
    RowSpMat A = RowSpMat::Map(DX.rows() * 4, n_free * 2, inner_indices.size(), outer_indices.data(), inner_indices.data(), std::vector<double>(inner_indices.size()).data());
    return std::make_pair(std::move(A), std::move(skip_indices));
}

/*
 * WJ = | w11 w12 | * | DxU DyU | = |w11 DxU + w12 DxV, w11 DyU + w12 DyV|
 *      | w21 w22 |   | DxV DyV | = |w21 DxU + w22 DxV, w21 DyV + w22 DyV|
 */
void assign_matrix_A(const RowSpMat& DX, const RowSpMat& DY, const VMat4& W, const std::vector<Eigen::Index>& skip_indices, const VMat2& uv, RowSpMat& A, Eigen::VectorXd& boundary_contribution) noexcept {
    using RowVecd = Eigen::Matrix<RowSpMat::Scalar, 1, Eigen::Dynamic>;
    using IVec = Eigen::Matrix<RowSpMat::StorageIndex, 1, Eigen::Dynamic, Eigen::RowMajor>;
    using Mat4 = Eigen::Matrix<RowSpMat::Scalar, 4, Eigen::Dynamic, Eigen::RowMajor>;
    const auto outer_indices = A.outerIndexPtr();
    const auto inner_indices = A.innerIndexPtr();
    boundary_contribution.setZero();
    for (Eigen::Index i = 0; i < DX.rows(); i++) {
        const auto idx = i * 4;
        const auto start = outer_indices[idx];
        const auto end = outer_indices[idx + 1];
        const auto size = end - start;
        const auto half_size = size / 2;
        const auto total_size = half_size + skip_indices[i];
        const auto dx = RowVecd::Map(DX.valuePtr() + DX.outerIndexPtr()[i], total_size);
        const auto dy = RowVecd::Map(DY.valuePtr() + DY.outerIndexPtr()[i], total_size);
        const auto wi = W.row(i).transpose().array().replicate(1, total_size);
        auto block = Mat4::Map(A.valuePtr() + start, 4, size);
        const auto dx_block = (dx.replicate<4, 1>().array() * wi).eval();
        const auto dy_block = (dy.replicate<4, 1>().array() * wi).eval();
        block.topRows(2).reshaped<Eigen::RowMajor>(4, half_size) = dx_block.rightCols(half_size);
        block.bottomRows(2).reshaped<Eigen::RowMajor>(4, half_size) = dy_block.rightCols(half_size);

        auto I = IVec::Map(DX.innerIndexPtr() + (DX.outerIndexPtr()[i]), skip_indices[i]);
        auto flatten_uv = uv(I, Eigen::placeholders::all).reshaped(skip_indices[i] * 2, 1);

        auto contribution_slice = boundary_contribution.segment(idx, 4);

        contribution_slice.segment(0, 2) += dx_block.leftCols(skip_indices[i]).reshaped<Eigen::RowMajor>(2, skip_indices[i] * 2).matrix() * flatten_uv;
        contribution_slice.segment(2, 2) += dy_block.leftCols(skip_indices[i]).reshaped<Eigen::RowMajor>(2, skip_indices[i] * 2).matrix() * flatten_uv;
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

struct AtAData {
    using StorageIndex = RowSpMat::StorageIndex;
    using IndexVector = RowSpMat::IndexVector;
    using ScalarVector = RowSpMat::ScalarVector;

    AtAData() noexcept = default;
    auto build(const RowSpMat& At) noexcept;
    void assign(const RowSpMat& At, const Eigen::VectorXd& M, RowSpMat& L) noexcept;

    std::vector<StorageIndex> entry_outer_indices;
    std::vector<StorageIndex> row_indices;
    std::vector<StorageIndex> col_indices;
    std::vector<StorageIndex> top_right_inner_indices;
    std::vector<StorageIndex> bottom_left_inner_indices;
    std::vector<StorageIndex> bottom_left_to_top_right_map;
};

auto AtAData::build(const RowSpMat& At) noexcept {
    const auto outer_indices = At.outerIndexPtr();
    const auto inner_indices = At.innerIndexPtr();
    const auto values = At.valuePtr();
    constexpr RowSpMat::StorageIndex INVALID = -1;

    entry_outer_indices.emplace_back(0);
    std::vector<StorageIndex> new_outer_indices{0};
    std::vector<StorageIndex> new_inner_indices;
    std::vector<StorageIndex> col_start(At.outerSize(), INVALID);
    std::vector<StorageIndex> col_end(At.outerSize(), INVALID);
    std::vector<StorageIndex> col_next;
    std::vector<StorageIndex> inner_row_indices;
    for (Eigen::Index r = 0; r < At.outerSize(); r++) {
        // setup left part
        auto curr = col_start[r];
        while(curr != INVALID) {
            bottom_left_inner_indices.emplace_back(new_inner_indices.size());
            bottom_left_to_top_right_map.emplace_back(curr);
            new_inner_indices.emplace_back(inner_row_indices[curr]);
            inner_row_indices.emplace_back(INVALID);
            col_next.emplace_back(INVALID);
            curr = col_next[curr];
        }
        for (Eigen::Index c = r; c < At.outerSize(); c++) {
            auto ri = outer_indices[c];
            for (StorageIndex li = outer_indices[r]; li < outer_indices[r + 1]; li++) {
                const auto left_inner_idx = inner_indices[li];
                while(ri < outer_indices[c + 1] && inner_indices[ri] < left_inner_idx) {
                    ri++;
                }

                if (ri == outer_indices[c + 1]) {
                    break;
                }

                if (inner_indices[ri] == left_inner_idx) {
                    row_indices.emplace_back(li);
                    col_indices.emplace_back(ri);
                    ri++;
                    if (ri == outer_indices[c + 1]) {
                        break;
                    }
                }
            }
            if (row_indices.size() != entry_outer_indices.back()) {
                entry_outer_indices.emplace_back(row_indices.size());
                const StorageIndex new_inner_idx = new_inner_indices.size();
                col_next.emplace_back(INVALID);
                inner_row_indices.emplace_back(r);
                new_inner_indices.emplace_back(c);
                top_right_inner_indices.emplace_back(new_inner_idx);
                if (col_start[c] == INVALID) {
                    col_start[c] = new_inner_idx;
                    col_end[c] = new_inner_idx;
                } else {
                    col_next[col_end[c]] = new_inner_idx;
                    col_end[c] = new_inner_idx;
                }
            }
        }
        new_outer_indices.emplace_back(new_inner_indices.size());
    }

    RowSpMat L(At.outerSize(), At.outerSize());
    L.resizeNonZeros(new_inner_indices.size());
    IndexVector::Map(L.outerIndexPtr(), new_outer_indices.size()) = IndexVector::Map(new_outer_indices.data(), new_outer_indices.size());
    IndexVector::Map(L.innerIndexPtr(), new_inner_indices.size()) = IndexVector::Map(new_inner_indices.data(), new_inner_indices.size());
    return L;
}

void AtAData::assign(const RowSpMat& At, const Eigen::VectorXd& M, RowSpMat& L) noexcept {
    const auto values_arr = ScalarVector::Map(At.valuePtr(), At.nonZeros());
    const auto block = (values_arr(row_indices).array()
        * M(IndexVector::Map(At.innerIndexPtr(), At.nonZeros())(row_indices)).array()
        * values_arr(col_indices).array()).eval();
    auto new_values_arr = ScalarVector::Map(L.valuePtr(), L.nonZeros());
    for (std::size_t i = 0; i < top_right_inner_indices.size(); i++) {
        new_values_arr[top_right_inner_indices[i]] = block.segment(entry_outer_indices[i], entry_outer_indices[i + 1] - entry_outer_indices[i]).sum();
    }

    new_values_arr(bottom_left_inner_indices) = new_values_arr(bottom_left_to_top_right_map);
}

void assign_rhs(const VMat4& W, const VMat4& R, Eigen::VectorXd& rhs) noexcept {
    Eigen::Index idx = 0;
    for (Eigen::Index i = 0; i < W.rows(); ++i) {
        rhs.segment(idx, 4).reshaped(2, 2) = W.row(i).reshaped<Eigen::RowMajor>(2, 2) * R.row(i).reshaped(2, 2);
        idx += 4;
    }
}

/*
 * A * M * b
 */
void get_At_M_b(const RowSpMat& At, const Eigen::VectorXd& M, const Eigen::VectorXd& b, Eigen::VectorXd& rhs) noexcept {
    const auto outer_indices = At.outerIndexPtr();
    const auto inner_indices = At.innerIndexPtr();
    const auto values = At.valuePtr();

    for (Eigen::Index i = 0; i < At.outerSize(); ++i) {
        const auto start = outer_indices[i];
        const auto end = outer_indices[i + 1];
        const auto size = end - start;
        const auto I =RowSpMat::IndexVector::Map(inner_indices + start, size);
        rhs[i] = (RowSpMat::ScalarVector::Map(values + start, size).array()
            * M(I).array())
            .matrix()
            .transpose()
            * b(I);
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
    VMat2 uv = this->uv;
    auto [B1, B2, N, DA, IFV, G] = grad_tri(this->V, this->F);
    DA /= DA.sum();

    auto get_derivate = [&](const auto& B, double* values) noexcept {
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
        get_derivate(B1, temp.data());
        DX = SpMat::Map(IFV.rows(), IFV.cols(), IFV.nonZeros(), IFV.outerIndexPtr(), IFV.innerIndexPtr(), temp.data());
        get_derivate(B2, temp.data());
        DY = SpMat::Map(IFV.rows(), IFV.cols(), IFV.nonZeros(), IFV.outerIndexPtr(), IFV.innerIndexPtr(), temp.data());
    }

    VMat4 J(B1.rows(), 4);
    VMat4 R(B1.rows(), 4);
    VMat4 W(B1.rows(), 4);

    Eigen::Matrix2d ji, ri, ti, ui, vi;
    Eigen::Vector2d si, swi;
    RowSpMat A;
    RowSpMat At;
    AtAData AtA_data;
    RowSpMat L;
    std::vector<Eigen::Index> skip_indices;
    RowSpMat::IndexVector A_to_At_indices;
    Eigen::VectorXd boundary_contribution(B1.rows() * 4);
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
            std::tie(A, skip_indices) = build_matrix_A(DX, n_bnd_points);
        }
        assign_matrix_A(DX, DY, W, skip_indices, uv, A, boundary_contribution);
        assign_rhs(W, R, rhs);
        rhs -= boundary_contribution;

        if (A_to_At_indices.size() == 0) {
            std::tie(At, A_to_At_indices) = transpose_sparse_matrix(A);
        } else {
            RowSpMat::ScalarVector::Map(At.valuePtr(), A.nonZeros())(A_to_At_indices) = RowSpMat::ScalarVector::Map(A.valuePtr(), A.nonZeros());
        }

        if (L.size() == 0) {
            L = AtA_data.build(At);
        }
        AtA_data.assign(At, DDA, L);
        get_At_M_b(At, DDA, rhs, real_rhs);

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
        new_uv.bottomRows(V.rows() - n_bnd_points).reshaped() = solver.compute(L).solve(real_rhs);
        energy = flip_avoiding_line_search(F, uv, new_uv, compute_energy, energy);
        uv = new_uv;
        std::cout << "Energy: " << energy << std::endl;

        write_uv("uv_new.obj", new_uv, F);
    }
}
