#include "flatten_surface.h"
#include <igl/doublearea.h>
#include <igl/harmonic.h>
#include <igl/local_basis.h>
#include <igl/grad.h>
#include <igl/polar_svd.h>
#include <igl/sparse_cached.h>
#include <igl/flip_avoiding_line_search.h>

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

template<typename S>
void print_sparse_mat(const Eigen::SparseMatrix<S>& mat, const std::size_t n_max) {
    std::size_t idx = 0;
    for (Eigen::Index i = 0; i < mat.outerSize(); i++) {
        for (typename Eigen::SparseMatrix<S>::InnerIterator it(mat, i); it; ++it) {
            std::cout << "(" << it.row() << ", " << it.col() << ") = " << it.value() << std::endl;
            if (++idx > n_max) {
                return;
            }
        }
    }
}

template<typename DerivedV1, typename DerivedV2, typename DerivedN>
void cross(const Eigen::MatrixBase<DerivedV1>& V1, const Eigen::MatrixBase<DerivedV2>& V2, Eigen::MatrixBase<DerivedN>& N) {
    N.col(0) = V1.col(1).cwiseProduct(V2.col(2)) - V1.col(2).cwiseProduct(V2.col(1));
    N.col(1) = V1.col(2).cwiseProduct(V2.col(0)) - V1.col(0).cwiseProduct(V2.col(2));
    N.col(2) = V1.col(0).cwiseProduct(V2.col(1)) - V1.col(1).cwiseProduct(V2.col(0));
}

template<typename DerivedV, typename DerivedF>
auto grad_tri(const Eigen::MatrixBase<DerivedV>& V, const Eigen::MatrixBase<DerivedF>& F) {
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
    pre_calc();
    {
        auto [B1, B2, N, DA, IFV, G] = grad_tri(this->V, this->F);

        auto get_derivate = [&](const auto& B, auto& D) noexcept {
            auto values = D.valuePtr();
            const auto outer_indices = D.outerIndexPtr();
            const auto inner_indices = D.innerIndexPtr();
            for (Eigen::Index i = 0; i < D.outerSize(); i++) {
                const auto start = outer_indices[i];
                const auto end = outer_indices[i + 1];
                const auto size = end - start;
                Eigen::Matrix<decltype(IFV)::Scalar, Eigen::Dynamic, 1>::ConstMapType I1(IFV.valuePtr() + start, size);
                typename Eigen::Matrix<std::remove_pointer_t<decltype(inner_indices)>, Eigen::Dynamic, 1>::ConstMapType I2(inner_indices + start, size);
                Eigen::VectorXd::MapType(values + start, size) = (G(I1, Eigen::placeholders::all).array() * B(I2, Eigen::placeholders::all).array()).rowwise().sum();
            }
        };

        SpMat DX = SpMat::Map(IFV.rows(), IFV.cols(), IFV.nonZeros(), IFV.outerIndexPtr(), IFV.innerIndexPtr(), std::vector<double>(IFV.nonZeros()).data());
        SpMat DY = DX;
        B1 = B1.array() / B1.rowwise().norm().replicate<1, 3>().array();
        B2 = B2.array() / B2.rowwise().norm().replicate<1, 3>().array();
        get_derivate(B1, DX);
        get_derivate(B2, DY);

        VMat4 J(B1.rows(), 4);
        VMat4 R(B1.rows(), 4);
        VMat4 W(B1.rows(), 4);
        J.col(0) = DX * uv.col(0);
        J.col(1) = DY * uv.col(0);
        J.col(2) = DX * uv.col(1);
        J.col(3) = DY * uv.col(1);

        Eigen::Matrix2d ji, ri, ti, ui, vi;
        Eigen::Vector2d si, swi;
        for (Eigen::Index i = 0; i < J.rows(); ++i) {
            igl::polar_svd(J.row(i).reshaped<Eigen::RowMajor>(Eigen::fix<2>, Eigen::fix<2>), ri, ti, ui, si, vi);
            const auto si_arr = si.array();
            swi = ((1.0 + si_arr) / si_arr * (1.0 + si_arr.square())).sqrt() / si_arr;
            R.row(i).reshaped<Eigen::RowMajor>(Eigen::fix<2>, Eigen::fix<2>) = ri;
            W.row(i).reshaped<Eigen::RowMajor>(Eigen::fix<2>, Eigen::fix<2>) = ui * swi.asDiagonal() * ui.transpose();
        }
    }
    Eigen::MatrixXi F_temp = F.cast<int>();
    for (std::size_t i = 0; i < n_iterations; ++i) {
        update_weights_and_closest_rotations();
        auto new_uv = solve_weighted_arap();
        std::function<double(Eigen::MatrixXd&)> compute_energy_fn = [this](Eigen::MatrixXd &aaa){ return this->compute_energy(aaa); };
        Eigen::MatrixXd uv_temp = uv;
        this->energy = igl::flip_avoiding_line_search(
            F_temp,
            uv_temp,
            new_uv,
            compute_energy_fn,
            this->energy * this->mesh_area
        ) / mesh_area;
        uv = uv_temp;
    }
}

void FlattenSurface::pre_calc() {
    igl::doublearea(V, F, M);
    mesh_area = M.sum() * 0.5;
    Eigen::MatrixXd F1, F2, F3;
    igl::local_basis(V, F, F1, F2, F3);
    Eigen::SparseMatrix<double> G;
    igl::grad(V, F, G);
    const auto n_faces = F.rows();

    auto face_proj = [n_faces](Eigen::MatrixXd& F){
        std::vector<Eigen::Triplet<double> >IJV;
        for(Eigen::Index i=0; i< F.rows(); i++) {
            IJV.push_back(Eigen::Triplet<double>(i, i, F(i, 0)));
            IJV.push_back(Eigen::Triplet<double>(i, i+ n_faces, F(i, 1)));
            IJV.push_back(Eigen::Triplet<double>(i, i+ 2 * n_faces, F(i, 2)));
        }
        Eigen::SparseMatrix<double> P(n_faces, 3 * n_faces);
        P.setFromTriplets(IJV.begin(), IJV.end());
        return P;
    };

    Dx = face_proj(F1) * G;
    Dy = face_proj(F2) * G;

    W.resize(n_faces, 4);
    Dx.makeCompressed();
    Dy.makeCompressed();
    Dz.makeCompressed();
    Ri.resize(n_faces, 4);
    Ji.resize(n_faces, 4);
    WGL_M.resize(n_faces * 4);
    for (Eigen::Index i = 0; i < 4; i++) {
        for (Eigen::Index j = 0; j < n_faces; j++) {
            WGL_M(i * n_faces + j) = M(j);
        }
    }
    this->energy = compute_energy(uv) / mesh_area;
}

void FlattenSurface::compute_jacobians(const VMat2& uv_new) {
    Ji.col(0) = Dx * uv_new.col(0);
    Ji.col(1) = Dy * uv_new.col(0);
    Ji.col(2) = Dx * uv_new.col(1);
    Ji.col(3) = Dy * uv_new.col(1);
}

void FlattenSurface::update_weights_and_closest_rotations() {
    compute_jacobians(uv);
    for (Eigen::Index i = 0; i < Ji.rows(); ++i) {
        using Mat2 = Eigen::Matrix2d;
        using RMat2 = Eigen::Matrix<double, 2, 2, Eigen::RowMajor>;
        using Vec2 = Eigen::Vector2d;
        Mat2 ji, ri, ti, ui, vi;
        Vec2 sing;
        Vec2 closest_sing_vec;
        RMat2 mat_W;
        Vec2 m_sing_new;
        double s1, s2;

        ji(0, 0) = Ji(i, 0);
        ji(0, 1) = Ji(i, 1);
        ji(1, 0) = Ji(i, 2);
        ji(1, 1) = Ji(i, 3);

        igl::polar_svd(ji, ri, ti, ui, sing, vi);

        s1 = sing(0);
        s2 = sing(1);

        double s1_g = 2 * (s1 - pow(s1, -3));
        double s2_g = 2 * (s2 - pow(s2, -3));
        // Limit is 4 if s==1 according to Equation (32) in Rabinovich et al.
        // [2017]
        m_sing_new <<
            (s1==1?4:sqrt(s1_g / (2 * (s1 - 1)))),
            (s2==1?4:sqrt(s2_g / (2 * (s2 - 1))));
        constexpr double eps = 1e-8;

        if (std::abs(s1 - 1) < eps) m_sing_new(0) = 1;
        if (std::abs(s2 - 1) < eps) m_sing_new(1) = 1;
        mat_W = ui * m_sing_new.asDiagonal() * ui.transpose();

        W.row(i) = Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>>(mat_W.data());
        // 2) Update local step (doesn't have to be a rotation, for instance in case of conformal energy)
        Ri.row(i) = Eigen::Map<Eigen::Matrix<double, 1,4,Eigen::RowMajor>>(ri.data());
    }
}

VMat2 FlattenSurface::solve_weighted_arap() {
    auto triplets = build_A();
    const auto v_n = V.rows();
    const auto f_n = F.rows();
    if (A.rows() == 0) {
        A = Eigen::SparseMatrix<double>(4 * f_n, 2 * v_n);
        igl::sparse_cached_precompute(triplets, A_data, A);
    } else {
        igl::sparse_cached(triplets, A_data, A);
    }

    Eigen::VectorXd bnd_uv_contribution = (A.leftCols(n_bnd_points) * uv.col(0).topRows(n_bnd_points)) + (A.middleCols(v_n, n_bnd_points) * uv.col(1).topRows(n_bnd_points));
    // Eigen::SparseMatrix<double> id_m(A.cols(), A.cols());
    // id_m.setIdentity();
    // AtA_data.W = WGL_M;
    // if (AtA.rows() == 0) {
    //     igl::AtA_cached_precompute(A, AtA_data, AtA);
    // } else {
    //     igl::AtA_cached(A,AtA_data,AtA);

    // }
    //
    Eigen::SparseMatrix<double> A1(A.rows(), A.cols() - 2 * n_bnd_points);
    A1.leftCols(v_n - n_bnd_points) = A.middleCols(n_bnd_points, v_n - n_bnd_points);
    A1.rightCols(v_n - n_bnd_points) = A.middleCols(v_n + n_bnd_points, v_n - n_bnd_points);
    A1.makeCompressed();

    Eigen::SparseMatrix<double> A1t = A1.transpose();
    A1t.makeCompressed();

    Eigen::SparseMatrix<double> L = A1t * WGL_M.asDiagonal() * A1;
    // Eigen::SparseMatrix<double> L = AtA + proximal_p * id_m; //add also a proximal
    L.makeCompressed();

    Eigen::VectorXd f_rhs(4 * f_n);
    f_rhs.setZero();
    for (int i = 0; i < f_n; i++)
    {
        f_rhs(i + 0 * f_n) = W(i, 0) * Ri(i, 0) + W(i, 1) * Ri(i, 1);
        f_rhs(i + 1 * f_n) = W(i, 0) * Ri(i, 2) + W(i, 1) * Ri(i, 3);
        f_rhs(i + 2 * f_n) = W(i, 2) * Ri(i, 0) + W(i, 3) * Ri(i, 1);
        f_rhs(i + 3 * f_n) = W(i, 2) * Ri(i, 2) + W(i, 3) * Ri(i, 3);
    }
    f_rhs -= bnd_uv_contribution;
    rhs = A1t * WGL_M.asDiagonal() * f_rhs;

    // build_rhs();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
    Eigen::VectorXd X = solver.compute(L).solve(rhs);
    VMat2 new_uv = uv;
    new_uv.col(0).bottomRows(v_n - n_bnd_points) = X.topRows(v_n - n_bnd_points);
    new_uv.col(1).bottomRows(v_n - n_bnd_points) = X.bottomRows(v_n - n_bnd_points);

    return new_uv;
}

std::vector<Eigen::Triplet<double>> FlattenSurface::build_A() {
    const auto v_n = V.rows();
    const auto f_n = F.rows();

    std::vector<Eigen::Triplet<double>> IJV;
    IJV.reserve(4 * (Dx.outerSize() + Dy.outerSize()));

    /*A = [W11*Dx, W12*Dx;
          W11*Dy, W12*Dy;
          W21*Dx, W22*Dx;
          W21*Dy, W22*Dy];*/
    for (int k = 0; k < Dx.outerSize(); ++k)
    {
      for (Eigen::SparseMatrix<double>::InnerIterator it(Dx, k); it; ++it)
      {
        int dx_r = it.row();
        int dx_c = it.col();
        double val = it.value();

        IJV.push_back(Eigen::Triplet<double>(dx_r, dx_c, val * W(dx_r, 0)));
        IJV.push_back(Eigen::Triplet<double>(dx_r, v_n + dx_c, val * W(dx_r, 1)));

        IJV.push_back(Eigen::Triplet<double>(2 * f_n + dx_r, dx_c, val * W(dx_r, 2)));
        IJV.push_back(Eigen::Triplet<double>(2 * f_n + dx_r, v_n + dx_c, val * W(dx_r, 3)));
      }
    }

    for (int k = 0; k < Dy.outerSize(); ++k)
    {
      for (Eigen::SparseMatrix<double>::InnerIterator it(Dy, k); it; ++it)
      {
        int dy_r = it.row();
        int dy_c = it.col();
        double val = it.value();

        IJV.push_back(Eigen::Triplet<double>(f_n + dy_r, dy_c, val * W(dy_r, 0)));
        IJV.push_back(Eigen::Triplet<double>(f_n + dy_r, v_n + dy_c, val * W(dy_r, 1)));

        IJV.push_back(Eigen::Triplet<double>(3 * f_n + dy_r, dy_c, val * W(dy_r, 2)));
        IJV.push_back(Eigen::Triplet<double>(3 * f_n + dy_r, v_n + dy_c, val * W(dy_r, 3)));
      }
    }
    return IJV;
}

void FlattenSurface::build_rhs() {
    const auto v_n = V.rows();
    const auto f_n = F.rows();
    Eigen::VectorXd f_rhs(4 * f_n);
    f_rhs.setZero();
    for (int i = 0; i < f_n; i++)
    {
        f_rhs(i + 0 * f_n) = W(i, 0) * Ri(i, 0) + W(i, 1) * Ri(i, 1);
        f_rhs(i + 1 * f_n) = W(i, 0) * Ri(i, 2) + W(i, 1) * Ri(i, 3);
        f_rhs(i + 2 * f_n) = W(i, 2) * Ri(i, 0) + W(i, 3) * Ri(i, 1);
        f_rhs(i + 3 * f_n) = W(i, 2) * Ri(i, 2) + W(i, 3) * Ri(i, 3);
    }
    Eigen::VectorXd uv_flat(2 * v_n);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < v_n; j++) {
            uv_flat(v_n * i + j) = uv(j, i);
        }
    }

    rhs = (f_rhs.transpose() * WGL_M.asDiagonal() * A).transpose() + proximal_p * uv_flat;
}

double FlattenSurface::compute_energy(const VMat2& uv_new) {
    compute_jacobians(uv_new);
    double energy = 0;
    Eigen::Matrix<double, 2, 2> ji;
    for (int i = 0; i < Ji.rows(); i++)
    {
        ji(0, 0) = Ji(i, 0);
        ji(0, 1) = Ji(i, 1);
        ji(1, 0) = Ji(i, 2);
        ji(1, 1) = Ji(i, 3);

        typedef Eigen::Matrix<double, 2, 2> Mat2;
        typedef Eigen::Matrix<double, 2, 1> Vec2;
        Mat2 ri, ti, ui, vi;
        Vec2 sing;
        igl::polar_svd(ji, ri, ti, ui, sing, vi);
        double s1 = sing(0);
        double s2 = sing(1);
        energy += M(i) * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2));
    }
    return energy;
}
