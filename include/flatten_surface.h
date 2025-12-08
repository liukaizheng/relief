#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <igl/AtA_cached.h>
#include <vector>

using VMat = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using VMat2 = Eigen::Matrix<double, Eigen::Dynamic, 2>;
using VMat4 = Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>;
using MatXu = Eigen::Matrix<std::size_t, Eigen::Dynamic, Eigen::Dynamic>;
using FMat = Eigen::Matrix<std::size_t, Eigen::Dynamic, 3, Eigen::RowMajor>;
using EMat = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using IVec = Eigen::Matrix<std::size_t, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using RowSpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;

class FlattenSurface {
public:
    FlattenSurface(VMat &&V, FMat &&F, const std::vector<std::size_t>& segment_offsets) noexcept;

    void slim_solve(const std::size_t n_iterations);
public:

    void pre_calc();
    void compute_jacobians(const VMat2& uv_new);
    void update_weights_and_closest_rotations();
    double compute_energy(const VMat2& uv_new);
    VMat2 solve_weighted_arap();
    std::vector<class Eigen::Triplet<double>> build_A();
    void build_rhs();
    VMat V;
    FMat F;

    VMat2 uv;
    const std::size_t n_bnd_points;
    const double proximal_p = 0.0001;
    double mesh_area;
    double mean_edge_length;
    double energy;

    Eigen::VectorXd M;
    Eigen::VectorXd WGL_M;
    Eigen::MatrixXd Ji;
    Eigen::MatrixXd Ri;
    Eigen::MatrixXd W;
    Eigen::VectorXd rhs;
    Eigen::SparseMatrix<double> Dx,Dy,Dz;

    Eigen::SparseMatrix<double> A;
    Eigen::VectorXi A_data;
    Eigen::SparseMatrix<double> AtA;
    igl::AtA_cached_data AtA_data;

    // rember to delete
    RowSpMat A1_global;
    Eigen::VectorXd bnd_uv_contri;
};
