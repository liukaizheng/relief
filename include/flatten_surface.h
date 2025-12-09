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

    VMat V;
    FMat F;

    VMat2 uv;
    const std::size_t n_bnd_points;
    double mean_edge_length;
};
