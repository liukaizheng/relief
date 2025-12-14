#pragma once

#include <eigen_alias.h>

#include <vector>

class FlattenSurface {
public:
    FlattenSurface(VMat &&V, FMat &&F, const std::vector<std::size_t>& segment_offsets) noexcept;
    FlattenSurface(VMat &&V, FMat &&F, VMat2& uv, const std::size_t n_bnd_points) noexcept;
    FlattenSurface(VMat &&V, FMat &&F, const std::size_t n_bnd_points) noexcept;

    void slim_solve(const std::size_t n_iterations);
public:

    VMat V;
    FMat F;
    VMat N;

    VMat2 uv;
    const std::size_t n_bnd_points;
    double mean_edge_length;
};
