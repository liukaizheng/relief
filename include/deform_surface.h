#pragma once

#include <eigen_alias.h>

class DeformSurface {
public:
    DeformSurface(VMat&& vertices, FMat&& faces, const Eigen::Index n_fixed_vertices);
    void deform(const std::size_t num_iterations, VMat& new_vertices);

private:
    VMat vertices;
    FMat faces;
    Eigen::Index n_fixed_vertices;
};
