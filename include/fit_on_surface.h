#pragma once

#include <array>
#include <cstddef>
#include <gpf/ids.hpp>
#include <vector>

#include <gpf/detail.hpp>
#include <gpf/manifold_mesh.hpp>
#include <gpf/mesh.hpp>

namespace fit_on_surface {
struct VertexProp {
    std::array<double, 3> pt;
};

struct FaceProp {
    gpf::FaceId parent;
    std::size_t polygon_id = gpf::kInvalidIndex;
};

using Mesh = gpf::ManifoldMesh<VertexProp, gpf::Empty, gpf::Empty, FaceProp>;

void fit_polygon_on_surface(
    fit_on_surface::Mesh& mesh,
    const std::vector<std::array<double, 2>>& polygon_points,
    const std::vector<std::vector<std::vector<std::size_t>>>& polygons,
    const std::array<double, 3>& surface_point,
    const gpf::FaceId face_idx,
    const std::array<double, 3>& direction
);
}

namespace image_relief {
struct VertexProp {
    std::array<double, 3> pt;
};

struct FaceProp {
    gpf::FaceId parent;
};

struct GridFaceIndex {
    gpf::FaceId combined_face;
    std::size_t grid_face_index;
    std::size_t grid_row;
    std::size_t grid_column;
    std::size_t triangle_index;
};

using Mesh = gpf::ManifoldMesh<VertexProp, gpf::Empty, gpf::Empty, FaceProp>;
std::vector<GridFaceIndex> make_relief_on_surface(
    Mesh& mesh,
    const std::span<const double> heights,
    const std::size_t width,
    const gpf::FaceId face_idx,
    const std::array<double, 3>& start_pt,
    const std::array<double, 3>& direction
);
}
