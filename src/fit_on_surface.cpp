#include "fit_on_surface.h"
#include "eigen_alias.h"
#include "flatten_surface.h"

#include <gpf/exp_map.hpp>
#include <gpf/ids.hpp>
#include <gpf/mesh_flood_fill.hpp>
#include <gpf/mesh_property.hpp>
#include <gpf/project_polylines_on_mesh.hpp>
#include <gpf/triangulation.hpp>
#include <igl/flipped_triangles.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <predicates/predicates.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace ranges = std::ranges;
namespace views = std::views;

// Shared patch data and preparation options.

namespace relief::surface {
using fit_on_surface::SurfaceFailure;

inline constexpr double kSupportPadding = 1.2;
inline constexpr double kProjectionTolerance = 1e-3;
inline constexpr std::size_t kSlimMinIterations = 5;
inline constexpr std::size_t kSlimMaxIterations = 15;
inline constexpr std::size_t kSlimFixedVertices = 1;

namespace uv {
    struct VertexProp {
        std::array<double, 2> pt;
    };
    using Mesh = gpf::ManifoldMesh<VertexProp, gpf::Empty, gpf::Empty, fit_on_surface::FaceProp>;
}

struct UvPlacementFrame {
    Eigen::Vector2d origin;
    Eigen::Vector2d xaxis;
    Eigen::Vector2d yaxis;
};

/** A symmetric support rectangle about domain coordinate (0.5, 0.5).
 * Convexity covers every used polygon segment, not just the input vertices. */
struct Footprint {
    Eigen::Vector2d half_extent = Eigen::Vector2d::Constant(0.5);
};

struct PlacementInput {
    gpf::FaceId source_face;
    std::array<gpf::VertexId, 3> source_vertices;
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> source_positions;
    std::array<double, 3> surface_point;
    Eigen::Vector3d normal;
    Eigen::Vector3d tangent_direction;
    double direction_magnitude;
    std::array<double, 2> lengths;
    Footprint footprint;
};

// Initializers place boundary vertices first for SLIM's fixed-vertex prefix.
struct InitialPatch {
    VMat positions;
    FMat triangles;
    VMat2 coordinates;
    std::vector<std::size_t> boundary;
    std::vector<gpf::FaceId> surface_faces;
    std::vector<gpf::VertexId> uv_to_surface_vertex;
    std::size_t center = gpf::kInvalidIndex;
};

struct OptimizedPatch {
    InitialPatch initial;
    VMat2 coordinates;
};

struct PreparedSurfacePatch {
    uv::Mesh uv_mesh;
    std::vector<gpf::FaceId> surface_faces;
    std::vector<gpf::VertexId> uv_to_surface_vertex;
    UvPlacementFrame frame;
};

[[nodiscard]] OptimizedPatch optimize_patch(InitialPatch patch);
[[nodiscard]] std::expected<UvPlacementFrame, SurfaceFailure> fit_placement(
    UvPlacementFrame frame, const Footprint& footprint, const VMat2& coordinates,
    std::span<const std::size_t> boundary);
[[nodiscard]] std::expected<UvPlacementFrame, SurfaceFailure> exp_map_placement(
    const OptimizedPatch& patch, const PlacementInput& input, const std::array<std::size_t, 3>& source_uv_indices);

[[nodiscard]] std::expected<PreparedSurfacePatch, SurfaceFailure> prepare_surface_patch(
    fit_on_surface::Mesh& mesh, const PlacementInput& input);

[[nodiscard]] std::expected<PreparedSurfacePatch, SurfaceFailure> prepare_surface_patch(
    image_relief::Mesh& mesh, const PlacementInput& input);
}

// Derived exp-map caches on the caller's surface mesh.

namespace relief::surface {
// Recompute before every attempt: downstream projection/subdivision may have
// invalidated any of these properties since the previous preparation.
void initialize_exp_map_properties(auto& mesh)
{
    gpf::update_edge_lengths<3>(mesh);
    gpf::update_corner_angles(mesh);
    gpf::update_vertex_angle_sums(mesh);
    gpf::update_halfedge_signpost_angles(mesh);
    gpf::update_halfedge_vectors(mesh);
}
}

// Shared mesh helpers and diagnostic declarations.

namespace relief::surface {
SurfaceFailure from_projection_failure(gpf::ProjectPolylinesOnMeshFailure failure) noexcept;
SurfaceFailure from_walk_failure(gpf::WalkOnMeshSurfaceFailure failure) noexcept;
std::array<int, 3> polygon_color(std::size_t polygon_id);
[[nodiscard]] std::expected<void, SurfaceFailure> write_mesh_as_off(const fit_on_surface::Mesh& mesh, const std::string& path);
[[nodiscard]] std::expected<void, SurfaceFailure> write_image_relief_mesh_as_off(const image_relief::Mesh& mesh, const std::string& path);
[[nodiscard]] std::expected<void, SurfaceFailure> write_uv_as_off(const VMat2& uv, const FMat& faces, const std::string& path);
[[nodiscard]] std::expected<void, SurfaceFailure> write_uv_mesh_as_off(const uv::Mesh& mesh, const std::string& path);
[[nodiscard]] std::expected<void, SurfaceFailure> write_polyline_as_obj(
    const std::vector<std::array<double, 2>>& points, std::span<const std::size_t> polyline, const std::string& path);

[[nodiscard]] std::expected<void, SurfaceFailure> write_faces_as_off(
    const auto& mesh, const std::span<const gpf::FaceId> face_ids, const std::string& path)
{
    std::vector<gpf::VertexId> vertices;
    std::vector<std::size_t> indices(mesh.n_vertices_capacity(), gpf::kInvalidIndex);
    std::vector<std::vector<std::size_t>> faces;
    for (const auto fid : face_ids) {
        std::vector<std::size_t> face;
        for (const auto he : mesh.face(fid).halfedges()) {
            const auto vid = he.to().id;
            if (indices[vid.idx] == gpf::kInvalidIndex) {
                indices[vid.idx] = vertices.size();
                vertices.push_back(vid);
            }
            face.push_back(indices[vid.idx]);
        }
        faces.push_back(std::move(face));
    }
    std::ofstream file(path);
    if (!file) {
        return std::unexpected(SurfaceFailure::SurfaceOffOpenFailed);
    }
    file << "OFF\n"
         << vertices.size() << ' ' << faces.size() << " 0\n";
    for (const auto vid : vertices) {
        const auto& pt = mesh.vertex_prop(vid).pt;
        file << pt[0] << ' ' << pt[1] << ' ' << pt[2] << '\n';
    }
    for (const auto& face : faces) {
        file << face.size();
        for (const auto vid : face) {
            file << ' ' << vid;
        }
        file << '\n';
    }
    return {};
}

auto make_base_uv_edges(const auto& mesh, const uv::Mesh& uv_mesh,
    const std::vector<gpf::VertexId>& local_to_mesh_vertex)
{
    std::vector<gpf::EdgeId> edges(uv_mesh.n_edges_capacity());
    for (const auto edge : uv_mesh.edges()) {
        const auto [a, b] = edge.vertices();
        edges[edge.id.idx] = mesh.e_from_vertices(local_to_mesh_vertex[a.id.idx], local_to_mesh_vertex[b.id.idx]);
    }
    return edges;
}
}

// Shared errors and diagnostic writers.

namespace relief::surface {
SurfaceFailure from_projection_failure(const gpf::ProjectPolylinesOnMeshFailure failure) noexcept
{
    switch (failure) {
    case gpf::ProjectPolylinesOnMeshFailure::PathNotFound:
        return SurfaceFailure::ProjectionPathNotFound;
    case gpf::ProjectPolylinesOnMeshFailure::InvalidTriangleIndex:
        return SurfaceFailure::ProjectionInvalidTriangleIndex;
    case gpf::ProjectPolylinesOnMeshFailure::ConstraintConflict:
        return SurfaceFailure::ProjectionConstraintConflict;
    }
    std::unreachable();
}

SurfaceFailure from_walk_failure(const gpf::WalkOnMeshSurfaceFailure failure) noexcept
{
    switch (failure) {
    case gpf::WalkOnMeshSurfaceFailure::BoundaryReached:
        return SurfaceFailure::WalkBoundaryReached;
    case gpf::WalkOnMeshSurfaceFailure::IterationLimitExceeded:
        return SurfaceFailure::WalkIterationLimitExceeded;
    case gpf::WalkOnMeshSurfaceFailure::DegenerateDirection:
        return SurfaceFailure::WalkDegenerateDirection;
    case gpf::WalkOnMeshSurfaceFailure::InvalidPath:
        return SurfaceFailure::WalkInvalidPath;
    case gpf::WalkOnMeshSurfaceFailure::DegenerateStep:
        return SurfaceFailure::WalkDegenerateStep;
    }
    std::unreachable();
}

[[nodiscard]] std::expected<void, SurfaceFailure> write_mesh_as_off(const fit_on_surface::Mesh& mesh, const std::string& path)
{
    std::vector<std::size_t> vertex_indices(mesh.n_vertices_capacity(), gpf::kInvalidIndex);
    std::vector<gpf::VertexId> vertices;
    vertices.reserve(mesh.n_vertices());
    for (const auto vertex : mesh.vertices()) {
        vertex_indices[vertex.id.idx] = vertices.size();
        vertices.push_back(vertex.id);
    }

    std::ofstream file(path);
    if (!file) {
        return std::unexpected(SurfaceFailure::MeshOffOpenFailed);
    }

    file << "OFF\n";
    file << vertices.size() << ' ' << mesh.n_faces() << " 0\n";
    for (const auto vid : vertices) {
        const auto& pt = mesh.vertex_prop(vid).pt;
        file << pt[0] << ' ' << pt[1] << ' ' << pt[2] << '\n';
    }
    for (const auto face : mesh.faces()) {
        std::vector<std::size_t> face_vertices;
        for (const auto halfedge : face.halfedges()) {
            const auto index = vertex_indices[halfedge.from().id.idx];
            if (index == gpf::kInvalidIndex) {
                return std::unexpected(SurfaceFailure::InvalidMeshVertexReference);
            }
            face_vertices.push_back(index);
        }

        file << face_vertices.size();
        for (const auto vid : face_vertices) {
            file << ' ' << vid;
        }
        const auto color = polygon_color(face.prop().polygon_id);
        file << ' ' << color[0] << ' ' << color[1] << ' ' << color[2] << " 255";
        file << '\n';
    }
    return {};
}

[[nodiscard]] std::expected<void, SurfaceFailure> write_image_relief_mesh_as_off(const image_relief::Mesh& mesh, const std::string& path)
{
    std::vector<std::size_t> vertex_indices(mesh.n_vertices_capacity(), gpf::kInvalidIndex);
    std::vector<gpf::VertexId> vertices;
    vertices.reserve(mesh.n_vertices());
    for (const auto vertex : mesh.vertices()) {
        vertex_indices[vertex.id.idx] = vertices.size();
        vertices.push_back(vertex.id);
    }

    std::ofstream file(path);
    if (!file) {
        return std::unexpected(SurfaceFailure::MeshOffOpenFailed);
    }

    file << "OFF\n";
    file << vertices.size() << ' ' << mesh.n_faces() << " 0\n";
    for (const auto vid : vertices) {
        const auto& point = mesh.vertex_prop(vid).pt;
        file << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
    }
    for (const auto face : mesh.faces()) {
        std::vector<std::size_t> face_vertices;
        for (const auto halfedge : face.halfedges()) {
            const auto index = vertex_indices[halfedge.from().id.idx];
            if (index == gpf::kInvalidIndex) {
                return std::unexpected(SurfaceFailure::InvalidMeshVertexReference);
            }
            face_vertices.push_back(index);
        }

        file << face_vertices.size();
        for (const auto vertex : face_vertices) {
            file << ' ' << vertex;
        }
        file << '\n';
    }
    return {};
}

[[nodiscard]] std::expected<void, SurfaceFailure> write_uv_as_off(const VMat2& uv, const FMat& faces, const std::string& path)
{
    std::ofstream file(path);
    if (!file) {
        return std::unexpected(SurfaceFailure::UvCoordinatesOffOpenFailed);
    }

    file << "OFF\n";
    file << uv.rows() << ' ' << faces.rows() << " 0\n";
    for (Eigen::Index i = 0; i < uv.rows(); ++i) {
        file << uv(i, 0) << ' ' << uv(i, 1) << " 0\n";
    }
    for (Eigen::Index i = 0; i < faces.rows(); ++i) {
        file << faces.cols();
        for (Eigen::Index j = 0; j < faces.cols(); ++j) {
            file << ' ' << faces(i, j);
        }
        file << '\n';
    }
    return {};
}

std::array<int, 3> polygon_color(const std::size_t polygon_id)
{
    static constexpr std::array<std::array<int, 3>, 12> kColors { {
        { 230, 25, 75 },
        { 60, 180, 75 },
        { 0, 130, 200 },
        { 245, 130, 48 },
        { 145, 30, 180 },
        { 70, 240, 240 },
        { 240, 50, 230 },
        { 210, 245, 60 },
        { 250, 190, 190 },
        { 0, 128, 128 },
        { 230, 190, 255 },
        { 170, 110, 40 },
    } };

    if (polygon_id == gpf::kInvalidIndex) {
        return { 180, 180, 180 };
    }
    return kColors[polygon_id % kColors.size()];
}

[[nodiscard]] std::expected<void, SurfaceFailure> write_uv_mesh_as_off(const uv::Mesh& mesh, const std::string& path)
{
    std::ofstream file(path);
    if (!file) {
        return std::unexpected(SurfaceFailure::UvMeshOffOpenFailed);
    }

    std::vector<std::size_t> vertex_indices(mesh.n_vertices_capacity(), gpf::kInvalidIndex);
    std::vector<gpf::VertexId> vertices;
    vertices.reserve(mesh.n_vertices());
    for (auto vertex : mesh.vertices()) {
        vertex_indices[vertex.id.idx] = vertices.size();
        vertices.push_back(vertex.id);
    }

    file << "OFF\n";
    file << vertices.size() << ' ' << mesh.n_faces() << " 0\n";
    for (const auto vid : vertices) {
        const auto& pt = mesh.vertex_prop(vid).pt;
        file << pt[0] << ' ' << pt[1] << " 0\n";
    }

    for (auto face : mesh.faces()) {
        std::vector<std::size_t> face_vertices;
        for (auto halfedge : face.halfedges()) {
            const auto vertex_idx = vertex_indices[halfedge.from().id.idx];
            if (vertex_idx == gpf::kInvalidIndex) {
                return std::unexpected(SurfaceFailure::InvalidUvVertexReference);
            }
            face_vertices.push_back(vertex_idx);
        }

        file << face_vertices.size();
        for (const auto vid : face_vertices) {
            file << ' ' << vid;
        }
        const auto color = polygon_color(face.prop().polygon_id);
        file << ' ' << color[0] << ' ' << color[1] << ' ' << color[2] << " 255";
        file << '\n';
    }
    return {};
}

[[nodiscard]] std::expected<void, SurfaceFailure> write_polyline_as_obj(
    const std::vector<std::array<double, 2>>& points,
    const std::span<const std::size_t> polyline,
    const std::string& path)
{
    std::ofstream file(path);
    if (!file) {
        return std::unexpected(SurfaceFailure::PolylineObjOpenFailed);
    }

    for (const auto& point : points) {
        file << "v " << point[0] << ' ' << point[1] << " 0\n";
    }

    for (std::size_t i = 1; i < polyline.size(); ++i) {
        const auto start_idx = polyline[i - 1];
        const auto end_idx = polyline[i];
        if (start_idx >= points.size() || end_idx >= points.size()) {
            return std::unexpected(SurfaceFailure::InvalidPolylinePointIndex);
        }
        file << "l " << start_idx + 1 << ' ' << end_idx + 1 << '\n';
    }
    return {};
}

}

// Legacy walk, extraction, harmonic initialization, and anchor placement.

// Private walk/project/extract and harmonic provider. Extraction already emits
// boundary-first vertices and maps its walk anchors into that ordering.
namespace relief::surface::legacy {
namespace ranges = std::ranges;
namespace views = std::views;
struct ExtractedSurfacePatch {
    std::vector<gpf::FaceId> surface_faces;
    std::vector<gpf::VertexId> uv_to_surface_vertex;
    std::array<std::size_t, 5> anchor_uv_indices; // Center, then the four walk corners.
    std::size_t n_boundary_vertices;
    VMat positions;
    FMat triangles;
};

ExtractedSurfacePatch extract_face_mesh(
    const auto& mesh,
    const std::span<const gpf::HalfedgeId> boundary_halfedges,
    std::vector<gpf::FaceId> inner_faces,
    const std::span<const gpf::VertexId, 5> anchor_surface_vertices)
{
    std::vector<bool> he_is_boundary(mesh.n_halfedges_capacity(), false);
    for (const auto hid : boundary_halfedges) {
        auto twin_hid = mesh.he_twin(hid);
        if (he_is_boundary[twin_hid.idx]) {
            he_is_boundary[twin_hid.idx] = false;
        } else {
            he_is_boundary[hid.idx] = true;
        }
    }

    std::vector<std::size_t> vertex_map(mesh.n_vertices_capacity(), gpf::kInvalidIndex);
    std::vector<gpf::VertexId> vertices;
    for (const auto hid : boundary_halfedges) {
        if (!he_is_boundary[hid.idx]) {
            continue;
        }
        const auto vid = mesh.he_to(hid);
        if (vertex_map[vid.idx] == gpf::kInvalidIndex) {
            vertex_map[vid.idx] = vertices.size();
            vertices.push_back(vid);
        }
    }
    const auto n_boundary_vertices = vertices.size();
    std::vector<std::size_t> face_vertices;
    for (const auto fid : inner_faces) {
        for (const auto he : mesh.face(fid).halfedges()) {
            auto vid = he.from().id;
            if (vertex_map[vid.idx] == gpf::kInvalidIndex) {
                vertex_map[vid.idx] = vertices.size();
                vertices.push_back(vid);
            }
            face_vertices.push_back(vertex_map[vid.idx]);
        }
    }
    VMat V(vertices.size(), 3);
    FMat F = FMat::Map(face_vertices.data(), face_vertices.size() / 3, 3);
    {
        auto v_data = V.data();
        for (const auto vid : vertices) {
            const auto& pt = mesh.vertex_prop(vid).pt;
            std::copy(pt.begin(), pt.end(), v_data);
            v_data += 3;
        }
    }
    std::array<std::size_t, 5> anchor_uv_indices;
    ranges::transform(anchor_surface_vertices, anchor_uv_indices.begin(), [&vertex_map](auto vid) { return vertex_map[vid.idx]; });
    return {
        std::move(inner_faces),
        std::move(vertices),
        anchor_uv_indices,
        n_boundary_vertices,
        std::move(V),
        std::move(F),
    };
}

inline auto face_point(const auto& mesh, const gpf::FaceId fid, const std::span<const double, 3> bary_coords)
{
    auto he = mesh.face(fid).halfedge();
    auto pa = Eigen::Vector3d::Map(he.from().prop().pt.data());
    he = he.next();
    auto pb = Eigen::Vector3d::Map(he.from().prop().pt.data());
    auto pc = Eigen::Vector3d::Map(he.to().prop().pt.data());
    std::array<double, 3> result {};
    Eigen::Vector3d::Map(result.data()) = pa * bary_coords[0] + pb * bary_coords[1] + pc * bary_coords[2];
    return result;
}

inline std::size_t find_anchor_corner_index(const VMat2& uv, const std::span<const std::size_t> vertices)
{
    Eigen::Vector2d center = uv.row(vertices[0]).transpose();
    return ranges::min(views::zip(vertices.subspan(1), ranges::iota_view { std::size_t { 1 }, std::size_t { 5 } }) | views::transform([&uv, &center](auto&& pair) { return std::make_pair(
                                                                                                                                                                        (uv.row(std::get<0>(pair)).transpose() - center).squaredNorm(),
                                                                                                                                                                        std::get<1>(pair)); }), {}, &std::pair<double, std::size_t>::first).second;
}

inline std::optional<double> boundary_contains_anchor_rectangle(const VMat2& uv, const std::span<const std::size_t> vertices, const std::size_t min_idx, const Eigen::VectorXi& bnd)
{
    const auto div = [](const auto& a, const auto& b) noexcept {
        return Eigen::Vector2d { a.x() * b.x() + a.y() * b.y(), a.y() * b.x() - a.x() * b.y() };
    };
    Eigen::Vector2d center = uv.row(vertices[0]).transpose();
    Eigen::Vector2d base_dir = uv.row(vertices[min_idx]).transpose() - center;
    std::vector<double> corners = { base_dir[0], base_dir[1], -base_dir[1], base_dir[0], -base_dir[0], -base_dir[1], base_dir[1], -base_dir[0] };
    const auto half_diag_len = base_dir.norm();
    base_dir /= half_diag_len;
    const auto half_len = half_diag_len / std::numbers::sqrt2;
    const Eigen::Rotation2Dd rot(std::numbers::pi * 0.25);
    // base_dir is an actual corner ray; +45 degrees gives a square edge axis.
    Eigen::Vector2d horizontal_dir = rot * base_dir;

    VMat2 uv_diff = uv(bnd, Eigen::placeholders::all).rowwise() - center.transpose();
    std::vector<std::size_t> quadrants(bnd.rows());
    const auto compute_quadrant = [](const double* data) {
        if (data[0] > 0.0 && data[1] >= 0.0) {
            return 0;
        } else if (data[0] <= 0.0 && data[1] > 0.0) {
            return 1;
        } else if (data[0] < 0.0 && data[1] <= 0.0) {
            return 2;
        } else {
            return 3;
        }
    };
    std::optional<double> scale {};
    for (Eigen::Index i = 0; i < uv_diff.rows(); i++) {
        Eigen::Vector2d vec = uv_diff.row(i).transpose();
        auto actual_len = vec.norm();
        vec /= actual_len;
        auto angle_vec = div(vec, horizontal_dir);
        auto expected_len = half_len / std::max(std::abs(angle_vec[0]), std::abs(angle_vec[1]));
        if (actual_len < expected_len) {
            const auto t = actual_len / expected_len;
            if (!scale.has_value() || t < *scale) {
                scale = t;
            }
        }

        angle_vec = (rot * angle_vec).eval(); // rotate 45 degree counterclockwise
        quadrants[i] = compute_quadrant(angle_vec.data());
    }
    Eigen::Vector2d zero { 0.0, 0.0 };
    for (Eigen::Index i = 0; i < uv_diff.rows(); i++) {
        const auto j = (i + 1) % uv_diff.rows();
        Eigen::Vector2d va = uv_diff.row(i);
        Eigen::Vector2d vb = uv_diff.row(j);
        auto q1 = quadrants[i];
        auto q2 = quadrants[j];
        if (q1 == q2) {
            continue;
        }
        const auto ori1 = predicates::orient2d(va.data(), vb.data(), zero.data());
        if (q2 < q1) {
            q2 += 4;
        }
        const auto quadrant_span = q2 - q1;
        if (quadrant_span > 2 || (quadrant_span == 2 && ori1 < 0.0)) {
            q2 %= 4;
            std::swap(q1, q2);
            if (q2 < q1) {
                q2 += 4;
            }
        }
        for (std::size_t q = q1 + 1; q <= q2; q++) {
            const auto ori2 = predicates::orient2d(va.data(), vb.data(), &corners[(q % 4) * 2]);
            if (ori1 * ori2 < 0.0) {
                const auto t = std::abs(ori1) / (std::abs(ori1) + std::abs(ori2));
                if (!scale.has_value() || t < *scale) {
                    scale = t;
                }
            }
        }
    }
    return scale;
}

inline UvPlacementFrame compute_anchor_uv_frame(const VMat2& uv, const std::span<const std::size_t> vertices, const std::size_t min_idx, const std::optional<double> scale)
{
    Eigen::Vector2d center = uv.row(vertices[0]).transpose();
    Eigen::Vector2d base_dir = uv.row(vertices[min_idx]).transpose() - center;
    if (scale.has_value()) {
        base_dir *= *scale;
    }
    // Anchors 1..4 follow x-y, x+y, -x+y, -x-y. Rotate the selected
    // corner to the x+y diagonal, then recover the two edge axes below.
    const auto angle = std::numbers::pi * (1.0 - 0.5 * min_idx);
    Eigen::Rotation2Dd rot(angle);
    Eigen::Vector2d dir = rot * base_dir;
    Eigen::Vector2d start_pt = center - dir;
    dir *= std::numbers::sqrt2;
    const auto pi_4 = std::numbers::pi * 0.25;
    Eigen::Vector2d xaxis = Eigen::Rotation2Dd(-pi_4) * dir;
    Eigen::Vector2d yaxis = Eigen::Rotation2Dd(pi_4) * dir;
    return { std::move(start_pt), std::move(xaxis), std::move(yaxis) };
}

namespace {
    constexpr double kWalkRoundoff = 64 * std::numeric_limits<double>::epsilon();

    struct WalkTriangle {
        std::array<Eigen::Vector3d, 3> positions;
        std::array<gpf::HalfedgeId, 3> halfedges;
        Eigen::Vector3d xaxis, yaxis, normal;
        std::array<double, 6> local;
        double tolerance, altitude;

        Eigen::Vector2d project(const Eigen::Vector3d& point) const
        {
            const Eigen::Vector3d offset = point - positions[0];
            return { offset.dot(xaxis), offset.dot(yaxis) };
        }

        std::array<double, 3> barycentric(const Eigen::Vector3d& point) const
        {
            const auto p = project(point);
            const std::array<double, 8> points { local[0], local[1], local[2], local[3], local[4], local[5], p.x(), p.y() };
            const auto bary = gpf::detail::compute_bary_coordinates(points);
            return { bary[0], bary[1], bary[2] };
        }

        bool contains(const Eigen::Vector3d& point) const
        {
            const auto bary = barycentric(point);
            const double bary_tolerance = std::min(tolerance / altitude, gpf::detail::BARY_EPS);
            return std::abs(normal.dot(point - positions[0])) <= tolerance
                && ranges::all_of(bary, [&](double b) { return b >= -bary_tolerance; });
        }

        // GPF's initial orientation search has no iteration bound. Mirror its
        // local projection/snapping and require a positive-distance forward exit
        // before handing it an edge/vertex start (including edge-aligned rays).
        bool has_forward_exit(const Eigen::Vector3d& point, const Eigen::Vector3d& direction) const
        {
            auto start = project(point);
            auto bary = barycentric(point);
            if (gpf::detail::normalize_barycentric(bary, gpf::detail::BARY_EPS)) {
                start = bary[0] * Eigen::Vector2d::Map(local.data())
                    + bary[1] * Eigen::Vector2d::Map(local.data() + 2)
                    + bary[2] * Eigen::Vector2d::Map(local.data() + 4);
            }
            const Eigen::Vector2d dir { direction.dot(xaxis), direction.dot(yaxis) };
            const Eigen::Vector2d ray = start + dir;
            std::array<double, 3> side;
            for (std::size_t i = 0; i < 3; ++i) {
                side[i] = predicates::orient2d(ray.data(), start.data(), local.data() + 2 * i);
            }
            for (std::size_t i = 0; i < 3; ++i) {
                const auto next = (i + 1) % 3;
                if (side[i] > 0 && side[next] <= 0) {
                    const Eigen::Vector2d exit = (-side[next] * Eigen::Vector2d::Map(local.data() + 2 * i)
                                                     + side[i] * Eigen::Vector2d::Map(local.data() + 2 * next))
                        / (side[i] - side[next]);
                    return (exit - start).dot(dir) > tolerance;
                }
            }
            return false;
        }
    };

    std::expected<WalkTriangle, SurfaceFailure> make_walk_triangle(std::array<Eigen::Vector3d, 3> positions)
    {
        WalkTriangle triangle;
        triangle.positions = std::move(positions);
        const Eigen::Vector3d ab = triangle.positions[1] - triangle.positions[0];
        const Eigen::Vector3d ac = triangle.positions[2] - triangle.positions[0];
        const double lab = ab.norm(), lac = ac.norm(), lbc = (triangle.positions[2] - triangle.positions[1]).norm();
        const double scale = std::max({ lab, lac, lbc });
        const double area = ab.cross(ac).norm();
        if (!std::isfinite(area) || area <= kWalkRoundoff * scale * scale) {
            return std::unexpected(SurfaceFailure::WalkDegenerateStep);
        }
        // Use the same axes and length-based apex as the GPF walker.
        triangle.xaxis = ab / lab;
        triangle.normal = triangle.xaxis.cross(ac / lac).normalized();
        triangle.yaxis = triangle.normal.cross(triangle.xaxis);
        const auto apex = gpf::triangle_apex_from_base_lengths(lab, lbc, lac, false);
        triangle.local = { 0, 0, lab, 0, apex.x(), apex.y() };
        triangle.altitude = area / scale;
        // Classification uses local differences, like the walker. A large
        // world-space translation must not turn an interior point into an edge
        // start or permit a positive-distance crossing to count as zero.
        triangle.tolerance = kWalkRoundoff * scale;
        if (!apex.allFinite() || apex.y() <= 0) {
            return std::unexpected(SurfaceFailure::WalkDegenerateStep);
        }
        return triangle;
    }

    std::expected<WalkTriangle, SurfaceFailure> read_walk_triangle(const auto& mesh, gpf::FaceId face)
    {
        if (!face.valid() || face.idx >= mesh.n_faces_capacity() || mesh.face_is_deleted(face)) {
            return std::unexpected(SurfaceFailure::WalkInvalidPath);
        }
        std::array<Eigen::Vector3d, 3> positions;
        std::array<gpf::HalfedgeId, 3> halfedges;
        auto hid = mesh.f_halfedge(face);
        // Do not rely on an unbounded face/vertex circulator to validate a start.
        for (std::size_t i = 0; i < 3; ++i) {
            if (!hid.valid() || hid.idx >= mesh.n_halfedges_capacity()
                || mesh.he_twin(hid).idx >= mesh.n_halfedges_capacity()
                || mesh.halfedge_is_deleted(hid) || mesh.he_face(hid) != face) {
                return std::unexpected(SurfaceFailure::WalkInvalidPath);
            }
            const auto vertex = mesh.he_from(hid);
            if (!vertex.valid() || vertex.idx >= mesh.n_vertices_capacity() || mesh.vertex_is_deleted(vertex)) {
                return std::unexpected(SurfaceFailure::WalkInvalidPath);
            }
            halfedges[i] = hid;
            positions[i] = Eigen::Vector3d::Map(mesh.vertex_prop(vertex).pt.data());
            hid = mesh.he_next(hid);
        }
        if (hid != halfedges[0] || halfedges[0] == halfedges[1] || halfedges[1] == halfedges[2]) {
            return std::unexpected(SurfaceFailure::WalkInvalidPath);
        }
        for (std::size_t i = 0; i < 3; ++i) {
            const auto next = halfedges[(i + 1) % 3];
            if (mesh.he_prev(next) != halfedges[i] || mesh.he_to(halfedges[i]) != mesh.he_from(next)) {
                return std::unexpected(SurfaceFailure::WalkInvalidPath);
            }
        }
        auto triangle = make_walk_triangle(std::move(positions));
        if (triangle) {
            triangle->halfedges = halfedges;
        }
        return triangle;
    }

    struct WalkStart {
        gpf::FaceId face;
        std::array<double, 3> point;
        Eigen::Vector3d direction;
    };

    /** Locate a source-connected child triangle, then cross only zero-distance
     * incident edges until this ray enters a face that GPF can safely walk.
     * Face parents are provenance, never adjacency. The original face ID is only
     * a seed: GPF reuses it for one child when splitting the captured triangle.
     */
    std::expected<WalkStart, SurfaceFailure> resolve_walk_start(
        const auto& mesh, const PlacementInput& input, const Eigen::Vector3d& direction)
    {
        auto source = make_walk_triangle({ input.source_positions.row(0), input.source_positions.row(1), input.source_positions.row(2) });
        if (!source) {
            return std::unexpected(source.error());
        }
        if (!direction.allFinite() || direction.squaredNorm() == 0) {
            return std::unexpected(SurfaceFailure::WalkDegenerateDirection);
        }
        for (std::size_t i = 0; i < 3; ++i) {
            const auto vertex = input.source_vertices[i];
            if (!vertex.valid() || vertex.idx >= mesh.n_vertices_capacity() || mesh.vertex_is_deleted(vertex)
                || (Eigen::Vector3d::Map(mesh.vertex_prop(vertex).pt.data()) - source->positions[i]).norm() > source->tolerance) {
                return std::unexpected(SurfaceFailure::WalkInvalidPath);
            }
        }
        Eigen::Vector3d point = Eigen::Vector3d::Map(input.surface_point.data());
        // Match legacy's projection onto the captured source plane, even if a
        // nearer disconnected sheet received the rejected exp-map subdivision.
        point -= source->normal.dot(point - source->positions[0]) * source->normal;
        if (!source->contains(point)) {
            return std::unexpected(SurfaceFailure::WalkInvalidPath);
        }
        std::vector<gpf::FaceId> faces { input.source_face };
        std::set<gpf::FaceId> visited { input.source_face };
        gpf::FaceId containing;
        for (std::size_t i = 0; i < faces.size(); ++i) {
            auto triangle = read_walk_triangle(mesh, faces[i]);
            if (!triangle) {
                return std::unexpected(triangle.error());
            }
            if (triangle->normal.dot(source->normal) <= 0
                || !ranges::all_of(triangle->positions, [&](const auto& p) { return source->contains(p); })) {
                continue;
            }
            if (triangle->contains(point)) {
                containing = faces[i];
                break;
            }
            for (const auto hid : triangle->halfedges) {
                const auto neighbor = mesh.he_face(mesh.he_twin(hid));
                if (neighbor.valid() && visited.insert(neighbor).second) {
                    faces.push_back(neighbor);
                }
            }
        }
        if (!containing.valid()) {
            return std::unexpected(SurfaceFailure::WalkInvalidPath);
        }

        std::vector<WalkStart> pending { { containing, { point.x(), point.y(), point.z() }, direction } };
        std::set<gpf::HalfedgeId> crossed;
        auto failure = SurfaceFailure::WalkDegenerateStep;
        // Each directed edge is crossed at most once, bounding even a curved
        // vertex fan with ambiguous continuation or an edge-aligned cycle.
        for (std::size_t i = 0; i < pending.size(); ++i) {
            auto start = pending[i];
            auto triangle = read_walk_triangle(mesh, start.face);
            if (!triangle) {
                return std::unexpected(triangle.error());
            }
            point = Eigen::Vector3d::Map(start.point.data());
            if (!triangle->contains(point)) {
                return std::unexpected(SurfaceFailure::WalkInvalidPath);
            }
            auto bary = triangle->barycentric(point);
            if (gpf::detail::normalize_barycentric(bary, gpf::detail::BARY_EPS)) {
                point = bary[0] * triangle->positions[0] + bary[1] * triangle->positions[1] + bary[2] * triangle->positions[2];
                start.point = { point.x(), point.y(), point.z() };
            }
            const double dc = start.direction.dot(triangle->yaxis) / triangle->local[5];
            const double db = (start.direction.dot(triangle->xaxis) - triangle->local[4] * dc) / triangle->local[2];
            const std::array<double, 3> rate { -db - dc, db, dc };
            const double rate_tolerance = kWalkRoundoff / triangle->altitude;
            std::vector<gpf::HalfedgeId> exits, aligned;
            for (std::size_t j = 0; j < 3; ++j) {
                // GPF snapping above makes incident-edge weights exactly zero.
                // Do not classify a positive weight as zero on a skinny triangle.
                if (bary[j] == 0) {
                    const auto opposite = triangle->halfedges[(j + 1) % 3];
                    if (rate[j] < -rate_tolerance) {
                        exits.push_back(opposite);
                    } else if (std::abs(rate[j]) <= rate_tolerance) {
                        aligned.push_back(opposite);
                    }
                }
            }
            if (exits.empty()) {
                if (triangle->has_forward_exit(point, start.direction)) {
                    return start;
                }
                // An aligned ray must use the side with a positive GPF right
                // orientation. Try the incident twin rather than perturbing it.
                exits = std::move(aligned);
            }
            ranges::sort(exits);
            for (const auto hid : exits) {
                if (!crossed.insert(hid).second) {
                    if (failure != SurfaceFailure::WalkBoundaryReached) {
                        failure = SurfaceFailure::WalkIterationLimitExceeded;
                    }
                    continue;
                }
                const auto neighbor = mesh.he_face(mesh.he_twin(hid));
                if (!neighbor.valid()) {
                    failure = SurfaceFailure::WalkBoundaryReached;
                    continue;
                }
                auto next = read_walk_triangle(mesh, neighbor);
                if (!next) {
                    return std::unexpected(next.error());
                }
                Eigen::Vector3d tangent = start.direction;
                if ((next->normal - triangle->normal).norm() > kWalkRoundoff) {
                    const Eigen::Vector3d axis = (Eigen::Vector3d::Map(mesh.vertex_prop(mesh.he_to(hid)).pt.data())
                        - Eigen::Vector3d::Map(mesh.vertex_prop(mesh.he_from(hid)).pt.data()))
                                                     .normalized();
                    const double angle = std::atan2(axis.dot(triangle->normal.cross(next->normal)), triangle->normal.dot(next->normal));
                    tangent = Eigen::AngleAxisd(angle, axis) * tangent;
                    tangent -= tangent.dot(next->normal) * next->normal;
                    if (!tangent.allFinite() || tangent.squaredNorm() == 0) {
                        return std::unexpected(SurfaceFailure::WalkDegenerateDirection);
                    }
                    tangent.normalize();
                }
                // Coplanar subdivisions preserve the original ray exactly.
                pending.push_back({ neighbor, start.point, tangent });
            }
        }
        return std::unexpected(failure);
    }
}

struct WalkedPatch {
    // Four outer corners, center, then four anchor corners.
    std::vector<std::array<double, 3>> points;
    gpf::FaceId center_face; // Resolved live start face, not the saved source-face ID.
};

[[nodiscard]] std::expected<WalkedPatch, SurfaceFailure> walk_patch_boundary_and_anchors(
    const auto& mesh, const PlacementInput& walk)
{
    const Eigen::Vector3d x = walk.tangent_direction.normalized();
    const auto& normal = walk.normal;
    const Eigen::Vector3d y = normal.cross(x);
    // The public direction is local +X; walks target corners, starting at x-y.
    // Successive +90-degree turns visit x+y, -x+y, and -x-y.
    Eigen::Vector3d dir = (x - y).normalized();
    constexpr double theta = std::numbers::pi / 2;
    constexpr double half_theta = theta / 2;
    const double cos_val = std::cos(half_theta);
    const double sin_val = std::sin(half_theta);
    Eigen::Quaterniond quat(cos_val, normal[0] * sin_val, normal[1] * sin_val, normal[2] * sin_val);
    std::vector<std::pair<gpf::FaceId, std::array<double, 3>>> corner_points;
    std::vector<std::array<double, 3>> outer_corner_points;
    std::optional<std::pair<gpf::FaceId, std::array<double, 3>>> start_info;
    for (std::size_t i { 0 }; i < std::size_t { 4 }; i++) {
        const auto start = resolve_walk_start(mesh, walk, dir);
        if (!start) {
            return std::unexpected(start.error());
        }
        const auto walk_ret = gpf::walk_on_mesh_surface(mesh, start->face, start->point,
            std::span<const double, 3> { start->direction.data(), 3 }, walk.lengths);
        if (walk_ret.has_value()) {
            if (!start_info.has_value()) {
                start_info = (*walk_ret)[0];
            }
            corner_points.push_back((*walk_ret)[1]);
            const auto [fid, bary_coords] = (*walk_ret)[2];
            outer_corner_points.push_back(face_point(mesh, fid, bary_coords));
        } else {
            return std::unexpected(from_walk_failure(walk_ret.error()));
        }
        dir = (quat * dir).eval();
    }
    {
        auto [fid, bary_coords] = start_info.value();
        outer_corner_points.push_back(face_point(mesh, fid, bary_coords));
    }
    for (auto [fid, bary_coords] : corner_points) {
        outer_corner_points.push_back(face_point(mesh, fid, bary_coords));
    }
    return WalkedPatch { std::move(outer_corner_points), start_info->first };
}

[[nodiscard]] std::expected<ExtractedSurfacePatch, SurfaceFailure> project_and_extract_surface_patch(
    auto& mesh, std::vector<std::array<double, 3>> points, std::optional<gpf::VertexId> center)
{
    if (center) {
        // The center is not part of the boundary polyline. Keep its live ID
        // rather than making the global projector rediscover this vertex.
        points.erase(points.begin() + 4);
    }
    auto projection = gpf::project_polylines_on_mesh(
        points,
        std::vector<std::vector<std::size_t>> { { 0, 1, 2, 3, 0 } },
        mesh, kProjectionTolerance);
    if (!projection) {
        // GPF retains earlier mesh, point, and parent-map mutations on failure.
        return std::unexpected(from_projection_failure(projection.error()));
    }
    auto [project_vertices, boundary_paths] = std::move(*projection);
    // Projection subdivides without deleting existing vertices. Its last four
    // results are the corner anchors whether or not the center was projected.
    std::array<gpf::VertexId, 5> anchors;
    anchors[0] = center ? *center : project_vertices[4];
    std::copy(project_vertices.end() - 4, project_vertices.end(), anchors.begin() + 1);
    auto inner_faces = gpf::surround_faces_by_halfedges(mesh, boundary_paths.front());
    return extract_face_mesh(mesh, boundary_paths.front(), std::move(inner_faces), anchors);
}

struct InitializedPatchUv {
    Eigen::VectorXi boundary;
    VMat2 coordinates;
};

inline InitializedPatchUv initialize_patch_uv(const ExtractedSurfacePatch& patch)
{
    auto uv_mesh = uv::Mesh::new_in(views::iota(Eigen::Index { 0 }, patch.triangles.rows()) | views::transform([&](auto i) {
        return std::span<const std::size_t, 3>(patch.triangles.data() + 3 * i, 3);
    }));
    auto boundary_halfedges = uv_mesh.halfedges() | views::filter([](const auto he) { return !he.face().id.valid(); });
    const auto boundary_count = ranges::distance(boundary_halfedges);
    if (boundary_count < 3 || static_cast<std::size_t>(boundary_count) != patch.n_boundary_vertices
        || uv_mesh.n_vertices() != static_cast<std::size_t>(patch.positions.rows())
        || uv_mesh.n_vertices() + uv_mesh.n_faces() != uv_mesh.n_edges() + 1) {
        return {};
    }
    // Preserve the original start when vertex zero is on the boundary, without
    // requiring boundary-first vertex numbering. Walking prev follows face winding.
    auto curr_he = *ranges::min_element(boundary_halfedges, {}, [](const auto he) { return he.to().id.idx; });
    const auto first_hid = curr_he.id;
    Eigen::VectorXi bnd(boundary_count);
    Eigen::Index idx = 0;
    do {
        if (idx == bnd.size()) {
            return {};
        }
        bnd(idx++) = static_cast<int>(curr_he.to().id.idx);
        curr_he = curr_he.prev();
    } while (curr_he.id != first_hid);
    // More than one boundary loop cannot be mapped to a single harmonic circle.
    if (idx != bnd.size()) {
        return {};
    }
    Eigen::MatrixXd bnd_uv;
    VMat2 uv;
    igl::map_vertices_to_circle(patch.positions, bnd, bnd_uv);
    if (!igl::harmonic(patch.positions, patch.triangles, bnd, bnd_uv, 1, uv)) {
        return {};
    }
    if (igl::flipped_triangles(uv, patch.triangles).size() != 0) {
        if (!igl::harmonic(patch.triangles, bnd, bnd_uv, 1, uv)) { // use uniform laplacian
            return {};
        }
    }
    return { std::move(bnd), std::move(uv) };
}
} // namespace relief::surface::legacy

// Common SLIM, placement fitting, and strategy selection.

namespace relief::surface {
namespace {
    namespace ranges = std::ranges;
    namespace views = std::views;

    constexpr auto parameterization_failed = std::unexpected(SurfaceFailure::ParameterizationFailed);
    // Projection can leave incomplete topology; later rejections permit fallback.
    enum class PreferredFailure { Rejected,
        ProjectionFailed };
    constexpr auto preferred_rejected = std::unexpected(PreferredFailure::Rejected);
    constexpr double kRelativeAreaTolerance = 64 * std::numeric_limits<double>::epsilon();

    using Point = Eigen::Vector2d;
    double orientation(const Point& a, const Point& b, const Point& c)
    {
        return predicates::orient2d(a.data(), b.data(), c.data());
    }
    bool on_segment(const Point& a, const Point& b, const Point& p)
    {
        return orientation(a, b, p) == 0 && (p.array() >= a.cwiseMin(b).array()).all()
            && (p.array() <= a.cwiseMax(b).array()).all();
    }
    bool strictly_inside(const Point& point, const VMat2& coordinates, std::span<const std::size_t> boundary)
    {
        int winding = 0;
        for (std::size_t i = 0; i < boundary.size(); ++i) {
            const Point a = coordinates.row(boundary[i]), b = coordinates.row(boundary[(i + 1) % boundary.size()]);
            if (on_segment(a, b, point)) {
                return false;
            }
            if (a.y() <= point.y() && b.y() > point.y() && orientation(a, b, point) > 0) {
                ++winding;
            } else if (a.y() > point.y() && b.y() <= point.y() && orientation(a, b, point) < 0) {
                --winding;
            }
        }
        return winding != 0;
    }

    bool valid_mappings(const auto& mesh, const InitialPatch& patch)
    {
        if (patch.surface_faces.size() != static_cast<std::size_t>(patch.triangles.rows())
            || patch.uv_to_surface_vertex.size() != static_cast<std::size_t>(patch.positions.rows())) {
            return false;
        }
        for (const auto vid : patch.uv_to_surface_vertex) {
            if (!vid.valid() || vid.idx >= mesh.n_vertices_capacity() || mesh.vertex_is_deleted(vid)) {
                return false;
            }
        }
        for (std::size_t i = 0; i < patch.surface_faces.size(); ++i) {
            const auto fid = patch.surface_faces[i];
            if (!fid.valid() || fid.idx >= mesh.n_faces_capacity() || mesh.face_is_deleted(fid)) {
                return false;
            }
            std::size_t j = 0;
            for (const auto he : mesh.face(fid).halfedges()) {
                if (j >= 3 || patch.triangles(i, j) >= patch.uv_to_surface_vertex.size()
                    || patch.uv_to_surface_vertex[patch.triangles(i, j++)] != he.from().id) {
                    return false;
                }
            }
            if (j != 3) {
                return false;
            }
        }
        return true;
    }

    PlacementInput make_placement_input(
        const auto& mesh, gpf::FaceId fid, const std::array<double, 3>& point,
        const std::array<double, 3>& direction, Footprint footprint = {})
    {
        PlacementInput input;
        auto he = mesh.face(fid).halfedge();
        for (std::size_t i = 0; i < 3; ++i, he = he.next()) {
            input.source_vertices[i] = he.from().id;
            input.source_positions.row(i) = Eigen::Vector3d::Map(he.from().prop().pt.data());
        }
        const Eigen::Vector3d ab = input.source_positions.row(1) - input.source_positions.row(0);
        const Eigen::Vector3d ac = input.source_positions.row(2) - input.source_positions.row(0);
        input.normal = ab.cross(ac);
        input.normal.normalize();
        const Eigen::Vector3d dir = Eigen::Vector3d::Map(direction.data());
        input.direction_magnitude = dir.norm();
        input.tangent_direction = dir - input.normal.dot(dir) * input.normal;
        input.tangent_direction.normalize();
        input.lengths = { input.direction_magnitude * std::numbers::sqrt2,
            input.direction_magnitude * std::numbers::sqrt2 * kSupportPadding };
        input.source_face = fid;
        input.surface_point = point;
        input.footprint = footprint;
        return input;
    }

    struct InitializedExpMapPatch {
        InitialPatch patch;
        std::array<std::size_t, 3> source_uv_indices { gpf::kInvalidIndex, gpf::kInvalidIndex, gpf::kInvalidIndex };
    };

    std::expected<InitializedExpMapPatch, PreferredFailure> initialize_exp_map_patch(auto& mesh, const PlacementInput& input)
    {
        initialize_exp_map_properties(mesh);
        auto result = gpf::exp_map(std::span<const double, 3>(input.surface_point), mesh, input.lengths[1]);
        if (!result) {
            return std::unexpected(result.error() == gpf::ExpMapFailure::ProjectionFailed
                    ? PreferredFailure::ProjectionFailed
                    : PreferredFailure::Rejected);
        }
        // Successful exp-map results contain live triangular faces and their
        // unique incident vertices, paired with UVs and including the center.
        InitializedExpMapPatch initialized;
        auto& patch = initialized.patch;
        patch.surface_faces = std::move(result->face_ids);
        // Put the supplied boundary loop first, then interior vertices in source
        // order. The source-ID lookup remaps every row and triangle consistently.
        patch.uv_to_surface_vertex = std::move(result->boundary_vertex_ids);
        patch.uv_to_surface_vertex.reserve(result->vertex_ids.size());
        patch.boundary.resize(patch.uv_to_surface_vertex.size());
        std::vector<std::size_t> lookup(mesh.n_vertices_capacity(), gpf::kInvalidIndex);
        for (std::size_t i = 0; i < patch.boundary.size(); ++i) {
            lookup[patch.uv_to_surface_vertex[i].idx] = i;
            patch.boundary[i] = i;
        }
        patch.positions.resize(result->vertex_ids.size(), 3);
        patch.coordinates.resize(result->vertex_ids.size(), 2);
        patch.triangles.resize(patch.surface_faces.size(), 3);
        for (std::size_t i = 0; i < result->vertex_ids.size(); ++i) {
            const auto vertex = result->vertex_ids[i];
            auto& local = lookup[vertex.idx];
            if (local == gpf::kInvalidIndex) {
                local = patch.uv_to_surface_vertex.size();
                patch.uv_to_surface_vertex.push_back(vertex);
            }
            patch.positions.row(local) = Eigen::Vector3d::Map(mesh.vertex_prop(vertex).pt.data());
            patch.coordinates.row(local) = Eigen::Vector2d::Map(result->uvs[i].data());
        }
        // Retain only the source triangle's rows from this lookup. Missing
        // vertices remain invalid until placement rejects them after SLIM.
        for (std::size_t i = 0; i < initialized.source_uv_indices.size(); ++i) {
            const auto vertex = input.source_vertices[i];
            if (vertex.valid() && vertex.idx < lookup.size()) {
                initialized.source_uv_indices[i] = lookup[vertex.idx];
            }
        }
        patch.center = lookup[result->center_vertex.idx];
        for (std::size_t f = 0; f < patch.surface_faces.size(); ++f) {
            std::size_t j = 0;
            for (const auto he : mesh.face(patch.surface_faces[f]).halfedges()) {
                patch.triangles(f, j++) = lookup[he.from().id.idx];
            }
        }
        return initialized;
    }

    std::expected<std::pair<InitialPatch, std::array<std::size_t, 5>>, SurfaceFailure> initialize_legacy_patch(
        auto& mesh, const PlacementInput& input)
    {
        auto walked = legacy::walk_patch_boundary_and_anchors(mesh, input);
        if (!walked) {
            return std::unexpected(walked.error());
        }
        // Reuse a nearby vertex of the resolved source-connected start face,
        // including an original corner. This also handles exp-map disk failures,
        // which retain the center vertex but do not return its ID.
        std::optional<gpf::VertexId> center;
        double nearest = kProjectionTolerance * kProjectionTolerance;
        const Eigen::Vector3d point = Eigen::Vector3d::Map(walked->points[4].data());
        for (const auto he : mesh.face(walked->center_face).halfedges()) {
            const auto vertex = he.from().id;
            const double distance = (Eigen::Vector3d::Map(he.from().prop().pt.data()) - point).squaredNorm();
            if (distance < nearest || (distance == nearest && center && vertex < *center)) {
                center = vertex;
                nearest = distance;
            }
        }
        auto extracted = legacy::project_and_extract_surface_patch(mesh, std::move(walked->points), center);
        if (!extracted) {
            return std::unexpected(extracted.error());
        }
        const auto anchors = extracted->anchor_uv_indices;
        if (ranges::any_of(anchors, [&](auto i) { return i >= static_cast<std::size_t>(extracted->positions.rows()); })) {
            return std::unexpected(SurfaceFailure::InvalidAnchorIndex);
        }
        auto initialized = legacy::initialize_patch_uv(*extracted);
        if (initialized.coordinates.rows() == 0) {
            return parameterization_failed;
        }
        InitialPatch patch;
        patch.positions = std::move(extracted->positions);
        patch.triangles = std::move(extracted->triangles);
        patch.coordinates = std::move(initialized.coordinates);
        patch.surface_faces = std::move(extracted->surface_faces);
        patch.uv_to_surface_vertex = std::move(extracted->uv_to_surface_vertex);
        patch.boundary.assign(initialized.boundary.data(), initialized.boundary.data() + initialized.boundary.size());
        patch.center = anchors[0];
        if (!valid_mappings(mesh, patch)) {
            return parameterization_failed;
        }
        return std::make_pair(std::move(patch), anchors);
    }

    PreparedSurfacePatch make_prepared(const auto& mesh, const OptimizedPatch& optimized, UvPlacementFrame frame)
    {
        const auto& patch = optimized.initial;
        auto uv_mesh = uv::Mesh::new_in(views::iota(Eigen::Index { 0 }, patch.triangles.rows()) | views::transform([&](auto i) {
            return std::span<const std::size_t, 3>(patch.triangles.data() + 3 * i, 3);
        }));
        for (const auto vertex : uv_mesh.vertices()) {
            const auto row = optimized.coordinates.row(vertex.id.idx);
            vertex.prop().pt = { row(0), row(1) };
        }
        for (const auto face : uv_mesh.faces()) {
            // UV labels start unset; polygon labeling is a downstream operation.
            face.prop().parent = mesh.face_prop(patch.surface_faces[face.id.idx]).parent;
        }
        return { std::move(uv_mesh), patch.surface_faces, patch.uv_to_surface_vertex, std::move(frame) };
    }

    std::expected<PreparedSurfacePatch, PreferredFailure> prepare_preferred(auto& mesh, const PlacementInput& input)
    {
        auto initial = initialize_exp_map_patch(mesh, input);
        if (!initial) {
            return std::unexpected(initial.error());
        }
        auto optimized = optimize_patch(std::move(initial->patch));
        auto frame = exp_map_placement(optimized, input, initial->source_uv_indices);
        if (!frame) {
            return preferred_rejected;
        }
        return make_prepared(mesh, optimized, *frame);
    }

    std::expected<PreparedSurfacePatch, SurfaceFailure> prepare_legacy(auto& mesh, const PlacementInput& input)
    {
        auto initialized = initialize_legacy_patch(mesh, input);
        if (!initialized) {
            return std::unexpected(initialized.error());
        }
        auto optimized = optimize_patch(std::move(initialized->first));
        const auto& anchors = initialized->second;
        const auto& coords = optimized.coordinates;
        const auto& bnd = optimized.initial.boundary;
        Eigen::VectorXi boundary(bnd.size());
        for (std::size_t i = 0; i < bnd.size(); ++i) {
            boundary[i] = static_cast<int>(bnd[i]);
        }
        const auto corner = legacy::find_anchor_corner_index(coords, anchors);
        const auto scale = legacy::boundary_contains_anchor_rectangle(coords, anchors, corner, boundary);
        auto frame = legacy::compute_anchor_uv_frame(coords, anchors, corner, scale);
        if (frame.xaxis.squaredNorm() == 0 || frame.yaxis.squaredNorm() == 0) {
            return parameterization_failed;
        }
        return make_prepared(mesh, optimized, frame);
    }

    std::expected<void, SurfaceFailure> write_selected_diagnostics(const auto& mesh, const PreparedSurfacePatch& patch)
    {
        VMat2 coordinates(patch.uv_mesh.n_vertices(), 2);
        FMat faces(patch.uv_mesh.n_faces(), 3);
        for (const auto v : patch.uv_mesh.vertices()) {
            coordinates.row(v.id.idx) = Eigen::Vector2d::Map(v.prop().pt.data());
        }
        for (const auto face : patch.uv_mesh.faces()) {
            std::size_t j = 0;
            for (const auto he : face.halfedges()) {
                faces(face.id.idx, j++) = he.from().id.idx;
            }
        }
        if (auto result = write_uv_as_off(coordinates, faces, "fit_polygon_on_surface_uv.off"); !result) {
            return result;
        }
        if (auto result = write_faces_as_off(mesh, patch.surface_faces, "fit_polygon_on_surface.off"); !result) {
            return result;
        }
        // Historically this auxiliary dump was best effort (unlike the OFF files).
        std::ofstream file("uv_new.obj");
        for (Eigen::Index i = 0; i < coordinates.rows(); ++i) {
            file << "v " << coordinates(i, 0) << ' ' << coordinates(i, 1) << " 0\n";
        }
        for (Eigen::Index i = 0; i < faces.rows(); ++i) {
            file << "f " << faces(i, 0) + 1 << ' ' << faces(i, 1) + 1 << ' ' << faces(i, 2) + 1 << '\n';
        }
        return {};
    }

    std::expected<PreparedSurfacePatch, SurfaceFailure> prepare_impl(auto& mesh, const PlacementInput& input)
    {
        auto preferred = prepare_preferred(mesh, input);
        if (!preferred && preferred.error() == PreferredFailure::ProjectionFailed) {
            // GPF may not have finished updating topology. Do not walk it.
            return parameterization_failed;
        }
        // Both providers use the same live mesh and the original placement input.
        // Only a recoverable rejection runs legacy; no rollback or eager work.
        std::expected<PreparedSurfacePatch, SurfaceFailure> selected = preferred
            ? std::expected<PreparedSurfacePatch, SurfaceFailure>(std::move(*preferred))
            : prepare_legacy(mesh, input);
        if (!selected) {
            return std::unexpected(selected.error());
        }
        // No retry after selection, including checked diagnostic failures.
        if (auto result = write_selected_diagnostics(mesh, *selected); !result) {
            return std::unexpected(result.error());
        }
        return selected;
    }
} // namespace

OptimizedPatch optimize_patch(InitialPatch patch)
{
    const auto a = patch.triangles(0, 0), b = patch.triangles(0, 1), c = patch.triangles(0, 2);
    if (orientation(patch.coordinates.row(a), patch.coordinates.row(b), patch.coordinates.row(c)) < 0) {
        patch.coordinates.col(1) *= -1;
    }
    // Boundary-first input makes vertex zero a boundary translation anchor.
    // The center and all remaining boundary vertices stay free.
    // See docs/SLIM_ldlt_nullspace_explanation.md.
    FlattenSurface solver(VMat(patch.positions), FMat(patch.triangles), VMat2(patch.coordinates), kSlimFixedVertices);
    // Keep FlattenSurface unchanged: its existing entry point returns no solve
    // status and writes uv_new.obj. Consume its UVs directly; selected diagnostics
    // later overwrite that auxiliary dump with the consumed UVs.
    solver.slim_solve(kSlimMinIterations, kSlimMaxIterations);
    return { std::move(patch), std::move(solver.uv) };
}

std::expected<UvPlacementFrame, SurfaceFailure> fit_placement(
    UvPlacementFrame frame, const Footprint& footprint, const VMat2& coordinates, std::span<const std::size_t> boundary)
{
    const Eigen::Vector2d center = frame.origin + 0.5 * (frame.xaxis + frame.yaxis);
    if (!strictly_inside(center, coordinates, boundary)) {
        return parameterization_failed;
    }
    Eigen::Matrix2d support;
    support.col(0) = frame.xaxis * footprint.half_extent.x();
    support.col(1) = frame.yaxis * footprint.half_extent.y();
    if (std::abs(support.determinant()) <= kRelativeAreaTolerance * support.squaredNorm()) {
        return parameterization_failed;
    }
    const Eigen::Matrix2d inverse = support.inverse();
    // Reserve the projector's distance tolerance in each support coordinate.
    // This bounds the image of a Euclidean tolerance ball even for a skew frame.
    const Eigen::Vector2d clearance { kProjectionTolerance * inverse.row(0).norm(), kProjectionTolerance * inverse.row(1).norm() };
    double limiting_scale = std::numeric_limits<double>::infinity();
    // Distance to each segment in the rectangle's Minkowski norm. The minimum
    // of max(|x|-clearance.x,|y|-clearance.y) is at an endpoint, an
    // absolute-value breakpoint, or an equality of the two affine pieces.
    // This includes concave intrusions crossing between otherwise valid corners.
    for (std::size_t i = 0; i < boundary.size(); ++i) {
        const Eigen::Vector2d a = inverse * (coordinates.row(boundary[i]).transpose() - center);
        const Eigen::Vector2d b = inverse * (coordinates.row(boundary[(i + 1) % boundary.size()]).transpose() - center);
        const Eigen::Vector2d delta = b - a;
        auto evaluate = [&](double t) noexcept {
            if (t >= 0 && t <= 1) {
                limiting_scale = std::min(limiting_scale, ((a + t * delta).cwiseAbs() - clearance).maxCoeff());
            }
        };
        evaluate(0.0);
        evaluate(1.0);
        if (delta[0] != 0.0) {
            evaluate(a[0] / delta[0]);
        }
        if (delta[1] != 0.0) {
            evaluate(a[1] / delta[1]);
        }
        for (const double xsign : { -1., 1. }) {
            for (const double ysign : { -1., 1. }) {
                const Eigen::Vector2d normal { xsign, -ysign };
                const double denominator = normal.dot(delta);
                if (denominator != 0) {
                    evaluate((clearance.x() - clearance.y() - normal.dot(a)) / denominator);
                }
            }
        }
    }
    const double scale = std::min(1.0, limiting_scale);
    if (!std::isfinite(scale) || scale <= 0) {
        return parameterization_failed;
    }
    frame.xaxis *= scale;
    frame.yaxis *= scale;
    frame.origin = center - 0.5 * (frame.xaxis + frame.yaxis);
    if (std::min(frame.xaxis.norm() * footprint.half_extent.x(), frame.yaxis.norm() * footprint.half_extent.y()) <= kProjectionTolerance) {
        return parameterization_failed;
    }
    return frame;
}

std::expected<UvPlacementFrame, SurfaceFailure> exp_map_placement(
    const OptimizedPatch& patch, const PlacementInput& input, const std::array<std::size_t, 3>& source_uv_indices)
{
    // Cached rows belong to the initializer's ordering and captured input.
    const auto& local = source_uv_indices;
    for (std::size_t i = 0; i < local.size(); ++i) {
        if (local[i] >= patch.initial.uv_to_surface_vertex.size()
            || patch.initial.uv_to_surface_vertex[local[i]] != input.source_vertices[i]) {
            return parameterization_failed;
        }
    }
    Eigen::Matrix<double, 3, 2> basis;
    basis.col(0) = (input.source_positions.row(1) - input.source_positions.row(0)).transpose();
    basis.col(1) = (input.source_positions.row(2) - input.source_positions.row(0)).transpose();
    const Eigen::Vector2d coefficients = basis.colPivHouseholderQr().solve(input.tangent_direction);
    if ((basis * coefficients - input.tangent_direction).norm() > 1e-8) {
        return parameterization_failed;
    }
    const auto& uv = patch.coordinates;
    const double uv_area = orientation(uv.row(local[0]), uv.row(local[1]), uv.row(local[2]));
    const double uv_scale = std::max((uv.row(local[1]) - uv.row(local[0])).squaredNorm(),
        (uv.row(local[2]) - uv.row(local[0])).squaredNorm());
    if (!std::isfinite(uv_area) || uv_area <= kRelativeAreaTolerance * uv_scale) {
        return parameterization_failed;
    }
    Point tangent = coefficients.x() * (uv.row(local[1]) - uv.row(local[0])).transpose()
        + coefficients.y() * (uv.row(local[2]) - uv.row(local[0])).transpose();
    if (tangent.squaredNorm() < 1e-24) {
        return parameterization_failed;
    }
    tangent.normalize();
    const Point perpendicular { -tangent.y(), tangent.x() };
    const Point center = uv.row(patch.initial.center);
    const Point xaxis = 2 * input.direction_magnitude * tangent;
    const Point yaxis = 2 * input.direction_magnitude * perpendicular;
    return fit_placement({ center - 0.5 * (xaxis + yaxis), xaxis, yaxis },
        input.footprint, uv, patch.initial.boundary);
}

std::expected<PreparedSurfacePatch, SurfaceFailure> prepare_surface_patch(
    fit_on_surface::Mesh& mesh, const PlacementInput& input)
{
    return prepare_impl(mesh, input);
}

std::expected<PreparedSurfacePatch, SurfaceFailure> prepare_surface_patch(
    image_relief::Mesh& mesh, const PlacementInput& input)
{
    return prepare_impl(mesh, input);
}
}

// Polygon projection, labeling, and subdivision transfer.

namespace {
using namespace relief::surface;
auto map_polygon_to_uv_frame(const std::vector<std::array<double, 2>>& polygon_points, const UvPlacementFrame& frame)
{
    return polygon_points | views::transform([&frame](auto&& point) {
        std::array<double, 2> result;
        Eigen::Vector2d::Map(result.data()) = frame.origin + frame.xaxis * point[0] + frame.yaxis * point[1];
        return result;
    }) | ranges::to<std::vector>();
}

enum class PolylineSide {
    Left,
    Right,
};

using OrientedPolylines = std::vector<std::vector<std::size_t>>;
using PolylinePolygonSides = std::vector<std::pair<std::size_t, std::size_t>>;

bool lexicographically_less(const std::vector<std::size_t>& lhs, const std::vector<std::size_t>& rhs)
{
    return ranges::lexicographical_compare(lhs, rhs);
}

struct PolylineVerticesLess {
    bool operator()(const std::vector<std::size_t>& lhs, const std::vector<std::size_t>& rhs) const
    {
        return lexicographically_less(lhs, rhs);
    }
};

std::vector<std::size_t> closed_rotation_from_anchor(
    const std::vector<std::size_t>& cycle,
    const std::size_t anchor,
    const bool reverse)
{
    auto vertices = cycle;
    std::rotate(vertices.begin(), vertices.begin() + static_cast<std::ptrdiff_t>(anchor), vertices.end());
    if (reverse) {
        ranges::reverse(vertices.begin() + 1, vertices.end());
    }
    vertices.push_back(vertices.front());
    return vertices;
}

std::pair<std::vector<std::size_t>, PolylineSide> canonicalize_polyline(const std::vector<std::size_t>& vertices)
{
    if (vertices.front() == vertices.back()) {
        const std::vector<std::size_t> cycle(vertices.begin(), vertices.end() - 1);
        const auto min_iter = ranges::min_element(cycle);
        const auto min_idx = static_cast<std::size_t>(std::distance(cycle.begin(), min_iter));
        auto forward = closed_rotation_from_anchor(cycle, min_idx, false);
        auto reversed = closed_rotation_from_anchor(cycle, min_idx, true);
        if (lexicographically_less(reversed, forward)) {
            return { std::move(reversed), PolylineSide::Right };
        }
        return { std::move(forward), PolylineSide::Left };
    }

    auto reversed = vertices;
    ranges::reverse(reversed);
    if (lexicographically_less(reversed, vertices)) {
        return { std::move(reversed), PolylineSide::Right };
    }
    return { vertices, PolylineSide::Left };
}

void add_oriented_polyline(
    OrientedPolylines& polylines,
    PolylinePolygonSides& polygon_sides,
    std::map<std::vector<std::size_t>, std::size_t, PolylineVerticesLess>& polyline_index,
    const std::vector<std::size_t>& vertices,
    const std::size_t polygon_idx)
{
    auto [canonical_vertices, polygon_side] = canonicalize_polyline(vertices);
    auto [iter, inserted] = polyline_index.emplace(canonical_vertices, polylines.size());
    if (inserted) {
        polylines.push_back(std::move(canonical_vertices));
        polygon_sides.emplace_back(gpf::kInvalidIndex, gpf::kInvalidIndex);
    }

    auto& side_polygon = polygon_side == PolylineSide::Left
        ? polygon_sides[iter->second].first
        : polygon_sides[iter->second].second;
    side_polygon = polygon_idx;
}

[[nodiscard]] std::expected<void, SurfaceFailure> set_face_polygon_id(uv::Mesh& mesh, const gpf::FaceId fid, const std::size_t polygon_id)
{
    if (!fid.valid() || polygon_id == gpf::kInvalidIndex) {
        return {};
    }

    auto& face_polygon_id = mesh.face_prop(fid).polygon_id;
    if (face_polygon_id != gpf::kInvalidIndex && face_polygon_id != polygon_id) {
        return std::unexpected(SurfaceFailure::ConflictingPolygonLabels);
    }
    face_polygon_id = polygon_id;
    return {};
}

[[nodiscard]] std::expected<void, SurfaceFailure> label_uv_mesh_polygon_ids(
    uv::Mesh& mesh,
    const std::vector<std::vector<gpf::HalfedgeId>>& projected_polyline_paths,
    const PolylinePolygonSides& polyline_polygon_sides)
{
    std::vector<bool> is_polygon_boundary(mesh.n_halfedges_capacity(), false);
    // Polygon paths may traverse both directions of the same mesh edge when
    // that edge is internal to the polygon region. Cancel those twin pairs so
    // flood fill can cross them instead of treating them as region boundaries.
    for (const auto& halfedges : projected_polyline_paths) {
        for (const auto hid : halfedges) {
            auto twin_hid = mesh.he_twin(hid);
            if (is_polygon_boundary[twin_hid.idx]) {
                is_polygon_boundary[twin_hid.idx] = false;
            } else {
                is_polygon_boundary[hid.idx] = true;
            }
        }
    }

    // After cancellation, remaining halfedges are true polygon boundaries.
    // The oriented halfedge sees the left polygon on its face and the right
    // polygon on its twin face.
    for (std::size_t path_idx = 0; path_idx < projected_polyline_paths.size(); ++path_idx) {
        const auto [left_polygon_id, right_polygon_id] = polyline_polygon_sides[path_idx];
        for (const auto hid : projected_polyline_paths[path_idx]) {
            if (!is_polygon_boundary[hid.idx]) {
                continue;
            }
            const auto twin_hid = mesh.he_twin(hid);
            is_polygon_boundary[twin_hid.idx] = true;
            if (auto result = set_face_polygon_id(mesh, mesh.he_face(hid), left_polygon_id); !result) {
                return std::unexpected(result.error());
            }
            if (auto result = set_face_polygon_id(mesh, mesh.he_face(twin_hid), right_polygon_id); !result) {
                return std::unexpected(result.error());
            }
        }
    }

    std::vector<gpf::FaceId> pending_faces;
    pending_faces.reserve(mesh.n_faces());
    for (const auto face : mesh.faces()) {
        if (face.prop().polygon_id != gpf::kInvalidIndex) {
            pending_faces.push_back(face.id);
        }
    }

    while (!pending_faces.empty()) {
        const auto fid = pending_faces.back();
        pending_faces.pop_back();
        const auto polygon_id = mesh.face_prop(fid).polygon_id;

        for (const auto halfedge : mesh.face(fid).halfedges()) {
            if (is_polygon_boundary[halfedge.id.idx]) {
                continue;
            }

            const auto adjacent_fid = halfedge.twin().face().id;
            if (!adjacent_fid.valid()) {
                continue;
            }

            auto& adjacent_polygon_id = mesh.face_prop(adjacent_fid).polygon_id;
            if (adjacent_polygon_id == gpf::kInvalidIndex) {
                adjacent_polygon_id = polygon_id;
                pending_faces.push_back(adjacent_fid);
            } else if (adjacent_polygon_id != polygon_id) {
                return std::unexpected(SurfaceFailure::ConflictingPolygonLabels);
            }
        }
    }
    return {};
}

struct EdgeSplitRequest {
    gpf::VertexId uv_vertex;
    double t;
    std::array<double, 3> point;
};

std::vector<gpf::VertexId> make_mesh_to_local_uv_vertex_map(
    const fit_on_surface::Mesh& mesh,
    const std::vector<gpf::VertexId>& local_to_mesh_vertex)
{
    std::vector<gpf::VertexId> mesh_to_local_uv_vertex(mesh.n_vertices_capacity(), gpf::VertexId {});
    for (std::size_t local_idx = 0; local_idx < local_to_mesh_vertex.size(); ++local_idx) {
        mesh_to_local_uv_vertex[local_to_mesh_vertex[local_idx].idx] = gpf::VertexId { local_idx };
    }
    return mesh_to_local_uv_vertex;
}

std::unordered_map<gpf::EdgeId, std::vector<EdgeSplitRequest>> collect_edge_split_requests(
    const fit_on_surface::Mesh& mesh,
    const uv::Mesh& uv_mesh,
    const std::vector<gpf::VertexId>& mesh_to_local_uv_vertex,
    const std::unordered_map<gpf::EdgeId, std::vector<gpf::EdgeId>>& subedges_by_parent,
    const std::vector<gpf::EdgeId>& base_uv_edges,
    std::vector<gpf::VertexId>& local_to_mesh_vertex)
{
    std::unordered_map<gpf::EdgeId, std::vector<EdgeSplitRequest>> edge_requests;
    for (const auto& [base_edge_id, subedges] : subedges_by_parent) {
        if (base_edge_id.idx >= base_uv_edges.size()) {
            continue;
        }
        const auto mesh_eid = base_uv_edges[base_edge_id.idx];
        if (!mesh_eid.valid()) {
            continue;
        }
        const auto [va, vb] = mesh.e_vertices(mesh_eid);
        const auto base_uv_va = mesh_to_local_uv_vertex[va.idx];
        const auto base_uv_vb = mesh_to_local_uv_vertex[vb.idx];

        auto uv_pa = Eigen::Vector2d::Map(uv_mesh.vertex_prop(base_uv_va).pt.data());
        auto uv_pb = Eigen::Vector2d::Map(uv_mesh.vertex_prop(base_uv_vb).pt.data());
        auto pa = Eigen::Vector3d::Map(mesh.vertex_prop(va).pt.data());
        auto pb = Eigen::Vector3d::Map(mesh.vertex_prop(vb).pt.data());

        Eigen::Vector2d edge_vec = uv_pb - uv_pa;
        const double edge_len_sq = edge_vec.squaredNorm();

        for (const auto subedge_id : subedges) {
            for (const auto uv_vertex : uv_mesh.edge(subedge_id).vertices()) {
                if (local_to_mesh_vertex[uv_vertex.id.idx].valid()) {
                    continue;
                }

                local_to_mesh_vertex[uv_vertex.id.idx] = gpf::VertexId { 0 };
                const double t = std::min(std::max((Eigen::Vector2d::Map(uv_vertex.prop().pt.data()) - uv_pa).dot(edge_vec) / edge_len_sq, 0.0), 1.0);
                std::array<double, 3> point {};
                Eigen::Vector3d::Map(point.data()) = pa * (1.0 - t) + t * pb;
                edge_requests[mesh_eid].push_back(EdgeSplitRequest {
                    .uv_vertex = uv_vertex.id,
                    .t = t,
                    .point = std::move(point),
                });
            }
        }
    }
    return edge_requests;
}

void apply_edge_split_requests(
    fit_on_surface::Mesh& mesh,
    std::unordered_map<gpf::EdgeId, std::vector<EdgeSplitRequest>>& edge_requests,
    std::vector<gpf::VertexId>& uv_to_mesh_vertex)
{
    for (auto& [eid, requests] : edge_requests) {
        if (requests.empty()) {
            continue;
        }

        ranges::sort(requests, {}, &EdgeSplitRequest::t);

        gpf::EdgeId current_eid = eid;

        for (const auto& request : requests) {
            const auto new_vertex = mesh.split_edge(current_eid);
            mesh.vertex_prop(new_vertex).pt = request.point;
            uv_to_mesh_vertex[request.uv_vertex.idx] = new_vertex;
            current_eid = mesh.vertex(new_vertex).halfedge().edge().id;
        }
    }
}

void add_face_inner_vertices(
    fit_on_surface::Mesh& mesh,
    const uv::Mesh& uv_mesh,
    const std::size_t n_old_vertices,
    const std::span<const gpf::FaceId> inner_faces,
    const std::vector<gpf::FaceId>& uv_vertex_root_face,
    const std::vector<gpf::VertexId>& mesh_to_local_uv_vertex,
    std::vector<gpf::VertexId>& local_to_mesh_vertex)
{
    std::unordered_map<gpf::FaceId, std::vector<gpf::VertexId>> face_inner_vertices;
    for (std::size_t idx { n_old_vertices }; idx < uv_mesh.n_vertices_capacity(); idx++) {
        const gpf::VertexId vid { idx };
        if (local_to_mesh_vertex[idx].valid()) {
            continue;
        }

        const auto root = uv_vertex_root_face[idx];
        face_inner_vertices[root].push_back(vid);
    }
    for (const auto& [fid, vertices] : face_inner_vertices) {
        std::vector<double> local_points;
        local_points.reserve((3 + vertices.size()) * 2);
        std::array<gpf::VertexId, 3> triangle_vertices;
        for (const auto [idx, he] : views::zip(ranges::iota_view { std::size_t { 0 }, std::size_t { 3 } }, mesh.face(inner_faces[fid.idx]).halfedges())) {
            triangle_vertices[idx] = he.from().id;
        }
        for (const auto vid : triangle_vertices) {
            assert(mesh_to_local_uv_vertex[vid.idx].valid());
            local_points.append_range(uv_mesh.vertex_prop(mesh_to_local_uv_vertex[vid.idx]).pt);
        }
        for (const auto vid : vertices) {
            local_points.append_range(uv_mesh.vertex_prop(vid).pt);
        }

        auto bary_coords = gpf::detail::compute_bary_coordinates(local_points);
        Eigen::Vector3d pa = Eigen::Vector3d::Map(mesh.vertex_prop(triangle_vertices[0]).pt.data());
        Eigen::Vector3d pb = Eigen::Vector3d::Map(mesh.vertex_prop(triangle_vertices[1]).pt.data());
        Eigen::Vector3d pc = Eigen::Vector3d::Map(mesh.vertex_prop(triangle_vertices[2]).pt.data());
        auto new_vid = mesh.new_vertices(vertices.size());
        for (std::size_t i = 0; i < vertices.size(); ++i) {
            std::span<double, 3> bary { bary_coords.data() + i * 3, 3 };
            // There is no need to normalize the barycentric coordinates: the point is
            // guaranteed to lie inside the triangle. Normalization could make two
            // previously disjoint segments appear to intersect after the adjustment.
            Eigen::Vector3d::Map(mesh.vertex_prop(new_vid).pt.data()) = bary[0] * pa + bary[1] * pb + bary[2] * pc;
            local_to_mesh_vertex[vertices[i].idx] = new_vid;
            new_vid.idx += 1;
        }
    }
}

void map_subdivided_uv_mesh_to_surface(
    fit_on_surface::Mesh& mesh,
    const uv::Mesh& uv_mesh,
    std::vector<gpf::VertexId>& local_to_mesh_vertex,
    const std::span<const gpf::FaceId> inner_faces,
    const std::unordered_map<gpf::FaceId, gpf::FaceId>& uv_face_parent_map,
    const std::unordered_map<gpf::EdgeId, gpf::EdgeId>& uv_edge_parent_map,
    const std::vector<gpf::EdgeId>& base_uv_edges)
{
    const auto n_old_vertices = local_to_mesh_vertex.size();
    std::unordered_map<gpf::EdgeId, std::vector<gpf::EdgeId>> subedges_by_parent;
    {
        for (const auto [child, parent] : uv_edge_parent_map) {
            subedges_by_parent[parent].push_back(child);
        }
    }
    const auto mesh_to_local_uv_vertex = make_mesh_to_local_uv_vertex_map(mesh, local_to_mesh_vertex);
    local_to_mesh_vertex.resize(uv_mesh.n_vertices_capacity());
    auto edge_requests = collect_edge_split_requests(
        mesh,
        uv_mesh,
        mesh_to_local_uv_vertex,
        subedges_by_parent,
        base_uv_edges,
        local_to_mesh_vertex);

    std::vector<gpf::FaceId> uv_vertex_root_face(uv_mesh.n_vertices_capacity());
    for (const auto face : uv_mesh.faces()) {
        const auto parent_iter = uv_face_parent_map.find(face.id);
        const auto root = parent_iter != uv_face_parent_map.end() ? parent_iter->second : face.id;
        for (const auto halfedge : face.halfedges()) {
            auto& incident_root = uv_vertex_root_face[halfedge.from().id.idx];
            if (!incident_root.valid()) {
                incident_root = root;
            }
        }
    }

    add_face_inner_vertices(
        mesh,
        uv_mesh,
        n_old_vertices,
        inner_faces,
        uv_vertex_root_face,
        mesh_to_local_uv_vertex,
        local_to_mesh_vertex);
    apply_edge_split_requests(mesh, edge_requests, local_to_mesh_vertex);

    std::vector<std::vector<gpf::VertexId>> replacement_triangles(inner_faces.size());
    std::vector<std::vector<gpf::FaceId>> replacement_uv_faces(inner_faces.size());
    for (const auto face : uv_mesh.faces()) {
        const auto parent_iter = uv_face_parent_map.find(face.id);
        const auto root = parent_iter != uv_face_parent_map.end() ? parent_iter->second : face.id;
        std::array<gpf::VertexId, 3> triangle_vertices;
        for (auto [idx, he] : views::zip(ranges::iota_view { std::size_t { 0 }, std::size_t { 3 } }, face.halfedges())) {
            triangle_vertices[idx] = local_to_mesh_vertex[he.from().id.idx];
        }
        replacement_triangles[root.idx].append_range(std::move(triangle_vertices));
        replacement_uv_faces[root.idx].push_back(face.id);
    }

    for (std::size_t face_idx = 0; face_idx < inner_faces.size(); ++face_idx) {
        auto& triangles = replacement_triangles[face_idx];
        const auto& uv_faces = replacement_uv_faces[face_idx];
        assert(triangles.size() == uv_faces.size() * std::size_t { 3 });
        if (triangles.empty()) {
            continue;
        }
        if (triangles.size() == std::size_t { 3 }) {
            mesh.face_prop(inner_faces[face_idx]) = uv_mesh.face_prop(uv_faces.front());
            continue;
        }
        const auto n_faces_before = mesh.n_faces_capacity();
        mesh.split_face_into_triangles(inner_faces[face_idx], triangles);
        mesh.face_prop(inner_faces[face_idx]) = uv_mesh.face_prop(uv_faces.front());
        for (std::size_t uv_face_idx = 1; uv_face_idx < uv_faces.size(); ++uv_face_idx) {
            mesh.face_prop(gpf::FaceId { n_faces_before + uv_face_idx - 1 }) = uv_mesh.face_prop(uv_faces[uv_face_idx]);
        }
    }
}

std::pair<OrientedPolylines, PolylinePolygonSides> divide_polygons_into_oriented_polylines(
    const std::vector<std::vector<std::vector<std::size_t>>>& polygons)
{
    std::vector<std::pair<std::size_t, std::vector<std::size_t>>> rings;
    for (std::size_t polygon_idx = 0; polygon_idx < polygons.size(); ++polygon_idx) {
        for (const auto& ring : polygons[polygon_idx]) {
            rings.emplace_back(polygon_idx, ring);
        }
    }

    std::unordered_map<std::size_t, std::size_t> vertex_incidence;
    for (const auto& [polygon_idx, ring] : rings) {
        for (std::size_t i = 0; i < ring.size(); ++i) {
            ++vertex_incidence[ring[i]];
            ++vertex_incidence[ring[(i + 1) % ring.size()]];
        }
    }

    auto is_split_vertex = [&vertex_incidence](const std::size_t vertex_idx) {
        const auto iter = vertex_incidence.find(vertex_idx);
        return iter != vertex_incidence.end() && iter->second > 2;
    };

    OrientedPolylines polylines;
    PolylinePolygonSides polygon_sides;
    std::map<std::vector<std::size_t>, std::size_t, PolylineVerticesLess> polyline_index;
    for (const auto& [polygon_idx, ring] : rings) {
        std::vector<std::size_t> split_positions;
        for (std::size_t i = 0; i < ring.size(); ++i) {
            if (is_split_vertex(ring[i])) {
                split_positions.push_back(i);
            }
        }

        if (split_positions.empty()) {
            auto vertices = ring;
            vertices.push_back(vertices.front());
            add_oriented_polyline(polylines, polygon_sides, polyline_index, vertices, polygon_idx);
            continue;
        }

        for (std::size_t i = 0; i < split_positions.size(); ++i) {
            const auto start = split_positions[i];
            const auto end = split_positions[(i + 1) % split_positions.size()];
            std::vector<std::size_t> vertices;

            auto cursor = start;
            while (true) {
                vertices.push_back(ring[cursor]);
                if (cursor == end && vertices.size() > 1) {
                    break;
                }
                cursor = (cursor + 1) % ring.size();
            }

            add_oriented_polyline(polylines, polygon_sides, polyline_index, vertices, polygon_idx);
        }
    }

    return { std::move(polylines), std::move(polygon_sides) };
}

std::expected<Footprint, SurfaceFailure> polygon_footprint(
    const std::vector<std::array<double, 2>>& points,
    const std::vector<std::vector<std::vector<std::size_t>>>& polygons)
{
    Footprint footprint { Eigen::Vector2d::Zero() };
    for (const auto& polygon : polygons) {
        for (const auto& ring : polygon) {
            if (ring.size() < 3) {
                return std::unexpected(SurfaceFailure::DegeneratePolygon);
            }
            for (const auto index : ring) {
                if (index >= points.size()) {
                    return std::unexpected(SurfaceFailure::InvalidPolylinePointIndex);
                }
            }
        }
    }
    // Projection also visits unreferenced supplied points, so include them in
    // the convex support rectangle along with every used segment.
    for (const auto& point : points) {
        const Eigen::Vector2d p = Eigen::Vector2d::Map(point.data());
        footprint.half_extent = footprint.half_extent.cwiseMax((p.array() - 0.5).abs().matrix());
    }
    if ((footprint.half_extent.array() <= 0).any()) {
        return std::unexpected(SurfaceFailure::DegeneratePolygon);
    }
    return footprint;
}
} // namespace
namespace fit_on_surface {
std::expected<void, SurfaceFailure> fit_polygon_on_surface(
    fit_on_surface::Mesh& mesh,
    const std::vector<std::array<double, 2>>& polygon_points,
    const std::vector<std::vector<std::vector<std::size_t>>>& polygons,
    const std::array<double, 3>& surface_point,
    const gpf::FaceId fid,
    const std::array<double, 3>& direction)
{
    auto footprint = polygon_footprint(polygon_points, polygons);
    if (!footprint) {
        return std::unexpected(footprint.error());
    }
    const auto input = make_placement_input(mesh, fid, surface_point, direction, *footprint);
    auto patch = prepare_surface_patch(mesh, input);
    if (!patch) {
        return std::unexpected(patch.error());
    }
    auto& uv_mesh = patch->uv_mesh;

    constexpr double EPS = kProjectionTolerance;
    auto poly_uv_pts = map_polygon_to_uv_frame(polygon_points, patch->frame);
    const auto [oriented_polylines, polyline_polygon_sides] = divide_polygons_into_oriented_polylines(polygons);
    auto base_uv_edges = make_base_uv_edges(mesh, uv_mesh, patch->uv_to_surface_vertex);
    std::unordered_map<gpf::FaceId, gpf::FaceId> uv_face_parent_map;
    std::unordered_map<gpf::EdgeId, gpf::EdgeId> uv_edge_parent_map;
    auto projection = gpf::project_polylines_on_mesh(poly_uv_pts, oriented_polylines, uv_mesh, EPS, &uv_face_parent_map, &uv_edge_parent_map);
    if (!projection) {
        return std::unexpected(from_projection_failure(projection.error()));
    }
    auto projected_polyline_paths = std::move(projection->second);
    if (auto result = label_uv_mesh_polygon_ids(uv_mesh, projected_polyline_paths, polyline_polygon_sides); !result) {
        return std::unexpected(result.error());
    }
    if (auto result = write_uv_mesh_as_off(uv_mesh, "fit_polygon_on_surface_projected_uv_mesh.off"); !result) {
        return std::unexpected(result.error());
    }
    map_subdivided_uv_mesh_to_surface(
        mesh,
        uv_mesh,
        patch->uv_to_surface_vertex,
        patch->surface_faces,
        uv_face_parent_map,
        uv_edge_parent_map,
        base_uv_edges);
    if (auto result = write_mesh_as_off(mesh, "fit_polygon_on_surface_final.off"); !result) {
        return std::unexpected(result.error());
    }
    return {};
}
}

// Relief sampling, boundary projection, assembly, and grid mappings.

namespace {
using namespace relief::surface;
[[nodiscard]] std::expected<void, SurfaceFailure> write_grid_mesh_as_off(const VMat& points, const std::size_t width, const std::string& path)
{
    if (width < 2 || points.rows() != static_cast<Eigen::Index>(width * width) || points.cols() != 3) {
        return std::unexpected(SurfaceFailure::GridMatrixDimensionMismatch);
    }

    const auto face_count = (width - 1) * (width - 1);
    std::ofstream file(path);
    if (!file) {
        return std::unexpected(SurfaceFailure::GridOffOpenFailed);
    }

    file << "OFF\n";
    file << points.rows() << ' ' << face_count << " 0\n";
    for (Eigen::Index i = 0; i < points.rows(); ++i) {
        file << points(i, 0) << ' ' << points(i, 1) << ' ' << points(i, 2) << '\n';
    }

    for (std::size_t j = 0; j + 1 < width; ++j) {
        for (std::size_t i = 0; i + 1 < width; ++i) {
            const auto lower_left = j * width + i;
            const auto lower_right = lower_left + 1;
            const auto upper_right = lower_right + width;
            const auto upper_left = lower_left + width;
            file << "4 " << lower_left << ' ' << lower_right << ' ' << upper_right << ' ' << upper_left << '\n';
        }
    }
    return {};
}

void smooth_grid_points(VMat& points, const std::size_t width)
{
    const VMat original_points = points;
    for (std::size_t j = 1; j + 1 < width; ++j) {
        for (std::size_t i = 1; i + 1 < width; ++i) {
            const auto point_idx = j * width + i;
            points.row(point_idx) = 0.25 * (original_points.row(point_idx - 1) + original_points.row(point_idx + 1) + original_points.row(point_idx - width) + original_points.row(point_idx + width));
        }
    }
}

struct SampledReliefGrid {
    std::vector<std::array<double, 2>> uv_points;
    VMat positions;
    double projection_tolerance;
};

[[nodiscard]] std::expected<SampledReliefGrid, SurfaceFailure> sample_relief_grid(
    const image_relief::Mesh& mesh,
    const PreparedSurfacePatch& patch,
    const std::span<const double> heights,
    const std::size_t width)
{
    const auto& uv_mesh = patch.uv_mesh;
    const auto& local_to_mesh_vertex = patch.uv_to_surface_vertex;
    const auto& [start_pt, xaxis, yaxis] = patch.frame;
    const auto point_count = width * width;
    assert(heights.size() == point_count);

    std::vector<std::array<double, 2>> points(point_count);
    const auto t = 1.0 / static_cast<double>(width - 1);
    const Eigen::Vector2d delta_x = xaxis * t;
    const Eigen::Vector2d delta_y = yaxis * t;
    std::size_t idx { 0 };
    for (std::size_t j { 0 }; j < width; j++) {
        Eigen::Vector2d::Map(points[idx++].data()) = start_pt + static_cast<double>(j) * delta_y;
        for (std::size_t i { 1 }; i < width; i++) {
            Eigen::Vector2d::Map(points[idx].data()) = Eigen::Vector2d::Map(points[idx - 1].data()) + delta_x;
            idx++;
        }
    }

    // The classifier uses this as a distance tolerance.  One whole grid cell
    // is large enough to classify valid interior samples as edge samples, so
    // keep the tolerance to a small fraction of the smaller cell dimension.
    const auto eps = std::min(kProjectionTolerance, 0.01 * delta_x.norm());
    const auto start = std::chrono::high_resolution_clock::now();
    auto [face_info_map, point_vertices, edge_to_points_map] = gpf::detail::prepare_projected_points(points, uv_mesh, eps);
    std::cout << "elapsed in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start) << "\n";

    VMat N = VMat::Constant(mesh.n_faces_capacity(), 3, std::numeric_limits<double>::quiet_NaN()).eval(); // face normals
    VMat P = VMat::Constant(points.size(), 3, std::numeric_limits<double>::quiet_NaN()).eval();
    using FaceRepr = std::variant<gpf::FaceId, std::array<gpf::VertexId, 3>>;

    auto get_face_vertices = [](const auto& mesh, const gpf::FaceId fid) -> std::array<gpf::VertexId, 3> {
        std::array<gpf::VertexId, 3> vertices {};
        auto he = mesh.face(fid).halfedge();
        vertices[0] = he.to().id;
        he = he.next();
        vertices[1] = he.to().id;
        he = he.next();
        vertices[2] = he.to().id;
        return vertices;
    };

    auto get_face_normal = [&mesh, &N, &get_face_vertices](const FaceRepr face_repr) -> Eigen::RowVector3d {
        const auto compute_normal = [&mesh](const std::array<gpf::VertexId, 3>& vertices) {
            const auto pa = Eigen::RowVector3d::Map(mesh.vertex_prop(vertices[0]).pt.data());
            const auto pb = Eigen::RowVector3d::Map(mesh.vertex_prop(vertices[1]).pt.data());
            const auto pc = Eigen::RowVector3d::Map(mesh.vertex_prop(vertices[2]).pt.data());
            return ((pb - pa).cross(pc - pa)).normalized().eval();
        };

        return std::visit([&mesh, &N, &get_face_vertices, &compute_normal](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, gpf::FaceId>) {
                auto row = N.row(arg.idx);
                if (!std::isnan(row[0])) {
                    return Eigen::RowVector3d { row };
                }
                row = compute_normal(get_face_vertices(mesh, arg));
                return Eigen::RowVector3d { row };
            } else {
                return compute_normal(arg);
            }
        },
            face_repr);
    };

    for (const auto& [uv_fid, info] : face_info_map) {
        const auto uv_vertices = get_face_vertices(uv_mesh, uv_fid);
        std::array<gpf::VertexId, 3> mesh_vertices = uv_vertices;
        mesh_vertices[0] = local_to_mesh_vertex[uv_vertices[0].idx];
        mesh_vertices[1] = local_to_mesh_vertex[uv_vertices[1].idx];
        mesh_vertices[2] = local_to_mesh_vertex[uv_vertices[2].idx];

        auto normal = get_face_normal(mesh_vertices);
        N.row(patch.surface_faces[uv_fid.idx].idx) = normal;

        std::vector<double> barycentric_input;
        barycentric_input.reserve(6 + 2 * info.point_indices.size());
        for (const auto uv_vid : uv_vertices) {
            barycentric_input.append_range(uv_mesh.vertex_prop(uv_vid).pt);
        }
        for (const auto pid : info.point_indices) {
            barycentric_input.append_range(points[pid]);
        }
        const auto bary_coords = gpf::detail::compute_bary_coordinates(barycentric_input);

        const auto pa = Eigen::RowVector3d::Map(mesh.vertex_prop(mesh_vertices[0]).pt.data());
        const auto pb = Eigen::RowVector3d::Map(mesh.vertex_prop(mesh_vertices[1]).pt.data());
        const auto pc = Eigen::RowVector3d::Map(mesh.vertex_prop(mesh_vertices[2]).pt.data());
        for (std::size_t i = 0; i < info.point_indices.size(); ++i) {
            const auto pid = info.point_indices[i];
            const auto bary = std::span<const double, 3> { bary_coords.data() + 3 * i, 3 };
            P.row(pid) = bary[0] * pa + bary[1] * pb + bary[2] * pc;
            P.row(pid) += heights[pid] * normal;
        }
    }

    auto base_uv_edges = make_base_uv_edges(mesh, uv_mesh, local_to_mesh_vertex);
    for (const auto& [uv_eid, point_indices] : edge_to_points_map) {
        const auto mesh_eid = base_uv_edges[uv_eid.idx];
        auto normal = Eigen::RowVector3d::Zero().eval();
        for (const auto he : mesh.edge(mesh_eid).halfedges()) {
            const auto mesh_fid = he.face().id;
            if (mesh_fid.valid()) {
                normal += get_face_normal(FaceRepr { mesh_fid });
            }
        }
        normal.normalize();

        const auto [uv_va, uv_vb] = uv_mesh.edge(uv_eid).vertices();
        const auto mesh_va = local_to_mesh_vertex[uv_va.id.idx];
        const auto mesh_vb = local_to_mesh_vertex[uv_vb.id.idx];
        const auto uv_pa = Eigen::Vector2d::Map(uv_mesh.vertex_prop(uv_va.id).pt.data());
        const auto uv_pb = Eigen::Vector2d::Map(uv_mesh.vertex_prop(uv_vb.id).pt.data());
        const auto uv_edge = (uv_pb - uv_pa).eval();
        const auto uv_edge_length_sq = uv_edge.squaredNorm();
        const auto pa = Eigen::RowVector3d::Map(mesh.vertex_prop(mesh_va).pt.data());
        const auto pb = Eigen::RowVector3d::Map(mesh.vertex_prop(mesh_vb).pt.data());
        for (const auto pid : point_indices) {
            const auto point = Eigen::Vector2d::Map(points[pid].data());
            const auto edge_t = std::clamp((point - uv_pa).dot(uv_edge) / uv_edge_length_sq, 0.0, 1.0);
            P.row(pid) = (1.0 - edge_t) * pa + edge_t * pb;
            P.row(pid) += heights[pid] * normal;
        }
    }

    for (std::size_t i { 0 }; i < points.size(); i++) {
        if (!std::isnan(P(i, 0))) {
            continue;
        }

        if (!point_vertices[i].valid() || point_vertices[i].idx >= local_to_mesh_vertex.size()) {
            return std::unexpected(SurfaceFailure::InvalidUvVertexReference);
        }
        const auto mesh_vid = local_to_mesh_vertex[point_vertices[i].idx];

        auto normal = Eigen::RowVector3d::Zero().eval();
        for (const auto he : mesh.vertex(mesh_vid).incoming_halfedges()) {
            const auto mesh_fid = he.face().id;
            if (mesh_fid.valid()) {
                normal += get_face_normal(FaceRepr { mesh_fid });
            }
        }
        normal.normalize();
        P.row(i) = Eigen::RowVector3d::Map(mesh.vertex_prop(mesh_vid).pt.data());
        P.row(i) += heights[i] * normal;
    }
    for (std::size_t iteration = 0; iteration < 15; ++iteration) {
        smooth_grid_points(P, width);
    }
    if (auto result = write_grid_mesh_as_off(P, width, "fit_polygon_on_surface_grid.off"); !result) {
        return std::unexpected(result.error());
    }
    return SampledReliefGrid { std::move(points), std::move(P), eps };
}

auto get_grid_boundary_points_and_indices(const std::vector<std::array<double, 2>>& points, std::size_t width)
{
    const auto index_mat = MatXu::Constant(width, width, gpf::kInvalidIndex).eval();
    std::vector<std::array<double, 2>> grid_boundary_points;
    std::vector<std::size_t> grid_boundary_point_indices;
    const auto cap = (width - 1) << 2;
    grid_boundary_points.reserve(cap);
    grid_boundary_point_indices.reserve(cap);
    // bottom
    for (std::size_t i = 0; i + 1 < width; i++) {
        grid_boundary_points.push_back(points[i]);
        grid_boundary_point_indices.push_back(i);
    }
    // right
    std::size_t idx { width - 1 };
    for (std::size_t j = 0; j + 1 < width; j++) {
        grid_boundary_points.push_back(points[idx]);
        grid_boundary_point_indices.push_back(idx);
        idx += width;
    }
    // top
    for (std::size_t i = 0; i + 1 < width; i++) {
        grid_boundary_points.push_back(points[idx]);
        grid_boundary_point_indices.push_back(idx);
        idx -= 1;
    }
    // left
    for (std::size_t j = 0; j + 1 < width; j++) {
        grid_boundary_points.push_back(points[idx]);
        grid_boundary_point_indices.push_back(idx);
        idx -= width;
    }
    return std::make_pair(std::move(grid_boundary_points), std::move(grid_boundary_point_indices));
}

[[nodiscard]] std::expected<std::vector<std::size_t>, SurfaceFailure> compute_boundary_vertex_separators(const uv::Mesh& mesh, const std::vector<gpf::VertexId>& vertices, const std::vector<gpf::HalfedgeId>& halfedges)
{
    std::vector<std::size_t> separators;
    separators.reserve(vertices.size() + 1);
    separators.push_back(0);
    std::size_t vidx { 1 };
    auto vid = vertices[vidx];
    for (std::size_t i = 0; i < halfedges.size(); i++) {
        if (mesh.he_to(halfedges[i]) == vid) {
            separators.push_back(i + 1);
            vidx += 1;
            if (vidx < vertices.size()) {
                vid = vertices[vidx];
            }
        }
    }
    if (separators.size() != vertices.size()) {
        return std::unexpected(SurfaceFailure::MissingBoundarySeparator);
    }
    separators.push_back(halfedges.size());
    return separators;
}

struct ProjectedGridBoundary {
    std::vector<std::size_t> grid_point_indices;
    std::vector<gpf::VertexId> uv_vertices;
    std::vector<gpf::HalfedgeId> uv_halfedges;
    std::vector<std::size_t> separators;
    std::vector<gpf::FaceId> retained_uv_faces;
    std::vector<std::array<gpf::VertexId, 2>> original_uv_edge_vertices;
    std::unordered_map<gpf::FaceId, gpf::FaceId> uv_face_parent_map;
    std::unordered_map<gpf::EdgeId, gpf::EdgeId> uv_edge_parent_map;
};

[[nodiscard]] std::expected<ProjectedGridBoundary, SurfaceFailure> project_grid_boundary(uv::Mesh& uv_mesh, const SampledReliefGrid& grid, const std::size_t width)
{
    auto [grid_boundary_points, grid_boundary_point_indices] = get_grid_boundary_points_and_indices(grid.uv_points, width);
    std::vector<std::size_t> boundary_polylines;
    boundary_polylines.reserve(grid_boundary_point_indices.size());
    for (std::size_t i { 0 }; i < grid_boundary_point_indices.size(); i++) {
        boundary_polylines.push_back(i);
    }
    boundary_polylines.push_back(0);

    if (auto result = write_polyline_as_obj(grid_boundary_points, boundary_polylines, "fit_polygon_on_surface_boundary_polylines.obj"); !result) {
        return std::unexpected(result.error());
    }
    std::vector<std::array<gpf::VertexId, 2>> original_uv_edge_vertices(uv_mesh.n_edges_capacity());
    for (const auto edge : uv_mesh.edges()) {
        const auto [a, b] = edge.vertices();
        original_uv_edge_vertices[edge.id.idx] = { a.id, b.id };
    }
    std::unordered_map<gpf::FaceId, gpf::FaceId> uv_face_parent_map;
    std::unordered_map<gpf::EdgeId, gpf::EdgeId> uv_edge_parent_map;
    auto projection = gpf::project_polylines_on_mesh(
        grid_boundary_points,
        { boundary_polylines },
        uv_mesh,
        grid.projection_tolerance,
        &uv_face_parent_map,
        &uv_edge_parent_map);
    if (!projection) {
        return std::unexpected(from_projection_failure(projection.error()));
    }
    auto [grid_boundary_vertices, grid_boundary_halfedges] = std::move(*projection);
    if (auto result = write_uv_mesh_as_off(uv_mesh, "fit_polygon_on_surface_grid_uv_mesh.off"); !result) {
        return std::unexpected(result.error());
    }

    const auto& projected_boundary_halfedges = grid_boundary_halfedges.front();
    auto separators = compute_boundary_vertex_separators(uv_mesh, grid_boundary_vertices, projected_boundary_halfedges);
    if (!separators) {
        return std::unexpected(separators.error());
    }
    auto reversed_grid_boundary_halfedges = projected_boundary_halfedges;
    std::ranges::reverse(reversed_grid_boundary_halfedges);
    for (auto& hid : reversed_grid_boundary_halfedges) {
        hid = uv_mesh.he_twin(hid);
    }
    auto kept_uv_faces = gpf::surround_faces_by_halfedges(uv_mesh, reversed_grid_boundary_halfedges);
    return ProjectedGridBoundary {
        std::move(grid_boundary_point_indices),
        std::move(grid_boundary_vertices),
        std::move(grid_boundary_halfedges.front()),
        std::move(*separators),
        std::move(kept_uv_faces),
        std::move(original_uv_edge_vertices),
        std::move(uv_face_parent_map),
        std::move(uv_edge_parent_map),
    };
}

struct ReliefOutputFace {
    std::vector<std::size_t> vertices;
    image_relief::FaceProp prop;
};

struct ReliefAssembly {
    const image_relief::Mesh& surface_mesh;
    const PreparedSurfacePatch& patch;
    std::vector<std::array<double, 3>> output_positions;
    std::vector<ReliefOutputFace> output_faces;
    std::vector<image_relief::GridFaceIndex> grid_face_indices;
    std::vector<std::size_t> surface_to_output_vertex;
    std::vector<std::size_t> uv_to_output_vertex;

    ReliefAssembly(const image_relief::Mesh& mesh, const PreparedSurfacePatch& patch)
        : surface_mesh(mesh)
        , patch(patch)
        , surface_to_output_vertex(mesh.n_vertices_capacity(), gpf::kInvalidIndex)
        , uv_to_output_vertex(patch.uv_mesh.n_vertices_capacity(), gpf::kInvalidIndex)
    {
    }

    std::size_t output_vertex_for_surface_vertex(const gpf::VertexId surface_vid)
    {
        if (auto output_vid = surface_to_output_vertex[surface_vid.idx]; output_vid != gpf::kInvalidIndex) {
            return output_vid;
        }
        const auto ret = surface_to_output_vertex[surface_vid.idx] = output_positions.size();
        output_positions.push_back(surface_mesh.vertex_prop(surface_vid).pt);
        return ret;
    }

    std::size_t output_vertex_for_uv_vertex(const gpf::VertexId uv_vid)
    {
        auto& output_vid = uv_to_output_vertex[uv_vid.idx];
        if (output_vid != gpf::kInvalidIndex) {
            return output_vid;
        }

        if (uv_vid.idx < patch.uv_to_surface_vertex.size()) {
            output_vid = output_vertex_for_surface_vertex(patch.uv_to_surface_vertex[uv_vid.idx]);
        }
        assert(output_vid != gpf::kInvalidIndex);
        return output_vid;
    }
};

void append_relief_edge_vertices(ReliefAssembly& assembly, const ProjectedGridBoundary& boundary)
{
    const auto& mesh = assembly.surface_mesh;
    const auto& uv_mesh = assembly.patch.uv_mesh;
    const auto& local_to_mesh_vertex = assembly.patch.uv_to_surface_vertex;
    auto& output_positions = assembly.output_positions;
    auto& uv_output_vertices = assembly.uv_to_output_vertex;
    std::unordered_map<gpf::EdgeId, std::vector<gpf::EdgeId>> parent_edge_to_edges_map;
    for (const auto [uv_eid, uv_parent_eid] : boundary.uv_edge_parent_map) {
        if (uv_parent_eid.idx < boundary.original_uv_edge_vertices.size()) {
            parent_edge_to_edges_map[uv_parent_eid].push_back(uv_eid);
        }
    }

    for (auto& [parent_eid, subedges] : parent_edge_to_edges_map) {
        const auto [uv_va, uv_vb] = boundary.original_uv_edge_vertices[parent_eid.idx];
        const auto va = local_to_mesh_vertex[uv_va.idx];
        const auto vb = local_to_mesh_vertex[uv_vb.idx];
        const auto uv_pa = Eigen::Vector2d::Map(uv_mesh.vertex_prop(uv_va).pt.data());
        const auto uv_pb = Eigen::Vector2d::Map(uv_mesh.vertex_prop(uv_vb).pt.data());
        const auto pa = Eigen::Vector3d::Map(mesh.vertex_prop(va).pt.data());
        const auto pb = Eigen::Vector3d::Map(mesh.vertex_prop(vb).pt.data());
        const auto uv_vab = (uv_pb - uv_pa).eval();
        const auto square_len = uv_vab.squaredNorm();

        subedges.push_back(parent_eid);
        for (const auto eid : subedges) {
            for (const auto vid : uv_mesh.e_vertices(eid)) {
                if (vid.idx < local_to_mesh_vertex.size() || uv_output_vertices[vid.idx] != gpf::kInvalidIndex) {
                    continue;
                }
                const auto uv_pt = Eigen::Vector2d::Map(uv_mesh.vertex_prop(vid).pt.data());
                const auto t = (uv_pt - uv_pa).dot(uv_vab) / square_len;
                uv_output_vertices[vid.idx] = output_positions.size();
                output_positions.push_back({});
                Eigen::Vector3d::Map(output_positions.back().data()) = (1.0 - t) * pa + t * pb;
            }
        }
    }
}

void append_relief_face_vertices(ReliefAssembly& assembly, const ProjectedGridBoundary& boundary)
{
    const auto& mesh = assembly.surface_mesh;
    const auto& uv_mesh = assembly.patch.uv_mesh;
    const auto& local_to_mesh_vertex = assembly.patch.uv_to_surface_vertex;
    auto& output_positions = assembly.output_positions;
    auto& uv_output_vertices = assembly.uv_to_output_vertex;
    std::vector<gpf::VertexId> mesh_to_uv(mesh.n_vertices_capacity());
    for (std::size_t uv_idx = 0; uv_idx < local_to_mesh_vertex.size(); ++uv_idx) {
        const auto mesh_vid = local_to_mesh_vertex[uv_idx];
        if (mesh_vid.valid()) {
            mesh_to_uv[mesh_vid.idx] = gpf::VertexId { uv_idx };
        }
    }

    std::unordered_map<gpf::FaceId, std::vector<gpf::FaceId>> parent_face_to_faces_map;
    for (const auto [uv_fid, uv_parent_fid] : boundary.uv_face_parent_map) {
        parent_face_to_faces_map[uv_parent_fid].push_back(uv_fid);
    }

    for (auto& [parent_fid, subfaces] : parent_face_to_faces_map) {
        const auto source_fid = assembly.patch.surface_faces[parent_fid.idx];
        std::array<gpf::VertexId, 3> source_vertices;
        std::array<gpf::VertexId, 3> source_uv_vertices;
        for (const auto [idx, he] : views::zip(
                 ranges::iota_view { std::size_t { 0 }, std::size_t { 3 } },
                 mesh.face(source_fid).halfedges())) {
            source_vertices[idx] = he.from().id;
            source_uv_vertices[idx] = mesh_to_uv[source_vertices[idx].idx];
        }

        const auto uv_pa = Eigen::Vector2d::Map(uv_mesh.vertex_prop(source_uv_vertices[0]).pt.data());
        const auto uv_pb = Eigen::Vector2d::Map(uv_mesh.vertex_prop(source_uv_vertices[1]).pt.data());
        const auto uv_pc = Eigen::Vector2d::Map(uv_mesh.vertex_prop(source_uv_vertices[2]).pt.data());
        const auto pa = Eigen::Vector3d::Map(mesh.vertex_prop(source_vertices[0]).pt.data());
        const auto pb = Eigen::Vector3d::Map(mesh.vertex_prop(source_vertices[1]).pt.data());
        const auto pc = Eigen::Vector3d::Map(mesh.vertex_prop(source_vertices[2]).pt.data());
        const auto denominator = (uv_pb - uv_pa).cross(uv_pc - uv_pa);

        subfaces.push_back(parent_fid);
        for (const auto fid : subfaces) {
            for (const auto he : uv_mesh.face(fid).halfedges()) {
                const auto vid = he.from().id;
                if (vid.idx < local_to_mesh_vertex.size() || uv_output_vertices[vid.idx] != gpf::kInvalidIndex) {
                    continue;
                }

                const auto uv_pt = Eigen::Vector2d::Map(uv_mesh.vertex_prop(vid).pt.data());
                const std::array<double, 3> barycentric {
                    (uv_pb - uv_pt).cross(uv_pc - uv_pt) / denominator,
                    (uv_pc - uv_pt).cross(uv_pa - uv_pt) / denominator,
                    (uv_pa - uv_pt).cross(uv_pb - uv_pt) / denominator,
                };
                uv_output_vertices[vid.idx] = output_positions.size();
                output_positions.push_back({});
                Eigen::Vector3d::Map(output_positions.back().data()) = barycentric[0] * pa + barycentric[1] * pb + barycentric[2] * pc;
            }
        }
    }
}

ReliefAssembly assemble_relief_surface(
    const image_relief::Mesh& mesh,
    const PreparedSurfacePatch& patch,
    const ProjectedGridBoundary& boundary,
    const std::size_t width)
{
    ReliefAssembly assembly(mesh, patch);
    const auto& uv_mesh = patch.uv_mesh;
    assembly.output_positions.reserve(mesh.n_vertices_capacity() + width * width + uv_mesh.n_vertices_capacity());
    std::vector<bool> is_inner_face(mesh.n_faces_capacity(), false);
    for (const auto inner_fid : patch.surface_faces) {
        is_inner_face[inner_fid.idx] = true;
    }

    assembly.output_faces.reserve(mesh.n_faces() - patch.surface_faces.size() + boundary.retained_uv_faces.size() + (width - 1) * (width - 1));
    for (const auto face : mesh.faces()) {
        if (is_inner_face[face.id.idx]) {
            continue;
        }
        ReliefOutputFace output_face { {}, face.prop() };
        for (const auto he : face.halfedges()) {
            output_face.vertices.push_back(assembly.output_vertex_for_surface_vertex(he.to().id));
        }
        assembly.output_faces.push_back(std::move(output_face));
    }

    append_relief_edge_vertices(assembly, boundary);
    append_relief_face_vertices(assembly, boundary);
    for (const auto uv_fid : boundary.retained_uv_faces) {
        ReliefOutputFace output_face { {}, image_relief::FaceProp { uv_mesh.face_prop(uv_fid).parent } };
        for (const auto he : uv_mesh.face(uv_fid).halfedges()) {
            output_face.vertices.push_back(assembly.output_vertex_for_uv_vertex(he.to().id));
        }
        assembly.output_faces.push_back(std::move(output_face));
    }
    return assembly;
}

struct GridPolygonVertex {
    std::size_t output_vid;
    std::array<double, 2> uv;
};

std::vector<std::vector<GridPolygonVertex>> build_grid_boundary_chains(
    ReliefAssembly& assembly,
    const SampledReliefGrid& grid,
    const ProjectedGridBoundary& boundary)
{
    const auto& uv_mesh = assembly.patch.uv_mesh;
    std::vector<std::vector<GridPolygonVertex>> boundary_chains(boundary.uv_vertices.size());
    for (std::size_t boundary_idx = 0; boundary_idx < boundary_chains.size(); ++boundary_idx) {
        const auto begin = boundary.separators[boundary_idx];
        const auto end = boundary.separators[boundary_idx + 1];
        auto& chain = boundary_chains[boundary_idx];
        chain.reserve(end - begin + 1);
        const auto grid_idx = boundary.grid_point_indices[boundary_idx];
        chain.push_back({
            assembly.output_vertex_for_uv_vertex(boundary.uv_vertices[boundary_idx]),
            grid.uv_points[grid_idx],
        });
        for (std::size_t path_idx = begin; path_idx < end; ++path_idx) {
            const auto uv_vertex = uv_mesh.he_to(boundary.uv_halfedges[path_idx]);
            chain.push_back({ assembly.output_vertex_for_uv_vertex(uv_vertex), uv_mesh.vertex_prop(uv_vertex).pt });
        }
    }
    return boundary_chains;
}

std::expected<std::vector<std::size_t>, SurfaceFailure> triangulate_grid_cell(
    const std::span<const GridPolygonVertex> polygon,
    const std::size_t row,
    const std::size_t col,
    const std::size_t width)
{
    std::vector<std::size_t> triangles;
    if (polygon.size() == 4) {
        triangles = { 0, 1, 3, 1, 2, 3 };
    } else if (row == 0 && col != 0 && col + 1 != width - 1) {
        for (std::size_t i { 0 }; i + 1 < polygon.size() - 2; i++) {
            triangles.push_back(i);
            triangles.push_back(i + 1);
            triangles.push_back(polygon.size() - 1);
        }
        triangles.push_back(polygon.size() - 3);
        triangles.push_back(polygon.size() - 2);
        triangles.push_back(polygon.size() - 1);
    } else if (col + 1 == width - 1 && row != 0 && row + 1 != width - 1) {
        for (std::size_t i { 1 }; i + 1 < polygon.size() - 1; i++) {
            triangles.push_back(i);
            triangles.push_back(i + 1);
            triangles.push_back(polygon.size() - 1);
        }
        triangles.push_back(0);
        triangles.push_back(1);
        triangles.push_back(polygon.size() - 1);
    } else if (row + 1 == width - 1 && col != 0 && col + 1 != width - 1) {
        for (std::size_t i { 2 }; i + 1 < polygon.size(); i++) {
            triangles.push_back(i);
            triangles.push_back(i + 1);
            triangles.push_back(1);
        }
        triangles.push_back(0);
        triangles.push_back(1);
        triangles.push_back(polygon.size() - 1);
    } else if (col == 0 && row != 0 && row + 1 != width - 1) {
        for (std::size_t i { 3 }; i + 1 <= polygon.size(); i++) {
            triangles.push_back(i);
            triangles.push_back((i + 1) % polygon.size());
            triangles.push_back(1);
        }
        triangles.push_back(1);
        triangles.push_back(2);
        triangles.push_back(3);
    } else {
        std::vector<double> triangulation_points;
        triangulation_points.reserve(polygon.size() * 2);
        std::vector<std::size_t> triangulation_segments;
        triangulation_segments.reserve(polygon.size() * 2);
        for (std::size_t i = 0; i < polygon.size(); ++i) {
            triangulation_points.push_back(polygon[i].uv[0]);
            triangulation_points.push_back(polygon[i].uv[1]);
            triangulation_segments.push_back(i);
            triangulation_segments.push_back((i + 1) % polygon.size());
        }
        triangles = gpf::triangulate_polygon(
            triangulation_points,
            triangulation_segments,
            true);
    }
    return triangles;
}

std::expected<void, SurfaceFailure> assemble_relief_grid(
    ReliefAssembly& assembly,
    const SampledReliefGrid& grid,
    const ProjectedGridBoundary& boundary,
    const std::size_t width,
    const gpf::FaceId fid)
{
    const auto& grid_boundary_vertices = boundary.uv_vertices;
    const auto& grid_boundary_point_indices = boundary.grid_point_indices;
    const auto& grid_points = grid.uv_points;
    const auto& uv_output_vertices = assembly.uv_to_output_vertex;
    auto& output_positions = assembly.output_positions;
    auto& output_faces = assembly.output_faces;
    auto& grid_face_indices = assembly.grid_face_indices;
    std::vector<std::size_t> grid_output_vertices(width * width, gpf::kInvalidIndex);
    for (std::size_t i = 0; i < grid_boundary_vertices.size(); i++) {
        assert(uv_output_vertices[grid_boundary_vertices[i].idx] != gpf::kInvalidIndex);
        grid_output_vertices[grid_boundary_point_indices[i]] = uv_output_vertices[grid_boundary_vertices[i].idx];
    }

    for (std::size_t grid_idx = 0; grid_idx < grid_output_vertices.size(); ++grid_idx) {
        if (grid_output_vertices[grid_idx] != gpf::kInvalidIndex) {
            continue;
        }
        grid_output_vertices[grid_idx] = output_positions.size();
        std::array<double, 3> point {};
        Eigen::Vector3d::Map(point.data()) = grid.positions.row(grid_idx);
        output_positions.push_back(std::move(point));
    }

    const auto boundary_chains = build_grid_boundary_chains(assembly, grid, boundary);

    const auto append_edge = [](std::vector<GridPolygonVertex>& polygon, const std::span<const GridPolygonVertex> edge) {
        for (const auto vertex : edge.subspan(0, edge.size() - 1)) {
            polygon.push_back(vertex);
        }
    };
    const auto grid_vertex = [&grid_output_vertices, &grid_points, width](const std::size_t row, const std::size_t column) {
        const auto grid_idx = row * width + column;
        return GridPolygonVertex { grid_output_vertices[grid_idx], grid_points[grid_idx] };
    };
    const auto n_boundary_segments_per_side = width - 1;
    const auto boundary_chain = [&boundary_chains](const std::size_t index) -> std::span<const GridPolygonVertex> {
        return boundary_chains[index];
    };

    for (std::size_t row = 0; row + 1 < width; ++row) {
        for (std::size_t col = 0; col + 1 < width; ++col) {
            std::vector<GridPolygonVertex> polygon;
            polygon.reserve(4);
            if (row == 0) {
                append_edge(polygon, boundary_chain(col));
            } else {
                polygon.push_back(grid_vertex(row, col));
            }

            if (col + 1 == width - 1) {
                append_edge(polygon, boundary_chain(n_boundary_segments_per_side + row));
            } else {
                polygon.push_back(grid_vertex(row, col + 1));
            }

            if (row + 1 == width - 1) {
                const auto boundary_idx = 2 * n_boundary_segments_per_side + (n_boundary_segments_per_side - 1 - col);
                append_edge(polygon, boundary_chain(boundary_idx));
            } else {
                polygon.push_back(grid_vertex(row + 1, col + 1));
            }

            if (col == 0) {
                const auto boundary_idx = 3 * n_boundary_segments_per_side + (n_boundary_segments_per_side - 1 - row);
                append_edge(polygon, boundary_chain(boundary_idx));
            } else {
                polygon.push_back(grid_vertex(row + 1, col));
            }

            // Projection snapping can make adjacent sides retrace an edge at
            // a corner (A -> B -> A). Flood fill already cancels these twins;
            // cancel the same zero-area spur before triangulating the grid cell.
            // Otherwise CDT receives a non-simple polygon and can dereference
            // an invalid halfedge. Do not remove genuine split boundary edges.
            bool changed = true;
            while (changed && polygon.size() >= 3) {
                changed = false;
                for (std::size_t i = 0; i < polygon.size(); ++i) {
                    const auto next = (i + 1) % polygon.size();
                    const auto prev = (i + polygon.size() - 1) % polygon.size();
                    if (polygon[i].output_vid == polygon[next].output_vid) {
                        polygon.erase(polygon.begin() + static_cast<std::ptrdiff_t>(next));
                        changed = true;
                        break;
                    }
                    if (polygon[prev].output_vid == polygon[next].output_vid) {
                        polygon.erase(polygon.begin() + static_cast<std::ptrdiff_t>(std::max(i, next)));
                        polygon.erase(polygon.begin() + static_cast<std::ptrdiff_t>(std::min(i, next)));
                        changed = true;
                        break;
                    }
                }
            }
            if (polygon.size() < 3) {
                return std::unexpected(SurfaceFailure::DegeneratePolygon);
            }
            const auto triangulation = triangulate_grid_cell(polygon, row, col, width);
            if (!triangulation) {
                return std::unexpected(triangulation.error());
            }
            const auto& triangles = *triangulation;
            for (std::size_t i = 0; i < triangles.size(); i += 3) {
                const auto triangle_index = i / 3;
                grid_face_indices.push_back(image_relief::GridFaceIndex {
                    gpf::FaceId { output_faces.size() },
                    row * (width - 1) + col,
                    row,
                    col,
                    triangle_index,
                });
                output_faces.push_back(ReliefOutputFace {
                    {
                        polygon[triangles[i]].output_vid,
                        polygon[triangles[i + 1]].output_vid,
                        polygon[triangles[i + 2]].output_vid,
                    },
                    image_relief::FaceProp { fid },
                });
            }
        }
    }
    return {};
}

[[nodiscard]] std::expected<std::vector<image_relief::GridFaceIndex>, SurfaceFailure> finalize_relief_surface(image_relief::Mesh& mesh, ReliefAssembly& assembly)
{
    auto& output_faces = assembly.output_faces;
    const auto& output_positions = assembly.output_positions;
    std::vector<std::vector<std::size_t>> polygons;
    std::vector<image_relief::FaceProp> face_properties;
    polygons.reserve(output_faces.size());
    face_properties.reserve(output_faces.size());
    for (auto& output_face : output_faces) {
        polygons.push_back(std::move(output_face.vertices));
        face_properties.push_back(output_face.prop);
    }

    for (const auto& polygon : polygons) {
        if (polygon.size() < 3) {
            return std::unexpected(SurfaceFailure::DegeneratePolygon);
        }
    }

    // Validate the polygon soup before handing it to ManifoldMesh::new_in().
    // new_in() pairs edges by key but intentionally does not diagnose a third
    // incident face or two faces using the same directed edge.
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> edge_incidence;
    std::map<std::pair<std::size_t, std::size_t>, std::pair<std::size_t, std::size_t>> edge_directions;
    for (const auto& polygon : polygons) {
        for (std::size_t i = 0; i < polygon.size(); ++i) {
            const auto from = polygon[i];
            const auto to = polygon[(i + 1) % polygon.size()];
            if (from == to) {
                return std::unexpected(SurfaceFailure::ZeroLengthTopologicalEdge);
            }
            const auto key = std::minmax(from, to);
            auto& incidence = edge_incidence[key];
            if (++incidence > 2) {
                return std::unexpected(SurfaceFailure::NonManifoldEdge);
            }
            const auto direction = std::make_pair(from, to);
            const auto direction_iter = edge_directions.find(key);
            if (direction_iter == edge_directions.end()) {
                edge_directions.emplace(key, direction);
            } else if (direction_iter->second == direction) {
                return std::unexpected(SurfaceFailure::ConflictingEdgeOrientation);
            }
        }
    }

    auto combined_mesh = image_relief::Mesh::new_in(polygons);
    for (const auto vertex : combined_mesh.vertices()) {
        if (vertex.id.idx >= output_positions.size()) {
            return std::unexpected(SurfaceFailure::MissingVertexPosition);
        }
        vertex.prop().pt = output_positions[vertex.id.idx];
    }
    std::size_t face_index = 0;
    for (const auto face : combined_mesh.faces()) {
        if (face_index >= face_properties.size()) {
            return std::unexpected(SurfaceFailure::FacePropertyCountMismatch);
        }
        face.prop() = face_properties[face_index++];
    }
    if (face_index != face_properties.size()) {
        return std::unexpected(SurfaceFailure::LostGeneratedFace);
    }
    mesh = std::move(combined_mesh);
    if (auto result = write_image_relief_mesh_as_off(mesh, "fit_polygon_on_surface_combined.off"); !result) {
        return std::unexpected(result.error());
    }
    return std::move(assembly.grid_face_indices);
}
} // namespace
namespace image_relief {
std::expected<std::vector<GridFaceIndex>, fit_on_surface::SurfaceFailure> make_relief_on_surface(
    Mesh& mesh,
    const std::span<const double> heights,
    const std::size_t width,
    const gpf::FaceId fid,
    const std::array<double, 3>& surface_point,
    const std::array<double, 3>& direction)
{
    if (width < 2) {
        return std::unexpected(SurfaceFailure::InvalidGridWidth);
    }
    if (width > std::numeric_limits<std::size_t>::max() / width || heights.size() != width * width) {
        return std::unexpected(SurfaceFailure::HeightCountMismatch);
    }
    if (ranges::any_of(heights, [](double h) { return !std::isfinite(h); })) {
        return std::unexpected(SurfaceFailure::MissingVertexPosition);
    }
    const auto input = make_placement_input(mesh, fid, surface_point, direction);
    auto patch = prepare_surface_patch(mesh, input);
    if (!patch) {
        return std::unexpected(patch.error());
    }
    const auto grid = sample_relief_grid(mesh, *patch, heights, width);
    if (!grid) {
        return std::unexpected(grid.error());
    }
    const auto boundary = project_grid_boundary(patch->uv_mesh, *grid, width);
    if (!boundary) {
        return std::unexpected(boundary.error());
    }
    // The prepared patch stays alive while assembly holds references to it.
    auto assembly = assemble_relief_surface(mesh, *patch, *boundary, width);
    if (auto result = assemble_relief_grid(assembly, *grid, *boundary, width, fid); !result) {
        return std::unexpected(result.error());
    }
    return finalize_relief_surface(mesh, assembly);
}
}

// Public failure descriptions.

namespace fit_on_surface {
std::string_view to_string(const SurfaceFailure failure) noexcept
{
    switch (failure) {
    case SurfaceFailure::InvalidGridWidth:
        return "grid width must be at least 2";
    case SurfaceFailure::HeightCountMismatch:
        return "height count must match the grid size";
    case SurfaceFailure::InvalidStartFace:
        return "relief start face is invalid";
    case SurfaceFailure::DegenerateStartFace:
        return "relief start face is degenerate";
    case SurfaceFailure::ZeroDirection:
        return "relief direction must be nonzero";
    case SurfaceFailure::MissingTangentComponent:
        return "relief direction must have a nonzero tangent component";
    case SurfaceFailure::ProjectionPathNotFound:
        return "project_polylines_on_mesh failed: path not found";
    case SurfaceFailure::ProjectionInvalidTriangleIndex:
        return "project_polylines_on_mesh failed: invalid triangle index";
    case SurfaceFailure::ProjectionConstraintConflict:
        return "project_polylines_on_mesh failed: constraint conflict";
    case SurfaceFailure::WalkBoundaryReached:
        return "walk_on_mesh_surface failed: boundary reached";
    case SurfaceFailure::WalkIterationLimitExceeded:
        return "walk_on_mesh_surface failed: iteration limit exceeded";
    case SurfaceFailure::WalkDegenerateDirection:
        return "walk_on_mesh_surface failed: degenerate direction";
    case SurfaceFailure::WalkInvalidPath:
        return "walk_on_mesh_surface failed: invalid path";
    case SurfaceFailure::WalkDegenerateStep:
        return "walk_on_mesh_surface failed: degenerate step";
    case SurfaceFailure::InvalidAnchorIndex:
        return "Invalid vertex index in corner_indices";
    case SurfaceFailure::ConflictingPolygonLabels:
        return "conflicting UV face polygon ids";
    case SurfaceFailure::MissingBoundarySeparator:
        return "could not find boundary separator for vertex";
    case SurfaceFailure::InvalidMeshVertexReference:
        return "face references invalid mesh vertex";
    case SurfaceFailure::InvalidUvVertexReference:
        return "face references invalid UV mesh vertex";
    case SurfaceFailure::InvalidPolylinePointIndex:
        return "polyline references invalid point index";
    case SurfaceFailure::GridMatrixDimensionMismatch:
        return "grid point matrix dimensions do not match the grid width";
    case SurfaceFailure::SurfaceOffOpenFailed:
        return "failed to open OFF output file";
    case SurfaceFailure::MeshOffOpenFailed:
        return "failed to open mesh OFF output file";
    case SurfaceFailure::UvCoordinatesOffOpenFailed:
        return "failed to open UV OFF output file";
    case SurfaceFailure::GridOffOpenFailed:
        return "failed to open grid mesh OFF output file";
    case SurfaceFailure::UvMeshOffOpenFailed:
        return "failed to open UV mesh OFF output file";
    case SurfaceFailure::PolylineObjOpenFailed:
        return "failed to open polyline OBJ output file";
    case SurfaceFailure::DegeneratePolygon:
        return "combined mesh contains a degenerate polygon";
    case SurfaceFailure::ZeroLengthTopologicalEdge:
        return "combined mesh contains a zero-length topological edge";
    case SurfaceFailure::NonManifoldEdge:
        return "combined mesh contains a non-manifold edge";
    case SurfaceFailure::ConflictingEdgeOrientation:
        return "combined mesh contains two faces with the same edge orientation";
    case SurfaceFailure::MissingVertexPosition:
        return "combined mesh vertex has no position";
    case SurfaceFailure::FacePropertyCountMismatch:
        return "combined mesh face/property count mismatch";
    case SurfaceFailure::LostGeneratedFace:
        return "combined mesh lost a generated face";
    case SurfaceFailure::ParameterizationFailed:
        return "surface patch parameterization failed";
    }
    return "unknown surface failure";
}

}
