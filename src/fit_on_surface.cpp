#include "fit_on_surface.h"
#include "eigen_alias.h"
#include "flatten_surface.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <gpf/mesh_flood_fill.hpp>
#include <gpf/project_polylines_on_mesh.hpp>

#include <igl/flipped_triangles.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>

#include <cmath>
#include <fstream>
#include <iterator>
#include <map>
#include <numbers>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ranges = std::ranges;
namespace views = std::views;

namespace {
void write_faces_as_off(const fit_on_surface::Mesh& mesh, const std::span<const gpf::FaceId> face_ids, const std::string& path)
{
    std::vector<gpf::VertexId> vertices;
    std::vector<std::size_t> vertex_indices(mesh.n_vertices_capacity(), gpf::kInvalidIndex);
    auto vertex_index = [&vertices, &vertex_indices](const gpf::VertexId vid) {
        auto& index = vertex_indices[vid.idx];
        if (index == gpf::kInvalidIndex) {
            index = vertices.size();
            vertices.push_back(vid);
        }
        return index;
    };

    std::vector<std::vector<std::size_t>> faces;
    faces.reserve(face_ids.size());
    for (const auto fid : face_ids) {
        std::vector<std::size_t> face_vertices;
        for (const auto halfedge : mesh.face(fid).halfedges()) {
            face_vertices.push_back(vertex_index(halfedge.to().id));
        }
        faces.push_back(std::move(face_vertices));
    }

    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open OFF output file");
    }

    file << "OFF\n";
    file << vertices.size() << ' ' << faces.size() << " 0\n";
    for (const auto vid : vertices) {
        const auto& pt = mesh.vertex_prop(vid).pt;
        file << pt[0] << ' ' << pt[1] << ' ' << pt[2] << '\n';
    }
    for (const auto& face_vertices : faces) {
        file << face_vertices.size();
        for (const auto vid : face_vertices) {
            file << ' ' << vid;
        }
        file << '\n';
    }
}

void write_uv_as_off(const VMat2& uv, const FMat& faces, const std::string& path)
{
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open UV OFF output file");
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
}

auto extract_face_mesh(
    const fit_on_surface::Mesh& mesh,
    const std::span<const gpf::HalfedgeId> boundary_halfedges,
    const std::span<const gpf::FaceId> inner_faces)
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
    return std::make_tuple(n_boundary_vertices, std::move(vertex_map), std::move(face_vertices), std::move(V), std::move(F));
}

inline auto face_point(const auto& mesh, const gpf::FaceId fid, const std::span<const double, 3> bary_coords) {
    auto he = mesh.face(fid).halfedge();
    auto pa = Eigen::Vector3d::Map(he.from().prop().pt.data());
    he = he.next();
    auto pb = Eigen::Vector3d::Map(he.from().prop().pt.data());
    auto pc = Eigen::Vector3d::Map(he.to().prop().pt.data());
    std::array<double, 3> result{};
    Eigen::Vector3d::Map(result.data()) = pa * bary_coords[0] + pb * bary_coords[1] + pc * bary_coords[2];
    return result;
}

inline std::size_t find_anchor_corner_index(const VMat2& uv, const std::span<const std::size_t> vertices) {
    Eigen::Vector2d center = uv.row(vertices[0]).transpose();
    return ranges::min(views::zip(vertices.subspan(1), ranges::iota_view{std::size_t{1}, std::size_t{5}}) |
        views::transform([&uv, &center](auto&& pair) { return std::make_pair(
            (uv.row(std::get<0>(pair)).transpose() - center).squaredNorm(),
            std::get<1>(pair)
        ); }), {}, &std::pair<double, std::size_t>::first).second;
}

bool boundary_contains_anchor_rectangle(const VMat2& uv, const std::span<const std::size_t> vertices, const std::size_t min_idx, const std::size_t n_boundaries) {
    Eigen::Vector2d center = uv.row(vertices[0]).transpose();
    Eigen::Vector2d base_dir = uv.row(vertices[min_idx]).transpose() - center;
    auto half_diag_len = base_dir.norm();
    base_dir /= half_diag_len;
    for (std::size_t i = 0; i < n_boundaries; ++i) {
        Eigen::Vector2d v = uv.row(i).transpose() - center;
        auto actual_len = v.norm();
        v /= actual_len;
        auto angle_vec = gpf::detail::complex_div(v, base_dir);
        auto expected_len = half_diag_len * std::abs(angle_vec[0]);  // cos(\theta) = angle_vec[0]
        if (actual_len < expected_len) {
            return false;
        }
    }
    return true;
}

auto compute_anchor_uv_frame(const VMat2& uv, const std::span<const std::size_t> vertices, const std::size_t min_idx) {
    Eigen::Vector2d center = uv.row(vertices[0]).transpose();
    Eigen::Vector2d base_dir = uv.row(vertices[min_idx]).transpose() - center;
    const auto angle = std::numbers::pi * (1.0 - 0.5 * min_idx); // rotate counterclockwise -0.5 * i * pi, then reverse
    Eigen::Rotation2Dd rot(angle);
    Eigen::Vector2d dir = rot * base_dir;
    Eigen::Vector2d start_pt = center - dir;
    dir *= std::numbers::sqrt2;
    const auto pi_4 = std::numbers::pi * 0.25;
    Eigen::Vector2d xaxis = Eigen::Rotation2Dd(-pi_4) * dir;
    Eigen::Vector2d yaxis = Eigen::Rotation2Dd(pi_4) * dir;
    return std::make_tuple(std::move(start_pt), std::move(xaxis), std::move(yaxis));
}

auto map_polygon_to_uv_frame(const std::vector<std::array<double, 2>>& polygon_points, const Eigen::Vector2d& start_pt, const Eigen::Vector2d& xaxis, const Eigen::Vector2d& yaxis) {
    return polygon_points | views::transform([&start_pt, &xaxis, &yaxis](auto&& point) {
        std::array<double, 2> result;
        Eigen::Vector2d::Map(result.data()) = start_pt + xaxis * point[0] + yaxis * point[1];
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

namespace uv {
struct VertexProp {
    std::array<double, 2> pt;
};

using Mesh = gpf::ManifoldMesh<VertexProp, gpf::Empty, gpf::Empty, fit_on_surface::FaceProp>;
}

void write_uv_mesh_as_obj(const uv::Mesh& mesh, const std::string& path)
{
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open UV mesh OBJ output file");
    }

    std::vector<std::size_t> vertex_indices(mesh.n_vertices_capacity(), gpf::kInvalidIndex);
    std::size_t obj_vertex_idx = 1;
    for (auto vertex : mesh.vertices()) {
        vertex_indices[vertex.id.idx] = obj_vertex_idx++;
        const auto& pt = vertex.prop().pt;
        file << "v " << pt[0] << ' ' << pt[1] << " 0\n";
    }

    for (auto face : mesh.faces()) {
        file << 'f';
        for (auto halfedge : face.halfedges()) {
            const auto vertex_idx = vertex_indices[halfedge.from().id.idx];
            if (vertex_idx == gpf::kInvalidIndex) {
                throw std::runtime_error("face references invalid UV mesh vertex");
            }
            file << ' ' << vertex_idx;
        }
        file << '\n';
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

} // unnamed namespace

void fit_polygon_on_surface(
    fit_on_surface::Mesh& mesh,
    const std::vector<std::array<double, 2>>& polygon_points,
    const std::vector<std::vector<std::vector<std::size_t>>>& polygons,
    const std::array<double, 3>& surface_point,
    const gpf::FaceId fid,
    const std::array<double, 3>& direction)
{
    const auto verts = mesh.face(fid).halfedges() | views::transform([&mesh](auto&& he) {
        return he.from().id;
    }) | ranges::to<std::vector>();
    auto pa = Eigen::Vector3d::Map(mesh.vertex_prop(verts[0]).pt.data());
    auto pb = Eigen::Vector3d::Map(mesh.vertex_prop(verts[1]).pt.data());
    auto pc = Eigen::Vector3d::Map(mesh.vertex_prop(verts[2]).pt.data());

    Eigen::Vector3d vab = pa - pb;
    Eigen::Vector3d vac = pa - pc;
    Eigen::Vector3d normal = vab.cross(vac).normalized();

    Eigen::Vector3d dir = Eigen::Vector3d::Map(direction.data());
    const auto diag_length = dir.norm() * std::sqrt(2.0);
    std::array<double, 2> lengths { diag_length, diag_length * 1.2 };
    dir = (dir - normal.dot(dir) * normal).normalized().eval();

    constexpr double theta = std::numbers::pi / 2;
    constexpr double half_theta = theta / 2;
    const double cos_val = std::cos(half_theta);
    const double sin_val = std::sin(half_theta);
    Eigen::Quaterniond quat(cos_val, normal[0] * sin_val, normal[1] * sin_val, normal[2] * sin_val);
    std::vector<std::pair<gpf::FaceId, std::array<double, 3>>> corner_points;
    std::vector<std::array<double, 3>> outer_corner_points;
    std::optional<std::pair<gpf::FaceId, std::array<double, 3>>> start_info;
    for (std::size_t i { 0 }; i < std::size_t { 4 }; i++) {
        const auto walk_ret = gpf::walk_on_mesh_surface(mesh, fid, surface_point, std::span<const double, 3> { dir.data(), 3 }, lengths);
        if (walk_ret.has_value()) {
            if (!start_info.has_value()) {
                start_info = (*walk_ret)[0];
            }
            corner_points.push_back((*walk_ret)[1]);
            const auto [fid, bary_coords] = (*walk_ret)[2];
            outer_corner_points.push_back(face_point(mesh, fid, bary_coords));
        } else {
            throw std::runtime_error("walk_on_mesh_surface failed");
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

    auto [project_vertices, boundary_paths] = gpf::project_polylines_on_mesh(
        outer_corner_points,
        std::vector<std::vector<std::size_t>> { { 0, 1, 2, 3, 0 } },
        mesh);
    auto inner_faces = gpf::surround_faces_by_halfedges(mesh, boundary_paths.front());
    auto [n_boundary_vertices, vertex_map, inner_face_indices, V, F] = extract_face_mesh(mesh, boundary_paths.front(), inner_faces);
    const auto bnd = Eigen::VectorXi::LinSpaced(
        static_cast<Eigen::Index>(n_boundary_vertices),
        0,
        static_cast<int>(n_boundary_vertices) - 1);
    // Eigen::MatrixXd bnd_uv, uv;
    Eigen::MatrixXd bnd_uv;
    VMat2 uv;
    igl::map_vertices_to_circle(V, bnd, bnd_uv);
    igl::harmonic(V, F, bnd, bnd_uv, 1, uv);
    if (igl::flipped_triangles(uv, F).size() != 0) {
        igl::harmonic(F, bnd, bnd_uv, 1, uv); // use uniform laplacian
    }
    // Treat vertex 0 as a fixed UV gauge during the SLIM global step.  The
    // SLIM matrix is assembled from gradients, so with 0 fixed vertices the
    // normal equation has two exact translation null modes: adding a constant
    // to every u coordinate or every v coordinate leaves all triangle
    // Jacobians unchanged.  That makes A^T M A positive semidefinite, and
    // Eigen::SimplicialLDLT can encounter a zero Schur-complement pivot even
    // when all original matrix entries are finite and the RHS is compatible.
    // Passing 1 removes vertex 0 from the free unknowns and anchors both its u
    // and v values to the harmonic initialization.  For a connected patch this
    // fixes only the translation gauge; it does not otherwise constrain the
    // local SLIM distortion minimization.  See
    // docs/SLIM_ldlt_nullspace_explanation.md for the full derivation.
    FlattenSurface fs(std::move(V), std::move(F), std::move(uv), 1);
    fs.slim_solve(5);
    write_uv_as_off(fs.uv, fs.F, "fit_polygon_on_surface_uv.off");
    write_faces_as_off(mesh, inner_faces, "fit_polygon_on_surface.off");

    const std::vector<std::size_t> corner_indices = std::span<const gpf::VertexId>{project_vertices.data() + project_vertices.size() - 5, 5} | views::transform([&vertex_map](auto vid) { return vertex_map[vid.idx]; }) | ranges::to<std::vector>();
    if (ranges::any_of(corner_indices, [](auto idx) { return idx == gpf::kInvalidIndex; })) {
        throw std::runtime_error("Invalid vertex index in corner_indices");
    }

    const auto anchor_idx = find_anchor_corner_index(fs.uv, corner_indices);
    if (!boundary_contains_anchor_rectangle(fs.uv, corner_indices, anchor_idx, n_boundary_vertices)) {
        throw std::runtime_error("Failed to fit polygon on surface");
    }

    auto uv_mesh = uv::Mesh::new_in(ranges::iota_view{std::size_t{0}, inner_faces.size()} | views::transform([&inner_face_indices](auto idx) { return  std::span<const std::size_t, 3>{inner_face_indices.data() + idx * 3, 3}; }));
    for (auto v : uv_mesh.vertices()) {
        auto row = fs.uv.row(static_cast<Eigen::Index>(v.id.idx));
        v.prop().pt = {row(0), row(1)};
    }

    for (auto face : uv_mesh.faces()) {
        face.prop().parent = mesh.face_prop(inner_faces[static_cast<std::size_t>(face.id.idx)]).parent;
        assert(face.prop().parent.valid());
    }

    const auto [start_pt, xaxis, yaxis] = compute_anchor_uv_frame(fs.uv, corner_indices, anchor_idx);
    auto poly_uv_pts = map_polygon_to_uv_frame(polygon_points, start_pt, xaxis, yaxis);
    const auto [oriented_polylines, polyline_polygon_sides] =
        divide_polygons_into_oriented_polylines(polygons);
    (void)polyline_polygon_sides;
    gpf::project_polylines_on_mesh(poly_uv_pts, oriented_polylines, uv_mesh);
    write_uv_mesh_as_obj(uv_mesh, "fit_polygon_on_surface_projected_uv_mesh.obj");
}
