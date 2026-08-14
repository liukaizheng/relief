#include "fit_on_surface.h"
#include "eigen_alias.h"
#include "flatten_surface.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <gpf/ids.hpp>
#include <gpf/mesh_flood_fill.hpp>
#include <gpf/project_polylines_on_mesh.hpp>
#include <gpf/triangulation.hpp>

#include <igl/flipped_triangles.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>

#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <predicates/predicates.hpp>

namespace ranges = std::ranges;
namespace views = std::views;

namespace {
std::array<int, 3> polygon_color(std::size_t polygon_id);

void write_faces_as_off(const auto& mesh, const std::span<const gpf::FaceId> face_ids, const std::string& path)
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

void write_mesh_as_off(const fit_on_surface::Mesh& mesh, const std::string& path)
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
        throw std::runtime_error("failed to open mesh OFF output file");
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
                throw std::runtime_error("face references invalid mesh vertex");
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
}

void write_image_relief_mesh_as_off(const image_relief::Mesh& mesh, const std::string& path)
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
        throw std::runtime_error("failed to open mesh OFF output file");
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
                throw std::runtime_error("face references invalid mesh vertex");
            }
            face_vertices.push_back(index);
        }

        file << face_vertices.size();
        for (const auto vertex : face_vertices) {
            file << ' ' << vertex;
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
    const auto& mesh,
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
    return std::make_tuple(n_boundary_vertices, std::move(vertex_map), std::move(vertices), std::move(face_vertices), std::move(V), std::move(F));
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

std::optional<double> boundary_contains_anchor_rectangle(const VMat2& uv, const std::span<const std::size_t> vertices, const std::size_t min_idx, const Eigen::VectorXi& bnd)
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

auto compute_anchor_uv_frame(const VMat2& uv, const std::span<const std::size_t> vertices, const std::size_t min_idx, const std::optional<double> scale)
{
    Eigen::Vector2d center = uv.row(vertices[0]).transpose();
    Eigen::Vector2d base_dir = uv.row(vertices[min_idx]).transpose() - center;
    if (scale.has_value()) {
        base_dir *= *scale * 0.99;
    }
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

auto map_polygon_to_uv_frame(const std::vector<std::array<double, 2>>& polygon_points, const Eigen::Vector2d& start_pt, const Eigen::Vector2d& xaxis, const Eigen::Vector2d& yaxis)
{
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

void set_face_polygon_id(uv::Mesh& mesh, const gpf::FaceId fid, const std::size_t polygon_id)
{
    if (!fid.valid() || polygon_id == gpf::kInvalidIndex) {
        return;
    }

    auto& face_polygon_id = mesh.face_prop(fid).polygon_id;
    if (face_polygon_id != gpf::kInvalidIndex && face_polygon_id != polygon_id) {
        throw std::runtime_error("conflicting UV face polygon ids");
    }
    face_polygon_id = polygon_id;
}

void label_uv_mesh_polygon_ids(
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
            set_face_polygon_id(mesh, mesh.he_face(hid), left_polygon_id);
            set_face_polygon_id(mesh, mesh.he_face(twin_hid), right_polygon_id);
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
                throw std::runtime_error("conflicting UV face polygon ids");
            }
        }
    }
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

void write_grid_mesh_as_off(const VMat& points, const std::size_t width, const std::string& path)
{
    if (width < 2 || points.rows() != static_cast<Eigen::Index>(width * width) || points.cols() != 3) {
        throw std::invalid_argument("grid point matrix dimensions do not match the grid width");
    }

    const auto face_count = (width - 1) * (width - 1);
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open grid mesh OFF output file");
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

void write_uv_mesh_as_off(const uv::Mesh& mesh, const std::string& path)
{
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open UV mesh OFF output file");
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
                throw std::runtime_error("face references invalid UV mesh vertex");
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
}

void write_polyline_as_obj(
    const std::vector<std::array<double, 2>>& points,
    const std::span<const std::size_t> polyline,
    const std::string& path)
{
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open polyline OBJ output file");
    }

    for (const auto& point : points) {
        file << "v " << point[0] << ' ' << point[1] << " 0\n";
    }

    for (std::size_t i = 1; i < polyline.size(); ++i) {
        const auto start_idx = polyline[i - 1];
        const auto end_idx = polyline[i];
        if (start_idx >= points.size() || end_idx >= points.size()) {
            throw std::runtime_error("polyline references invalid point index");
        }
        file << "l " << start_idx + 1 << ' ' << end_idx + 1 << '\n';
    }
}

struct EdgeSplitRequest {
    gpf::VertexId uv_vertex;
    double t;
    std::array<double, 3> point;
};

auto make_base_uv_edges(
    const auto& mesh,
    const uv::Mesh& uv_mesh,
    const std::vector<gpf::VertexId>& local_to_mesh_vertex)
{
    std::vector<gpf::EdgeId> base_edges(uv_mesh.n_edges_capacity());
    for (const auto edge : uv_mesh.edges()) {
        const auto [uv_va, uv_vb] = edge.vertices();

        const auto mesh_edge = mesh.e_from_vertices(local_to_mesh_vertex[uv_va.id.idx], local_to_mesh_vertex[uv_vb.id.idx]);
        base_edges[edge.id.idx] = mesh_edge;
    }
    return base_edges;
}

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
    const std::span<const std::size_t> inner_face_indices,
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

auto compute_grid_points_and_positions(
    const image_relief::Mesh& mesh,
    const uv::Mesh& uv_mesh,
    const std::span<const double> heights,
    const std::size_t width,
    const Eigen::Vector2d& start_pt,
    const Eigen::Vector2d& xaxis,
    const Eigen::Vector2d& yaxis,
    const std::vector<gpf::VertexId>& local_to_mesh_vertex)
{
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
    const auto eps = std::min(1e-3, 0.01 * delta_x.norm());
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
        N.row(uv_mesh.face_prop(uv_fid).parent.idx) = normal;

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
    return std::make_tuple(std::move(points), std::move(P), eps);
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

auto compute_boundary_vertex_separators(const uv::Mesh& mesh, const std::vector<gpf::VertexId>& vertices, const std::vector<gpf::HalfedgeId>& halfedges)
{
    std::vector<std::size_t> separators;
    separators.reserve(vertices.size() + 1);
    separators.push_back(0);
    std::size_t vidx { 1 };
    auto vid = vertices[vidx];
    for (std::size_t i = 0; i < halfedges.size(); i++) {
        if (mesh.he_to(halfedges[i]) == vid) {
            separators.push_back(i + 1);
            vid = vertices[++vidx];
        }
    }
    if (separators.size() != vertices.size()) {
        const auto missing_vertex = vidx < vertices.size() ? std::to_string(vertices[vidx].idx) : "unknown";
        throw std::runtime_error(
            "could not find boundary separator for vertex " + missing_vertex + ": found " + std::to_string(separators.size() - 1) + " of " + std::to_string(vertices.size() - 1) + " expected vertices while scanning " + std::to_string(halfedges.size()) + " halfedges");
    }
    separators.push_back(halfedges.size());
    return separators;
}
} // unnamed namespace
namespace fit_on_surface {
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
    constexpr double EPS { 1e-3 };

    auto [project_vertices, boundary_paths] = gpf::project_polylines_on_mesh(
        outer_corner_points,
        std::vector<std::vector<std::size_t>> { { 0, 1, 2, 3, 0 } },
        mesh, EPS);
    auto inner_faces = gpf::surround_faces_by_halfedges(mesh, boundary_paths.front());
    auto [n_boundary_vertices, vertex_map, local_to_mesh_vertex, inner_face_indices, V, F] = extract_face_mesh(mesh, boundary_paths.front(), inner_faces);
    auto uv_mesh = uv::Mesh::new_in(ranges::iota_view { std::size_t { 0 }, inner_faces.size() } | views::transform([&inner_face_indices](auto idx) { return std::span<const std::size_t, 3> { inner_face_indices.data() + idx * 3, 3 }; }));
    Eigen::VectorXi bnd(n_boundary_vertices);
    {
        auto curr_he = uv_mesh.vertex(gpf::VertexId { 0 }).halfedge().prev();
        Eigen::Index idx { 0 };
        const auto first_hid = curr_he.id;
        while (true) {
            bnd(idx++) = static_cast<int>(curr_he.to().id.idx);
            curr_he = curr_he.prev();
            if (curr_he.id == first_hid) {
                break;
            }
        }
    }
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
    fs.slim_solve(5, 15);
    write_uv_as_off(fs.uv, fs.F, "fit_polygon_on_surface_uv.off");
    write_faces_as_off(mesh, inner_faces, "fit_polygon_on_surface.off");

    const std::vector<std::size_t> corner_indices = std::span<const gpf::VertexId> { project_vertices.data() + project_vertices.size() - 5, 5 } | views::transform([&vertex_map](auto vid) { return vertex_map[vid.idx]; }) | ranges::to<std::vector>();
    if (ranges::any_of(corner_indices, [](auto idx) { return idx == gpf::kInvalidIndex; })) {
        throw std::runtime_error("Invalid vertex index in corner_indices");
    }

    const auto anchor_idx = find_anchor_corner_index(fs.uv, corner_indices);
    const auto scale = boundary_contains_anchor_rectangle(fs.uv, corner_indices, anchor_idx, bnd);

    for (auto v : uv_mesh.vertices()) {
        auto row = fs.uv.row(static_cast<Eigen::Index>(v.id.idx));
        v.prop().pt = { row(0), row(1) };
    }

    for (auto face : uv_mesh.faces()) {
        face.prop().parent = mesh.face_prop(inner_faces[static_cast<std::size_t>(face.id.idx)]).parent;
        assert(face.prop().parent.valid());
    }

    const auto [start_pt, xaxis, yaxis] = compute_anchor_uv_frame(fs.uv, corner_indices, anchor_idx, scale);
    auto poly_uv_pts = map_polygon_to_uv_frame(polygon_points, start_pt, xaxis, yaxis);
    const auto [oriented_polylines, polyline_polygon_sides] = divide_polygons_into_oriented_polylines(polygons);
    auto base_uv_edges = make_base_uv_edges(mesh, uv_mesh, local_to_mesh_vertex);
    std::unordered_map<gpf::FaceId, gpf::FaceId> uv_face_parent_map;
    std::unordered_map<gpf::EdgeId, gpf::EdgeId> uv_edge_parent_map;
    auto projected_polyline_paths = std::get<1>(gpf::project_polylines_on_mesh(poly_uv_pts, oriented_polylines, uv_mesh, EPS, &uv_face_parent_map, &uv_edge_parent_map));
    label_uv_mesh_polygon_ids(uv_mesh, projected_polyline_paths, polyline_polygon_sides);
    write_uv_mesh_as_off(uv_mesh, "fit_polygon_on_surface_projected_uv_mesh.off");
    map_subdivided_uv_mesh_to_surface(
        mesh,
        uv_mesh,
        local_to_mesh_vertex,
        inner_faces,
        inner_face_indices,
        uv_face_parent_map,
        uv_edge_parent_map,
        base_uv_edges);
    write_mesh_as_off(mesh, "fit_polygon_on_surface_final.off");
}
}

namespace image_relief {
std::vector<GridFaceIndex> make_relief_on_surface(
    Mesh& mesh,
    const std::span<const double> heights,
    const std::size_t width,
    const gpf::FaceId fid,
    const std::array<double, 3>& surface_point,
    const std::array<double, 3>& direction)
{
    if (width < 2) {
        throw std::invalid_argument("grid width must be at least 2");
    }
    if (heights.size() != width * width) {
        throw std::invalid_argument("height count must match the grid size");
    }
    if (!fid.valid() || fid.idx >= mesh.n_faces_capacity() || mesh.face_is_deleted(fid)) {
        throw std::invalid_argument("relief start face is invalid");
    }

    const auto verts = mesh.face(fid).halfedges() | views::transform([&mesh](auto&& he) {
        return he.from().id;
    }) | ranges::to<std::vector>();
    auto pa = Eigen::Vector3d::Map(mesh.vertex_prop(verts[0]).pt.data());
    auto pb = Eigen::Vector3d::Map(mesh.vertex_prop(verts[1]).pt.data());
    auto pc = Eigen::Vector3d::Map(mesh.vertex_prop(verts[2]).pt.data());

    Eigen::Vector3d vab = pa - pb;
    Eigen::Vector3d vac = pa - pc;
    const auto face_normal_vector = vab.cross(vac);
    if (face_normal_vector.squaredNorm() < 1e-24) {
        throw std::invalid_argument("relief start face is degenerate");
    }
    Eigen::Vector3d normal = face_normal_vector.normalized();

    Eigen::Vector3d dir = Eigen::Vector3d::Map(direction.data());
    if (dir.squaredNorm() < 1e-24) {
        throw std::invalid_argument("relief direction must be nonzero");
    }
    const auto diag_length = dir.norm() * std::sqrt(2.0);
    std::array<double, 2> lengths { diag_length, diag_length * 1.2 };
    const auto tangent_direction = dir - normal.dot(dir) * normal;
    if (tangent_direction.squaredNorm() < 1e-24) {
        throw std::invalid_argument("relief direction must have a nonzero tangent component");
    }
    dir = tangent_direction.normalized().eval();

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
        mesh, 1e-3);
    auto inner_faces = gpf::surround_faces_by_halfedges(mesh, boundary_paths.front());
    auto [n_boundary_vertices, vertex_map, local_to_mesh_vertex, inner_face_indices, V, F] = extract_face_mesh(mesh, boundary_paths.front(), inner_faces);
    auto uv_mesh = uv::Mesh::new_in(ranges::iota_view { std::size_t { 0 }, inner_faces.size() } | views::transform([&inner_face_indices](auto idx) { return std::span<const std::size_t, 3> { inner_face_indices.data() + idx * 3, 3 }; }));
    Eigen::VectorXi bnd(n_boundary_vertices);
    {
        auto curr_he = uv_mesh.vertex(gpf::VertexId { 0 }).halfedge().prev();
        Eigen::Index idx { 0 };
        const auto first_hid = curr_he.id;
        while (true) {
            bnd(idx++) = static_cast<int>(curr_he.to().id.idx);
            curr_he = curr_he.prev();
            if (curr_he.id == first_hid) {
                break;
            }
        }
    }
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
    fs.slim_solve(5, 15);
    write_uv_as_off(fs.uv, fs.F, "fit_polygon_on_surface_uv.off");
    write_faces_as_off(mesh, inner_faces, "fit_polygon_on_surface.off");

    const std::vector<std::size_t> corner_indices = std::span<const gpf::VertexId> { project_vertices.data() + project_vertices.size() - 5, 5 } | views::transform([&vertex_map](auto vid) { return vertex_map[vid.idx]; }) | ranges::to<std::vector>();
    if (ranges::any_of(corner_indices, [](auto idx) { return idx == gpf::kInvalidIndex; })) {
        throw std::runtime_error("Invalid vertex index in corner_indices");
    }

    const auto anchor_idx = find_anchor_corner_index(fs.uv, corner_indices);
    const auto scale = boundary_contains_anchor_rectangle(fs.uv, corner_indices, anchor_idx, bnd);

    for (auto v : uv_mesh.vertices()) {
        auto row = fs.uv.row(static_cast<Eigen::Index>(v.id.idx));
        v.prop().pt = { row(0), row(1) };
    }

    for (auto face : uv_mesh.faces()) {
        face.prop().parent = mesh.face_prop(inner_faces[static_cast<std::size_t>(face.id.idx)]).parent;
        assert(face.prop().parent.valid());
    }

    const auto [start_pt, xaxis, yaxis] = compute_anchor_uv_frame(fs.uv, corner_indices, anchor_idx, scale);
    auto [grid_points, P, eps] = compute_grid_points_and_positions(mesh, uv_mesh, heights, width, start_pt, xaxis, yaxis, local_to_mesh_vertex);
    for (std::size_t iteration = 0; iteration < 15; ++iteration) {
        smooth_grid_points(P, width);
    }
    write_grid_mesh_as_off(P, width, "fit_polygon_on_surface_grid.off");
    auto [grid_boundary_points, grid_boundary_point_indices] = get_grid_boundary_points_and_indices(grid_points, width);
    std::vector<std::size_t> boundary_polylines;
    boundary_polylines.reserve(grid_boundary_point_indices.size());
    for (std::size_t i { 0 }; i < grid_boundary_point_indices.size(); i++) {
        boundary_polylines.push_back(i);
    }
    boundary_polylines.push_back(0);

    write_polyline_as_obj(grid_boundary_points, boundary_polylines, "fit_polygon_on_surface_boundary_polylines.obj");
    std::vector<std::array<gpf::VertexId, 2>> original_uv_edge_vertices(uv_mesh.n_edges_capacity());
    for (const auto edge : uv_mesh.edges()) {
        const auto [a, b] = edge.vertices();
        original_uv_edge_vertices[edge.id.idx] = { a.id, b.id };
    }
    std::unordered_map<gpf::FaceId, gpf::FaceId> uv_face_parent_map;
    std::unordered_map<gpf::EdgeId, gpf::EdgeId> uv_edge_parent_map;
    auto [grid_boundary_vertices, grid_boundary_halfedges] = gpf::project_polylines_on_mesh(
        grid_boundary_points,
        { boundary_polylines },
        uv_mesh,
        eps,
        &uv_face_parent_map,
        &uv_edge_parent_map);
    write_uv_mesh_as_off(uv_mesh, "fit_polygon_on_surface_grid_uv_mesh.off");

    const auto& projected_boundary_halfedges = grid_boundary_halfedges.front();
    auto separators = compute_boundary_vertex_separators(uv_mesh, grid_boundary_vertices, projected_boundary_halfedges);
    auto reversed_grid_boundary_halfedges = projected_boundary_halfedges;
    std::ranges::reverse(reversed_grid_boundary_halfedges);
    for (auto& hid : reversed_grid_boundary_halfedges) {
        hid = uv_mesh.he_twin(hid);
    }
    const auto kept_uv_faces = gpf::surround_faces_by_halfedges(uv_mesh, reversed_grid_boundary_halfedges);

    struct OutputFace {
        std::vector<std::size_t> vertices;
        FaceProp prop;
    };

    const auto n_old_mesh_vertices = mesh.n_vertices_capacity();
    std::vector<std::array<double, 3>> output_positions;
    output_positions.reserve(n_old_mesh_vertices + width * width + uv_mesh.n_vertices_capacity());
    std::vector<bool> is_inner_face(mesh.n_faces_capacity(), false);
    for (const auto inner_fid : inner_faces) {
        is_inner_face[inner_fid.idx] = true;
    }
    std::vector<std::size_t> old_mesh_vertex_map(n_old_mesh_vertices, gpf::kInvalidIndex);
    const auto add_mesh_vertices_to_output = [&mesh, &old_mesh_vertex_map, &output_positions](const gpf::VertexId old_vid) {
        if (auto new_vid = old_mesh_vertex_map[old_vid.idx]; new_vid != gpf::kInvalidIndex) {
            return new_vid;
        }
        const auto ret = old_mesh_vertex_map[old_vid.idx] = output_positions.size();
        output_positions.push_back(mesh.vertex_prop(old_vid).pt);
        return ret;
    };

    std::vector<OutputFace> output_faces;
    std::vector<GridFaceIndex> grid_face_indices;
    output_faces.reserve(mesh.n_faces() - inner_faces.size() + kept_uv_faces.size() + (width - 1) * (width - 1));
    for (const auto face : mesh.faces()) {
        if (is_inner_face[face.id.idx]) {
            continue;
        }
        OutputFace output_face { {}, face.prop() };
        for (const auto he : face.halfedges()) {
            output_face.vertices.push_back(add_mesh_vertices_to_output(he.to().id));
        }
        output_faces.push_back(std::move(output_face));
    }

    std::vector<gpf::VertexId> mesh_to_uv(n_old_mesh_vertices);
    for (std::size_t uv_idx = 0; uv_idx < local_to_mesh_vertex.size(); ++uv_idx) {
        const auto mesh_vid = local_to_mesh_vertex[uv_idx];
        if (mesh_vid.valid()) {
            mesh_to_uv[mesh_vid.idx] = gpf::VertexId { uv_idx };
        }
    }

    std::vector<std::size_t> uv_output_vertices(uv_mesh.n_vertices_capacity(), gpf::kInvalidIndex);

    std::unordered_map<gpf::EdgeId, std::vector<gpf::EdgeId>> parent_edge_to_edges_map;
    for (const auto [uv_eid, uv_parent_eid] : uv_edge_parent_map) {
        parent_edge_to_edges_map[uv_parent_eid].push_back(uv_eid);
    }

    const auto add_points_on_edge = [&](const gpf::EdgeId parent_eid, std::vector<gpf::EdgeId>& subedges) {
        const auto [uv_va, uv_vb] = original_uv_edge_vertices[parent_eid.idx];
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
    };

    for (auto& [parent_eid, subedges] : parent_edge_to_edges_map) {
        add_points_on_edge(parent_eid, subedges);
    }

    std::unordered_map<gpf::FaceId, std::vector<gpf::FaceId>> parent_face_to_faces_map;
    for (const auto [uv_fid, uv_parent_fid] : uv_face_parent_map) {
        parent_face_to_faces_map[uv_parent_fid].push_back(uv_fid);
    }

    const auto add_points_on_face = [&](const gpf::FaceId parent_fid, std::vector<gpf::FaceId>& subfaces) {
        const auto source_fid = inner_faces[parent_fid.idx];
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
    };

    for (auto& [parent_fid, subfaces] : parent_face_to_faces_map) {
        add_points_on_face(parent_fid, subfaces);
    }

    auto output_vertex_for_uv_vertex = [&](const gpf::VertexId uv_vid) {
        auto& output_vid = uv_output_vertices[uv_vid.idx];
        if (output_vid != gpf::kInvalidIndex) {
            return output_vid;
        }

        if (uv_vid.idx < local_to_mesh_vertex.size()) {
            output_vid = add_mesh_vertices_to_output(local_to_mesh_vertex[uv_vid.idx]);
        }
        assert(output_vid != gpf::kInvalidIndex);
        return output_vid;
    };

    for (const auto uv_fid : kept_uv_faces) {
        OutputFace output_face { {}, FaceProp { uv_mesh.face_prop(uv_fid).parent } };
        for (const auto he : uv_mesh.face(uv_fid).halfedges()) {
            output_face.vertices.push_back(output_vertex_for_uv_vertex(he.to().id));
        }
        output_faces.push_back(std::move(output_face));
    }

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
        Eigen::Vector3d::Map(point.data()) = P.row(grid_idx);
        output_positions.push_back(std::move(point));
    }

    struct GridPolygonVertex {
        std::size_t output_vid;
        std::array<double, 2> uv;
    };

    std::vector<std::vector<GridPolygonVertex>> boundary_chains(grid_boundary_vertices.size());
    for (std::size_t boundary_idx = 0; boundary_idx < boundary_chains.size(); ++boundary_idx) {
        const auto begin = separators[boundary_idx];
        const auto end = separators[boundary_idx + 1];
        auto& chain = boundary_chains[boundary_idx];
        chain.reserve(end - begin + 1);
        const auto grid_idx = grid_boundary_point_indices[boundary_idx];
        chain.push_back({
            output_vertex_for_uv_vertex(grid_boundary_vertices[boundary_idx]),
            grid_points[grid_idx],
        });
        for (std::size_t path_idx = begin; path_idx < end; ++path_idx) {
            const auto uv_vertex = uv_mesh.he_to(projected_boundary_halfedges[path_idx]);
            chain.push_back({ output_vertex_for_uv_vertex(uv_vertex), uv_mesh.vertex_prop(uv_vertex).pt });
        }
    }

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

            for (std::size_t i = 0; i < triangles.size(); i += 3) {
                const auto triangle_index = i / 3;
                grid_face_indices.push_back(GridFaceIndex {
                    gpf::FaceId { output_faces.size() },
                    row * (width - 1) + col,
                    row,
                    col,
                    triangle_index,
                });
                output_faces.push_back(OutputFace {
                    {
                        polygon[triangles[i]].output_vid,
                        polygon[triangles[i + 1]].output_vid,
                        polygon[triangles[i + 2]].output_vid,
                    },
                    FaceProp { fid },
                });
            }
        }
    }

    std::vector<std::vector<std::size_t>> polygons;
    std::vector<FaceProp> face_properties;
    polygons.reserve(output_faces.size());
    face_properties.reserve(output_faces.size());
    for (auto& output_face : output_faces) {
        polygons.push_back(std::move(output_face.vertices));
        face_properties.push_back(output_face.prop);
    }

    for (const auto& polygon : polygons) {
        if (polygon.size() < 3) {
            throw std::runtime_error("combined mesh contains a degenerate polygon");
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
                throw std::runtime_error("combined mesh contains a zero-length topological edge");
            }
            const auto key = std::minmax(from, to);
            auto& incidence = edge_incidence[key];
            if (++incidence > 2) {
                throw std::runtime_error("combined mesh contains a non-manifold edge");
            }
            const auto direction = std::make_pair(from, to);
            const auto direction_iter = edge_directions.find(key);
            if (direction_iter == edge_directions.end()) {
                edge_directions.emplace(key, direction);
            } else if (direction_iter->second == direction) {
                throw std::runtime_error("combined mesh contains two faces with the same edge orientation");
            }
        }
    }

    auto combined_mesh = Mesh::new_in(polygons);
    for (const auto vertex : combined_mesh.vertices()) {
        if (vertex.id.idx >= output_positions.size()) {
            throw std::runtime_error("combined mesh vertex has no position");
        }
        vertex.prop().pt = output_positions[vertex.id.idx];
    }
    std::size_t face_index = 0;
    for (const auto face : combined_mesh.faces()) {
        if (face_index >= face_properties.size()) {
            throw std::runtime_error("combined mesh face/property count mismatch");
        }
        face.prop() = face_properties[face_index++];
    }
    if (face_index != face_properties.size()) {
        throw std::runtime_error("combined mesh lost a generated face");
    }
    mesh = std::move(combined_mesh);
    write_image_relief_mesh_as_off(mesh, "fit_polygon_on_surface_combined.off");
    return grid_face_indices;
}
}
