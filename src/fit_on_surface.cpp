#include "fit_on_surface.h"
#include "eigen_alias.h"
#include "flatten_surface.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <gpf/ids.hpp>
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
#include <predicates/predicates.hpp>
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
std::array<int, 3> polygon_color(std::size_t polygon_id);

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

std::optional<double> boundary_contains_anchor_rectangle(const VMat2& uv, const std::span<const std::size_t> vertices, const std::size_t min_idx, const Eigen::VectorXi& bnd) {
    const auto div = [](const auto& a, const auto& b) noexcept {
        return Eigen::Vector2d{ a.x() * b.x() + a.y() * b.y(), a.y() * b.x() - a.x() * b.y() };
    };
    Eigen::Vector2d center = uv.row(vertices[0]).transpose();
    Eigen::Vector2d base_dir = uv.row(vertices[min_idx]).transpose() - center;
    std::vector<double> corners = {base_dir[0], base_dir[1], -base_dir[1], base_dir[0], -base_dir[0], -base_dir[1], base_dir[1], -base_dir[0]};
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
    std::optional<double> scale{};
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
    Eigen::Vector2d zero{0.0, 0.0};
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

struct EdgeSplitRequest {
    gpf::VertexId uv_vertex;
    double t;
    std::array<double, 3> point;
};

auto make_base_uv_edges(
    const fit_on_surface::Mesh& mesh,
    const uv::Mesh& uv_mesh,
    const std::vector<gpf::VertexId>& local_to_mesh_vertex)
{
    std::vector<gpf::EdgeId> base_edges(mesh.n_edges_capacity());
    base_edges.reserve(uv_mesh.n_edges());
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
            gpf::detail::normalize_barycentric(bary, gpf::detail::BARY_EPS);
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
    auto [n_boundary_vertices, vertex_map, local_to_mesh_vertex, inner_face_indices, V, F] = extract_face_mesh(mesh, boundary_paths.front(), inner_faces);
    auto uv_mesh = uv::Mesh::new_in(ranges::iota_view { std::size_t { 0 }, inner_faces.size() } | views::transform([&inner_face_indices](auto idx) { return std::span<const std::size_t, 3> { inner_face_indices.data() + idx * 3, 3 }; }));
    Eigen::VectorXi bnd(n_boundary_vertices);
    {
        auto curr_he = uv_mesh.vertex(gpf::VertexId{0}).halfedge().prev();
        Eigen::Index idx{0};
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
    fs.slim_solve(5);
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
    auto projected_polyline_paths = std::get<1>(gpf::project_polylines_on_mesh(poly_uv_pts, oriented_polylines, uv_mesh, &uv_face_parent_map, &uv_edge_parent_map));
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
