#include "fit_on_surface.h"
#include "eigen_alias.h"
#include "flatten_surface.h"

#include <array>
#include <gpf/ids.hpp>
#include <gpf/mesh_flood_fill.hpp>
#include <gpf/project_polylines_on_mesh.hpp>

#include <igl/flipped_triangles.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>

#include <cmath>
#include <fstream>
#include <numbers>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
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
    return std::make_tuple(n_boundary_vertices, std::move(V), std::move(F));
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

    auto boundary_paths = std::get<1>(gpf::project_polylines_on_mesh(
        outer_corner_points,
        std::vector<std::vector<std::size_t>> { { 0, 1, 2, 3, 0 } },
        mesh));
    if (boundary_paths.empty() || boundary_paths.front().empty()) {
        throw std::runtime_error("failed to project boundary polyline on mesh");
    }
    auto inner_faces = gpf::surround_faces_by_halfedges(mesh, boundary_paths.front());
    auto [n_boundary_vertices, V, F] = extract_face_mesh(mesh, boundary_paths.front(), inner_faces);
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
    FlattenSurface fs(std::move(V), std::move(F), uv, 0);
    fs.slim_solve(5);
    write_uv_as_off(fs.uv, fs.F, "fit_polygon_on_surface_uv.off");
    write_faces_as_off(mesh, inner_faces, "fit_polygon_on_surface.off");

    const auto a = 2;
}
