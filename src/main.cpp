#include <CGAL/AABB_traits_2.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_face_base_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/boost/graph/Euler_operations.h>
#include <CGAL/boost/graph/helpers.h>

#include <CLI/CLI.hpp>

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <cstdio>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/os.h>

#include <geometrycentral/surface/flip_geodesics.h>
#include <geometrycentral/surface/halfedge_element_types.h>
#include <geometrycentral/surface/mesh_graph_algorithms.h>
#include <geometrycentral/surface/meshio.h>
#include <geometrycentral/surface/surface_mesh.h>
#include <geometrycentral/surface/surface_mesh_factories.h>
#include <geometrycentral/surface/surface_point.h>
#include <geometrycentral/surface/trace_geodesic.h>
#include <geometrycentral/utilities/vector2.h>
#include <geometrycentral/utilities/vector3.h>

#include <igl/barycentric_coordinates.h>
#include <igl/project_to_line_segment.h>
#include <igl/triangle/triangulate.h>
#include <igl/per_face_normals.h>
#include <igl/read_triangle_mesh.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/flipped_triangles.h>
#include <igl/write_triangle_mesh.h>

#include <boost/functional/hash.hpp>

#include "eigen_alias.h"
#include "flatten_surface.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/utilities/utilities.h"
#include "igl/harmonic.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <ranges>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <array>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_2 = Kernel::Point_2;
using Point_3 = Kernel::Point_3;
using Mesh_2 = CGAL::Surface_mesh<Point_2>;
using Mesh = CGAL::Surface_mesh<Point_3>;
using VI = Mesh::Vertex_index;
using HI = Mesh::Halfedge_index;
using EI = Mesh::Edge_index;
using FI = Mesh::Face_index;
using Vb = CGAL::Triangulation_vertex_base_with_info_2<VI, Kernel>;
using Fb_info = CGAL::Triangulation_face_base_with_info_2<bool, Kernel>;
using Fb = CGAL::Constrained_triangulation_face_base_2<Kernel, Fb_info>;
using Tds = CGAL::Triangulation_data_structure_2<Vb, Fb>;
using CDT = CGAL::Constrained_Delaunay_triangulation_2<Kernel, Tds>;

using Triangle_2 = Kernel::Triangle_2;
using Iterator = std::vector<Triangle_2>::const_iterator;
using Primitive = CGAL::AABB_triangle_primitive_2<Kernel, Iterator>;
using AABB_triangle_traits = CGAL::AABB_traits_2<Kernel, Primitive>;
using Tree = CGAL::AABB_tree<AABB_triangle_traits>;

using GV3 = geometrycentral::Vector3;
using GMesh = geometrycentral::surface::SurfaceMesh;

namespace gs = geometrycentral::surface;
namespace PMP = CGAL::Polygon_mesh_processing;

const auto& view_ts = std::views::transform;

namespace {
struct Bounds {
    Kernel::Point_3 min;
    Kernel::Point_3 max;
};

auto write_grid(const std::string& name, const std::vector<Point_2>& points)
{
    std::ofstream file(name);
    for (Eigen::Index i = 0; i < points.size(); ++i) {
        file << "v " << points[i].x() << " " << points[i].y() << " 0\n";
    }

    for (int i = 0; i < 4; i++) {
        file << "l " << i + 1 << " " << (i + 1) % 4 + 1 << "\n";
    }
    file.close();
}

auto write_uv(const std::string& name, const VMat2& uv, const FMat& F)
{
    std::ofstream file(name);
    for (Eigen::Index i = 0; i < uv.rows(); ++i) {
        file << "v " << uv(i, 0) << " " << uv(i, 1) << " 0\n";
    }
    for (Eigen::Index i = 0; i < F.rows(); ++i) {
        file << "f " << F(i, 0) + 1 << " " << F(i, 1) + 1 << " " << F(i, 2) + 1 << "\n";
    }
    file.close();
}

auto write_uv1(const std::string& name, const std::vector<Point_2>& uv, const std::vector<std::vector<std::size_t>>& F)
{
    std::ofstream file(name);
    for (Eigen::Index i = 0; i < uv.size(); ++i) {
        file << "v " << uv[i].x() << " " << uv[i].y() << " 0\n";
    }

    for (Eigen::Index i = 0; i < F.size(); i++) {
        file << "f ";
        for (Eigen::Index j = 0; j < F[i].size(); j++) {
            file << F[i][j] + 1 << " ";
        }
        file << "\n";
    }
    file.close();
}

auto write_mesh2(const std::string& name, const Mesh_2& mesh)
{
    std::ofstream file(name);
    for (const auto& point : mesh.points()) {
        file << "v " << point.x() << " " << point.y() << " " << 0.0 << "\n";
    }
    for (const auto fid : mesh.faces()) {
        file << "f ";
        for (const auto& vertex : mesh.vertices_around_face(mesh.halfedge(fid))) {
            file << vertex.idx() + 1 << " ";
        }
        file << "\n";
    }
    file.close();
}

auto write_mesh_with_colors(
    const std::string& name,
    const std::vector<Point_3>& points,
    const std::vector<std::vector<std::size_t>>& faces,
    const std::size_t face_start_index,
    const std::size_t face_end_index
) {
    std::ofstream file(name);
    file << "mtllib colors.mtl\n";
    for (const auto& point : points) {
        file << "v " << point.x() << " " << point.y() << " " << point.z() << "\n";
    }
    file << "usemtl white_mat\n";
    for (std::size_t fid = 0; fid < faces.size(); fid++) {
        if (fid == face_start_index) {
            file << "usemtl red_mat\n";
        } else if (fid == face_end_index) {
            file << "usemtl white_mat\n";
        }
        file << "f ";
        for (const auto& vid : faces[fid]) {
            file << vid + 1 << " ";
        }
        file << "\n";
    }
    file.close();

}

void compute_bary_coordinates(
    const Point_2& pa,
    const Point_2& pb,
    const Point_2& pc,
    const std::vector<Point_2>& points,
    const std::vector<std::size_t>& point_indices,
    std::vector<double>& bary_coordinates
) {
    auto B = VMat::Map(bary_coordinates.data(), bary_coordinates.size() / 3, 3);

    const auto v0 = pb - pa;
    const auto v1 = pc - pa;
    const auto d00 = v0 * v0;
    const auto d01 = v0 * v1;
    const auto d11 = v1 * v1;
    const auto denom = d00 * d11 - d01 * d01;
    for (const auto pid : point_indices) {
        const auto v2 = points[pid] - pa;
        const auto d20 = v2 * v0;
        const auto d21 = v2 * v1;
        const auto b1 = (d11 * d20 - d01 * d21) / denom;
        const auto b2 = (d00 * d21 - d01 * d20) / denom;
        const auto b0 = 1.0 - b1 - b2;
        if (b1 < -1e-8 || b2 < -1e-8 || b0 < -1e-8) {
            const auto a = 2;
        }
        B.row(pid) << b0, b1, b2;
    }
}

std::optional<Bounds> compute_bounds(Mesh& mesh)
{
    if (mesh.number_of_vertices() == 0) {
        return std::nullopt;
    }

    for (auto& pt : mesh.points()) {
        const auto x = pt.x();
        const auto y = pt.y();
        const auto z = pt.z();
        pt = Point_3(x, z, -y);
    }

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();

    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();

    for (Mesh::Vertex_index v : mesh.vertices()) {
        const auto& point = mesh.point(v);
        min_x = std::min(min_x, point.x());
        min_y = std::min(min_y, point.y());
        min_z = std::min(min_z, point.z());

        max_x = std::max(max_x, point.x());
        max_y = std::max(max_y, point.y());
        max_z = std::max(max_z, point.z());
    }

    return Bounds { Kernel::Point_3(min_x, min_y, min_z),
        Kernel::Point_3(max_x, max_y, max_z) };
}

void scale_box(Eigen::Vector2d& min_pt, Eigen::Vector2d& max_pt, const double s)
{
    const auto center = ((min_pt + max_pt) * 0.5).eval();
    const auto vec = ((max_pt - center) * s).eval();
    max_pt = center + vec;
    min_pt = center - vec;
}


std::size_t remove_faces_on_min_plane(Mesh& mesh, double min_z,
    double tolerance)
{
    std::vector<Mesh::Face_index> faces_to_remove;
    faces_to_remove.reserve(mesh.number_of_faces());

    for (Mesh::Face_index face : mesh.faces()) {
        bool should_remove = false;
        for (Mesh::Vertex_index v :
            CGAL::vertices_around_face(mesh.halfedge(face), mesh)) {
            const auto& point = mesh.point(v);
            if (std::abs(point.z() - min_z) <= tolerance) {
                should_remove = true;
                break;
            }
        }

        if (should_remove) {
            faces_to_remove.push_back(face);
        }
    }

    for (Mesh::Face_index face : faces_to_remove) {
        CGAL::Euler::remove_face(mesh.halfedge(face), mesh);
    }

    if (!faces_to_remove.empty()) {
        mesh.collect_garbage();
    }

    return faces_to_remove.size();
}

void write_halfedges(
    const std::string& name,
    gs::ManifoldSurfaceMesh& mesh,
    gs::VertexPositionGeometry& geom,
    const std::vector<gs::Halfedge>& halfedges
) {
    auto out = fmt::output_file(name);
    for (const auto& halfedge : halfedges) {
        const auto& source = halfedge.tailVertex();
        const auto& source_point = geom.vertexPositions[source];
        out.print("v {} {} {}\n", source_point.x, source_point.y, source_point.z);
    }
    const auto& he = halfedges.back();
    const auto& target = he.tipVertex();
    const auto& target_point = geom.vertexPositions[target];
    out.print("v {} {} {}\n", target_point.x, target_point.y, target_point.z);

    for (std::size_t i = 0; i < halfedges.size(); i++) {
        const auto j = i + 1;
        out.print("l {} {}\n", i + 1, j + 1);
    }
    out.close();
}
void write_halfedges(
    const std::string& name,
    const Mesh_2& mesh,
    const std::vector<HI>& halfedges
) {
    const auto closed = mesh.source(halfedges.front()) == mesh.target(halfedges.back());
    auto out = fmt::output_file(name);
    for (const auto& halfedge : halfedges) {
        const auto source = mesh.source(halfedge);
        const auto& source_point = mesh.point(source);
        out.print("v {} {} {}\n", source_point.x(), source_point.y(), 0.0);
    }
    if (!closed) {
        const auto& he = halfedges.back();
        const auto& target_point = mesh.point(mesh.target(he));
        out.print("v {} {} {}\n", target_point.x(), target_point.y(), 0.0);
    }

    const auto len = closed ? (halfedges.size() - 1) : halfedges.size();
    for (std::size_t i = 0; i < len; i++) {
        const auto j = i + 1;
        out.print("l {} {}\n", i + 1, j + 1);
    }
    if (closed) {
        out.print("l {} {}\n", halfedges.size(), 1);
    }
    out.close();
}


void write_polyline(const std::string& name, const std::vector<gs::SurfacePoint>& points, gs::VertexData<geometrycentral::Vector3>& vertexData)
{
    auto out = fmt::output_file(name);
    for (const auto& point : points) {
        const auto p = point.interpolate(vertexData);
        out.print("v {} {} {}\n", p.x, p.y, p.z);
    }

    for (std::size_t i = 0; i + 1 < points.size(); i++) {
        const auto j = i + 1;
        out.print("l {} {}\n", i + 1, j + 1);
    }
    out.close();
}
template <typename Point>
void split_edges(
    CGAL::Surface_mesh<Point>& mesh,
    const std::unordered_map<EI, std::vector<std::size_t>>& edge_points_map,
    const std::vector<Point>& points,
    std::vector<VI>& point_vertex_indices,
    const double tol)
{
    for (auto& [eid, edge_point_indices] : edge_points_map) {
        if (edge_point_indices.size() == 1) {
            const auto new_hid = CGAL::Euler::split_edge(mesh.halfedge(eid), mesh);
            const auto new_vid = mesh.target(new_hid);
            const auto pid = edge_point_indices[0];
            mesh.point(new_vid) = points[pid];
            point_vertex_indices[pid] = new_vid;
        } else {
            auto curr_hid = mesh.halfedge(eid);
            const auto va = mesh.source(curr_hid);
            const auto& pa = mesh.point(va);
            std::vector<std::size_t> indices(edge_point_indices.size());
            std::iota(indices.begin(), indices.end(), 0);
            const auto distances = edge_point_indices
                | std::views::transform([&](const auto pid) { return std::sqrt((points[pid] - pa).squared_length()); })
                | std::ranges::to<std::vector>();
            std::sort(indices.begin(), indices.end(), [&](auto i, auto j) { return distances[i] < distances[j]; });
            std::size_t j = 0;
            for (std::size_t i = 0; i < indices.size(); i++) {
                const auto pid = edge_point_indices[indices[i]];
                if (i == 0 || (i > 0 && (distances[indices[i]] - distances[indices[j]]) > tol)) {
                    const auto new_hid = CGAL::Euler::split_edge(mesh.halfedge(eid), mesh);
                    const auto new_vid = mesh.target(new_hid);
                    mesh.point(new_vid) = points[pid];
                    point_vertex_indices[pid] = new_vid;
                    curr_hid = mesh.next(new_hid);
                    j = i;
                } else {
                    point_vertex_indices[pid] = point_vertex_indices[edge_point_indices[indices[j]]];
                }
            }
        }
    }
}

auto record_boundary_point(
    const gs::SurfacePoint& surface_point,
    const gs::VertexData<geometrycentral::Vector3>& vertex_positions,
    const Mesh& target_mesh,
    std::vector<Point_3>& inserted_points,
    std::vector<VI>& point_vertex_handles,
    std::unordered_map<EI, std::vector<std::size_t>>& edge_split_map,
    std::unordered_map<FI, std::vector<std::size_t>>& face_split_map)
{

    const auto pos = surface_point.interpolate(vertex_positions);
    const auto point_idx = inserted_points.size();
    inserted_points.emplace_back(pos.x, pos.y, pos.z);
    switch (surface_point.type) {
    case gs::SurfacePointType::Vertex: {
        point_vertex_handles.emplace_back(VI(surface_point.vertex.getIndex()));
        break;
    }
    case gs::SurfacePointType::Edge: {
        auto [va, vb] = surface_point.edge.adjacentVertices();
        const auto eid = target_mesh.edge(target_mesh.halfedge(VI(va.getIndex()), VI(vb.getIndex())));
        edge_split_map[eid].emplace_back(point_idx);
        point_vertex_handles.emplace_back();
        break;
    }
    case gs::SurfacePointType::Face: {
        face_split_map[FI(surface_point.face.getIndex())].emplace_back(point_idx);
        point_vertex_handles.emplace_back();
        break;
    }
    }
}

auto record_boundary_point_2d(
    const gs::SurfacePoint& surface_point,
    const gs::VertexData<geometrycentral::Vector3>& vertex_positions,
    const Mesh_2& target_mesh,
    std::vector<Point_2>& inserted_points,
    std::vector<VI>& point_vertex_handles,
    std::unordered_map<EI, std::vector<std::size_t>>& edge_split_map,
    std::unordered_map<FI, std::vector<std::size_t>>& face_split_map)
{

    const auto pos = surface_point.interpolate(vertex_positions);
    const auto point_idx = inserted_points.size();
    inserted_points.emplace_back(pos.x, pos.y);
    switch (surface_point.type) {
    case gs::SurfacePointType::Vertex: {
        point_vertex_handles.emplace_back(VI(surface_point.vertex.getIndex()));
        break;
    }
    case gs::SurfacePointType::Edge: {
        auto [va, vb] = surface_point.edge.adjacentVertices();
        const auto eid = target_mesh.edge(target_mesh.halfedge(VI(va.getIndex()), VI(vb.getIndex())));
        edge_split_map[eid].emplace_back(point_idx);
        point_vertex_handles.emplace_back();
        break;
    }
    case gs::SurfacePointType::Face: {
        face_split_map[FI(surface_point.face.getIndex())].emplace_back(point_idx);
        point_vertex_handles.emplace_back();
        break;
    }
    }
}

void add_points_on_face(
    const Mesh::Face_index fid,
    Mesh& mesh,
    const std::vector<Point_3>& points,
    const std::vector<std::size_t>& point_indices,
    const std::size_t n_old_vertices,
    std::vector<Mesh::Vertex_index>& point_vertex_map)
{
    std::vector<Mesh::Halfedge_index> halfedges;
    for (const auto hid : mesh.halfedges_around_face(mesh.halfedge(fid))) {
        halfedges.emplace_back(hid);
    }
    auto vertices = halfedges | std::views::transform([&](const auto hid) {
        return mesh.source(hid);
    }) | std::ranges::to<std::vector>();
    const auto n_face_verts = vertices.size();
    const auto tri = vertices
        | std::views::filter([&, n_old_vertices](const auto vid) { return vid.idx() < n_old_vertices; })
        | std::views::transform([&](const auto vid) { return mesh.point(vid); })
        | std::ranges::to<std::vector>();
    for (const auto pid : point_indices) {
        const auto new_vid = mesh.add_vertex(points[pid]);
        vertices.emplace_back(new_vid);
        point_vertex_map[pid] = new_vid;
    }
    const auto normal = CGAL::unit_normal(tri[0], tri[1], tri[2]);
    auto x_axis = tri[1] - tri[0];
    x_axis / std::sqrt(x_axis.squared_length());
    const auto y_axis = CGAL::cross_product(normal, x_axis);
    const auto project = [&x_axis, &y_axis, &tri](const Point_3& p) {
        const auto v = p - tri[0];
        return Point_2(
            CGAL::scalar_product(v, x_axis),
            CGAL::scalar_product(v, y_axis));
    };
    CDT cdt;
    const auto cdt_vertices = vertices | std::views::transform([&](const auto vid) {
        CDT::Vertex_handle vhandle = cdt.insert(project(mesh.point(vid)));
        vhandle->info() = vid;
        return vhandle;
    }) | std::ranges::to<std::vector>();
    std::unordered_map<Mesh::Vertex_index, Mesh::Vertex_index> constrained_vertex_map;
    for (std::size_t i = 0; i < n_face_verts; i++) {
        const auto j = (i + 1) % n_face_verts;
        cdt.insert_constraint(cdt_vertices[i], cdt_vertices[j]);
        constrained_vertex_map.emplace(vertices[i], vertices[j]);
    }
    for (auto face : cdt.all_face_handles()) {
        face->info() = false;
    }
    std::vector<CDT::Face_handle> cdt_valid_faces;
    std::vector<int> bdy_edge_indices;
    for (auto edge : cdt.constrained_edges()) {
        const auto va = edge.first->vertex((edge.second + 1) % 3)->info();
        const auto vb = edge.first->vertex((edge.second + 2) % 3)->info();
        if (constrained_vertex_map[va] != vb) {
            edge = cdt.mirror_edge(edge);
        }
        cdt.mirror_edge(edge).first->info() = true;
        auto face = edge.first;
        auto index = edge.second;
        if (face->info()) {
            continue;
        }
        face->info() = true;
        cdt_valid_faces.emplace_back(face);
        bdy_edge_indices.emplace_back(index);
    }
    const auto n_bdy_faces = cdt_valid_faces.size();
    for (std::size_t i = 0; i < n_bdy_faces; i++) {
        const auto fh = cdt_valid_faces[i];
        const auto index = bdy_edge_indices[i];
        for (int i = 1; i < 3; i++) {
            auto nei_fh = fh->neighbor((i + index) % 3);
            if (nei_fh->info() || cdt.is_infinite(nei_fh)) {
                continue;
            }
            nei_fh->info() = true;
            cdt_valid_faces.emplace_back(nei_fh);
        }
    }
    for (std::size_t i = n_bdy_faces; i < cdt_valid_faces.size(); i++) {
        const auto fh = cdt_valid_faces[i];
        for (int i = 0; i < 3; i++) {
            auto nei_fh = fh->neighbor(i);
            if (nei_fh->info()) {
                continue;
            }
            nei_fh->info() = true;
            cdt_valid_faces.emplace_back(nei_fh);
        }
    }
    CGAL::Euler::remove_face(mesh.halfedge(fid), mesh);
    for (const auto& fh : cdt_valid_faces) {
        std::vector<Mesh::Vertex_index> face_vertices { fh->vertex(0)->info(), fh->vertex(1)->info(), fh->vertex(2)->info() };
        const auto new_fid = mesh.add_face(face_vertices);
        if (!new_fid.is_valid()) {
            std::cout << "failed to add faces\n";
        }
    }
}

void triangulate_face(
    const std::vector<Point_2>& points,
    const std::vector<std::size_t>& vertices,
    std::vector<std::vector<std::size_t>>& new_faces
) {

    VMat2 V(vertices.size(), 2);
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const auto vid = vertices[i];
        const auto& point = points[vid];
        V.row(i) << point.x(), point.y();
    }
    EMat E(vertices.size(), 2);
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        E.row(i) << i, (i + 1) % vertices.size();
    }

    VMat2 H;
    VMat2 WV;
    FMat WF;
    igl::triangle::triangulate(V, E, H, "Q", WV, WF);
    for (Eigen::Index i = 0; i < WF.rows(); i++) {
        std::vector<std::size_t> face;
        for (Eigen::Index j = 0; j < 3; j++) {
            const auto vid = vertices[WF(i, j)];
            face.emplace_back(vid);
        }
        new_faces.emplace_back(std::move(face));
    }
}

void triangualte_on_face(
    const Mesh_2& mesh,
    const FI fid,
    const std::size_t real_fid,
    std::vector<Point_2>& points,
    const std::vector<Point_2>& face_points,
    const std::vector<std::size_t>& face_point_indices,
    std::vector<VI>& point_vertex_indices,
    std::vector<std::vector<std::size_t>>& new_faces,
    const FMat& F,
    std::vector<double>& bary_coordinates)
{
    std::vector<std::size_t> vertices;
    for (const auto vid : mesh.vertices_around_face(mesh.halfedge(fid))) {
        vertices.emplace_back(vid.id());
    }
    const auto n_old_points = points.size();
    const auto n_old_vertices = vertices.size();
    for (const auto pid : face_point_indices) {
        const auto vid = points.size();
        points.emplace_back(face_points[pid]);
        point_vertex_indices[pid] = VI(vid);
        vertices.emplace_back(vid);
    }

    VMat2 V(vertices.size(), 2);
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const auto vid = vertices[i];
        const auto& point = points[vid];
        V.row(i) << point.x(), point.y();
    }

    EMat E(n_old_vertices, 2);
    for (std::size_t i = 0; i < n_old_vertices; ++i) {
        E.row(i) << i, (i + 1) % n_old_vertices;
    }

    VMat2 H;
    VMat2 WV;
    FMat WF;
    igl::triangle::triangulate(V, E, H, "Q", WV, WF);
    for (Eigen::Index i = 0; i < WF.rows(); i++) {
        std::vector<std::size_t> face;
        for (Eigen::Index j = 0; j < 3; j++) {
            const auto vid = vertices[WF(i, j)];
            face.emplace_back(vid);
        }
        new_faces.emplace_back(std::move(face));
    }

    if (!face_point_indices.empty()) {
        const auto fv = F.row(real_fid);
        const auto& pa = mesh.point(VI(fv[0]));
        const auto& pb = mesh.point(VI(fv[1]));
        const auto& pc = mesh.point(VI(fv[2]));

        compute_bary_coordinates(pa, pb, pc, face_points, face_point_indices, bary_coordinates);
    }
}

void add_points_on_face_2d(
    const FI fid,
    Mesh_2& mesh,
    const std::vector<Point_2>& points,
    const std::vector<std::size_t>& point_indices,
    const std::size_t n_old_vertices,
    std::vector<Mesh::Vertex_index>& point_vertex_map,
    std::vector<std::vector<std::size_t>>& new_faces,
    VMat& bary_centers)
{
    std::vector<Mesh::Halfedge_index> halfedges;
    for (const auto hid : mesh.halfedges_around_face(mesh.halfedge(fid))) {
        halfedges.emplace_back(hid);
    }
    auto vertices = halfedges | std::views::transform([&](const auto hid) {
        return mesh.source(hid);
    }) | std::ranges::to<std::vector>();
    const auto n_face_verts = vertices.size();
    const auto tri = vertices
        | std::views::filter([&, n_old_vertices](const auto vid) { return vid.idx() < n_old_vertices; })
        | std::views::transform([&](const auto vid) { return mesh.point(vid); })
        | std::ranges::to<std::vector>();
    for (const auto pid : point_indices) {
        const auto new_vid = mesh.add_vertex(points[pid]);
        vertices.emplace_back(new_vid);
        point_vertex_map[pid] = new_vid;
    }
    const auto project = [](const Point_2& p) {
        return p;
    };
    CDT cdt;
    const auto cdt_vertices = vertices | std::views::transform([&](const auto vid) {
        CDT::Vertex_handle vhandle = cdt.insert(project(mesh.point(vid)));
        vhandle->info() = vid;
        return vhandle;
    }) | std::ranges::to<std::vector>();
    std::unordered_map<Mesh::Vertex_index, Mesh::Vertex_index> constrained_vertex_map;
    for (std::size_t i = 0; i < n_face_verts; i++) {
        const auto j = (i + 1) % n_face_verts;
        cdt.insert_constraint(cdt_vertices[i], cdt_vertices[j]);
        constrained_vertex_map.emplace(vertices[i], vertices[j]);
    }

    for (auto face : cdt.all_face_handles()) {
        face->info() = false;
    }
    std::vector<CDT::Face_handle> cdt_valid_faces;
    std::vector<int> bdy_edge_indices;
    for (auto edge : cdt.constrained_edges()) {
        const auto va = edge.first->vertex((edge.second + 1) % 3)->info();
        const auto vb = edge.first->vertex((edge.second + 2) % 3)->info();
        if (constrained_vertex_map[va] != vb) {
            edge = cdt.mirror_edge(edge);
        }
        cdt.mirror_edge(edge).first->info() = true;
        auto face = edge.first;
        auto index = edge.second;
        if (face->info()) {
            continue;
        }
        face->info() = true;
        cdt_valid_faces.emplace_back(face);
        bdy_edge_indices.emplace_back(index);
    }
    const auto n_bdy_faces = cdt_valid_faces.size();
    for (std::size_t i = 0; i < n_bdy_faces; i++) {
        const auto fh = cdt_valid_faces[i];
        const auto index = bdy_edge_indices[i];
        for (int i = 1; i < 3; i++) {
            auto nei_fh = fh->neighbor((i + index) % 3);
            if (nei_fh->info() || cdt.is_infinite(nei_fh)) {
                continue;
            }
            nei_fh->info() = true;
            cdt_valid_faces.emplace_back(nei_fh);
        }
    }
    for (std::size_t i = n_bdy_faces; i < cdt_valid_faces.size(); i++) {
        const auto fh = cdt_valid_faces[i];
        for (int i = 0; i < 3; i++) {
            auto nei_fh = fh->neighbor(i);
            if (nei_fh->info()) {
                continue;
            }
            nei_fh->info() = true;
            cdt_valid_faces.emplace_back(nei_fh);
        }
    }
    // CGAL::Euler::remove_face(mesh.halfedge(fid), mesh);
    for (const auto& fh : cdt_valid_faces) {
        std::vector<std::size_t> face_vertices { fh->vertex(0)->info().id(), fh->vertex(1)->info().id(), fh->vertex(2)->info().id() };
        new_faces.emplace_back(std::move(face_vertices));
        // const auto new_fid = mesh.add_face(face_vertices);
    }
}

auto insert_point_into_mesh(
    Mesh& mesh,
    const std::vector<Point_3>& points,
    std::vector<VI>& point_vertex_indices,
    std::unordered_map<EI, std::vector<std::size_t>> edge_points_map,
    std::unordered_map<FI, std::vector<std::size_t>> face_points_map)
{
    const auto n_old_vertices = mesh.number_of_vertices();
    split_edges(mesh, edge_points_map, points, point_vertex_indices, 0.001);
    for (const auto& [fid, point_indices] : face_points_map) {
        add_points_on_face(fid, mesh, points, point_indices, n_old_vertices, point_vertex_indices);
    }
    if (!face_points_map.empty()) {
        mesh.collect_garbage();
        PMP::triangulate_faces(mesh);
    }
}

template <typename P>
auto surround_faces(const CGAL::Surface_mesh<P>& mesh, const std::vector<Mesh ::Halfedge_index>& halfedges)
{
    std::vector<bool> visited(mesh.number_of_faces() + mesh.number_of_removed_faces(), false);
    std::vector<Mesh::Face_index> faces;
    for (const auto hid : halfedges) {
        visited[mesh.face(mesh.opposite(hid))] = true;
        const auto fid = mesh.face(hid);
        if (!visited[fid]) {
            faces.emplace_back(fid);
            visited[fid] = true;
        }
    }
    for (std::size_t i = 0; i < faces.size(); i++) {
        const auto curr_fid = faces[i];
        for (const auto hid : mesh.halfedges_around_face(mesh.halfedge(curr_fid))) {
            const auto oppo_fid = mesh.face(mesh.opposite(hid));
            if (oppo_fid.is_valid() && !visited[oppo_fid]) {
                faces.emplace_back(oppo_fid);
                visited[oppo_fid] = true;
            }
        }
    }
    return faces;
}

template <typename P>
auto extract_faces(const CGAL::Surface_mesh<P>& mesh, const std::vector<FI>& faces, const std::vector<HI>& boundary_halfedges)
{
    constexpr auto INVALID = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> point_map(mesh.number_of_vertices() + mesh.number_of_removed_vertices(), INVALID);
    std::vector<P> points;
    std::vector<VI> point_vertices;
    for (const auto hid : boundary_halfedges) {
        const auto vid = mesh.source(hid);
        if (point_map[vid] == INVALID) {
            point_map[vid] = points.size();
            points.emplace_back(mesh.point(vid));
            point_vertices.emplace_back(vid);
        }
    }

    std::vector<std::vector<std::size_t>> face_vertices;
    for (const auto fid : faces) {
        std::vector<std::size_t> vertices;
        for (const auto vid : mesh.vertices_around_face(mesh.halfedge(fid))) {
            if (point_map[vid] == INVALID) {
                point_map[vid] = points.size();
                points.emplace_back(mesh.point(vid));
                point_vertices.emplace_back(vid);
            }
            vertices.push_back(point_map[vid]);
        }
        face_vertices.emplace_back(vertices);
    }
    return std::make_tuple(std::move(points), std::move(point_map), std::move(point_vertices), std::move(face_vertices));
}

auto mesh_to_eigen_mat(const std::vector<Point_3>& points, std::vector<std::vector<std::size_t>>& faces)
{
    VMat V(points.size(), 3);
    for (std::size_t i = 0; i < points.size(); ++i) {
        V.row(i) << points[i].x(), points[i].y(), points[i].z();
    }

    FMat F(faces.size(), 3);
    for (std::size_t i = 0; i < faces.size(); ++i) {
        F.row(i) << faces[i][0], faces[i][1], faces[i][2];
    }

    return std::make_pair(std::move(V), std::move(F));
}

auto trace_bounding_box_outline(
    const std::string& input_mesh_path,
    Eigen::Vector2d bounding_min,
    Eigen::Vector2d bounding_max,
    const std::size_t center_vid
) {
    auto [gmesh, geom] = gs::readManifoldSurfaceMesh(input_mesh_path);
    std::vector<Point_3> points;
    std::vector<std::vector<std::size_t>> faces;
    for (const auto& vid : gmesh->vertices()) {
        const auto& pt = geom->vertexPositions[vid];
        points.emplace_back(pt.x, pt.y, pt.z);
    }
    for (const auto& fid : gmesh->faces()) {
        std::vector<std::size_t> face;
        for (const auto vid : fid.adjacentVertices()) {
            face.emplace_back(vid.getIndex());
        }
        faces.emplace_back(face);
    }
    Mesh mesh;
    PMP::polygon_soup_to_polygon_mesh(points, faces, mesh);

    const auto center_vertex = gmesh->vertex(center_vid);
    const auto bounding_box_center = ((bounding_min + bounding_max) * 0.5).eval();
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> bounding_box_corners {
        bounding_min,
        {bounding_max.x(), bounding_min.y()},
        bounding_max,
        {bounding_min.x(), bounding_max.y()}
    };

    constexpr double scale_factor = 1.2;
    scale_box(bounding_min, bounding_max, scale_factor);
    bounding_box_corners.emplace_back(bounding_min.x(), bounding_box_center.y());
    bounding_box_corners.emplace_back(bounding_min);
    bounding_box_corners.emplace_back(bounding_box_center.x(), bounding_min.y());
    bounding_box_corners.emplace_back(bounding_max.x(), bounding_min.y());
    bounding_box_corners.emplace_back(bounding_max.x(), bounding_box_center.y());
    bounding_box_corners.emplace_back(bounding_max);
    bounding_box_corners.emplace_back(bounding_box_center.x(), bounding_max.y());
    bounding_box_corners.emplace_back(bounding_min.x(), bounding_max.y());


    std::unordered_map<EI, std::vector<std::size_t>> outline_edge_points_map;
    std::unordered_map<FI, std::vector<std::size_t>> outline_face_points_map;
    std::vector<Point_3> outline_points;
    std::vector<VI> outline_vertex_indices;
    for (const auto& corner : bounding_box_corners) {
        const auto dir = (corner - bounding_box_center).eval();
        const auto trace_result = gs::traceGeodesic(*geom, center_vertex, geometrycentral::Vector2 { dir[0], dir[1] }, { .includePath = true });
        record_boundary_point(trace_result.endPoint, geom->vertexPositions, mesh, outline_points, outline_vertex_indices, outline_edge_points_map, outline_face_points_map);
    }
    insert_point_into_mesh(mesh, outline_points, outline_vertex_indices, outline_edge_points_map, outline_face_points_map);
    CGAL::IO::write_polygon_mesh("mesh1.obj", mesh);
    return std::make_tuple(std::move(mesh), std::move(outline_vertex_indices));
}

template<typename P>
auto split_face_and_return_edge(
    CGAL::Surface_mesh<P>& target_mesh,
    const VI start_vertex,
    const VI end_vertex,
    const std::size_t original_vertex_threshold)
{
    auto first_v = start_vertex;
    auto second_v = end_vertex;
    if (first_v.id() < second_v.id()) {
        std::swap(first_v, second_v);
    }
    if (first_v.id() < original_vertex_threshold) {
        const auto common_edge = target_mesh.halfedge(start_vertex, end_vertex);
        if (!common_edge.is_valid()) {
            throw std::runtime_error("cannot find old halfedge");
        }
        return common_edge;
    } else {
        const auto first_out_halfedge = target_mesh.halfedge(first_v);
        Mesh::Face_index common_face;
        Mesh::Halfedge_index second_halfedge;
        for (const auto fid : target_mesh.faces_around_target(first_out_halfedge)) {
            for (const auto hid : target_mesh.halfedges_around_face(target_mesh.halfedge(fid))) {
                if (target_mesh.target(hid) == second_v) {
                    second_halfedge = hid;
                    break;
                }
            }
            if (second_halfedge.is_valid()) {
                common_face = fid;
                break;
            }
        }
        if (!common_face.is_valid()) {
            throw std::runtime_error("cannot find common face");
        }
        Mesh::Halfedge_index first_halfedge;
        for (const auto hid : target_mesh.halfedges_around_face(target_mesh.halfedge(common_face))) {
            if (target_mesh.target(hid) == first_v) {
                first_halfedge = hid;
                break;
            }
        }
        Mesh::Halfedge_index split_halfedge;
        if (first_v == start_vertex) {
            split_halfedge = CGAL::Euler::split_face(first_halfedge, second_halfedge, target_mesh);
        } else {
            split_halfedge = CGAL::Euler::split_face(second_halfedge, first_halfedge, target_mesh);
        }
        return split_halfedge;
    }
}

auto generate_grid_points(const std::size_t grid_dimension, const double edge_length)
{
    const auto X = Eigen::VectorXd::LinSpaced(grid_dimension, 0.0, edge_length).eval();
    std::vector<Point_2> points;
    points.reserve(X.size() * X.size());
    for (std::size_t j = 0; j < X.size(); ++j) {
        for (std::size_t i = 0; i < X.size(); ++i) {
            points.push_back(Point_2(X(i), X(j)));
        }
    }
    return points;
}

void locate_points_on_face(
    const Mesh_2& mesh,
    std::vector<Point_2>& all_query_points,
    const FI fid,
    std::vector<std::size_t>& face_points,
    std::unordered_map<EI, std::vector<std::size_t>>& edge_points_map,
    std::vector<VI>& point_vertex_indices,
    VMat& bary_centers,
    const double sq_tol)
{
    const auto h1 = mesh.halfedge(fid);
    const auto h2 = mesh.next(h1);
    const auto h3 = mesh.next(h2);
    const std::array<HI, 3> halfedges = { h1, h2, h3 };
    const std::array<VI, 3> vertices = { mesh.source(h1), mesh.source(h2), mesh.source(h3) };
    const auto pts = vertices | view_ts([&mesh](VI vi) {
        const auto& p = mesh.point(vi);
        return Eigen::RowVector2d(p.x(), p.y());
    }) | std::ranges::to<std::vector<Eigen::RowVector2d, Eigen::aligned_allocator<Eigen::RowVector2d>>>();

    Eigen::MatrixXd P(face_points.size(), 2);
    for (std::size_t i = 0; i < face_points.size(); ++i) {
        const auto& p = all_query_points[face_points[i]];
        P.row(i) << p.x(), p.y();
    }

    std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> parameters { Eigen::VectorXd(P.rows()), Eigen::VectorXd(P.rows()), Eigen::VectorXd(P.rows()) };
    std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> distances { Eigen::VectorXd(P.rows()), Eigen::VectorXd(P.rows()), Eigen::VectorXd(P.rows()) };

    for (std::size_t i = 0; i < 3; i++) {
        const auto j = (i + 1) % 3;
        igl::project_to_line_segment(P, pts[i], pts[j], parameters[i], distances[i]);
    }

    std::size_t idx = 0;
    for (std::size_t i = 0; i < face_points.size(); i++) {
        const auto pid = face_points[i];
        std::size_t min_idx = 0;
        double min_dist = distances[0][i];
        for (std::size_t j = 1; j < 3; j++) {
            if (distances[j][i] < min_dist) {
                min_idx = j;
                min_dist = distances[j][i];
            }
        }
        if (min_dist > sq_tol) {
            if (idx != i) {
                face_points[idx] = face_points[i];
            }
            idx += 1;
            continue;
        }

        const auto next_min_idx = (min_idx + 1) % 3;
        const auto& pa = pts[min_idx];
        const auto& pb = pts[next_min_idx];
        const auto& p = P.row(i);
        if ((p - pa).squaredNorm() < sq_tol) {
            point_vertex_indices[pid] = vertices[min_idx];
            bary_centers(pid, min_idx) = 1.0;
        } else if ((p - pb).squaredNorm() < sq_tol) {
            point_vertex_indices[pid] = vertices[next_min_idx];
            bary_centers(pid, next_min_idx) = 1.0;
        } else {
            const auto eid = mesh.edge(halfedges[min_idx]);
            edge_points_map[eid].emplace_back(pid);
            const auto t = parameters[min_idx][i];
            const auto new_pt = (1.0 - t) * pa + t * pb;
            all_query_points[pid] = Point_2(new_pt.x(), new_pt.y());
            bary_centers(pid, min_idx) = 1.0 - t;
            bary_centers(pid, next_min_idx) = t;
        }
    }
    face_points.resize(idx);
}

void locate_points_on_face1(
    const Mesh_2& mesh,
    std::vector<Point_2>& all_query_points,
    const FI fid,
    std::vector<std::size_t>& face_points,
    std::unordered_map<EI, std::vector<std::size_t>>& edge_points_map,
    std::vector<VI>& point_vertex_indices,
    std::vector<double>& bary_center_arr,
    const double sq_tol)
{
    auto B = VMat::Map(bary_center_arr.data(), point_vertex_indices.size(), 3);
    const auto h1 = mesh.halfedge(fid);
    const auto h2 = mesh.next(h1);
    const auto h3 = mesh.next(h2);
    const std::array<HI, 3> halfedges = { h1, h2, h3 };
    const std::array<VI, 3> vertices = { mesh.source(h1), mesh.source(h2), mesh.source(h3) };
    const auto pts = vertices | view_ts([&mesh](VI vi) {
        const auto& p = mesh.point(vi);
        return Eigen::RowVector2d(p.x(), p.y());
    }) | std::ranges::to<std::vector<Eigen::RowVector2d, Eigen::aligned_allocator<Eigen::RowVector2d>>>();

    Eigen::MatrixXd P(face_points.size(), 2);
    for (std::size_t i = 0; i < face_points.size(); ++i) {
        const auto& p = all_query_points[face_points[i]];
        P.row(i) << p.x(), p.y();
    }

    std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> parameters { Eigen::VectorXd(P.rows()), Eigen::VectorXd(P.rows()), Eigen::VectorXd(P.rows()) };
    std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> distances { Eigen::VectorXd(P.rows()), Eigen::VectorXd(P.rows()), Eigen::VectorXd(P.rows()) };

    for (std::size_t i = 0; i < 3; i++) {
        const auto j = (i + 1) % 3;
        igl::project_to_line_segment(P, pts[i], pts[j], parameters[i], distances[i]);
    }

    std::size_t idx = 0;
    for (std::size_t i = 0; i < face_points.size(); i++) {
        const auto pid = face_points[i];
        std::size_t min_idx = 0;
        double min_dist = distances[0][i];
        for (std::size_t j = 1; j < 3; j++) {
            if (distances[j][i] < min_dist) {
                min_idx = j;
                min_dist = distances[j][i];
            }
        }
        if (min_dist > sq_tol) {
            if (idx != i) {
                face_points[idx] = face_points[i];
            }
            idx += 1;
            continue;
        }

        const auto next_min_idx = (min_idx + 1) % 3;
        const auto& pa = pts[min_idx];
        const auto& pb = pts[next_min_idx];
        const auto& p = P.row(i);
        if ((p - pa).squaredNorm() < sq_tol) {
            point_vertex_indices[pid] = vertices[min_idx];
        } else if ((p - pb).squaredNorm() < sq_tol) {
            point_vertex_indices[pid] = vertices[next_min_idx];
        } else {
            const auto eid = mesh.edge(halfedges[min_idx]);
            edge_points_map[eid].emplace_back(pid);
            const auto t = parameters[min_idx][i];
            const auto new_pt = (1.0 - t) * pa + t * pb;
            all_query_points[pid] = Point_2(new_pt.x(), new_pt.y());
            B(pid, min_idx) = 1.0 - t;
            B(pid, next_min_idx) = t;
        }
    }
    face_points.resize(idx);
}

auto add_outline(
    std::vector<Point_2>& points,
    std::vector<std::vector<std::size_t>>& faces,
    const std::vector<VI>& boundary_vertices,
    const std::vector<std::size_t>& face_map,
    const std::size_t grid_dimension,
    const std::vector<Point_2>& relief_points,
    const MatXu& relief_index_mat,
    const std::vector<std::size_t>& relief_boundary_point_indices,
    const std::vector<std::size_t>& relief_inner_point_indices,
    std::vector<std::size_t>& relief_vertex_indices,
    std::vector<std::size_t>& vertex_faces,
    std::vector<double>& vertex_bary_coords,
    const FMat& F,
    const double tol
) {

    Mesh_2 mesh;
    PMP::polygon_soup_to_polygon_mesh(points, faces, mesh);
    const auto points_3d = points | view_ts([] (const Point_2& p) { return GV3(p.x(), p.y(), 0.0); }) | std::ranges::to<std::vector>();
    auto [gc_mesh, gc_geometry] = gs::makeManifoldSurfaceMeshAndGeometry(faces, points_3d);
    const auto gc_boundary_vertices = boundary_vertices | view_ts([&] (const VI vid) {return gc_mesh->vertex(vid.id());}) | std::ranges::to<std::vector>();
    gs::VertexData<bool> is_boundary_vertex(*gc_mesh, false);
    for (const auto& vid : boundary_vertices) {
        is_boundary_vertex[vid] = true;
    }

    // std::vector<std::vector<gs::Halfedge>> initial_paths;
    std::vector<gs::SurfacePoint> outline;
    std::vector<std::size_t> separators{0};
    for (std::size_t i = 0; i < boundary_vertices.size(); i++) {
        const auto j = (i + 1) % boundary_vertices.size();
        auto initial_path = gs::shortestEdgePathAvoidingMarkedVertices(*gc_geometry, gc_boundary_vertices[i], gc_boundary_vertices[j], is_boundary_vertex);
        gs::FlipEdgeNetwork flip_network(*gc_mesh, *gc_geometry, {initial_path}, is_boundary_vertex);
        flip_network.straightenAroundMarkedVertices = false;
        flip_network.iterativeShorten();
        const auto paths = flip_network.getPathPolyline();
        const auto& path = paths[0];
        outline.append_range(path | std::views::take(path.size() - 1));
        separators.emplace_back(outline.size());
    }

    std::vector<std::size_t> point_faces(outline.size());
    for (std::size_t i = 0; i < outline.size(); i++) {
        const auto srf_pt = outline[i].inSomeFace();
        point_faces[i] = face_map[srf_pt.face.getIndex()];
    }

    std::vector<std::size_t> boundary_halfedge_faces(outline.size());
    for (std::size_t i = 0; i < outline.size(); i++) {
        const auto j = (i + 1) % outline.size();
        const auto& pa = outline[i];
        const auto& pb = outline[j];
        const auto shared_face = gs::sharedFace(pa, pb);
        if (shared_face.getIndex() == geometrycentral::INVALID_IND) {
            fmt::println(stderr, "No shared face found between points {} and {}", i, j);
        } else {
            boundary_halfedge_faces[i] = face_map[shared_face.getIndex()];
        }
    }

    std::vector<Point_2> new_points;
    std::vector<VI> point_vertices;
    std::unordered_map<EI, std::vector<std::size_t>> edge_split_points;
    std::unordered_map<FI, std::vector<std::size_t>> face_split_points;
    for (const auto& pt : outline) {
        record_boundary_point_2d(pt, gc_geometry->vertexPositions, mesh, new_points, point_vertices, edge_split_points, face_split_points);
    }
    const auto n_old_points = mesh.num_vertices();
    split_edges(mesh, edge_split_points, new_points, point_vertices, tol);
    std::vector<HI> outline_halfedges(outline.size());
    for (std::size_t i = 0; i < outline.size(); i++) {
        const auto va = point_vertices[i];
        const auto vb = point_vertices[(i + 1) % outline.size()];
        outline_halfedges[i] = split_face_and_return_edge(mesh, va, vb, n_old_points);
    }
    // write_mesh2("mesh2.obj", mesh);
    std::vector<double> boundary_edge_length(outline.size());
    for (std::size_t i = 0; i < outline.size(); i++) {
        const auto& pa = new_points[i];
        const auto& pb = new_points[(i + 1) % outline.size()];
        const auto v = pa - pb;
        boundary_edge_length[i] = std::sqrt(v * v);
    }
    std::vector<Point_2> mesh_points;
    mesh_points.append_range(mesh.points());

    std::vector<std::vector<std::size_t>> edge_vertex_indices(outline.size());
    const auto hash = [](const std::pair<std::size_t, std::size_t>& p) noexcept {
        return boost::hash_value(p);
    };
    std::unordered_map<std::pair<std::size_t, std::size_t>, std::vector<std::size_t>, decltype(hash)> grid_edge_middle_vertices_map;
    std::vector<std::size_t> middle_vertices;
    for (std::size_t i = 0; i < boundary_vertices.size(); i++) {
        std::vector<double> grid_edge_lengths;
        grid_edge_lengths.reserve(separators[i + 1] - separators[i] + 1);
        grid_edge_lengths.emplace_back(0.0);
        for (std::size_t j = separators[i]; j < separators[i + 1]; j++) {
            grid_edge_lengths.emplace_back(grid_edge_lengths.back() + boundary_edge_length[j]);
        }
        const auto stride = grid_edge_lengths.back() / (grid_dimension - 1);
        double curr_len = 0.0;
        std::size_t i1 = 0;
        const auto start = i * (grid_dimension - 1);
        const auto end = (i + 1) * (grid_dimension - 1);
        for (std::size_t i2 = start; i2 < end; i2++) {
            const auto relief_pid = relief_boundary_point_indices[i2];
            middle_vertices.clear();
            while(grid_edge_lengths[i1] + tol < curr_len) {
                const auto vid = point_vertices[separators[i] + i1];
                if (i2 != start && relief_vertex_indices[relief_boundary_point_indices[i2 - 1]] != vid.id()) {
                    middle_vertices.push_back(vid.id());
                }
                i1 += 1;
            }
            if (std::abs(grid_edge_lengths[i1] - curr_len) < tol) {
                relief_vertex_indices[relief_pid] = point_vertices[separators[i] + i1];
            } else {
                const auto vid = mesh_points.size();
                relief_vertex_indices[relief_pid] = vid;
                const auto& pt = relief_points[relief_pid];
                new_points.emplace_back(pt);
                mesh_points.emplace_back(pt);
                point_vertices.push_back(VI(vid));
                const auto edge_index = separators[i] + i1 - 1;
                edge_vertex_indices[edge_index].emplace_back(vid);
                point_faces.emplace_back(boundary_halfedge_faces[edge_index]);
            }
            if (!middle_vertices.empty()) {
                grid_edge_middle_vertices_map.emplace(
                    std::make_pair(relief_vertex_indices[relief_boundary_point_indices[i2 - 1]], relief_vertex_indices[relief_pid]), middle_vertices);
            }
            curr_len += stride;
        }

        middle_vertices.clear();
        const auto last_vid = relief_vertex_indices[relief_boundary_point_indices[end - 1]];
        while(i1 + 1 < grid_edge_lengths.size()) {
            const auto vid = point_vertices[separators[i] + i1];
            if (vid.id() != last_vid) {
                middle_vertices.push_back(vid.id());
            }
            i1++;
        }
        if (!middle_vertices.empty()) {
            const std::size_t end_vid = i == 3 ? point_vertices[0].id() : point_vertices[separators[i + 1]].id();
            grid_edge_middle_vertices_map.emplace(
                std::make_pair(last_vid, end_vid), middle_vertices);
        }
    }

    vertex_faces.resize(mesh_points.size());
    vertex_bary_coords.resize(mesh_points.size() * 3);
    std::unordered_map<std::size_t, std::vector<std::size_t>> face_vertices_map;
    for (std::size_t pid = 0; pid < point_faces.size(); pid++) {
        const auto fid = point_faces[pid];
        const std::size_t vid = point_vertices[pid];
        face_vertices_map[fid].push_back(vid);
        vertex_faces[vid] = fid;
    }

    for (const auto& [fid, face_vertex_indices] : face_vertices_map) {
        const auto& pa = mesh_points[F(fid, 0)];
        const auto& pb = mesh_points[F(fid, 1)];
        const auto& pc = mesh_points[F(fid, 2)];
        compute_bary_coordinates(pa, pb, pc, mesh_points, face_vertex_indices, vertex_bary_coords);
    }

    const auto reversed_outline_halfedges = std::ranges::reverse_view{ outline_halfedges }
        | view_ts([&](const HI hid) { return mesh.opposite(hid); })
        | std::ranges::to<std::vector>();
    // write_halfedges("reversed_outline.obj", mesh, reversed_outline_halfedges);
    const auto outer_faces = surround_faces(mesh, reversed_outline_halfedges);
    // auto [extracted_points, _1, _2, extracted_faces] = extract_faces(mesh, outer_faces, reversed_outline_halfedges);
    // write_uv1("extracted.obj", extracted_points, extracted_faces);
    std::unordered_map<HI, std::size_t> outline_halfedge_index_map(outline_halfedges.size());
    for (std::size_t i = 0; i < outline_halfedges.size(); i++) {
        outline_halfedge_index_map.emplace(mesh.opposite(outline_halfedges[i]), i);
    }

    const auto add_halfedge = [&](std::vector<std::size_t>& vertices, const HI hid) {
        vertices.emplace_back(mesh.source(hid).id());
        const auto it = outline_halfedge_index_map.find(hid);
        if (it != outline_halfedge_index_map.end()) {
            vertices.append_range(std::ranges::reverse_view{edge_vertex_indices[it->second]});
        }
    };
    std::vector<std::vector<std::size_t>> result_face_veritces;
    for (const auto fid : outer_faces) {
        std::vector<std::size_t> vertices;
        for (const auto hid : mesh.halfedges_around_face(mesh.halfedge(fid))) {
            add_halfedge(vertices, hid);
        }
        if (vertices.size() == 3) {
            result_face_veritces.emplace_back(std::move(vertices));
        } else {
            triangulate_face(mesh_points, vertices, result_face_veritces);
        }
    }
    const auto relief_face_start_index = result_face_veritces.size();

    for (const auto pid : relief_inner_point_indices) {
        const auto vid = mesh_points.size();
        mesh_points.emplace_back(relief_points[pid]);
        relief_vertex_indices[pid] = vid;
    }

    const auto n_inner_points = grid_dimension - 1;
    for (std::size_t j = 0; j < n_inner_points; ++j) {

        for (std::size_t i = 0; i < n_inner_points; i++) {
            std::vector<std::size_t> vertices;
            const auto add_vertex = [&](const std::size_t v1, const std::size_t v2) {
                const auto it = grid_edge_middle_vertices_map.find(std::make_pair(v1, v2));
                if (it != grid_edge_middle_vertices_map.end()) {
                    vertices.append_range(it->second);
                }
            };
            const auto va = relief_vertex_indices[relief_index_mat(i, j)];
            const auto vb = relief_vertex_indices[relief_index_mat(i + 1, j)];
            const auto vc = relief_vertex_indices[relief_index_mat(i + 1, j + 1)];
            const auto vd = relief_vertex_indices[relief_index_mat(i, j + 1)];

            vertices.emplace_back(va);
            if (j == 0) {
                add_vertex(va, vb);
            }
            vertices.emplace_back(vb);
            if (i + 1 == n_inner_points) {
                add_vertex(vb, vc);
            }
            vertices.emplace_back(vc);
            if (j + 1== n_inner_points) {
                add_vertex(vc, vd);
            }
            vertices.emplace_back(vd);
            if (i == 0) {
                add_vertex(vd, va);
            }
            if (vertices.size() == 4) {
                result_face_veritces.emplace_back(std::vector{vd, va, vb});
                result_face_veritces.emplace_back(std::vector{vd, vb, vc});
            } else {
                triangulate_face(mesh_points, vertices, result_face_veritces);
            }
        }
    }

    write_uv1("outer.obj", mesh_points, result_face_veritces);
    points.swap(mesh_points);
    faces.swap(result_face_veritces);
    return relief_face_start_index;
}

auto generate_grid(const std::vector<Point_2>& corner_points, const std::size_t grid_dimension) {
    auto v1 = corner_points[1] - corner_points[0];
    const auto len = std::sqrt(v1 * v1);
    v1 /= len;
    Eigen::Matrix2d rot;
    rot << v1[0], -v1[1], v1[1], v1[0];
    VMat2 V(grid_dimension * grid_dimension, 2);
    MatXu I(grid_dimension, grid_dimension);
    const auto X = Eigen::VectorXd::LinSpaced(grid_dimension, 0.0, len).eval();
    V.col(0).reshaped<Eigen::RowMajor>(grid_dimension, grid_dimension).rowwise() = X.transpose();
    V.col(1).reshaped<Eigen::RowMajor>(grid_dimension, grid_dimension).colwise() = X;
    V.transpose() = (rot * V.transpose()).eval();
    V.rowwise() += Eigen::RowVector2d{corner_points[0][0], corner_points[0][1]};

    FMat F;
    write_uv("grid.obj", V, F);

    std::vector<std::size_t> inner_point_indices;
    std::size_t count = 0;
    for (std::size_t j = 0; j < X.size(); ++j) {
        for (std::size_t i = 0; i < X.size(); ++i) {
            I(i, j) = count;
            if (i != 0 && j != 0 && i != X.size() - 1 && j != X.size() - 1) {
                inner_point_indices.push_back(count);
            }
            count += 1;
        }

    }

    const auto n_points = grid_dimension - 1;
    std::vector<std::size_t> boundary_point_indices(n_points * 4);
    IVec::Map(boundary_point_indices.data(), n_points) = IVec::LinSpaced(n_points, 0, n_points - 1);
    IVec::Map(boundary_point_indices.data() + n_points, n_points) = IVec::LinSpaced(n_points, n_points, n_points + (n_points - 1) * grid_dimension);
    IVec::Map(boundary_point_indices.data() + 2 * n_points, n_points).reverse() = IVec::LinSpaced(n_points, n_points * grid_dimension + 1, grid_dimension * grid_dimension - 1);
    IVec::Map(boundary_point_indices.data() + 3 * n_points, n_points).reverse() = IVec::LinSpaced(n_points, grid_dimension, n_points * grid_dimension);
    std::vector<Point_2> points;
    points.reserve(V.rows());
    for (Eigen::Index i = 0; i < V.rows(); ++i) {
        points.emplace_back(V(i, 0), V(i, 1));
    }

    return std::make_tuple(std::move(points), std::move(I), std::move(boundary_point_indices), std::move(inner_point_indices));
}

auto add_points_into_mesh(Mesh_2& mesh, std::vector<Point_2>& corner_points, const std::size_t grid_dimension, const double tol) {
    auto[relief_points, relief_index_mat, relief_boundary_point_indices, relief_inner_point_indices] = generate_grid(corner_points, grid_dimension);
    std::vector<Triangle_2> triangles;
    triangles.reserve(mesh.number_of_faces());
    for (const auto fid : mesh.faces()) {
        const auto h1 = mesh.halfedge(fid);
        const auto h2 = mesh.next(h1);
        const auto h3 = mesh.next(h2);
        triangles.emplace_back(
            mesh.point(mesh.target(h1)),
            mesh.point(mesh.target(h2)),
            mesh.point(mesh.target(h3)));
    }
    Tree tree(triangles.begin(), triangles.end());
    tree.accelerate_distance_queries();
    std::unordered_map<std::size_t, std::vector<std::size_t>> face_points_map(mesh.num_faces());
    IVec point_faces(corner_points.size());
    for (std::size_t i = 0; i < corner_points.size(); ++i) {
        auto res = tree.closest_point_and_primitive(corner_points[i]);
        const auto fid = res.second - triangles.begin();
        point_faces[i] = fid;
        face_points_map[fid].push_back(i);
    }

    FMat F(mesh.number_of_faces(), 3);
    for (const auto fid : mesh.faces()) {
        const auto h1 = mesh.halfedge(fid);
        const auto h2 = mesh.next(h1);
        const auto h3 = mesh.next(h2);

        F.row(fid.id()) << mesh.source(h1).id(), mesh.source(h2).id(), mesh.source(h3).id();
    }

    const std::size_t n_old_vertices = mesh.number_of_vertices();

    std::vector<VI> point_vertex_indices(corner_points.size());
    std::unordered_map<EI, std::vector<std::size_t>> edge_points_map;
    std::vector<double> point_bary_coords(corner_points.size() * 3, 0.0);
    for (auto& [fid, face_point_indices] : face_points_map) {
        locate_points_on_face1(mesh, corner_points, FI(fid), face_point_indices, edge_points_map, point_vertex_indices, point_bary_coords, tol * tol);
    }
    split_edges(mesh, edge_points_map, corner_points, point_vertex_indices, tol); // notices: assume no points on edges
    for (const auto& [eid, edge_point_indices] : edge_points_map) {
        const auto fid = point_faces[edge_point_indices[0]];
        const auto& pa = mesh.point(VI(F(fid, 0)));
        const auto& pb = mesh.point(VI(F(fid, 1)));
        const auto& pc = mesh.point(VI(F(fid, 2)));
        compute_bary_coordinates(pa, pb, pc, corner_points, edge_point_indices, point_bary_coords);
        for (const auto pid : edge_point_indices) {
            point_faces[pid] = fid;
        }
    }

    std::vector<Point_2> mesh_points;
    mesh_points.append_range(mesh.points());
    std::vector<std::vector<std::size_t>> new_faces;
    std::vector<std::size_t> face_evolutions;
    for (const auto fid : mesh.faces()) {
        const auto it = face_points_map.find(fid);
        if (it == face_points_map.end()) {
            triangualte_on_face(mesh, FI(fid), fid, mesh_points, corner_points, {}, point_vertex_indices, new_faces, F, point_bary_coords);
        } else {
            triangualte_on_face(mesh, FI(fid), fid, mesh_points, corner_points, it->second, point_vertex_indices, new_faces, F, point_bary_coords);
        }
        face_evolutions.resize(new_faces.size(), fid.id());
    }

    std::vector<std::size_t> vertex_faces(mesh_points.size());
    std::vector<double> vertex_bary_coordinates(mesh_points.size() * 3);
    for (std::size_t i = 0; i < corner_points.size(); i++) {
        const auto vid = point_vertex_indices[i];
        vertex_faces[vid] = point_faces[i];
        vertex_bary_coordinates[3 * vid] = point_bary_coords[3 * i];
        vertex_bary_coordinates[3 * vid + 1] = point_bary_coords[3 * i + 1];
        vertex_bary_coordinates[3 * vid + 2] = point_bary_coords[3 * i + 2];
    }

    write_uv1("uv_clip.obj", mesh_points, new_faces);

    std::vector<std::size_t> relief_vertices(relief_points.size());
    const auto relief_face_start_index = add_outline(
        mesh_points,
        new_faces,
        point_vertex_indices,
        face_evolutions,
        grid_dimension,
        relief_points,
        relief_index_mat,
        relief_boundary_point_indices,
        relief_inner_point_indices,
        relief_vertices,
        vertex_faces,
        vertex_bary_coordinates,
        F,
        tol
    );

    face_points_map.clear();
    vertex_faces.resize(mesh_points.size());
    vertex_bary_coordinates.resize(mesh_points.size() * 3);
    for (const auto pid : relief_inner_point_indices) {
        const auto vid = relief_vertices[pid];
        auto res = tree.closest_point_and_primitive(mesh_points[vid]);
        const auto fid = res.second - triangles.begin();
        vertex_faces[vid] = fid;
        face_points_map[fid].push_back(vid);
    }

    for (const auto& [fid, face_vertices] : face_points_map) {
        const auto& pa = mesh_points[F(fid, 0)];
        const auto& pb = mesh_points[F(fid, 1)];
        const auto& pc = mesh_points[F(fid, 2)];
        compute_bary_coordinates(pa, pb, pc, mesh_points, face_vertices, vertex_bary_coordinates);
    }

    return std::make_tuple(
        std::move(mesh_points),
        std::move(new_faces),
        std::move(F),
        std::move(vertex_faces),
        std::move(vertex_bary_coordinates),
        std::move(relief_index_mat),
        std::move(relief_vertices),
        relief_face_start_index
    );
}

auto clip_region(
    const VMat2& V,
    const std::vector<std::vector<std::size_t>>& faces,
    const std::size_t center_id,
    const std::vector<std::size_t>& edge_point_indices,
    const std::size_t grid_dimension,
    const double alpha
) {
    std::array<double, 2> angles{{M_PI - alpha * 2.0, alpha * 2.0}};
    const auto center = V.row(center_id).eval();
    const auto E = (V(edge_point_indices, Eigen::placeholders::all)- center.replicate<4, 1>()).eval();
    const auto L = E.rowwise().norm().eval();
    Eigen::Index min_index;
    L.minCoeff(&min_index);

    std::vector<Point_2> points;
    points.reserve(V.rows());
    for (Eigen::Index i = 0; i < V.rows(); ++i) {
        points.emplace_back(V(i, 0), V(i, 1));
    }
    Mesh_2 mesh;
    PMP::polygon_soup_to_polygon_mesh(points, faces, mesh);
    std::vector<Point_2> corner_points;
    const auto base_dir = (V.row(edge_point_indices[min_index]) - center).transpose().eval();
    double angle = 0;
    for (Eigen::Index i = 0; i < 4; i++) {
        Eigen::Rotation2D<double> rot(angle);
        const auto corner_pt = (center + (rot * base_dir).transpose()).eval();
        corner_points.emplace_back(corner_pt[0], corner_pt[1]);
        angle += angles[(i + min_index) % 2];
    }
    std::rotate(corner_points.begin(), corner_points.begin() + ((4 - min_index) % 4), corner_points.end());
    write_grid("grid.obj", corner_points);
    const auto edge_length = std::sqrt((corner_points[1] - corner_points[0]).squared_length());

    return add_points_into_mesh(mesh, corner_points, grid_dimension, edge_length * 0.01 / (grid_dimension - 1));
}


auto embed_planar_grid_boundary1(
    const std::string& input_mesh_path,
    const Eigen::Vector2d& grid_min_corner,
    const Eigen::Vector2d& grid_max_corner,
    const std::size_t grid_dimension,
    const Eigen::MatrixXd& height_mat
){
    // back: 7008, tail: 6714,
    const std::size_t center_vid_index = 2347;
    auto [cgal_mesh, initial_boundary_indices] = trace_bounding_box_outline(input_mesh_path, grid_min_corner, grid_max_corner, center_vid_index);
    const auto vertex_positions = cgal_mesh.points() | view_ts([](const auto& p) { return GV3 { p.x(), p.y(), p.z() }; }) | std::ranges::to<std::vector>();
    const auto mesh_faces = cgal_mesh.faces() | view_ts([&](const auto fid) { return cgal_mesh.vertices_around_face(cgal_mesh.halfedge(fid)) | view_ts([](const auto vid) { return (std::size_t)vid.idx(); }) | std::ranges::to<std::vector>(); }) | std::ranges::to<std::vector>();
    auto [gc_mesh, gc_geometry] = gs::makeManifoldSurfaceMeshAndGeometry(mesh_faces, vertex_positions);
    const auto boundary_vertices_gc = initial_boundary_indices | std::views::drop(4) | view_ts([&gc_mesh](const auto vid) { return gc_mesh->vertex(vid.idx()); }) | std::ranges::to<std::vector>();

    gs::VertexData<bool> is_boundary_vertex(*gc_mesh, false);
    for (const auto& vid : boundary_vertices_gc) {
        is_boundary_vertex[vid] = true;
    }

    std::vector<gs::SurfacePoint> large_outline;
    for (std::size_t i = 0; i < boundary_vertices_gc.size(); i++) {
        const auto j = (i + 1) % boundary_vertices_gc.size();
        auto initial_path = gs::shortestEdgePathAvoidingMarkedVertices(*gc_geometry, boundary_vertices_gc[i], boundary_vertices_gc[j], is_boundary_vertex);
        gs::FlipEdgeNetwork flip_network(*gc_mesh, *gc_geometry, {initial_path}, is_boundary_vertex);
        flip_network.straightenAroundMarkedVertices = false;
        flip_network.iterativeShorten();
        const auto paths = flip_network.getPathPolyline();
        const auto& path = paths[0];
        large_outline.append_range(path | std::views::take(path.size() - 1));
    }
    std::vector<Point_3> new_vertex_positions;
    std::vector<VI> new_boundary_indices;
    std::unordered_map<EI, std::vector<std::size_t>> edge_split_points;
    std::unordered_map<FI, std::vector<std::size_t>> face_split_points;
    for (const auto& pt : large_outline) {
        record_boundary_point(pt, gc_geometry->vertexPositions, cgal_mesh, new_vertex_positions, new_boundary_indices, edge_split_points, face_split_points);
    }

    const auto original_vertex_count = cgal_mesh.num_vertices();
    insert_point_into_mesh(cgal_mesh, new_vertex_positions, new_boundary_indices, edge_split_points, face_split_points);
    std::vector<HI> boundary_halfedges;
    for (std::size_t i = 0; i < new_boundary_indices.size(); i++) {
        const auto hid = split_face_and_return_edge(cgal_mesh, new_boundary_indices[i], new_boundary_indices[(i + 1) % new_boundary_indices.size()], original_vertex_count);
        boundary_halfedges.emplace_back(hid);
    }
    PMP::triangulate_faces(cgal_mesh);

    const auto patch_faces = surround_faces(cgal_mesh, boundary_halfedges);
    auto [local_patch_points, vertex_point_map, patch_vertices, local_patch_faces] = extract_faces(cgal_mesh, patch_faces, boundary_halfedges);
    CGAL::IO::write_polygon_soup("mesh2.obj", local_patch_points, local_patch_faces);
    auto eigen_data = mesh_to_eigen_mat(local_patch_points, local_patch_faces);
    FlattenSurface fs(std::move(eigen_data.first), std::move(eigen_data.second), boundary_halfedges.size());
    fs.slim_solve(20);
    write_uv("uv.obj", fs.uv, fs.F);
    const auto local_center_vid = vertex_point_map[center_vid_index];

    const auto edge_point_indices = initial_boundary_indices | std::views::take(4) | view_ts([&](const auto vid) { return vertex_point_map[vid]; }) | std::ranges::to<std::vector>();
    const auto dir = (grid_max_corner - grid_min_corner).eval();
    const auto theta = std::atan2(dir.y(), dir.x());
    auto [new_vertices, new_faces, F, vertex_faces, vertex_bary_coordinates, relief_index_mat, relief_vertices, relief_face_start_index] = clip_region(fs.uv, local_patch_faces, vertex_point_map[center_vid_index], edge_point_indices, grid_dimension, theta);
    const auto relief_face_end_index = new_faces.size();

    const auto n_old_points = local_patch_points.size();
    local_patch_points.resize(vertex_faces.size());
    auto P = VMat::Map(reinterpret_cast<double*>(local_patch_points.data()), vertex_faces.size(), 3);
    const auto B = VMat::Map(vertex_bary_coordinates.data(), vertex_faces.size(), 3);
    for (std::size_t i = n_old_points; i < local_patch_points.size(); ++i) {
        const auto fid = vertex_faces[i];
        const auto pa = P.row(F(fid, 0));
        const auto pb = P.row(F(fid, 1));
        const auto pc = P.row(F(fid, 2));
        P.row(i) = pa * B(i, 0) + pb * B(i, 1) + pc * B(i, 2);
    }

    for (std::size_t j = 1; j + 1 < grid_dimension; j++) {
        for (std::size_t i = 1; i + 1 < grid_dimension; i++) {
            const auto vid = relief_vertices[relief_index_mat(i, j)];
            const auto fid = vertex_faces[vid];
            P.row(vid) += height_mat(i, j) * fs.N.row(fid);
        }
    }
    std::vector<bool> face_removed(cgal_mesh.number_of_removed_faces() + cgal_mesh.number_of_faces(), false);
    for (const auto fid : patch_faces) {
        face_removed[fid] = true;
    }
    std::unordered_map<VI, std::size_t> patch_vertex_map;
    for (std::size_t i = 0; i < patch_vertices.size(); i++) {
        patch_vertex_map.emplace(patch_vertices[i], i);
    }

    for (const auto fid : cgal_mesh.faces()) {
        if (face_removed[fid]) {
            continue;
        }

        std::vector<std::size_t> vertices;
        for (const auto vid : cgal_mesh.vertices_around_face(cgal_mesh.halfedge(fid))) {
            const auto it = patch_vertex_map.emplace(vid, local_patch_points.size());
            vertices.push_back(it.first->second);
            if (it.second) {
                local_patch_points.emplace_back(cgal_mesh.point(vid));
            }
        }
        new_faces.emplace_back(vertices);
    }

    write_mesh_with_colors("mesh3.obj", local_patch_points, new_faces, relief_face_start_index, relief_face_end_index);
    // CGAL::IO::write_polygon_soup("mesh3.obj", local_patch_points, new_faces);
    const auto a = 2;
}


auto get_height_mat(Mesh& mesh, const Point_3& min_pt, const std::size_t grid_dimension, const double stride)
{
    Eigen::MatrixXd height_mat = Eigen::MatrixXd::Constant(grid_dimension, grid_dimension, std::numeric_limits<double>::quiet_NaN());
    MatXu index_mat = MatXu::Zero(grid_dimension, grid_dimension);
    for (const auto vid : mesh.vertices()) {
        const auto& pt = mesh.point(vid);
        const auto x = static_cast<std::size_t>(std::round((pt.x() - min_pt.x()) / stride));
        const auto y = static_cast<std::size_t>(std::round((pt.y() - min_pt.y()) / stride));
        if (x < grid_dimension && y < grid_dimension) {
            height_mat(x, y) = pt.z();
            index_mat(x, y) = vid.id();
        } else {
            fmt::println(stderr, "Point out of bounds: ({}, {}, {})", pt.x(), pt.y(), pt.z());
        }
    }
    const auto not_nan = height_mat.cwiseNotEqual(std::numeric_limits<double>::quiet_NaN()).all();
    if (!not_nan) {
        fmt::println(stderr, "Mesh contains NaN values");
    }
    return std::make_pair(std::move(height_mat), std::move(index_mat));
}

auto build_relief_base(const std::vector<Point_3>& points, const std::size_t n_points, const IVec& point_faces, FMat& F, VMat& B, std::vector<VI>& point_vertex_indices) {
    std::vector<Point_3> result = points;
    result.resize(n_points);
    for (Eigen::Index i = 0; i < B.rows(); i++) {
        const auto vid = point_vertex_indices[i].id();
        if (vid < points.size()) {
            continue;
        }
        const auto f = F.row(point_faces[i]);
        const auto& pa = points[f[0]];
        const auto& pb = points[f[1]];
        const auto& pc = points[f[2]];
        const double a = B(i, 0);
        const double b = B(i, 1);
        const double c = B(i, 2);

        result[vid] = Point_3(a * pa.x() + b * pb.x() + c * pc.x(),
                              a * pa.y() + b * pb.y() + c * pc.y(),
                              a * pa.z() + b * pb.z() + c * pc.z());
    }
    return result;
}

void build_relief_top(
    Mesh& mesh,
    const std::vector<Point_3>& points,
    const IVec& point_faces,
    Eigen::MatrixXd& height_mat,
    MatXu& index_mat,
    VMat& N,
    const std::vector<VI>& point_vertex_indices
) {
    // std::vector<Point_3> result(height_mat.size());
    for (Eigen::Index i = 0; i <  point_faces.size(); i++) {
        auto r = i / height_mat.cols();
        auto c = i % height_mat.cols();
        auto h = height_mat(r, c);
        const auto vid = point_vertex_indices[i];
        const auto fid = point_faces[i];
        const auto& base_pt = points[vid];
        const auto& normal = N.row(fid);

        auto idx = index_mat(r, c);
        mesh.point(VI(idx)) = Point_3(base_pt.x() + h * normal(0),
                              base_pt.y() + h * normal(1),
                              base_pt.z() + h * normal(2));
    }
}
auto get_base_boundary(const Mesh& mesh, const std::size_t grid_dimension, const std::vector<VI>& point_vertex_indices) {
    std::vector<VI> vertices;
    vertices.reserve(grid_dimension * 4);
    const auto add_vertex = [&](std::size_t r, std::size_t c) {
        vertices.emplace_back(point_vertex_indices[grid_dimension * c + r]);
    };
    for (std::size_t i = 0; i < grid_dimension; i++) {
        add_vertex(i, 0);
    }
    for (std::size_t i = 0; i < grid_dimension; i++) {
        add_vertex(grid_dimension, i);
    }
    for (std::size_t i = 0; i < grid_dimension; i++) {
        add_vertex(grid_dimension - i, grid_dimension);
    }
    for (std::size_t i = 0; i < grid_dimension; i++) {
        add_vertex(0, grid_dimension - i);
    }
    vertices.emplace_back(vertices.back());

    auto hid = mesh.halfedge(vertices[0]);
    std::vector<HI> halfedges{};
    std::vector<std::size_t> separators{0};
    std::size_t i = 1;
    auto vb = vertices[i];
    while (i < vertices.size()) {
        const auto oppo_hid = mesh.opposite(hid);
        halfedges.emplace_back(oppo_hid);
        const auto vid = mesh.target(oppo_hid);
        hid = mesh.prev(hid);
        if (vid == vb) {
            separators.emplace_back(halfedges.size());
            i += 1;
            if (i < vertices.size()) {
                vb = vertices[i];
            }
        }
    }
    const auto a = 2;
}

auto get_top_boundary(const std::size_t grid_dimensioin, const MatXu& index_mat) {
    std::vector<VI> vertices;
    vertices.resize(grid_dimensioin * 4);
    const auto add_vertex = [&](std::size_t r, std::size_t c) {
        vertices.emplace_back(index_mat(r, c));
    };
    for (std::size_t i = 0; i < grid_dimensioin; i++) {
        add_vertex(i, 0);
    }
    for (std::size_t i = 0; i < grid_dimensioin; i++) {
        add_vertex(grid_dimensioin, i);
    }
    for (std::size_t i = 0; i < grid_dimensioin; i++) {
        add_vertex(grid_dimensioin - i, grid_dimensioin);
    }
    for (std::size_t i = 0; i < grid_dimensioin; i++) {
        add_vertex(0, grid_dimensioin - i);
    }
    return vertices;
}

} // namespace

int main(int argc, char** argv)
{
    CLI::App app { "Relief App" };
    std::string mesh_path;
    std::string relief_path;
    app.add_option("-m,--mesh", mesh_path, "Path to mesh file")->required();
    app.add_option("-r,--relief", relief_path, "Path to relief file")->required();
    CLI11_PARSE(app, argc, argv);

    namespace fs = std::filesystem;
    const fs::path relief_input(relief_path);
    const std::string output_path = (relief_input.parent_path() / (relief_input.stem().string() + "_cleaned" + relief_input.extension().string()))
                                        .string();

    Mesh relief;

    if (!CGAL::IO::read_polygon_mesh(relief_path, relief)) {
        fmt::print(stderr, "Failed to load relief mesh from {}\n", relief_path);
        return 1;
    }

    const auto bounds = compute_bounds(relief);
    if (!bounds) {
        fmt::print(stderr, "Relief mesh '{}' contains no vertices\n", relief_path);
        return 1;
    }

    constexpr double kZTolerance = 1e-6;
    const double min_z = bounds->min.z();
    const std::size_t n_removed_faces = remove_faces_on_min_plane(relief, min_z, kZTolerance);

    fmt::print(
        "Loaded relief mesh '{}'\n- Vertices: {}\n- Faces: {}\n- Min point: "
        "({}, {}, {})\n- Max point: ({}, {}, {})\n- Faces removed near z ~= {}\n",
        relief_path, relief.number_of_vertices(), relief.number_of_faces(),
        bounds->min.x(), bounds->min.y(), bounds->min.z(), bounds->max.x(),
        bounds->max.y(), bounds->max.z(), min_z);
    fmt::print("Faces removed near base plane: {}\n", n_removed_faces);

    // if (!CGAL::IO::write_polygon_mesh(output_path, relief)) {
    //     fmt::print(stderr, "Failed to write cleaned relief mesh to {}\n",
    //                output_path);
    //     return 1;
    // }
    // fmt::print("Wrote cleaned relief mesh to {}\n", output_path);

    const std::size_t grid_dimension = static_cast<std::size_t>(std::sqrt(relief.number_of_vertices()));
    fmt::print("Estimated grid dimension: {}\n", grid_dimension);

    const double scale = 0.0095;
    auto [height_mat, index_mat] = get_height_mat(relief, bounds->min, grid_dimension, (bounds->max.x() - bounds->min.x()) / (grid_dimension - 1));
    height_mat = (height_mat.array() - min_z) * scale;
    height_mat = height_mat.array() - height_mat.minCoeff();

    Eigen::Vector2d min_pt(bounds->min.x(), bounds->min.y());
    Eigen::Vector2d max_pt(bounds->max.x(), bounds->max.y());
    scale_box(min_pt, max_pt, scale);
    embed_planar_grid_boundary1(mesh_path, min_pt, max_pt, grid_dimension, height_mat);
    return 0;
}

 int main1(int argc, char** argv) {
     VMat V;
     FMat F;
     igl::read_triangle_mesh("cylinder_cut.obj", V, F);
     Eigen::MatrixXi F1 = F.cast<int>();
     Eigen::VectorXi bnd;

     igl::boundary_loop(F1, bnd);
     Eigen::VectorXi map(V.rows());
     const auto INVALID = V.rows();
     map.setConstant(INVALID);

     map(bnd) = Eigen::VectorXi::LinSpaced(bnd.size(), 0, bnd.size() - 1);
     Eigen::Index count = bnd.size();
     for (Eigen::Index i = 0; i < V.rows(); ++i) {
         if (map(i) == INVALID) {
             map[i] = count++;
         }
     }
     VMat V1 = V;
     V(map, Eigen::placeholders::all) = V1;
     F.col(0) = map(F.col(0)).cast<std::size_t>();
     F.col(1) = map(F.col(1)).cast<std::size_t>();
     F.col(2) = map(F.col(2)).cast<std::size_t>();

     // igl::write_triangle_mesh("mesh2.obj", V, F.cast<int>());

     bnd = Eigen::VectorXi::LinSpaced(bnd.size(), 0, bnd.size() - 1);
     Eigen::MatrixXd bnd_uv, uv;
     igl::map_vertices_to_circle(V, bnd, bnd_uv);
     igl::harmonic(V,F,bnd,bnd_uv,1,uv);
     if (igl::flipped_triangles(uv,F).size() != 0) {
       igl::harmonic(F,bnd,bnd_uv,1,uv); // use uniform laplacian
     }

     write_uv("init.obj", uv, F);

     VMat newV = V;
     newV.setZero();
     newV.leftCols(2) = uv;

     // DeformSurface ds(std::move(V), std::move(F), bnd.size());
     // ds.deform(10, newV);
     VMat2 uv1 = uv;
     FlattenSurface fs(std::move(V), std::move(F), uv1, 0);
     fs.slim_solve(10);

     return 0;
 }
