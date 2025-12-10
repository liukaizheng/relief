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

#include "flatten_surface.h"
#include "igl/PI.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <ranges>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
    for (Eigen::Index i = 0; i < F.size(); ++i) {
        file << "f " << F[i][0] + 1 << " " << F[i][1] + 1 << " " << F[i][2] + 1 << "\n";
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
    std::unordered_map<EI, std::vector<std::size_t>>& edge_points_map,
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

void triangualte_on_face(
    const Mesh_2& mesh,
    const FI fid,
    std::vector<Point_2>& points,
    const std::vector<Point_2>& face_points,
    const std::vector<std::size_t>& face_point_indices,
    std::vector<VI>& point_vertex_indices,
    std::vector<std::vector<std::size_t>>& new_faces,
    const FMat& F,
    VMat& baray_center)
{
    std::vector<std::size_t> vertices;
    for (const auto vid : mesh.vertices_around_face(mesh.halfedge(fid))) {
        vertices.emplace_back(vid.id());
    }
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
        const auto find1 = std::ranges::find(face, 359) != face.end();
        const auto find2 = std::ranges::find(face, 172) != face.end();
        new_faces.emplace_back(std::move(face));
    }

    if (!face_point_indices.empty()) {
        const auto fv = F.row(fid.id());
        Eigen::RowVector2d p1(points[fv[0]].x(), points[fv[0]].y());
        Eigen::RowVector2d p2(points[fv[1]].x(), points[fv[1]].y());
        Eigen::RowVector2d p3(points[fv[2]].x(), points[fv[2]].y());

        auto I = IVec::LinSpaced(face_point_indices.size(), n_old_vertices, vertices.size() - 1).eval();
        Eigen::MatrixXd B;
        igl::barycentric_coordinates(V(I, Eigen::placeholders::all), p1.replicate(I.size(), 1), p2.replicate(I.size(), 1), p3.replicate(I.size(), 1), B);
        baray_center(face_point_indices, Eigen::placeholders::all) = B;
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

    // {
    //     std::unordered_map<VI, std::size_t> vertex_index_map;
    //     for (std::size_t i = 0; i < vertices.size(); i++) {
    //         vertex_index_map[vertices[i]] = i;
    //     }
    //     const auto points_2d = vertices | std::views::transform([&](const auto vid) {
    //         return mesh.point(vid);
    //     }) | std::ranges::to<std::vector>();

    //     std::vector<std::vector<std::size_t>> face_vertices;
    //     for (auto face : cdt.finite_face_handles()) {
    //         const auto v1 = vertex_index_map[face->vertex(0)->info()];
    //         const auto v2 = vertex_index_map[face->vertex(1)->info()];
    //         const auto v3 = vertex_index_map[face->vertex(2)->info()];
    //         face_vertices.push_back(std::vector{v1, v2, v3});
    //     }
    //     write_uv1("test.obj", points_2d, face_vertices);
    // }

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

auto trace_bounding_box_outline(const std::string& mesh_path, Eigen::Vector2d bounding_min, Eigen::Vector2d bounding_max, const std::size_t samples_per_edge)
{
    auto [gmesh, geometry] = gs::readManifoldSurfaceMesh(mesh_path);
    fmt::println(
        "Loaded mesh '{}'\n- Vertices: {}\n- Faces: {}\n",
        mesh_path, gmesh->nVertices(), gmesh->nFaces());

    std::vector<Point_3> points;
    std::vector<std::vector<std::size_t>> faces;
    for (const auto& vid : gmesh->vertices()) {
        const auto& pt = geometry->vertexPositions[vid];
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
    const auto sample_stride = 1.0 / samples_per_edge;

    // gs::SurfacePoint geodesic_seed_vertex(gmesh->vertex(7008));
    gs::SurfacePoint geodesic_seed_vertex(gmesh->vertex(6714));
    const auto bounding_box_center = ((bounding_min + bounding_max) * 0.5).eval();
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> bounding_box_corners { bounding_min, { bounding_max.x(), bounding_min.y() }, bounding_max, { bounding_min.x(), bounding_max.y() } };
    std::vector<Point_3> outline_points;
    std::vector<VI> outline_vertex_indices;
    outline_points.reserve(samples_per_edge * 4);
    outline_vertex_indices.reserve(samples_per_edge * 4);
    std::unordered_map<EI, std::vector<std::size_t>> outline_edge_points_map;
    std::unordered_map<FI, std::vector<std::size_t>> outline_face_points_map;

    for (std::size_t edge_idx = 0; edge_idx < bounding_box_corners.size(); edge_idx++) {
        const auto& edge_start = bounding_box_corners[edge_idx];
        const auto& edge_end = bounding_box_corners[(edge_idx + 1) % bounding_box_corners.size()];
        for (std::size_t sample_idx = 0; sample_idx < samples_per_edge; sample_idx++) {
            const auto t = sample_idx * sample_stride;
            const auto sampled_point_2d = (1.0 - t) * edge_start + t * edge_end;
            const auto offset_from_center = sampled_point_2d - bounding_box_center;
            const auto trace_result = gs::traceGeodesic(*geometry, geodesic_seed_vertex, geometrycentral::Vector2 { offset_from_center[0], offset_from_center[1] }, { .includePath = true });
            // write_polyline(fmt::format("path_{}_{}.obj", edge_idx, sample_idx), trace_result.pathPoints, geometry->vertexPositions);
            record_boundary_point(trace_result.endPoint, geometry->vertexPositions, mesh, outline_points, outline_vertex_indices, outline_edge_points_map, outline_face_points_map);
        }
    }

    insert_point_into_mesh(mesh, outline_points, outline_vertex_indices, outline_edge_points_map, outline_face_points_map);
    // CGAL::IO::write_polygon_mesh("mesh1.obj", mesh);
    return std::make_pair(std::move(outline_vertex_indices), std::move(mesh));
}
auto split_face_and_return_edge(
    Mesh& target_mesh,
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

auto embed_planar_grid_boundary(const std::string& input_mesh_path, const Eigen::Vector2d& grid_min_corner, const Eigen::Vector2d& grid_max_corner, std::size_t boundary_samples_per_edge, std::size_t grid_cells_per_axis)
{
    auto [initial_boundary_indices, cgal_mesh] = trace_bounding_box_outline(input_mesh_path, grid_min_corner, grid_max_corner, boundary_samples_per_edge);
    const auto vertex_positions = cgal_mesh.points() | view_ts([](const auto& p) { return GV3 { p.x(), p.y(), p.z() }; }) | std::ranges::to<std::vector>();
    const auto mesh_faces = cgal_mesh.faces() | view_ts([&](const auto fid) { return cgal_mesh.vertices_around_face(cgal_mesh.halfedge(fid)) | view_ts([](const auto vid) { return (std::size_t)vid.idx(); }) | std::ranges::to<std::vector>(); }) | std::ranges::to<std::vector>();
    auto [gc_mesh, gc_geometry] = gs::makeManifoldSurfaceMeshAndGeometry(mesh_faces, vertex_positions);
    const auto boundary_vertices_gc = initial_boundary_indices | view_ts([&gc_mesh](const auto vid) { return gc_mesh->vertex(vid.idx()); }) | std::ranges::to<std::vector>();

    gs::VertexData<bool> is_boundary_vertex(*gc_mesh, false);
    for (const auto& vid : boundary_vertices_gc) {
        is_boundary_vertex[vid] = true;
    }

    std::vector<std::vector<gs::Halfedge>> initial_paths;
    for (std::size_t i = 0; i < boundary_vertices_gc.size(); i++) {
        const auto j = (i + 1) % boundary_vertices_gc.size();
        initial_paths.push_back(gs::shortestEdgePathAvoidingMarkedVertices(*gc_geometry, boundary_vertices_gc[i], boundary_vertices_gc[j], is_boundary_vertex));
    }

    gs::FlipEdgeNetwork flip_network(*gc_mesh, *gc_geometry, initial_paths, is_boundary_vertex);
    flip_network.straightenAroundMarkedVertices = false;
    flip_network.iterativeShorten();
    const auto& optimized_paths = flip_network.getPathPolyline();
    std::vector<std::size_t> corner_path_indices { 0, boundary_samples_per_edge, boundary_samples_per_edge * 2, boundary_samples_per_edge * 3, boundary_samples_per_edge * 4 };
    std::vector<std::vector<gs::SurfacePoint>> boundary_segments;
    for (std::size_t i = 0; i + 1 < corner_path_indices.size(); i++) {
        const auto start = corner_path_indices[i];
        const auto end = corner_path_indices[i + 1];
        std::vector<gs::SurfacePoint> segment;
        for (std::size_t j = start; j < end; j++) {
            const auto& path = optimized_paths[j];
            segment.append_range(path | std::views::take(path.size() - 1));
        }
        boundary_segments.push_back(segment);
    }

    // for (std::size_t i = 0; i < boundary_segments.size(); i++) {
    //     auto temp = boundary_segments[i];
    //     const auto j = (i + 1) % boundary_segments.size();
    //     temp.emplace_back(boundary_segments[j].front());
    //     write_polyline(fmt::format("outline_{}.obj", i), temp, gc_geometry->vertexPositions);
    // }
    // CGAL::IO::write_polygon_mesh("mesh2.obj", cgal_mesh);

    std::vector<Point_3> new_vertex_positions;
    std::vector<VI> new_boundary_indices;
    std::unordered_map<EI, std::vector<std::size_t>> edge_split_points;
    std::unordered_map<FI, std::vector<std::size_t>> face_split_points;
    std::vector<std::size_t> segment_offsets { 0 };
    for (const auto& segment : boundary_segments) {
        for (const auto& pt : segment) {
            record_boundary_point(pt, gc_geometry->vertexPositions, cgal_mesh, new_vertex_positions, new_boundary_indices, edge_split_points, face_split_points);
        }
        segment_offsets.emplace_back(segment_offsets.back() + segment.size());
    }

    const auto original_vertex_count = cgal_mesh.num_vertices();
    insert_point_into_mesh(cgal_mesh, new_vertex_positions, new_boundary_indices, edge_split_points, face_split_points);
    std::vector<HI> boundary_halfedges;
    for (std::size_t i = 0; i < new_boundary_indices.size(); i++) {
        const auto hid = split_face_and_return_edge(cgal_mesh, new_boundary_indices[i], new_boundary_indices[(i + 1) % new_boundary_indices.size()], original_vertex_count);
        boundary_halfedges.emplace_back(hid);
    }
    PMP::triangulate_faces(cgal_mesh);

    // const double grid_stride = 1.0 / grid_cells_per_axis;
    // const auto grid_sample_t_arr = std::ranges::iota_view{0ull, grid_cells_per_axis} | view_ts([grid_stride] (const auto i)  { return grid_stride * i; }) | std::ranges::to<std::vector>();

    return make_tuple(std::move(cgal_mesh), std::move(boundary_halfedges), std::move(segment_offsets));
}
auto surround_faces(const Mesh& mesh, const std::vector<Mesh ::Halfedge_index>& halfedges)
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
            if (!visited[oppo_fid]) {
                faces.emplace_back(oppo_fid);
                visited[oppo_fid] = true;
            }
        }
    }
    return faces;
}

void scale_box(Eigen::Vector2d& min_pt, Eigen::Vector2d& max_pt, const double s)
{
    const auto center = ((min_pt + max_pt) * 0.5).eval();
    const auto vec = ((max_pt - center) * s).eval();
    max_pt = center + vec;
    min_pt = center - vec;
}

auto extract_faces(const Mesh& mesh, const std::vector<FI>& faces, std::vector<HI>& boundary_halfedges)
{
    constexpr auto INVALID = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> point_map(mesh.number_of_vertices() + mesh.number_of_removed_vertices(), INVALID);
    std::vector<Point_3> points;
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
    return std::make_tuple(std::move(points), std::move(point_vertices), std::move(face_vertices));
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

auto add_grid_into_uv_domain(Mesh_2& mesh, const std::size_t grid_dimension, const double edge_length)
{
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
    auto points = generate_grid_points(grid_dimension, edge_length);
    std::unordered_map<std::size_t, std::vector<std::size_t>> face_points_map(mesh.num_faces());
    face_points_map.reserve(mesh.num_faces());
    IVec point_faces(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        auto res = tree.closest_point_and_primitive(points[i]);
        const auto fid = res.second - triangles.begin();
        point_faces[i] = fid;
        face_points_map[fid].push_back(i);
    }

    VMat bary_centers = VMat::Zero(points.size(), 3);
    std::vector<VI> point_vertex_indices(points.size());
    std::unordered_map<EI, std::vector<std::size_t>> edge_points_map;
    const double tol = edge_length / (grid_dimension - 1) * 0.001;
    for (auto& [fid, face_point_indices] : face_points_map) {
        locate_points_on_face(mesh, points, FI(fid), face_point_indices, edge_points_map, point_vertex_indices, bary_centers, tol * tol);
    }
    const std::size_t n_old_vertices = mesh.number_of_vertices();
    FMat F(mesh.number_of_faces(), 3);
    for (const auto fid : mesh.faces()) {
        const auto h1 = mesh.halfedge(fid);
        const auto h2 = mesh.next(h1);
        const auto h3 = mesh.next(h2);

        F.row(fid.id()) << mesh.source(h1).id(), mesh.source(h2).id(), mesh.source(h3).id();
    }

    split_edges(mesh, edge_points_map, points, point_vertex_indices, tol);
    std::vector<std::vector<std::size_t>> new_faces;
    std::vector<Point_2> mesh_points;
    mesh_points.append_range(mesh.points());
    for (const auto fid : mesh.faces()) {
        const auto it = face_points_map.find(fid);
        if (it == face_points_map.end()) {
            triangualte_on_face(mesh, FI(fid), mesh_points, points, {}, point_vertex_indices, new_faces, F, bary_centers);
        } else {
            triangualte_on_face(mesh, FI(fid), mesh_points, points, it->second, point_vertex_indices, new_faces, F, bary_centers);
        }
    }

    const auto error = (bary_centers.rowwise().sum().array() - 1.00).abs().maxCoeff();
    if (error > 1e-3) {
        fmt::println(stderr, "Barycenter error: {}", error);
    }

    return std::make_tuple(std::move(mesh_points), std::move(new_faces), std::move(point_faces), std::move(point_vertex_indices), std::move(F), std::move(bary_centers));

    // Mesh_2 new_mesh;
    // PMP::polygon_soup_to_polygon_mesh(mesh_points, new_faces, new_mesh);
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

int main1(int argc, char** argv)
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

    const double scale = 0.02;
    auto [height_mat, index_mat] = get_height_mat(relief, bounds->min, grid_dimension, (bounds->max.x() - bounds->min.x()) / (grid_dimension - 1));
    height_mat = (height_mat.array() - min_z) * scale;

    Eigen::Vector2d min_pt(bounds->min.x(), bounds->min.y());
    Eigen::Vector2d max_pt(bounds->max.x(), bounds->max.y());
    scale_box(min_pt, max_pt, scale);
    auto [mesh, boundary_halfedges, segment_offset] = embed_planar_grid_boundary(mesh_path, min_pt, max_pt, 8, grid_dimension);
    const auto faces = surround_faces(mesh, boundary_halfedges);
    auto [patch_points, patch_vertices, patch_faces] = extract_faces(mesh, faces, boundary_halfedges);

    auto fv_mat = mesh_to_eigen_mat(patch_points, patch_faces);
    FlattenSurface flatten_surface(std::move(fv_mat.first), std::move(fv_mat.second), segment_offset);
    flatten_surface.slim_solve(10);

    VMat N;
    igl::per_face_normals(flatten_surface.V, flatten_surface.F, N);

    std::vector<Point_2> uv_points;
    uv_points.reserve(flatten_surface.uv.rows());
    for (std::size_t i = 0; i < flatten_surface.uv.rows(); ++i) {
        uv_points.emplace_back(flatten_surface.uv(i, 0), flatten_surface.uv(i, 1));
    }
    Mesh_2 uv_mesh;
    PMP::polygon_soup_to_polygon_mesh(uv_points, patch_faces, uv_mesh);

    auto [
        relief_base_uvs, relief_base_faces, point_faces, grid_point_vertex_indices, path_face_vertices, bary_centers
    ] = add_grid_into_uv_domain(uv_mesh, grid_dimension, flatten_surface.mean_edge_length);
    auto relief_base_points = build_relief_base(patch_points, relief_base_uvs.size(), point_faces, path_face_vertices, bary_centers, grid_point_vertex_indices);
    // CGAL::IO::write_polygon_soup("mesh2.obj", relief_base_points, relief_base_faces);

    Mesh base_relief_mesh;
    PMP::polygon_soup_to_polygon_mesh(relief_base_points, relief_base_faces, base_relief_mesh);

    build_relief_top(relief, relief_base_points, point_faces, height_mat, index_mat, N, grid_point_vertex_indices);
    CGAL::IO::write_polygon_mesh("mesh3.obj", relief);

    get_base_boundary(base_relief_mesh, grid_dimension, grid_point_vertex_indices);
    const auto base_outline = get_top_boundary(grid_dimension, index_mat);

    return 0;
}
 int main(int argc, char** argv) {
     Eigen::Matrix2d uv;
     uv << 1, 0, 0, 1;
     using Mat23 = Eigen::Matrix<double, 2, 3>;
     Mat23 X;
     X << 1, 0, 0,
          1, 1, 1;

     Eigen::Vector3d axis = Eigen::Vector3d::Random();
     axis.normalize();

     Eigen::Transform<double, 3, Eigen::Affine> t(Eigen::AngleAxis(igl::PI / 6.0, axis));
     X.row(0).transpose() = t * X.row(0).transpose();
     X.row(1).transpose() = t * X.row(1).transpose();

     Eigen::JacobiSVD<Mat23, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(X);

     auto U = svd.matrixU();
     auto V = svd.matrixV();
     auto S = svd.singularValues();

     Mat23 D = (U * S.asDiagonal() * V.transpose().topRows(2) -  X).eval();

     std::cout << "Transformed Matrix X:\n" << X << std::endl;
     std::cout << "matrix U:\n" << svd.matrixU() << std::endl;
     std::cout << "matrix V:\n" << svd.matrixV() << std::endl;
     std::cout << "singular values:\n" << svd.singularValues() << std::endl;
     std::cout << "matrix D:\n" << D << std::endl;
     return 0;
 }
