#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_face_base_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/boost/graph/Euler_operations.h>
#include <CGAL/boost/graph/helpers.h>

#include <CLI/CLI.hpp>

#include <Eigen/Core>
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

#include "flatten_surface.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_2 = Kernel::Point_2;
using Point_3 = Kernel::Point_3;
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

using VMat2 = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor, Eigen::Dynamic, 2>;
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
void split_edges(
    Mesh& mesh,
    std::unordered_map<Mesh::Edge_index, std::vector<std::size_t>>& edge_points_map,
    const std::vector<Point_3>& points,
    std::vector<Mesh::Vertex_index>& point_vertex_indices)
{
    for (auto& [eid, edge_point_indices] : edge_points_map) {
        if (edge_point_indices.size() == 1) {
            const auto new_hid = CGAL::Euler::split_edge(mesh.halfedge(eid), mesh);
            const auto new_vid = mesh.target(new_hid);
            const auto pid = edge_point_indices[0];
            mesh.point(new_vid) = points[pid];
            point_vertex_indices[pid] = new_vid;
        } else {
            constexpr double tol = 0.001;
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

auto insert_point_into_mesh(
    Mesh& mesh,
    const std::vector<Point_3>& points,
    std::vector<VI>& point_vertex_indices,
    std::unordered_map<EI, std::vector<std::size_t>> edge_points_map,
    std::unordered_map<FI, std::vector<std::size_t>> face_points_map)
{
    const auto n_old_vertices = mesh.number_of_vertices();
    split_edges(mesh, edge_points_map, points, point_vertex_indices);
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

    gs::SurfacePoint geodesic_seed_vertex(gmesh->vertex(7008));
    // gs::SurfacePoint geodesic_seed_vertex(gmesh->vertex(6714));
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
    //     write_polyline(fmt::format("outline_{}.obj", i), boundary_segments[i], gc_geometry->vertexPositions);
    // }

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
    // CGAL::IO::write_polygon_mesh("mesh1.obj", cgal_mesh);

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

auto extract_faces(const Mesh& mesh, const std::vector<FI>& faces, std::vector<HI>& boundary_halfedges) {
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

    const std::size_t grid_dimension = static_cast<std::size_t>(std::sqrt(relief.number_of_vertices())) - 1;
    fmt::print("Estimated grid dimension: {}\n", grid_dimension);

    VMat2 V(relief.number_of_vertices(), 2);
    std::size_t idx = 0;
    for (const auto& pt : relief.points()) {
        V.row(idx++) << pt.x(), pt.y();
    }
    auto min_pt = V.colwise().minCoeff().transpose().eval();
    auto max_pt = V.colwise().maxCoeff().transpose().eval();
    fmt::println("Min point: ({}, {})", min_pt.x(), min_pt.y());
    fmt::println("Max point: ({}, {})", max_pt.x(), max_pt.y());

    scale_box(min_pt, max_pt, 0.02);
    auto [mesh, boundary_halfedges, segment_offset] = embed_planar_grid_boundary(mesh_path, min_pt, max_pt, 16, grid_dimension);
    const auto faces = surround_faces(mesh, boundary_halfedges);
    auto [patch_points, patch_vertices, patch_faces] = extract_faces(mesh, faces, boundary_halfedges);
    CGAL::IO::write_polygon_soup("mesh2.obj", patch_points, patch_faces);
    return 0;
}
