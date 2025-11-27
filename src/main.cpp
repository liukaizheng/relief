#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/Euler_operations.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/IO/polygon_mesh_io.h>

#include <CLI/CLI.hpp>

#include <fmt/core.h>
#include <fmt/os.h>
#include <fmt/format.h>
#include <Eigen/Core>

#include <format>
#include <geometrycentral/surface/meshio.h>
#include <geometrycentral/surface/trace_geodesic.h>
#include <geometrycentral/surface/surface_mesh.h>
#include <geometrycentral/surface/surface_point.h>
#include <geometrycentral/utilities/vector2.h>
#include <geometrycentral/utilities/vector3.h>
#include <geometrycentral/surface/mesh_graph_algorithms.h>
#include "geometrycentral/surface/halfedge_element_types.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include <geometrycentral/surface/flip_geodesics.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include <ranges>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;
using VI = Mesh::Vertex_index;
using HI = Mesh::Halfedge_index;
using EI = Mesh::Edge_index;
using FI = Mesh::Face_index;

using VMat2 = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor, Eigen::Dynamic, 2>;
using GV3 = geometrycentral::Vector3;
using GMesh = geometrycentral::surface::SurfaceMesh;


namespace gs = geometrycentral::surface;
namespace PMP =  CGAL::Polygon_mesh_processing;

const auto& view_ts = std::views::transform;

namespace {
struct Bounds {
    Kernel::Point_3 min;
    Kernel::Point_3 max;
};

std::optional<Bounds> compute_bounds(Mesh& mesh) {
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

    return Bounds{Kernel::Point_3(min_x, min_y, min_z),
                  Kernel::Point_3(max_x, max_y, max_z)};
}

std::size_t remove_faces_on_min_plane(Mesh& mesh, double min_z,
                                      double tolerance) {
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

void write_polyline(const std::string& name, const std::vector<gs::SurfacePoint>& points, gs::VertexData<geometrycentral::Vector3>& vertexData) {
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

auto split_edge(const EI& eid, Mesh& mesh, Point_3 pt) {
    const auto new_hid = CGAL::Euler::split_edge(mesh.halfedge(eid), mesh);
    const auto new_vid = mesh.target(new_hid);
    mesh.point(new_vid) = std::move(pt);
    return new_vid;
}

auto split_face(const FI& fid, Mesh& mesh, Point_3 pt) {
    const auto new_hid = CGAL::Euler::add_center_vertex(mesh.halfedge(fid), mesh);
    const auto new_vid = mesh.target(new_hid);
    mesh.point(new_vid) = std::move(pt);
    return new_vid;
}

auto add_corner_points(const std::string& name) {
    auto [gmesh, geometry] = gs::readManifoldSurfaceMesh(name);
    fmt::println(
        "Loaded mesh '{}'\n- Vertices: {}\n- Faces: {}\n",
        name, gmesh->nVertices(), gmesh->nFaces()
    );

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

    gs::SurfacePoint center(gmesh->vertex(7008));
    const double length = 0.3;

    std::vector<VI> corners;
    for (const auto [x, y] : {std::pair{-1., -1.}, {1., -1.}, {1., 1.}, {-1., 1.}}) {
        const auto trace_ret = gs::traceGeodesic(*geometry, center, geometrycentral::Vector2{x, y} * length, {.includePath = true });
        // write_polyline(fmt::format("polyline_{}_{}.obj", (int(x) + 1) >> 1, (int(y) + 1) >> 1), trace_ret.pathPoints, geometry->vertexPositions);
        const auto& end_point = trace_ret.endPoint;
        switch(end_point.type) {
            case gs::SurfacePointType::Vertex:
                corners.emplace_back(VI(end_point.vertex.getIndex()));
                break;
            case gs::SurfacePointType::Edge: {
                auto [va, vb] = end_point.edge.adjacentVertices();
                const auto pt = end_point.interpolate(geometry->vertexPositions);
                corners.emplace_back(split_edge(mesh.edge(mesh.halfedge(VI(va.getIndex()), VI(vb.getIndex()))), mesh, {pt.x, pt.y, pt.z}));
                break;
            }
            case gs::SurfacePointType::Face:
                const auto pt = end_point.interpolate(geometry->vertexPositions);
                corners.emplace_back(split_face(FI(end_point.face.getIndex()), mesh, {pt.x, pt.y, pt.z}));
                break;
        }
    }
    // CGAL::IO::write_polygon_mesh("corner.obj", mesh);
    return std::make_pair(std::move(mesh), std::move(corners));
}

auto add_outline(const std::string& mesh_path) {
    auto [mesh, corners] = add_corner_points(mesh_path);
    const auto points = mesh.points() | view_ts([](const auto& p) { return GV3{p.x(), p.y(), p.z()}; }) | std::ranges::to<std::vector>();
    const auto polygons = mesh.faces() | view_ts([&](const auto fid) { return mesh.vertices_around_face(mesh.halfedge(fid)) | view_ts([](const auto vid) { return (std::size_t) vid.idx();}) | std::ranges::to<std::vector>();}) | std::ranges::to<std::vector>();
    auto [gmesh, geom] = gs::makeManifoldSurfaceMeshAndGeometry(polygons, points);
    const auto gs_corners = corners | view_ts([&gmesh](const auto vid) { return gmesh->vertex(vid.idx()); }) | std::ranges::to<std::vector>();

    std::vector<std::vector<gs::Halfedge>> path_halfedges;
    for (std::size_t i = 0; i < gs_corners.size(); i++) {
        const auto j = (i + 1) % gs_corners.size();
        path_halfedges.push_back(gs::shortestEdgePath(*geom, gs_corners[i], gs_corners[j]));
    }

    gs::VertexData<bool> vertex_is_marked(*gmesh, false);
    for (const auto& vid : gs_corners) {
        vertex_is_marked[vid] = true;
    }

    gs::FlipEdgeNetwork network(*gmesh, *geom, path_halfedges, vertex_is_marked);
    network.straightenAroundMarkedVertices = false;
    network.iterativeShorten();
    const auto paths = network.getPathPolyline();
    for (std::size_t i = 0; i < paths.size(); i++) {
        write_polyline(std::format("path{}.obj", i), paths[i], geom->vertexPositions);
    }
}

}  // namespace

int main1(int argc, char** argv) {
    CLI::App app{"Relief App"};
    std::string mesh_path;
    std::string relief_path;
    app.add_option("-m,--mesh", mesh_path, "Path to mesh file")->required();
    app.add_option("-r,--relief", relief_path, "Path to relief file")->required();
    CLI11_PARSE(app, argc, argv);

    namespace fs = std::filesystem;
    const fs::path relief_input(relief_path);
    const std::string output_path =
        (relief_input.parent_path() /
         (relief_input.stem().string() + "_cleaned" +
          relief_input.extension().string()))
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
    const std::size_t n_removed_faces =
        remove_faces_on_min_plane(relief, min_z, kZTolerance);

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

    const std::size_t grid_dimension =
        static_cast<std::size_t>(std::sqrt(relief.number_of_vertices()));
    fmt::print("Estimated grid dimension: {}\n", grid_dimension);

    VMat2 V(relief.number_of_vertices(), 2);
    std::size_t idx = 0;
    for (const auto& pt : relief.points()) {
        V.row(idx++) << pt.x(), pt.y();
    }
    const auto min_pt = V.colwise().minCoeff().eval();
    const auto max_pt = V.colwise().maxCoeff().eval();
    fmt::println("Min point: ({}, {})", min_pt.x(), min_pt.y());
    fmt::println("Max point: ({}, {})", max_pt.x(), max_pt.y());

    add_outline(mesh_path);
    return 0;
}
int main(int argc, char** argv) {
    CLI::App app{"Relief App"};
    std::string mesh_path;
    app.add_option("-m,--mesh", mesh_path, "Path to mesh file")->required();
    CLI11_PARSE(app, argc, argv);
    auto [mesh, geom] = gs::readManifoldSurfaceMesh(mesh_path);
    std::vector<std::size_t> path_indices1{0, 1, 5, 7};
    std::vector<std::size_t> path_indices2{7, 5, 4};

    auto to_halfedges_path = [&](const std::vector<std::size_t>& path) {
        std::vector<gs::Halfedge> halfedges;
        for (std::size_t i = 0; i < path.size() - 1; ++i) {
            const auto va = mesh->vertex(path[i]);
            const auto vb = mesh->vertex(path[i + 1]);
            const auto edge = mesh->connectingEdge(va, vb);
            const auto he = edge.halfedge();
            if (he.tailVertex() == va) {
                halfedges.push_back(he);
            } else {
                halfedges.emplace_back(he.twin());
            }
        }
        return halfedges;
    };

    const auto path1 = to_halfedges_path(path_indices1);
    const auto path2 = to_halfedges_path(path_indices2);
    std::vector<std::vector<gs::Halfedge>> path_halfedges {path1, path2};
    gs::VertexData<bool> vertex_is_marked(*mesh, false);
    vertex_is_marked[mesh->vertex(0)] = true;
    vertex_is_marked[mesh->vertex(4)] = true;
    vertex_is_marked[mesh->vertex(7)] = true;
    gs::FlipEdgeNetwork network(*mesh, *geom, path_halfedges, vertex_is_marked);
    network.straightenAroundMarkedVertices = false;
    network.iterativeShorten();
    const auto paths = network.getPathPolyline();
    write_polyline("path1.obj", paths[0], geom->vertexPositions);
    write_polyline("path2.obj", paths[1], geom->vertexPositions);

    for (const auto vid : mesh->vertices()) {
        const auto& pt = geom->vertexPositions[vid];
        fmt::println("Vertex position: ({}, {}, {})", pt.x, pt.y, pt.z);
    }
}
