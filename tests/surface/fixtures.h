#pragma once

// Compile the single implementation in each white-box test executable to access
// its private preparation stages without splitting them into internal headers.
#include "fit_on_surface.cpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>

namespace relief::test {
using namespace relief::surface;

struct Geometry {
    std::vector<std::array<double, 3>> points;
    std::vector<std::array<std::size_t, 3>> triangles;
};

inline Geometry cube_geometry(int subdivisions = 3, bool curved = false)
{
    Geometry g {
        { { -1, -1, -1 }, { 1, -1, -1 }, { 1, 1, -1 }, { -1, 1, -1 },
            { -1, -1, 1 }, { 1, -1, 1 }, { 1, 1, 1 }, { -1, 1, 1 } },
        { { 0, 2, 1 }, { 0, 3, 2 }, { 4, 5, 6 }, { 4, 6, 7 },
            { 0, 1, 5 }, { 0, 5, 4 }, { 1, 2, 6 }, { 1, 6, 5 },
            { 2, 3, 7 }, { 2, 7, 6 }, { 3, 0, 4 }, { 3, 4, 7 } }
    };
    for (int iteration = 0; iteration < subdivisions; ++iteration) {
        std::map<std::pair<std::size_t, std::size_t>, std::size_t> midpoints;
        auto midpoint = [&](std::size_t a, std::size_t b) {
            auto [it, inserted] = midpoints.emplace(std::minmax(a, b), g.points.size());
            if (inserted) {
                std::array<double, 3> p;
                for (int j = 0; j < 3; ++j) {
                    p[j] = 0.5 * (g.points[a][j] + g.points[b][j]);
                }
                g.points.push_back(p);
            }
            return it->second;
        };
        std::vector<std::array<std::size_t, 3>> triangles;
        for (const auto& t : g.triangles) {
            const auto a = midpoint(t[0], t[1]), b = midpoint(t[1], t[2]), c = midpoint(t[2], t[0]);
            triangles.push_back({ t[0], a, c });
            triangles.push_back({ a, t[1], b });
            triangles.push_back({ c, b, t[2] });
            triangles.push_back({ a, b, c });
        }
        g.triangles = std::move(triangles);
    }
    if (curved) {
        for (auto& p : g.points) {
            Eigen::Vector3d::Map(p.data()).normalize();
        }
    }
    return g;
}

template <class Mesh = fit_on_surface::Mesh>
Mesh make_mesh(const Geometry& g)
{
    auto mesh = Mesh::new_in(g.triangles);
    for (const auto vertex : mesh.vertices()) {
        vertex.prop().pt = g.points[vertex.id.idx];
    }
    for (const auto face : mesh.faces()) {
        face.prop().parent = face.id;
    }
    return mesh;
}

template <class Mesh = fit_on_surface::Mesh>
Mesh cube(bool curved = false) { return make_mesh<Mesh>(cube_geometry(3, curved)); }

template <class Mesh = fit_on_surface::Mesh>
Mesh tetrahedron()
{
    return make_mesh<Mesh>({ { { 1, 1, 1 }, { -1, -1, 1 }, { -1, 1, -1 }, { 1, -1, -1 } },
        { { 0, 2, 1 }, { 0, 1, 3 }, { 0, 3, 2 }, { 1, 2, 3 } } });
}

template <class Mesh = fit_on_surface::Mesh>
Mesh open_plane()
{
    return make_mesh<Mesh>({ { { -2, -2, 0 }, { 2, -2, 0 }, { 2, 2, 0 }, { -2, 2, 0 } },
        { { 0, 1, 2 }, { 0, 2, 3 } } });
}

inline std::array<double, 3> centroid(const auto& mesh, gpf::FaceId face)
{
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    for (const auto he : mesh.face(face).halfedges()) {
        p += Eigen::Vector3d::Map(he.from().prop().pt.data()) / 3;
    }
    return { p.x(), p.y(), p.z() };
}

inline gpf::FaceId top_face(const auto& mesh)
{
    double distance = std::numeric_limits<double>::infinity();
    gpf::FaceId result;
    for (const auto face : mesh.faces()) {
        const auto p = centroid(mesh, face.id);
        const double d = (Eigen::Vector3d::Map(p.data()) - Eigen::Vector3d { 0.1, 0.05, 1 }).squaredNorm();
        if (d < distance) {
            distance = d;
            result = face.id;
        }
    }
    return result;
}

inline PlacementInput placement(const auto& mesh, std::array<double, 3> direction = { 0.12, 0.03, 0 })
{
    const auto face = top_face(mesh);
    return make_placement_input(mesh, face, centroid(mesh, face), direction);
}

inline std::string fingerprint(const auto& mesh)
{
    std::ostringstream out;
    out << std::setprecision(17) << mesh.n_vertices() << ' ' << mesh.n_faces() << ' ' << mesh.n_edges() << '\n';
    out << mesh.n_vertices_capacity() << ' ' << mesh.n_faces_capacity() << ' ' << mesh.n_halfedges_capacity() << '\n';
    for (std::size_t i = 0; i < mesh.n_vertices_capacity(); ++i) {
        const auto& data = mesh.vertex_data(gpf::VertexId { i });
        out << data.halfedge.idx;
        for (const auto v : data.property.pt) {
            out << ' ' << v;
        }
        out << '\n';
    }
    for (std::size_t i = 0; i < mesh.n_halfedges_capacity(); ++i) {
        const auto& d = mesh.halfedge_data(gpf::HalfedgeId { i });
        out << d.vertex.idx << ' ' << d.next.idx << ' ' << d.prev.idx << ' ' << d.face.idx << '\n';
    }
    for (std::size_t i = 0; i < mesh.n_faces_capacity(); ++i) {
        const auto& d = mesh.face_data(gpf::FaceId { i });
        out << d.halfedge.idx << ' ' << d.property.parent.idx;
        if constexpr (requires { d.property.polygon_id; }) {
            out << ' ' << d.property.polygon_id;
        }
        out << '\n';
    }
    return out.str();
}

inline VMat2 coordinates(const PreparedSurfacePatch& patch)
{
    VMat2 result(patch.uv_mesh.n_vertices(), 2);
    for (const auto vertex : patch.uv_mesh.vertices()) {
        result.row(vertex.id.idx) = Eigen::Vector2d::Map(vertex.prop().pt.data());
    }
    return result;
}

inline void expect_mesh_connectivity(const auto& mesh)
{
    for (const auto face : mesh.faces()) {
        EXPECT_TRUE(face.prop().parent.valid());
        for (const auto he : face.halfedges()) {
            EXPECT_EQ(he.from().id, he.twin().to().id);
            EXPECT_EQ(he.to().id, he.twin().from().id);
        }
    }
}

class SurfaceFixture : public ::testing::Test {
protected:
    std::filesystem::path previous_directory;
    void SetUp() override
    {
        previous_directory = std::filesystem::current_path();
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        auto directory = previous_directory / (std::string(info->test_suite_name()) + "_" + info->name());
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        std::filesystem::current_path(directory);
    }
    void TearDown() override
    {
        std::filesystem::current_path(previous_directory);
    }
};
}
