#include "fixtures.h"

#include <limits>
#include <set>

namespace relief::test {
class FitOnSurfaceTest : public SurfaceFixture { };
const std::vector<std::array<double, 2>> square { { 0.17, 0.19 }, { 0.83, 0.19 }, { 0.83, 0.81 }, { 0.17, 0.81 } };
const std::vector<std::vector<std::vector<std::size_t>>> square_polygon { { { 0, 1, 2, 3 } } };

TEST_F(FitOnSurfaceTest, PreferredCubeRetainsSubdivisionsAndLabels)
{
    auto mesh = cube();
    for (auto face : mesh.faces()) {
        face.prop().polygon_id = 99;
    }
    const auto original = mesh;
    const auto input = placement(mesh);
    auto prepared_mesh = mesh;
    auto patch = prepare_preferred(prepared_mesh, input);
    ASSERT_TRUE(patch);
    const std::set<gpf::FaceId> patch_faces(patch->surface_faces.begin(), patch->surface_faces.end());
    auto result = fit_on_surface::fit_polygon_on_surface(mesh, square, square_polygon, input.surface_point, input.source_face, { 0.12, 0.03, 0 });
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    EXPECT_GT(mesh.n_faces(), original.n_faces());
    int labeled = 0;
    for (const auto face : mesh.faces()) {
        labeled += face.prop().polygon_id == 0;
        EXPECT_LT(face.prop().parent.idx, original.n_faces_capacity());
        if (face.id.idx < original.n_faces_capacity() && !patch_faces.contains(face.id)) {
            EXPECT_EQ(face.prop().polygon_id, 99);
        }
    }
    EXPECT_GT(labeled, 0);
    expect_mesh_connectivity(mesh);
}

TEST_F(FitOnSurfaceTest, SharedPolygonBoundaryKeepsBothLabels)
{
    auto mesh = cube();
    const auto input = placement(mesh);
    const std::vector<std::array<double, 2>> points { { 0.15, 0.15 }, { 0.5, 0.15 }, { 0.85, 0.15 }, { 0.85, 0.85 }, { 0.5, 0.85 }, { 0.15, 0.85 } };
    const std::vector<std::vector<std::vector<std::size_t>>> polygons { { { 0, 1, 4, 5 } }, { { 1, 2, 3, 4 } } };
    auto result = fit_on_surface::fit_polygon_on_surface(mesh, points, polygons, input.surface_point, input.source_face, { 0.12, 0.03, 0 });
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    std::array<int, 2> counts {};
    int shared = 0;
    for (const auto face : mesh.faces()) {
        if (face.prop().polygon_id < 2) {
            ++counts[face.prop().polygon_id];
        }
    }
    for (const auto edge : mesh.edges()) {
        const auto he = edge.halfedge();
        const auto a = he.face().prop().polygon_id, b = he.twin().face().prop().polygon_id;
        if ((a == 0 && b == 1) || (a == 1 && b == 0)) {
            ++shared;
        }
    }
    EXPECT_GT(counts[0], 0);
    EXPECT_GT(counts[1], 0);
    EXPECT_GT(shared, 0);
    expect_mesh_connectivity(mesh);
}

TEST_F(FitOnSurfaceTest, HoleRemainsUnlabeled)
{
    auto mesh = cube();
    const auto input = placement(mesh);
    const std::vector<std::array<double, 2>> points { { 0.05, 0.05 }, { 0.95, 0.05 }, { 0.95, 0.95 }, { 0.05, 0.95 },
        { 0.3, 0.3 }, { 0.3, 0.7 }, { 0.7, 0.7 }, { 0.7, 0.3 } };
    const std::vector<std::vector<std::vector<std::size_t>>> polygons { { { 0, 1, 2, 3 }, { 4, 5, 6, 7 } } };
    auto result = fit_on_surface::fit_polygon_on_surface(mesh, points, polygons, input.surface_point, input.source_face, { 0.12, 0.03, 0 });
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    double nearest = std::numeric_limits<double>::infinity();
    std::size_t label = 0;
    int labeled = 0;
    for (const auto face : mesh.faces()) {
        labeled += face.prop().polygon_id == 0;
        const auto p = centroid(mesh, face.id);
        const auto distance = (Eigen::Vector3d::Map(p.data()) - Eigen::Vector3d::Map(input.surface_point.data())).norm();
        if (distance < nearest) {
            nearest = distance;
            label = face.prop().polygon_id;
        }
    }
    EXPECT_GT(labeled, 0);
    EXPECT_EQ(label, gpf::kInvalidIndex);
}

TEST_F(FitOnSurfaceTest, ProjectionMatchesPreparedChart)
{
    auto mesh = cube();
    const auto input = placement(mesh);
    auto prepared_mesh = mesh;
    auto patch = prepare_preferred(prepared_mesh, input);
    ASSERT_TRUE(patch);
    // This planar patch has one affine UV-to-surface map across all its faces.
    std::array<Eigen::Vector2d, 3> uv;
    std::array<Eigen::Vector3d, 3> xyz;
    int i = 0;
    for (const auto he : patch->uv_mesh.face(gpf::FaceId { 0 }).halfedges()) {
        uv[i] = Eigen::Vector2d::Map(he.from().prop().pt.data());
        xyz[i] = Eigen::Vector3d::Map(prepared_mesh.vertex_prop(patch->uv_to_surface_vertex[he.from().id.idx]).pt.data());
        ++i;
    }
    Eigen::Matrix2d basis;
    basis.col(0) = uv[1] - uv[0];
    basis.col(1) = uv[2] - uv[0];
    std::vector<Eigen::Vector3d> expected;
    for (const auto& point : square) {
        const Eigen::Vector2d q = patch->frame.origin + point[0] * patch->frame.xaxis + point[1] * patch->frame.yaxis;
        const Eigen::Vector2d bary = basis.inverse() * (q - uv[0]);
        expected.push_back(xyz[0] + bary.x() * (xyz[1] - xyz[0]) + bary.y() * (xyz[2] - xyz[0]));
    }
    auto result = fit_on_surface::fit_polygon_on_surface(mesh, square, square_polygon, input.surface_point, input.source_face, { 0.12, 0.03, 0 });
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    for (const auto& p : expected) {
        double nearest = std::numeric_limits<double>::infinity();
        for (const auto vertex : mesh.vertices()) {
            nearest = std::min(nearest, (Eigen::Vector3d::Map(vertex.prop().pt.data()) - p).norm());
        }
        EXPECT_LT(nearest, 1e-8);
    }
}

TEST_F(FitOnSurfaceTest, PublicPlacementMapsAsymmetricLandmarksAlongLocalAxes)
{
    // An asymmetric pentagon distinguishes rotation, reflection, and swapped axes.
    const std::vector<std::array<double, 2>> points { { 0.13, 0.22 }, { 0.79, 0.22 },
        { 0.86, 0.64 }, { 0.36, 0.87 }, { 0.13, 0.63 } };
    const std::vector<std::vector<std::vector<std::size_t>>> polygons { { { 0, 1, 2, 3, 4 } } };
    for (const bool fallback : { false, true }) {
        for (const std::array<double, 3> direction : { std::array<double, 3> { 0.12, 0, 0 },
                 { 0.12, 0.03, 0 }, { -0.12, -0.03, 0 }, { 0.12, 0, 0.16 } }) {
            SCOPED_TRACE((testing::Message() << "fallback=" << fallback << " direction="
                                             << Eigen::Vector3d::Map(direction.data()).transpose()));
            // Both public paths use closed meshes; the footprint stays on one known plane.
            auto mesh = fallback ? tetrahedron() : cube();
            const auto face = fallback ? gpf::FaceId { 1 } : top_face(mesh);
            const auto point = centroid(mesh, face);
            const Eigen::Vector3d center = Eigen::Vector3d::Map(point.data());
            const Eigen::Vector3d normal = fallback ? Eigen::Vector3d { 1, -1, 1 }.normalized() : Eigen::Vector3d { 0, 0, 1 };
            const Eigen::Vector3d d = Eigen::Vector3d::Map(direction.data());
            const Eigen::Vector3d xaxis = 2 * d.norm() * (d - normal.dot(d) * normal).normalized();
            const Eigen::Vector3d yaxis = normal.cross(xaxis);
            auto candidate = mesh;
            auto preferred = prepare_preferred(candidate, make_placement_input(mesh, face, point, direction));
            if (fallback) {
                ASSERT_FALSE(preferred);
                EXPECT_EQ(preferred.error(), PreferredFailure::Rejected);
                EXPECT_GT(candidate.n_vertices(), mesh.n_vertices());
            } else {
                ASSERT_TRUE(preferred);
            }
            auto result = fit_on_surface::fit_polygon_on_surface(mesh, points, polygons, point, face, direction);
            ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
            for (const auto& uv : points) {
                // Expectations depend only on the original input and known plane, not a prepared frame.
                const Eigen::Vector3d expected = center + (uv[0] - 0.5) * xaxis + (uv[1] - 0.5) * yaxis;
                double nearest = std::numeric_limits<double>::infinity();
                gpf::VertexId landmark;
                for (const auto vertex : mesh.vertices()) {
                    const double distance = (Eigen::Vector3d::Map(vertex.prop().pt.data()) - expected).norm();
                    if (distance < nearest) {
                        nearest = distance;
                        landmark = vertex.id;
                    }
                }
                // Preferred planar charts are isometric apart from projection snapping.
                // Legacy's unchanged solve leaves small distortion: allow 2% of a side,
                // well below the systematic 45-degree error these landmarks detect.
                const double tolerance = fallback ? 0.02 * xaxis.norm() : kProjectionTolerance;
                EXPECT_LT(nearest, tolerance);
                ASSERT_TRUE(landmark.valid());
                EXPECT_TRUE(ranges::any_of(mesh.vertex(landmark).incoming_halfedges(),
                    [](const auto he) { return he.face().prop().polygon_id == 0; }));
            }
            expect_mesh_connectivity(mesh);
        }
    }
}

TEST_F(FitOnSurfaceTest, FailedExpMapUsesLegacyPath)
{
    auto mesh = tetrahedron();
    const auto input = placement(mesh);
    auto candidate = mesh;
    auto rejected = prepare_preferred(candidate, input);
    ASSERT_FALSE(rejected);
    EXPECT_GT(candidate.n_vertices(), mesh.n_vertices());
    auto result = fit_on_surface::fit_polygon_on_surface(mesh, square, square_polygon, input.surface_point, input.source_face, { 0.12, 0.03, 0 });
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    EXPECT_TRUE(ranges::any_of(mesh.faces(), [](const auto face) { return face.prop().polygon_id == 0; }));
    expect_mesh_connectivity(mesh);
}

TEST_F(FitOnSurfaceTest, PolygonValidationDoesNotModifyMesh)
{
    auto mesh = cube();
    const auto input = placement(mesh);
    const auto original = fingerprint(mesh);
    auto check = [&](const auto& polygons, SurfaceFailure error) {
        auto result = fit_on_surface::fit_polygon_on_surface(mesh, square, polygons, input.surface_point, input.source_face, { 0.1, 0, 0 });
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error(), error);
        EXPECT_EQ(fingerprint(mesh), original);
    };
    auto polygons = square_polygon;
    polygons[0][0][0] = 999;
    check(polygons, SurfaceFailure::InvalidPolylinePointIndex);
    polygons[0][0] = { 0, 1 };
    check(polygons, SurfaceFailure::DegeneratePolygon);
}

TEST_F(FitOnSurfaceTest, DownstreamDiagnosticFailureDoesNotRetry)
{
    auto mesh = cube();
    const auto input = placement(mesh);
    auto expected_mesh = mesh;
    auto expected_patch = prepare_preferred(expected_mesh, input);
    ASSERT_TRUE(expected_patch);
    std::filesystem::create_directory("fit_polygon_on_surface_projected_uv_mesh.off");
    auto result = fit_on_surface::fit_polygon_on_surface(mesh, square, square_polygon, input.surface_point, input.source_face, { 0.12, 0.03, 0 });
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), SurfaceFailure::UvMeshOffOpenFailed);
    EXPECT_EQ(fingerprint(mesh), fingerprint(expected_mesh));
}
}
