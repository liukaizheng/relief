#include "fixtures.h"
#include "flatten_surface.h"

#include <fstream>
#include <gpf/exp_map.hpp>

namespace relief::test {
class SurfacePatchTest : public SurfaceFixture { };

InitialPatch planar_patch()
{
    InitialPatch p;
    p.positions.resize(5, 3);
    p.positions << -2, -2, 0, 2, -2, 0, 2, 2, 0, -2, 2, 0, 0, 0, 0;
    p.coordinates = p.positions.leftCols<2>();
    p.triangles.resize(4, 3);
    p.triangles << 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4;
    p.boundary = { 0, 1, 2, 3 };
    p.center = 4;
    for (std::size_t i = 0; i < 5; ++i) {
        p.uv_to_surface_vertex.push_back(gpf::VertexId { i });
    }
    for (std::size_t i = 0; i < 4; ++i) {
        p.surface_faces.push_back(gpf::FaceId { i });
    }
    return p;
}

void expect_planar_frame(const auto& mesh, const PreparedSurfacePatch& patch, const Eigen::Vector3d& center,
    const Eigen::Vector3d& xaxis, const Eigen::Vector3d& yaxis, double tolerance = 1e-7)
{
    // Observe the piecewise UV-to-surface map. Even on a plane, legacy SLIM
    // need not converge to a single affine map in its fixed iteration budget.
    for (const Eigen::Vector2d domain : { Eigen::Vector2d { 0.5, 0.5 }, { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } }) {
        SCOPED_TRACE((testing::Message() << "domain=" << domain.transpose()));
        const Eigen::Vector2d point = patch.frame.origin + domain.x() * patch.frame.xaxis + domain.y() * patch.frame.yaxis;
        std::optional<Eigen::Vector3d> actual;
        for (const auto face : patch.uv_mesh.faces()) {
            std::array<Eigen::Vector2d, 3> uv;
            std::array<Eigen::Vector3d, 3> xyz;
            std::size_t i = 0;
            for (const auto he : face.halfedges()) {
                uv[i] = Eigen::Vector2d::Map(he.from().prop().pt.data());
                xyz[i] = Eigen::Vector3d::Map(mesh.vertex_prop(patch.uv_to_surface_vertex[he.from().id.idx]).pt.data());
                ++i;
            }
            Eigen::Matrix2d basis;
            basis.col(0) = uv[1] - uv[0];
            basis.col(1) = uv[2] - uv[0];
            const Eigen::Vector2d bary = basis.inverse() * (point - uv[0]);
            if (bary.minCoeff() >= -1e-10 && bary.sum() <= 1 + 1e-10) {
                actual = xyz[0] + bary.x() * (xyz[1] - xyz[0]) + bary.y() * (xyz[2] - xyz[0]);
                break;
            }
        }
        ASSERT_TRUE(actual);
        // Expected positions use only the known plane, center, and public axes.
        const Eigen::Vector3d expected = center + (domain.x() - 0.5) * xaxis + (domain.y() - 0.5) * yaxis;
        EXPECT_LT((*actual - expected).norm(), domain.x() == 0.5 ? 1e-10 : tolerance);
    }
}

std::array<std::size_t, 3> reference_source_uv_indices(const InitialPatch& patch, const PlacementInput& input)
{
    std::array<std::size_t, 3> indices { gpf::kInvalidIndex, gpf::kInvalidIndex, gpf::kInvalidIndex };
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const auto found = ranges::find(patch.uv_to_surface_vertex, input.source_vertices[i]);
        if (found != patch.uv_to_surface_vertex.end()) {
            indices[i] = std::distance(patch.uv_to_surface_vertex.begin(), found);
        }
    }
    return indices;
}

void expect_exp_map_mapping(const auto& mesh, const InitializedExpMapPatch& initialized,
    const PlacementInput& input, const gpf::ExpMapResult& raw)
{
    const auto& patch = initialized.patch;
    EXPECT_EQ(initialized.source_uv_indices, reference_source_uv_indices(patch, input));
    for (std::size_t i = 0; i < initialized.source_uv_indices.size(); ++i) {
        const auto local = initialized.source_uv_indices[i];
        ASSERT_LT(local, patch.uv_to_surface_vertex.size());
        EXPECT_EQ(patch.uv_to_surface_vertex[local], input.source_vertices[i]);
    }
    ASSERT_EQ(patch.positions.rows(), raw.vertex_ids.size());
    ASSERT_EQ(patch.coordinates.rows(), raw.uvs.size());
    ASSERT_EQ(patch.uv_to_surface_vertex.size(), raw.vertex_ids.size());
    EXPECT_EQ(patch.surface_faces, raw.face_ids);
    ASSERT_EQ(patch.triangles.rows(), raw.face_ids.size());
    ASSERT_EQ(patch.boundary.size(), raw.boundary_vertex_ids.size());
    for (std::size_t i = 0; i < patch.boundary.size(); ++i) {
        EXPECT_EQ(patch.boundary[i], i);
        EXPECT_EQ(patch.uv_to_surface_vertex[i], raw.boundary_vertex_ids[i]);
    }
    for (std::size_t i = 0; i < raw.vertex_ids.size(); ++i) {
        const auto vertex = raw.vertex_ids[i];
        const auto found = ranges::find(patch.uv_to_surface_vertex, vertex);
        ASSERT_NE(found, patch.uv_to_surface_vertex.end());
        const auto local = std::distance(patch.uv_to_surface_vertex.begin(), found);
        EXPECT_TRUE(patch.positions.row(local).transpose().isApprox(Eigen::Vector3d::Map(mesh.vertex_prop(vertex).pt.data())));
        EXPECT_DOUBLE_EQ(patch.coordinates(local, 0), raw.uvs[i][0]);
        EXPECT_DOUBLE_EQ(patch.coordinates(local, 1), raw.uvs[i][1]);
    }
    for (std::size_t f = 0; f < raw.face_ids.size(); ++f) {
        std::size_t corner = 0;
        for (const auto he : mesh.face(raw.face_ids[f]).halfedges()) {
            const auto local = patch.triangles(f, corner++);
            ASSERT_LT(local, patch.uv_to_surface_vertex.size());
            EXPECT_EQ(patch.uv_to_surface_vertex[local], he.from().id);
        }
    }
    ASSERT_LT(patch.center, patch.uv_to_surface_vertex.size());
    EXPECT_EQ(patch.uv_to_surface_vertex[patch.center], raw.center_vertex);
    EXPECT_GE(patch.center, patch.boundary.size());
}

TEST_F(SurfacePatchTest, ValidPlanarInitializationUsesOneVertexGaugeAndSharedSettings)
{
    auto p = planar_patch();
    EXPECT_EQ(kSlimMinIterations, 5);
    EXPECT_EQ(kSlimMaxIterations, 15);
    EXPECT_EQ(kSlimFixedVertices, 1);
    auto result = optimize_patch(p);
    EXPECT_EQ(result.initial.coordinates, p.coordinates);
    EXPECT_EQ(result.coordinates.row(0), p.coordinates.row(0));
    EXPECT_TRUE(std::filesystem::exists("uv_new.obj")); // Existing solver diagnostic is unchanged.
}

TEST_F(SurfacePatchTest, UniformlyReversedInitializationIsNormalizedOnce)
{
    auto p = planar_patch();
    p.coordinates.col(1) *= -1;
    auto result = optimize_patch(p);
    EXPECT_EQ(result.initial.coordinates, planar_patch().coordinates);
}

TEST_F(SurfacePatchTest, ExistingTwoArgumentSlimEntryPointStillWritesDebugUvs)
{
    auto p = planar_patch();
    FlattenSurface solver(VMat(p.positions), FMat(p.triangles), VMat2(p.coordinates), 1);
    solver.slim_solve(5, 15);
    std::ifstream file("uv_new.obj");
    ASSERT_TRUE(file);
    for (Eigen::Index i = 0; i < solver.uv.rows(); ++i) {
        std::string tag;
        double x, y, z;
        file >> tag >> x >> y >> z;
        EXPECT_EQ(tag, "v");
        EXPECT_TRUE(std::isfinite(x));
        EXPECT_TRUE(std::isfinite(y));
        EXPECT_EQ(z, 0);
    }
}

TEST_F(SurfacePatchTest, CubePreferredUsesActualExpMapCoordinatesAndOptimizedOutput)
{
    auto mesh = cube();
    const auto input = placement(mesh);
    auto raw_mesh = mesh;
    initialize_exp_map_properties(raw_mesh);
    auto raw = gpf::exp_map(std::span<const double, 3>(input.surface_point), raw_mesh, input.lengths[1]);
    ASSERT_TRUE(raw);
    auto candidate = mesh;
    auto initial = initialize_exp_map_patch(candidate, input);
    ASSERT_TRUE(initial);
    expect_exp_map_mapping(candidate, *initial, input, *raw);
    EXPECT_FALSE(std::filesystem::exists("uv_new.obj"));
    auto optimized = optimize_patch(std::move(initial->patch));
    auto frame = exp_map_placement(optimized, input, initial->source_uv_indices);
    ASSERT_TRUE(frame);
    auto result = prepare_surface_patch(mesh, input);
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    EXPECT_EQ(result->frame.origin, frame->origin);
    EXPECT_EQ(result->frame.xaxis, frame->xaxis);
    EXPECT_EQ(result->frame.yaxis, frame->yaxis);
    EXPECT_EQ(coordinates(*result), optimized.coordinates);
    EXPECT_EQ(fingerprint(mesh), fingerprint(raw_mesh));
    expect_mesh_connectivity(mesh);
}

TEST_F(SurfacePatchTest, ExpMapMovesLowestIdInteriorVertexBehindBoundaryBeforeSlim)
{
    auto geometry = cube_geometry();
    const auto reference = make_mesh(geometry);
    const auto face = top_face(reference);
    const auto old_center = reference.face(face).halfedge().from().id.idx;
    ASSERT_NE(old_center, 0);
    std::swap(geometry.points[0], geometry.points[old_center]);
    for (auto& triangle : geometry.triangles) {
        for (auto& vertex : triangle) {
            if (vertex == old_center) {
                vertex = 0;
            } else if (vertex == 0) {
                vertex = old_center;
            }
        }
    }
    auto mesh = make_mesh(geometry);
    const auto input = make_placement_input(mesh, face, mesh.vertex_prop(gpf::VertexId { 0 }).pt, { 0.12, 0.03, 0 });
    auto raw_mesh = mesh;
    initialize_exp_map_properties(raw_mesh);
    auto raw = gpf::exp_map(std::span<const double, 3>(input.surface_point), raw_mesh, input.lengths[1]);
    ASSERT_TRUE(raw);
    EXPECT_EQ(raw->center_vertex, gpf::VertexId { 0 });
    EXPECT_EQ(raw->vertex_ids.front(), raw->center_vertex);
    auto candidate = mesh;
    auto initial = initialize_exp_map_patch(candidate, input);
    ASSERT_TRUE(initial);
    expect_exp_map_mapping(candidate, *initial, input, *raw);
    EXPECT_NE(initial->patch.uv_to_surface_vertex.front(), raw->center_vertex);
    auto optimized = optimize_patch(std::move(initial->patch));
    EXPECT_EQ(optimized.coordinates.row(0), optimized.initial.coordinates.row(0));
    auto result = prepare_surface_patch(mesh, input);
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    EXPECT_EQ(result->uv_to_surface_vertex, optimized.initial.uv_to_surface_vertex);
    EXPECT_EQ(coordinates(*result), optimized.coordinates);
    EXPECT_EQ(fingerprint(mesh), fingerprint(raw_mesh));
    expect_mesh_connectivity(mesh);
}

TEST_F(SurfacePatchTest, CurvedClosedPatchMatchesDirectExpMapInitializedSlim)
{
    auto mesh = cube(true);
    const auto input = placement(mesh);
    auto raw_mesh = mesh;
    initialize_exp_map_properties(raw_mesh);
    auto raw = gpf::exp_map(std::span<const double, 3>(input.surface_point), raw_mesh, input.lengths[1]);
    ASSERT_TRUE(raw);
    auto candidate = mesh;
    auto initial = initialize_exp_map_patch(candidate, input);
    ASSERT_TRUE(initial);
    expect_exp_map_mapping(candidate, *initial, input, *raw);
    FlattenSurface solver(VMat(initial->patch.positions), FMat(initial->patch.triangles), VMat2(initial->patch.coordinates), 1);
    solver.slim_solve(5, 15);
    EXPECT_GT((solver.uv - initial->patch.coordinates).norm(), 1e-8);
    auto result = prepare_surface_patch(mesh, input);
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    EXPECT_TRUE(coordinates(*result).isApprox(solver.uv, 1e-12));
    EXPECT_EQ(result->surface_faces, initial->patch.surface_faces);
    EXPECT_EQ(result->uv_to_surface_vertex, initial->patch.uv_to_surface_vertex);
}

TEST_F(SurfacePatchTest, TetrahedronRejectionRetainsSubdivisionForFallback)
{
    auto mesh = tetrahedron();
    auto direct = mesh;
    const auto input = placement(mesh);
    auto candidate = mesh;
    initialize_exp_map_properties(candidate);
    auto failed = gpf::exp_map(std::span<const double, 3>(input.surface_point), candidate, input.lengths[1]);
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error(), gpf::ExpMapFailure::NotTopologicalDisk);
    EXPECT_GT(candidate.n_vertices(), mesh.n_vertices());
    auto rejected = prepare_preferred(direct, input);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error(), PreferredFailure::Rejected);
    EXPECT_FALSE(std::filesystem::exists("uv_new.obj"));
    auto result = prepare_surface_patch(mesh, input);
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    auto expected = prepare_legacy(direct, input);
    ASSERT_TRUE(expected);
    EXPECT_EQ(fingerprint(mesh), fingerprint(direct));
    EXPECT_EQ(coordinates(*result), coordinates(*expected));
    EXPECT_EQ(result->frame.origin, expected->frame.origin);
    // The selected tetrahedron face is planar with this outward normal.
    const Eigen::Vector3d normal = Eigen::Vector3d { 1, -1, 1 }.normalized();
    const Eigen::Vector3d direction { 0.12, 0.03, 0 };
    const Eigen::Vector3d xaxis = 2 * direction.norm() * (direction - normal.dot(direction) * normal).normalized();
    const Eigen::Vector3d yaxis = normal.cross(xaxis);
    // Allow 2% of a side for legacy chart distortion, not the old 45-degree rotation.
    expect_planar_frame(mesh, *result, Eigen::Vector3d { 1. / 3, -1. / 3, 1. / 3 }, xaxis, yaxis, 0.02 * xaxis.norm());
    expect_mesh_connectivity(mesh);
}

TEST_F(SurfacePatchTest, LegacyBoundaryWalkDoesNotAssumeVertexZeroIsOnBoundary)
{
    legacy::ExtractedSurfacePatch patch {};
    patch.positions.resize(5, 3);
    patch.positions << 0, 0, 0, -2, -2, 0, 2, -2, 0, 2, 2, 0, -2, 2, 0;
    patch.triangles.resize(4, 3);
    patch.triangles << 0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1;
    patch.n_boundary_vertices = 4;
    auto initialized = legacy::initialize_patch_uv(patch);
    ASSERT_EQ(initialized.boundary.size(), 4);
    ASSERT_EQ(initialized.coordinates.rows(), 5);
    for (Eigen::Index i = 0; i < initialized.boundary.size(); ++i) {
        EXPECT_EQ(initialized.boundary(i), i + 1);
    }
    EXPECT_NEAR(initialized.coordinates.row(0).norm(), 0, 1e-12);
    EXPECT_EQ(igl::flipped_triangles(initialized.coordinates, patch.triangles).size(), 0);
}

TEST_F(SurfacePatchTest, LegacyBoundaryWalkRejectsIncompleteOrMultipleLoops)
{
    auto expect_rejected = [](const legacy::ExtractedSurfacePatch& patch) {
        auto initialized = legacy::initialize_patch_uv(patch);
        EXPECT_EQ(initialized.boundary.size(), 0);
        EXPECT_EQ(initialized.coordinates.rows(), 0);
    };
    expect_rejected({});
    const auto planar = planar_patch();
    legacy::ExtractedSurfacePatch patch {};
    patch.positions = planar.positions;
    patch.triangles = planar.triangles;
    for (const std::size_t count : { 3, 5 }) {
        patch.n_boundary_vertices = count;
        expect_rejected(patch);
    }
    // An annulus has two boundary loops rather than one harmonic-circle boundary.
    patch.positions.resize(8, 3);
    patch.positions << -2, -2, 0, 2, -2, 0, 2, 2, 0, -2, 2, 0,
        -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0;
    patch.triangles.resize(8, 3);
    patch.triangles << 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5,
        2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7;
    patch.n_boundary_vertices = 8;
    expect_rejected(patch);
    // Adding a disconnected disk makes the total Euler characteristic one.
    // Walking only one of the three loops must still reject this patch.
    patch.positions.conservativeResize(12, 3);
    patch.positions.bottomRows<4>() << 3, -2, 0, 7, -2, 0, 7, 2, 0, 3, 2, 0;
    patch.triangles.conservativeResize(10, 3);
    patch.triangles.bottomRows<2>() << 8, 9, 10, 8, 10, 11;
    patch.n_boundary_vertices = 12;
    expect_rejected(patch);
    EXPECT_FALSE(std::filesystem::exists("uv_new.obj"));
}

TEST_F(SurfacePatchTest, LegacyPreparationSupportsOpenTriangles)
{
    auto mesh = open_plane();
    auto direct = mesh;
    const auto input = placement(mesh);
    auto initial = initialize_legacy_patch(direct, input);
    ASSERT_TRUE(initial);
    auto boundary = initial->first.boundary;
    ranges::sort(boundary);
    for (std::size_t i = 0; i < boundary.size(); ++i) {
        EXPECT_EQ(boundary[i], i);
    }
    EXPECT_EQ(initial->first.center, initial->second[0]);
    auto expected = optimize_patch(std::move(initial->first));
    EXPECT_EQ(expected.coordinates.row(0), expected.initial.coordinates.row(0));
    auto result = prepare_legacy(mesh, input);
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    EXPECT_EQ(fingerprint(mesh), fingerprint(direct));
    EXPECT_EQ(coordinates(*result), expected.coordinates);
    expect_planar_frame(mesh, *result, Eigen::Vector3d::Map(input.surface_point.data()),
        Eigen::Vector3d { 0.24, 0.06, 0 }, Eigen::Vector3d { -0.06, 0.24, 0 });
}

TEST_F(SurfacePatchTest, CenterReuseAndInsertionPreserveFaceProperties)
{
    for (bool reuse : { false, true }) {
        auto mesh = cube();
        for (auto face : mesh.faces()) {
            face.prop().polygon_id = 50 + face.id.idx;
        }
        const auto original = mesh;
        auto input = placement(mesh);
        if (reuse) {
            input.surface_point = mesh.vertex_prop(input.source_vertices[0]).pt;
        }
        auto result = prepare_surface_patch(mesh, input);
        ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
        EXPECT_EQ(mesh.n_vertices(), original.n_vertices() + (reuse ? 0 : 1));
        for (const auto face : mesh.faces()) {
            ASSERT_LT(face.prop().parent.idx, original.n_faces_capacity());
            EXPECT_EQ(face.prop().polygon_id, original.face_prop(face.prop().parent).polygon_id);
        }
    }
}

TEST_F(SurfacePatchTest, SparseCapacityPropertiesAndFreshInitializationAfterEdit)
{
    auto mesh = cube();
    for (auto face : mesh.faces()) {
        face.prop().polygon_id = face.id.idx + 13;
    }
    const auto edge = mesh.face(gpf::FaceId { 0 }).halfedge().edge().id;
    const auto [a, b] = mesh.e_vertices(edge);
    const auto midpoint = (0.5 * (Eigen::Vector3d::Map(mesh.vertex_prop(a).pt.data()) + Eigen::Vector3d::Map(mesh.vertex_prop(b).pt.data()))).eval();
    mesh.collapse_edge(edge, a, b);
    Eigen::Vector3d::Map(mesh.vertex_prop(a).pt.data()) = midpoint;
    ASSERT_GT(mesh.n_vertices_capacity(), mesh.n_vertices());
    ASSERT_GT(mesh.n_faces_capacity(), mesh.n_faces());
    auto expected_mesh = mesh;
    initialize_exp_map_properties(expected_mesh);
    for (const auto e : expected_mesh.edges()) {
        const auto [va, vb] = e.vertices();
        EXPECT_DOUBLE_EQ(e.prop().len, (Eigen::Vector3d::Map(va.prop().pt.data()) - Eigen::Vector3d::Map(vb.prop().pt.data())).norm());
    }
    const auto input = placement(mesh);
    auto expected = prepare_preferred(expected_mesh, input);
    ASSERT_TRUE(expected);
    auto result = prepare_surface_patch(mesh, input);
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    EXPECT_EQ(fingerprint(mesh), fingerprint(expected_mesh));
    EXPECT_EQ(coordinates(*result), coordinates(*expected));
    expect_mesh_connectivity(mesh);
}

TEST_F(SurfacePatchTest, ExpMapBoundaryIdsMapToLocalIndicesOnSparseMesh)
{
    auto mesh = cube();
    const auto edge = mesh.face(gpf::FaceId { 0 }).halfedge().edge().id;
    const auto [a, b] = mesh.e_vertices(edge);
    const Eigen::Vector3d midpoint = 0.5 * (Eigen::Vector3d::Map(mesh.vertex_prop(a).pt.data()) + Eigen::Vector3d::Map(mesh.vertex_prop(b).pt.data()));
    mesh.collapse_edge(edge, a, b);
    Eigen::Vector3d::Map(mesh.vertex_prop(a).pt.data()) = midpoint;
    ASSERT_GT(mesh.n_vertices_capacity(), mesh.n_vertices());
    ASSERT_GT(mesh.n_faces_capacity(), mesh.n_faces());

    const auto input = placement(mesh);
    auto raw_mesh = mesh;
    initialize_exp_map_properties(raw_mesh);
    auto raw = gpf::exp_map(std::span<const double, 3>(input.surface_point), raw_mesh, input.lengths[1]);
    ASSERT_TRUE(raw);
    auto candidate = mesh;
    auto initial = initialize_exp_map_patch(candidate, input);
    ASSERT_TRUE(initial);
    expect_exp_map_mapping(candidate, *initial, input, *raw);
    EXPECT_NE(initial->patch.uv_to_surface_vertex.front().idx, 0);
}

TEST_F(SurfacePatchTest, ExpMapSourceIndicesFollowSourceTriangleOrder)
{
    const auto mesh = cube();
    const auto original = placement(mesh);
    auto reference_mesh = mesh;
    auto reference = initialize_exp_map_patch(reference_mesh, original);
    ASSERT_TRUE(reference);
    std::array<std::size_t, 3> order { 0, 1, 2 };
    do {
        SCOPED_TRACE((testing::Message() << "order=" << order[0] << ',' << order[1] << ',' << order[2]));
        auto input = original;
        for (std::size_t i = 0; i < order.size(); ++i) {
            input.source_vertices[i] = original.source_vertices[order[i]];
            input.source_positions.row(i) = original.source_positions.row(order[i]);
        }
        auto candidate = mesh;
        auto initial = initialize_exp_map_patch(candidate, input);
        ASSERT_TRUE(initial);
        EXPECT_EQ(initial->patch.uv_to_surface_vertex, reference->patch.uv_to_surface_vertex);
        EXPECT_EQ(initial->patch.coordinates, reference->patch.coordinates);
        EXPECT_EQ(initial->source_uv_indices, reference_source_uv_indices(initial->patch, input));
        for (std::size_t i = 0; i < order.size(); ++i) {
            const auto local = initial->source_uv_indices[i];
            ASSERT_LT(local, initial->patch.uv_to_surface_vertex.size());
            EXPECT_EQ(local, reference->source_uv_indices[order[i]]);
            EXPECT_EQ(initial->patch.uv_to_surface_vertex[local], input.source_vertices[i]);
        }
    } while (ranges::next_permutation(order).found);
}

TEST_F(SurfacePatchTest, ExpMapUnavailableSourceIndicesRemainInvalid)
{
    const InitializedExpMapPatch empty;
    const std::array<std::size_t, 3> invalid_indices { gpf::kInvalidIndex, gpf::kInvalidIndex, gpf::kInvalidIndex };
    EXPECT_EQ(empty.source_uv_indices, invalid_indices);

    const auto mesh = cube();
    const auto original = placement(mesh);
    auto reference_mesh = mesh;
    auto reference = initialize_exp_map_patch(reference_mesh, original);
    ASSERT_TRUE(reference);
    // Vertex zero is live but lies on the opposite side of the cube, outside
    // this patch. The lookup bound is measured after exp-map inserts its center.
    const gpf::VertexId absent { 0 };
    EXPECT_FALSE(mesh.vertex_is_deleted(absent));
    ASSERT_EQ(ranges::find(reference->patch.uv_to_surface_vertex, absent), reference->patch.uv_to_surface_vertex.end());
    const auto capacity = reference_mesh.n_vertices_capacity();
    for (std::size_t slot = 0; slot < original.source_vertices.size(); ++slot) {
        for (const auto vertex : { absent, gpf::VertexId {}, gpf::VertexId { capacity }, gpf::VertexId { capacity + 17 } }) {
            SCOPED_TRACE((testing::Message() << "slot=" << slot << " source_id=" << vertex.idx));
            auto input = original;
            input.source_vertices[slot] = vertex;
            auto candidate = mesh;
            auto initial = initialize_exp_map_patch(candidate, input);
            ASSERT_TRUE(initial); // Missing source metadata must not reject initialization.
            EXPECT_EQ(initial->patch.uv_to_surface_vertex, reference->patch.uv_to_surface_vertex);
            auto expected = reference->source_uv_indices;
            expected[slot] = gpf::kInvalidIndex;
            EXPECT_EQ(initial->source_uv_indices, expected);
            EXPECT_EQ(initial->source_uv_indices, reference_source_uv_indices(initial->patch, input));
        }
    }
    EXPECT_FALSE(std::filesystem::exists("uv_new.obj"));
}

TEST_F(SurfacePatchTest, PlacementRejectsOutOfRangeOrMismatchedSourceIndices)
{
    auto p = planar_patch();
    PlacementInput input {};
    input.source_vertices = { gpf::VertexId { 0 }, gpf::VertexId { 1 }, gpf::VertexId { 3 } };
    input.source_positions << -2, -2, 0, 2, -2, 0, -2, 2, 0;
    input.tangent_direction << 1, 0, 0;
    input.direction_magnitude = 0.2;
    const std::array<std::size_t, 3> source_uv_indices { 0, 1, 3 };
    const OptimizedPatch optimized { p, p.coordinates };
    ASSERT_TRUE(exp_map_placement(optimized, input, source_uv_indices));
    for (std::size_t slot = 0; slot < source_uv_indices.size(); ++slot) {
        for (const auto local : { gpf::kInvalidIndex, p.uv_to_surface_vertex.size(),
                 p.uv_to_surface_vertex.size() + 1, source_uv_indices[(slot + 1) % source_uv_indices.size()] }) {
            SCOPED_TRACE((testing::Message() << "slot=" << slot << " local=" << local));
            auto indices = source_uv_indices;
            indices[slot] = local;
            auto rejected = exp_map_placement(optimized, input, indices);
            ASSERT_FALSE(rejected);
            EXPECT_EQ(rejected.error(), SurfaceFailure::ParameterizationFailed);
        }
        // A valid row from an earlier input must not silently bind to a different
        // source vertex, even when that vertex is also present in the patch.
        auto changed_input = input;
        changed_input.source_vertices[slot] = input.source_vertices[(slot + 1) % input.source_vertices.size()];
        auto rejected = exp_map_placement(optimized, changed_input, source_uv_indices);
        ASSERT_FALSE(rejected);
        EXPECT_EQ(rejected.error(), SurfaceFailure::ParameterizationFailed);
    }
}

TEST_F(SurfacePatchTest, PlacementUsesOptimizedCenterAndSignedLocalAxes)
{
    for (const double tilt_angle : { 0., 0.73 }) {
        const Eigen::AngleAxisd tilt(tilt_angle, Eigen::Vector3d { 1, -2, 3 }.normalized());
        auto source = open_plane();
        auto p = planar_patch();
        p.positions = (p.positions * tilt.toRotationMatrix().transpose()).eval();
        for (const auto vertex : source.vertices()) {
            Eigen::Vector3d::Map(vertex.prop().pt.data()) = p.positions.row(vertex.id.idx).transpose();
        }
        for (const Eigen::Vector3d direction : { Eigen::Vector3d { 0.2, 0, 0 }, { 0.12, 0.16, 0 },
                 { -0.2, 0, 0 }, { -0.12, -0.16, 0 }, { 0.12, 0, 0.16 } }) {
            const Eigen::Vector3d world_direction = tilt * direction;
            const auto input = make_placement_input(source, gpf::FaceId { 0 }, { 0, 0, 0 },
                { world_direction.x(), world_direction.y(), world_direction.z() });
            // This synthetic patch's source IDs equal its local indices.
            const std::array<std::size_t, 3> source_uv_indices {
                input.source_vertices[0].idx, input.source_vertices[1].idx, input.source_vertices[2].idx
            };
            for (const double chart_angle : { 0., 0.41 }) {
                const Eigen::Rotation2Dd rotation(chart_angle);
                const Eigen::Vector2d x = rotation * direction.head<2>().normalized();
                const Eigen::Vector2d y = rotation * Eigen::Vector2d { -direction.y(), direction.x() }.normalized();
                for (const double chart_scale : { 1., 0.04 }) {
                    SCOPED_TRACE((testing::Message() << "tilt=" << tilt_angle << " direction=" << direction.transpose()
                                                     << " chart_angle=" << chart_angle << " chart_scale=" << chart_scale));
                    VMat2 optimized = chart_scale * p.coordinates * rotation.toRotationMatrix().transpose();
                    optimized.rowwise() += Eigen::RowVector2d { 7, -3 };
                    auto frame = exp_map_placement({ p, optimized }, input, source_uv_indices);
                    ASSERT_TRUE(frame);
                    const Eigen::Vector2d center = frame->origin + 0.5 * (frame->xaxis + frame->yaxis);
                    EXPECT_TRUE(center.isApprox(Eigen::Vector2d { 7, -3 }, 1e-12));
                    EXPECT_TRUE(frame->xaxis.normalized().isApprox(x, 1e-12));
                    EXPECT_TRUE(frame->yaxis.normalized().isApprox(y, 1e-12));
                    EXPECT_NEAR(frame->xaxis.norm(), frame->yaxis.norm(), 1e-12);
                    EXPECT_NEAR(frame->xaxis.dot(frame->yaxis), 0, 1e-12);
                    EXPECT_GT(frame->xaxis.x() * frame->yaxis.y() - frame->xaxis.y() * frame->yaxis.x(), 0);
                    if (chart_scale == 1.) {
                        EXPECT_NEAR(frame->xaxis.norm(), 2 * direction.norm(), 1e-12);
                        EXPECT_NEAR(frame->yaxis.norm(), 2 * direction.norm(), 1e-12);
                    } else {
                        EXPECT_LT(frame->xaxis.norm(), 2 * direction.norm());
                        EXPECT_GT(frame->xaxis.norm(), 2 * kProjectionTolerance);
                    }
                }
            }
        }
    }
}

TEST_F(SurfacePatchTest, PlacementInputCapturesSourceTriangleAndFootprint)
{
    const auto mesh = cube();
    const auto face = top_face(mesh);
    const auto point = centroid(mesh, face);
    const Footprint footprint { Eigen::Vector2d { 0.7, 0.4 } };
    const auto input = make_placement_input(mesh, face, point, { 0.12, 0.03, 0 }, footprint);
    EXPECT_EQ(input.source_face, face);
    EXPECT_EQ(input.surface_point, point);
    EXPECT_TRUE(input.normal.isApprox(Eigen::Vector3d::UnitZ()));
    EXPECT_TRUE(input.footprint.half_extent.isApprox(footprint.half_extent));
    std::size_t i = 0;
    for (const auto he : mesh.face(face).halfedges()) {
        EXPECT_EQ(input.source_vertices[i], he.from().id);
        EXPECT_TRUE(input.source_positions.row(i).transpose().isApprox(Eigen::Vector3d::Map(he.from().prop().pt.data())));
        ++i;
    }
}

TEST_F(SurfacePatchTest, NormalComponentControlsSizeButNotTangent)
{
    auto mesh = cube();
    auto a = placement(mesh, { 0.12, 0, 0 });
    auto b = placement(mesh, { 0.12, 0, 0.16 });
    EXPECT_NEAR(a.direction_magnitude, 0.12, 1e-12);
    EXPECT_NEAR(b.direction_magnitude, 0.2, 1e-12);
    EXPECT_TRUE(a.tangent_direction.isApprox(b.tangent_direction));
    EXPECT_TRUE(b.tangent_direction.isApprox(Eigen::Vector3d::UnitX()));
    EXPECT_NEAR(b.lengths[0], std::sqrt(2.) * 0.2, 1e-12);
    EXPECT_NEAR(b.lengths[1], 1.2 * std::sqrt(2.) * 0.2, 1e-12);
    EXPECT_TRUE(b.footprint.half_extent.isApprox(Eigen::Vector2d::Constant(0.5)));
}

TEST_F(SurfacePatchTest, ShrinkingUsesEntireConcaveBoundaryAndKeepsCenterAndAspect)
{
    // All four requested corners are inside; the top notch crosses its top edge.
    VMat2 boundary(8, 2);
    boundary << -2, -2, 2, -2, 2, 2, 0.2, 2, 0.2, 0.4, -0.2, 0.4, -0.2, 2, -2, 2;
    std::vector<std::size_t> indices(8);
    std::iota(indices.begin(), indices.end(), 0);
    UvPlacementFrame requested { { -1, -1 }, { 2, 0 }, { 0, 2 } };
    auto result = fit_placement(requested, {}, boundary, indices);
    ASSERT_TRUE(result);
    EXPECT_NEAR(result->xaxis.norm() / requested.xaxis.norm(), 0.4 - kProjectionTolerance, 1e-12);
    EXPECT_NEAR(result->xaxis.norm(), result->yaxis.norm(), 1e-12);
    EXPECT_TRUE((result->origin + 0.5 * (result->xaxis + result->yaxis)).isZero(1e-12));
    // Every point on every support segment stays below the notch, not merely corners.
    for (int i = 0; i <= 100; ++i) {
        const Eigen::Vector2d p = result->origin + (i / 100.) * result->xaxis + result->yaxis;
        EXPECT_LT(p.y(), 0.4);
    }
}

TEST_F(SurfacePatchTest, PlacementKeepsSizeWhenSafeAndUsesOptimizedNotInitialBoundary)
{
    auto p = planar_patch();
    const UvPlacementFrame requested { { -0.5, -0.5 }, { 1, 0 }, { 0, 1 } };
    auto large = fit_placement(requested, {}, p.coordinates, p.boundary);
    ASSERT_TRUE(large);
    EXPECT_EQ(large->origin, requested.origin);
    EXPECT_EQ(large->xaxis, requested.xaxis);
    VMat2 smaller = p.coordinates * 0.2;
    auto small = fit_placement(requested, {}, smaller, p.boundary);
    ASSERT_TRUE(small);
    EXPECT_NEAR(small->xaxis.norm(), 0.8 - 2 * kProjectionTolerance, 1e-12);
    auto collapsed = fit_placement(requested, {}, (p.coordinates * 1e-6).eval(), p.boundary);
    EXPECT_FALSE(collapsed);
    auto outside = requested;
    outside.origin << 20, 20;
    EXPECT_FALSE(fit_placement(outside, {}, p.coordinates, p.boundary));
}

TEST_F(SurfacePatchTest, PlacementKeepsSizeWhenProjectionClearanceFits)
{
    const auto p = planar_patch();
    const UvPlacementFrame requested { { -0.5, -0.5 }, { 1, 0 }, { 0, 1 } };
    // The boundary clears the requested footprint by twice the projection tolerance.
    const VMat2 boundary = p.coordinates * (0.25 + kProjectionTolerance);
    auto result = fit_placement(requested, {}, boundary, p.boundary);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->origin, requested.origin);
    EXPECT_EQ(result->xaxis, requested.xaxis);
    EXPECT_EQ(result->yaxis, requested.yaxis);
}

TEST_F(SurfacePatchTest, LegacyCornerWalksFollowLocalAxesWithOriginalMagnitude)
{
    const auto mesh = open_plane();
    const auto face = gpf::FaceId { 0 };
    const auto point = centroid(mesh, face);
    const Eigen::Vector3d center = Eigen::Vector3d::Map(point.data());
    // Both inputs project along +X and have magnitude 0.2, even with a normal component.
    for (const std::array<double, 3> direction : { std::array<double, 3> { 0.2, 0, 0 }, { 0.12, 0, 0.16 } }) {
        const auto input = make_placement_input(mesh, face, point, direction);
        auto walked = legacy::walk_patch_boundary_and_anchors(mesh, input);
        ASSERT_TRUE(walked) << fit_on_surface::to_string(walked.error());
        ASSERT_EQ(walked->points.size(), 9);
        EXPECT_TRUE(Eigen::Vector3d::Map(walked->points[4].data()).isApprox(center, 1e-12));
        const std::array<Eigen::Vector3d, 4> offsets { Eigen::Vector3d { 0.2, -0.2, 0 },
            { 0.2, 0.2, 0 }, { -0.2, 0.2, 0 }, { -0.2, -0.2, 0 } };
        for (std::size_t i = 0; i < offsets.size(); ++i) {
            EXPECT_TRUE(Eigen::Vector3d::Map(walked->points[5 + i].data()).isApprox(center + offsets[i], 1e-12));
            EXPECT_TRUE(Eigen::Vector3d::Map(walked->points[i].data()).isApprox(center + 1.2 * offsets[i], 1e-12));
        }
    }
}

TEST_F(SurfacePatchTest, LegacyAnchorFrameUsesProvidedScaleWithoutExtraShrink)
{
    VMat2 coordinates(5, 2);
    coordinates << 3, -2, 4, -3, 4, -1, 2, -1, 2, -3;
    const std::array<std::size_t, 5> anchors { 0, 1, 2, 3, 4 };
    for (std::size_t corner = 1; corner <= 4; ++corner) {
        for (const auto scale : { std::optional<double> {}, std::optional<double> { 1. }, std::optional<double> { 0.4 } }) {
            SCOPED_TRACE((testing::Message() << "corner=" << corner << " scale=" << scale.value_or(1.)));
            const auto frame = legacy::compute_anchor_uv_frame(coordinates, anchors, corner, scale);
            const double half_side = scale.value_or(1.);
            EXPECT_TRUE(frame.xaxis.isApprox(Eigen::Vector2d { 2 * half_side, 0 }, 1e-12));
            EXPECT_TRUE(frame.yaxis.isApprox(Eigen::Vector2d { 0, 2 * half_side }, 1e-12));
            EXPECT_TRUE(frame.origin.isApprox(Eigen::Vector2d { 3 - half_side, -2 - half_side }, 1e-12));
            const Eigen::Vector2d center = frame.origin + 0.5 * (frame.xaxis + frame.yaxis);
            EXPECT_TRUE(center.isApprox(Eigen::Vector2d { 3, -2 }, 1e-12));
        }
    }
}

TEST_F(SurfacePatchTest, LegacyAnchorContainmentMatchesLocalAxisCornerOrder)
{
    VMat2 coordinates(9, 2);
    coordinates.topRows<5>() << 3, -2, 4, -3, 4, -1, 2, -1, 2, -3;
    const std::array<std::size_t, 5> anchors { 0, 1, 2, 3, 4 };
    const Eigen::Vector4i boundary { 5, 6, 7, 8 };
    for (const double half_side : { 1.2, 0.4 }) {
        coordinates.bottomRows<4>() << 3 - half_side, -2 - half_side, 3 + half_side, -2 - half_side,
            3 + half_side, -2 + half_side, 3 - half_side, -2 + half_side;
        for (std::size_t corner = 1; corner <= 4; ++corner) {
            const auto scale = legacy::boundary_contains_anchor_rectangle(coordinates, anchors, corner, boundary);
            if (half_side > 1) {
                EXPECT_FALSE(scale);
            } else {
                ASSERT_TRUE(scale);
                EXPECT_NEAR(*scale, half_side, 1e-12);
            }
            const auto frame = legacy::compute_anchor_uv_frame(coordinates, anchors, corner, scale);
            EXPECT_TRUE(frame.xaxis.isApprox(Eigen::Vector2d { 2 * std::min(1., half_side), 0 }, 1e-12));
            EXPECT_TRUE(frame.yaxis.isApprox(Eigen::Vector2d { 0, 2 * std::min(1., half_side) }, 1e-12));
        }
    }
}

TEST_F(SurfacePatchTest, PreferredUnusablePlacementFallsBackWithRetainedSubdivision)
{
    auto mesh = cube();
    auto direct = mesh;
    auto input = placement(mesh);
    // A finite but extremely elongated footprint makes the support frame singular
    // at the placement tolerance. Legacy retains its original anchor sizing.
    input.footprint.half_extent.x() = 1e16;
    auto rejected = prepare_preferred(direct, input);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error(), PreferredFailure::Rejected);
    auto result = prepare_surface_patch(mesh, input);
    ASSERT_TRUE(result);
    auto expected = prepare_legacy(direct, input);
    ASSERT_TRUE(expected);
    EXPECT_EQ(fingerprint(mesh), fingerprint(direct));
    EXPECT_EQ(coordinates(*result), coordinates(*expected));
    EXPECT_EQ(result->frame.origin, expected->frame.origin);
}

TEST_F(SurfacePatchTest, SelectedDumpsContainOptimizedCoordinates)
{
    auto mesh = cube(true);
    auto result = prepare_surface_patch(mesh, placement(mesh));
    ASSERT_TRUE(result);
    const VMat2 optimized = coordinates(*result);
    std::ifstream obj("uv_new.obj"), off("fit_polygon_on_surface_uv.off");
    ASSERT_TRUE(obj);
    ASSERT_TRUE(off);
    std::string tag;
    std::size_t vertices, faces, edges;
    off >> tag >> vertices >> faces >> edges;
    EXPECT_EQ(tag, "OFF");
    EXPECT_EQ(vertices, static_cast<std::size_t>(optimized.rows()));
    for (Eigen::Index i = 0; i < optimized.rows(); ++i) {
        double x, y, z, ox, oy, oz;
        obj >> tag >> x >> y >> z;
        off >> ox >> oy >> oz;
        EXPECT_EQ(tag, "v");
        EXPECT_NEAR(x, optimized(i, 0), 1e-5);
        EXPECT_NEAR(y, optimized(i, 1), 1e-5);
        EXPECT_DOUBLE_EQ(x, ox);
        EXPECT_DOUBLE_EQ(y, oy);
    }
}

TEST_F(SurfacePatchTest, LegacyWalkFailureDoesNotWriteDiagnostics)
{
    auto mesh = open_plane();
    const auto original = fingerprint(mesh);
    const auto input = placement(mesh, { 50, 0, 0 });
    auto result = prepare_legacy(mesh, input);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), SurfaceFailure::WalkBoundaryReached);
    EXPECT_EQ(fingerprint(mesh), original);
    EXPECT_FALSE(std::filesystem::exists("uv_new.obj"));
    EXPECT_FALSE(std::filesystem::exists("fit_polygon_on_surface_uv.off"));
}

TEST_F(SurfacePatchTest, PreferredPlacementRejectionRetainsExistingSolverDump)
{
    auto mesh = cube();
    const auto original = fingerprint(mesh);
    const auto count = mesh.n_vertices();
    auto input = placement(mesh);
    input.footprint.half_extent.x() = 1e16;
    auto result = prepare_preferred(mesh, input);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), PreferredFailure::Rejected);
    EXPECT_GT(mesh.n_vertices(), count);
    EXPECT_NE(fingerprint(mesh), original);
    // The unchanged solver writes its auxiliary dump before placement can fail.
    EXPECT_TRUE(std::filesystem::exists("uv_new.obj"));
    EXPECT_FALSE(std::filesystem::exists("fit_polygon_on_surface_uv.off"));
}

TEST_F(SurfacePatchTest, PreferredShrinksLargeFootprintWithoutFallback)
{
    auto mesh = cube();
    auto input = placement(mesh);
    input.footprint.half_extent << 2, 1;
    auto candidate = mesh;
    auto expected = prepare_preferred(candidate, input);
    ASSERT_TRUE(expected);
    auto result = prepare_surface_patch(mesh, input);
    ASSERT_TRUE(result);
    EXPECT_EQ(fingerprint(mesh), fingerprint(candidate));
    EXPECT_EQ(coordinates(*result), coordinates(*expected));
    EXPECT_LT(result->frame.xaxis.norm(), 2 * input.direction_magnitude);
    EXPECT_NEAR(result->frame.xaxis.norm(), result->frame.yaxis.norm(), 1e-12);
}

TEST_F(SurfacePatchTest, NearbySheetWithMissingSourceVerticesIsRejectedAfterSlim)
{
    auto geometry = cube_geometry();
    const auto original = geometry;
    const auto offset = geometry.points.size();
    for (auto p : original.points) {
        p[2] += 0.0005;
        geometry.points.push_back(p);
    }
    for (auto face : original.triangles) {
        for (auto& vertex : face) {
            vertex += offset;
        }
        geometry.triangles.push_back(face);
    }
    auto mesh = make_mesh(geometry);
    const auto source = cube();
    auto input = placement(source);
    input.surface_point[2] += 0.0004; // closer to the other sheet than the supplied source face
    const auto original_mesh = fingerprint(mesh);
    auto candidate = mesh;
    initialize_exp_map_properties(candidate);
    auto raw = gpf::exp_map(std::span<const double, 3>(input.surface_point), candidate, input.lengths[1]);
    ASSERT_TRUE(raw);
    for (const auto face : raw->face_ids) {
        EXPECT_GE(face.idx, original.triangles.size());
    }
    auto direct = mesh;
    auto rejected = prepare_preferred(direct, input);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error(), PreferredFailure::Rejected);
    EXPECT_NE(fingerprint(direct), original_mesh);
    // Placement still needs the original source vertices' UVs, but the center
    // no longer has a separate compatibility gate before the solver runs.
    EXPECT_TRUE(std::filesystem::exists("uv_new.obj"));
    // The other sheet's subdivision remains. Fallback still walks from the
    // captured source triangle; its projection may itself select the other sheet.
    auto result = prepare_surface_patch(mesh, input);
    auto expected = prepare_legacy(direct, input);
    ASSERT_EQ(result.has_value(), expected.has_value());
    EXPECT_EQ(fingerprint(mesh), fingerprint(direct));
    if (result) {
        EXPECT_EQ(coordinates(*result), coordinates(*expected));
    } else {
        EXPECT_EQ(result.error(), expected.error());
    }
}

TEST_F(SurfacePatchTest, DegenerateOptimizedSourceCorrespondenceIsRejected)
{
    auto p = planar_patch();
    PlacementInput input {};
    input.source_vertices = { gpf::VertexId { 0 }, gpf::VertexId { 1 }, gpf::VertexId { 3 } };
    input.source_positions << -2, -2, 0, 2, -2, 0, -2, 2, 0;
    input.tangent_direction << 1, 0, 0;
    input.direction_magnitude = 0.2;
    VMat2 optimized = p.coordinates;
    optimized.row(3) = 0.5 * (optimized.row(0) + optimized.row(1));
    const std::array<std::size_t, 3> source_uv_indices { 0, 1, 3 };
    auto degenerate = exp_map_placement({ p, optimized }, input, source_uv_indices);
    ASSERT_FALSE(degenerate);
    EXPECT_EQ(degenerate.error(), SurfaceFailure::ParameterizationFailed);
    input.source_vertices[0] = gpf::VertexId { 999 };
    auto missing = exp_map_placement({ p, p.coordinates }, input, source_uv_indices);
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error(), SurfaceFailure::ParameterizationFailed);
}

TEST_F(SurfacePatchTest, SurfaceDiagnosticFailureDoesNotRetry)
{
    auto mesh = cube();
    const auto input = placement(mesh);
    auto candidate = mesh;
    auto expected = prepare_preferred(candidate, input);
    ASSERT_TRUE(expected);
    std::filesystem::create_directory("fit_polygon_on_surface.off");
    auto result = prepare_surface_patch(mesh, input);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), SurfaceFailure::SurfaceOffOpenFailed);
    EXPECT_EQ(fingerprint(mesh), fingerprint(candidate));
}

TEST_F(SurfacePatchTest, DiagnosticFailureAfterSelectionDoesNotRetry)
{
    auto mesh = cube();
    const auto input = placement(mesh);
    const auto count = mesh.n_vertices();
    auto candidate = mesh;
    auto expected = prepare_preferred(candidate, input);
    ASSERT_TRUE(expected);
    std::filesystem::create_directory("fit_polygon_on_surface_uv.off");
    auto result = prepare_surface_patch(mesh, input);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), SurfaceFailure::UvCoordinatesOffOpenFailed);
    EXPECT_EQ(fingerprint(mesh), fingerprint(candidate));
    EXPECT_GT(mesh.n_vertices(), count);
}
}
