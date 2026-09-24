#include "fixtures.h"

#include <set>

namespace relief::test {
class ImageReliefTest : public SurfaceFixture { };

void expect_grid_mapping(const image_relief::Mesh& mesh, const std::vector<image_relief::GridFaceIndex>& mapping,
    std::size_t width, gpf::FaceId parent)
{
    std::set<gpf::FaceId> faces;
    std::map<std::size_t, std::set<std::size_t>> cells;
    for (const auto& entry : mapping) {
        EXPECT_TRUE(entry.combined_face.valid());
        ASSERT_LT(entry.combined_face.idx, mesh.n_faces_capacity());
        EXPECT_FALSE(mesh.face_is_deleted(entry.combined_face));
        EXPECT_TRUE(faces.insert(entry.combined_face).second);
        EXPECT_EQ(mesh.face_prop(entry.combined_face).parent, parent);
        EXPECT_LT(entry.grid_row, width - 1);
        EXPECT_LT(entry.grid_column, width - 1);
        EXPECT_EQ(entry.grid_face_index, entry.grid_row * (width - 1) + entry.grid_column);
        EXPECT_TRUE(cells[entry.grid_face_index].insert(entry.triangle_index).second);
    }
    EXPECT_EQ(cells.size(), (width - 1) * (width - 1));
    for (const auto& [cell, triangles] : cells) {
        EXPECT_GE(triangles.size(), 2);
        std::size_t i = 0;
        for (const auto index : triangles) {
            EXPECT_EQ(index, i++);
        }
    }
}

void expect_closed_oriented(const image_relief::Mesh& mesh)
{
    std::map<std::pair<std::size_t, std::size_t>, std::pair<int, int>> incidence;
    for (const auto face : mesh.faces()) {
        for (const auto he : face.halfedges()) {
            const auto a = he.from().id.idx, b = he.to().id.idx;
            auto& [count, orientation] = incidence[std::minmax(a, b)];
            ++count;
            orientation += a < b ? 1 : -1;
            EXPECT_TRUE(he.twin().face().id.valid());
        }
    }
    for (const auto& [edge, counts] : incidence) {
        EXPECT_EQ(counts.first, 2);
        EXPECT_EQ(counts.second, 0);
    }
}

TEST_F(ImageReliefTest, PreferredCubeZeroAndNonzeroGridsHaveCompleteMappings)
{
    for (const double height : { 0., 0.03 }) {
        auto mesh = cube<image_relief::Mesh>();
        const auto input = placement(mesh);
        constexpr std::size_t width = 4;
        const std::vector<double> heights(width * width, height);
        auto prepared_mesh = mesh;
        auto patch = prepare_preferred(prepared_mesh, input);
        ASSERT_TRUE(patch);
        auto result = image_relief::make_relief_on_surface(mesh, heights, width, input.source_face, input.surface_point, { 0.12, 0.03, 0 });
        ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
        expect_mesh_connectivity(mesh);
        expect_closed_oriented(mesh);
        expect_grid_mapping(mesh, *result, width, input.source_face);
        double max_z = -1;
        for (const auto v : mesh.vertices()) {
            max_z = std::max(max_z, v.prop().pt[2]);
        }
        EXPECT_NEAR(max_z, 1 + height, 1e-8);
    }
}

TEST_F(ImageReliefTest, PublicAffineRampMapsColumnsToXAndRowsToY)
{
    constexpr std::size_t width = 4;
    std::vector<double> heights(width * width);
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t col = 0; col < width; ++col) {
            // Unequal row/column slopes distinguish transposition and reversal.
            // An affine ramp is unchanged by the existing interior smoothing.
            heights[row * width + col] = 0.004 + 0.002 * col + 0.005 * row;
        }
    }
    for (const bool fallback : { false, true }) {
        for (const std::array<double, 3> direction : { std::array<double, 3> { 0.12, 0, 0 },
                 { 0.12, 0.03, 0 }, { -0.12, -0.03, 0 }, { 0.12, 0, 0.16 } }) {
            SCOPED_TRACE((testing::Message() << "fallback=" << fallback << " direction="
                                             << Eigen::Vector3d::Map(direction.data()).transpose()));
            auto mesh = fallback ? tetrahedron<image_relief::Mesh>() : cube<image_relief::Mesh>();
            const auto face = fallback ? gpf::FaceId { 1 } : top_face(mesh);
            const auto point = centroid(mesh, face);
            const Eigen::Vector3d center = Eigen::Vector3d::Map(point.data());
            const Eigen::Vector3d normal = fallback ? Eigen::Vector3d { 1, -1, 1 }.normalized() : Eigen::Vector3d { 0, 0, 1 };
            const Eigen::Vector3d d = Eigen::Vector3d::Map(direction.data());
            const double side = 2 * d.norm();
            const Eigen::Vector3d x = (d - normal.dot(d) * normal).normalized();
            const Eigen::Vector3d y = normal.cross(x);
            auto candidate = mesh;
            auto preferred = prepare_preferred(candidate, make_placement_input(mesh, face, point, direction));
            if (fallback) {
                ASSERT_FALSE(preferred);
                EXPECT_EQ(preferred.error(), PreferredFailure::Rejected);
                EXPECT_GT(candidate.n_vertices(), mesh.n_vertices());
            } else {
                ASSERT_TRUE(preferred);
            }
            auto result = image_relief::make_relief_on_surface(mesh, heights, width, face, point, direction);
            ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
            expect_grid_mapping(mesh, *result, width, face);
            expect_closed_oriented(mesh);
            expect_mesh_connectivity(mesh);

            // Legacy's fixed-budget solve retains slight planar distortion.
            // A 2%-of-side allowance is far smaller than an axis swap or 45-degree turn.
            const double position_tolerance = fallback ? 0.02 * side : kProjectionTolerance;
            for (std::size_t row = 1; row + 1 < width; ++row) {
                for (std::size_t col = 1; col + 1 < width; ++col) {
                    const double u = static_cast<double>(col) / (width - 1);
                    const double v = static_cast<double>(row) / (width - 1);
                    const Eigen::Vector3d base = center + side * ((u - 0.5) * x + (v - 0.5) * y);
                    const Eigen::Vector3d expected = base + heights[row * width + col] * normal;
                    double nearest = std::numeric_limits<double>::infinity();
                    Eigen::Vector3d actual = Eigen::Vector3d::Zero();
                    for (const auto vertex : mesh.vertices()) {
                        const Eigen::Vector3d position = Eigen::Vector3d::Map(vertex.prop().pt.data());
                        const double distance = (position - expected).norm();
                        if (distance < nearest) {
                            nearest = distance;
                            actual = position;
                        }
                    }
                    EXPECT_LT(nearest, position_tolerance);
                    EXPECT_NEAR(normal.dot(actual - base), heights[row * width + col], 1e-8);
                }
            }
            // Every mapped triangle must occupy its independently predicted cell.
            // Boundary vertices remain on the original surface for stitching.
            const double tolerance = position_tolerance / side;
            for (const auto& entry : *result) {
                std::vector<Eigen::Vector3d> triangle;
                for (const auto he : mesh.face(entry.combined_face).halfedges()) {
                    const Eigen::Vector3d position = Eigen::Vector3d::Map(he.from().prop().pt.data());
                    const double u = 0.5 + (position - center).dot(x) / side;
                    const double v = 0.5 + (position - center).dot(y) / side;
                    EXPECT_GE(u, static_cast<double>(entry.grid_column) / (width - 1) - tolerance);
                    EXPECT_LE(u, static_cast<double>(entry.grid_column + 1) / (width - 1) + tolerance);
                    EXPECT_GE(v, static_cast<double>(entry.grid_row) / (width - 1) - tolerance);
                    EXPECT_LE(v, static_cast<double>(entry.grid_row + 1) / (width - 1) + tolerance);
                    triangle.push_back(position);
                }
                ASSERT_EQ(triangle.size(), 3);
                EXPECT_GT((triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]).dot(normal), 0);
            }
        }
    }
}

TEST_F(ImageReliefTest, MinimalGridRetainsClosedOrientationAndCompleteMapping)
{
    auto mesh = cube<image_relief::Mesh>();
    const auto input = placement(mesh);
    auto result = image_relief::make_relief_on_surface(mesh, std::vector<double>(4), 2,
        input.source_face, input.surface_point, { 0.12, 0.03, 0 });
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    expect_grid_mapping(mesh, *result, 2, input.source_face);
    expect_closed_oriented(mesh);
    expect_mesh_connectivity(mesh);
}

TEST_F(ImageReliefTest, SamplingUsesOptimizedChart)
{
    auto mesh = cube<image_relief::Mesh>();
    const auto input = placement(mesh);
    auto initial = initialize_exp_map_patch(mesh, input);
    ASSERT_TRUE(initial);
    auto optimized = optimize_patch(std::move(initial->patch));
    // Construct a distinct valid chart in the test, then call the real placement,
    // materialization, and sampling stages directly.
    optimized.coordinates *= 2;
    optimized.coordinates.rowwise() += Eigen::RowVector2d { -5, 7 };
    auto frame = exp_map_placement(optimized, input, initial->source_uv_indices);
    ASSERT_TRUE(frame);
    auto patch = make_prepared(mesh, optimized, *frame);
    EXPECT_EQ(coordinates(patch), optimized.coordinates);

    std::array<Eigen::Vector2d, 3> uv;
    std::array<Eigen::Vector3d, 3> xyz;
    int i = 0;
    for (const auto he : patch.uv_mesh.face(gpf::FaceId { 0 }).halfedges()) {
        uv[i] = Eigen::Vector2d::Map(he.from().prop().pt.data());
        xyz[i] = Eigen::Vector3d::Map(mesh.vertex_prop(patch.uv_to_surface_vertex[he.from().id.idx]).pt.data());
        ++i;
    }
    Eigen::Matrix2d basis;
    basis.col(0) = uv[1] - uv[0];
    basis.col(1) = uv[2] - uv[0];
    auto result = sample_relief_grid(mesh, patch, std::vector<double>(16), 4);
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    for (int row = 1; row <= 2; ++row) {
        for (int col = 1; col <= 2; ++col) {
            const Eigen::Vector2d q = frame->origin + (col / 3.) * frame->xaxis + (row / 3.) * frame->yaxis;
            const Eigen::Vector2d bary = basis.inverse() * (q - uv[0]);
            const Eigen::Vector3d expected = xyz[0] + bary.x() * (xyz[1] - xyz[0]) + bary.y() * (xyz[2] - xyz[0]);
            EXPECT_TRUE(result->positions.row(row * 4 + col).transpose().isApprox(expected, 1e-8));
        }
    }
}

TEST_F(ImageReliefTest, LegacyOpenPatchMaintainsGridMappings)
{
    auto mesh = open_plane<image_relief::Mesh>();
    const auto input = placement(mesh);
    auto patch = prepare_legacy(mesh, input);
    ASSERT_TRUE(patch);
    auto grid = sample_relief_grid(mesh, *patch, std::vector<double>(9), 3);
    ASSERT_TRUE(grid);
    auto boundary = project_grid_boundary(patch->uv_mesh, *grid, 3);
    ASSERT_TRUE(boundary);
    auto assembly = assemble_relief_surface(mesh, *patch, *boundary, 3);
    auto assembled = assemble_relief_grid(assembly, *grid, *boundary, 3, input.source_face);
    ASSERT_TRUE(assembled);
    auto result = finalize_relief_surface(mesh, assembly);
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    expect_grid_mapping(mesh, *result, 3, input.source_face);
    expect_mesh_connectivity(mesh);
}

TEST_F(ImageReliefTest, FailedExpMapFallsBackWithRetainedReliefSubdivision)
{
    auto mesh = tetrahedron<image_relief::Mesh>();
    const auto input = placement(mesh);
    auto direct_mesh = mesh;
    auto rejected = prepare_preferred(direct_mesh, input);
    ASSERT_FALSE(rejected);
    EXPECT_GT(direct_mesh.n_vertices(), mesh.n_vertices());
    auto selected_mesh = mesh;
    auto expected = prepare_legacy(direct_mesh, input);
    auto selected = prepare_surface_patch(selected_mesh, input);
    ASSERT_TRUE(expected);
    ASSERT_TRUE(selected);
    EXPECT_EQ(fingerprint(selected_mesh), fingerprint(direct_mesh));
    EXPECT_EQ(coordinates(*selected), coordinates(*expected));
    auto result = image_relief::make_relief_on_surface(mesh, std::vector<double>(9), 3, input.source_face, input.surface_point, { 0.12, 0.03, 0 });
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    expect_grid_mapping(mesh, *result, 3, input.source_face);
    expect_closed_oriented(mesh);
}

TEST_F(ImageReliefTest, SourceParentsAreMetadataNotNormalCacheIndices)
{
    auto mesh = cube<image_relief::Mesh>();
    for (auto face : mesh.faces()) {
        face.prop().parent = gpf::FaceId { face.id.idx + 10000 };
    }
    const auto input = placement(mesh);
    auto result = image_relief::make_relief_on_surface(mesh, std::vector<double>(9, 0.01), 3, input.source_face, input.surface_point, { 0.12, 0.03, 0 });
    ASSERT_TRUE(result) << fit_on_surface::to_string(result.error());
    expect_mesh_connectivity(mesh);
    expect_grid_mapping(mesh, *result, 3, input.source_face);
}

TEST_F(ImageReliefTest, GridValidationErrorsPreserveOriginalMesh)
{
    auto mesh = cube<image_relief::Mesh>();
    const auto input = placement(mesh);
    const auto original = fingerprint(mesh);
    auto check = [&](std::vector<double> heights, std::size_t width, SurfaceFailure failure) {
        auto result = image_relief::make_relief_on_surface(mesh, heights, width, input.source_face, input.surface_point, { 1, 0, 0 });
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error(), failure);
        EXPECT_EQ(fingerprint(mesh), original);
    };
    check({}, 1, SurfaceFailure::InvalidGridWidth);
    check({}, 2, SurfaceFailure::HeightCountMismatch);
    check({}, std::numeric_limits<std::size_t>::max(), SurfaceFailure::HeightCountMismatch);
    check(std::vector<double>(4, std::numeric_limits<double>::quiet_NaN()), 2, SurfaceFailure::MissingVertexPosition);
}

TEST_F(ImageReliefTest, DownstreamDiagnosticsNeverRetryPreparation)
{
    for (const auto& [path, failure] : std::vector<std::pair<std::string, SurfaceFailure>> {
             { "fit_polygon_on_surface_grid.off", SurfaceFailure::GridOffOpenFailed },
             { "fit_polygon_on_surface_boundary_polylines.obj", SurfaceFailure::PolylineObjOpenFailed },
             { "fit_polygon_on_surface_grid_uv_mesh.off", SurfaceFailure::UvMeshOffOpenFailed },
             { "fit_polygon_on_surface_combined.off", SurfaceFailure::MeshOffOpenFailed } }) {
        auto mesh = cube<image_relief::Mesh>();
        const auto input = placement(mesh);
        auto expected_mesh = mesh;
        if (failure == SurfaceFailure::MeshOffOpenFailed) {
            auto expected = image_relief::make_relief_on_surface(expected_mesh, std::vector<double>(9), 3,
                input.source_face, input.surface_point, { 0.12, 0.03, 0 });
            ASSERT_TRUE(expected);
        } else {
            auto expected = prepare_surface_patch(expected_mesh, input);
            ASSERT_TRUE(expected);
        }
        std::filesystem::remove_all(path);
        std::filesystem::create_directory(path);
        auto result = image_relief::make_relief_on_surface(mesh, std::vector<double>(9), 3, input.source_face, input.surface_point, { 0.12, 0.03, 0 });
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error(), failure);
        EXPECT_EQ(fingerprint(mesh), fingerprint(expected_mesh));
        std::filesystem::remove_all(path);
    }
}

TEST_F(ImageReliefTest, ErrorDescriptionsRemainStableAndNewFailureIsAppended)
{
    EXPECT_EQ(static_cast<int>(SurfaceFailure::ParameterizationFailed), static_cast<int>(SurfaceFailure::LostGeneratedFace) + 1);
    EXPECT_EQ(fit_on_surface::to_string(SurfaceFailure::InvalidStartFace), "relief start face is invalid");
    EXPECT_EQ(fit_on_surface::to_string(SurfaceFailure::WalkBoundaryReached), "walk_on_mesh_surface failed: boundary reached");
    EXPECT_EQ(fit_on_surface::to_string(SurfaceFailure::UvCoordinatesOffOpenFailed), "failed to open UV OFF output file");
    EXPECT_EQ(fit_on_surface::to_string(SurfaceFailure::ParameterizationFailed), "surface patch parameterization failed");
    for (int i = 0; i <= static_cast<int>(SurfaceFailure::ParameterizationFailed); ++i) {
        EXPECT_NE(fit_on_surface::to_string(static_cast<SurfaceFailure>(i)), "unknown surface failure");
    }
}
}
