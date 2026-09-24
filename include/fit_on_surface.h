#pragma once

#include <gpf/detail.hpp>
#include <gpf/ids.hpp>
#include <gpf/manifold_mesh.hpp>
#include <gpf/mesh.hpp>

#include <array>
#include <cstddef>
#include <expected>
#include <span>
#include <string_view>
#include <vector>

namespace fit_on_surface {
/** @brief Operational failures shared by polygon fitting and image relief. */
enum class SurfaceFailure {
    // Relief validation.
    InvalidGridWidth,
    HeightCountMismatch,
    InvalidStartFace,
    DegenerateStartFace,
    ZeroDirection,
    MissingTangentComponent,

    // GPF projection and surface walking.
    ProjectionPathNotFound,
    ProjectionInvalidTriangleIndex,
    ProjectionConstraintConflict,
    WalkBoundaryReached,
    WalkIterationLimitExceeded,
    WalkDegenerateDirection,
    WalkInvalidPath,
    WalkDegenerateStep,

    // UV and indexing checks.
    InvalidAnchorIndex,
    ConflictingPolygonLabels,
    MissingBoundarySeparator,
    InvalidMeshVertexReference,
    InvalidUvVertexReference,
    InvalidPolylinePointIndex,
    GridMatrixDimensionMismatch,

    // Diagnostic output (both surface mesh writers share MeshOffOpenFailed).
    SurfaceOffOpenFailed,
    MeshOffOpenFailed,
    UvCoordinatesOffOpenFailed,
    GridOffOpenFailed,
    UvMeshOffOpenFailed,
    PolylineObjOpenFailed,

    // Relief finalization.
    DegeneratePolygon,
    ZeroLengthTopologicalEdge,
    NonManifoldEdge,
    ConflictingEdgeOrientation,
    MissingVertexPosition,
    FacePropertyCountMismatch,
    LostGeneratedFace,

    // Patch initialization or placement fitting failed.
    ParameterizationFailed,
};

/** @brief Return a stable, printable description of an operational failure. */
[[nodiscard]] std::string_view to_string(SurfaceFailure failure) noexcept;

struct VertexProp {
    std::array<double, 3> pt;
    double angle_sum {};
};

/** @brief Derived exp-map caches shared by both public surface mesh types.
 * Preparation recomputes these and VertexProp::angle_sum before every exp-map
 * attempt. Later geometry/topology edits can invalidate them, including edits
 * within these APIs; they are not guaranteed-valid output metadata.
 * The enriched mesh layouts require dependents to rebuild.
 */
struct HalfedgeProp {
    double angle {};
    double signpost_angle {};
    std::array<double, 2> vector {};
};

struct EdgeProp {
    double len {};
};

struct FaceProp {
    gpf::FaceId parent;
    std::size_t polygon_id = gpf::kInvalidIndex;
};

using Mesh = gpf::ManifoldMesh<VertexProp, HalfedgeProp, EdgeProp, FaceProp>;

/**
 * @brief Fit polygons onto the surface mesh in place.
 * Initializes a local patch with exp-map and optimizes it with SLIM.
 * The projection of direction onto the source-face tangent plane defines local
 * +X (increasing polygon u); local +Y follows the face orientation (increasing v).
 * The nominal unit-square side is 2 * norm(direction), using the original 3D
 * magnitude even when direction has a normal component. Domain coordinate
 * (0.5, 0.5) remains the placement center. The optimized placement is uniformly
 * shrunk about that center if needed.
 * Preparation mutates this mesh directly. Recoverable preferred rejections run
 * the legacy walk/harmonic plus SLIM fallback on the same mesh, retaining its
 * subdivisions. Legacy reuses a nearby vertex of the resolved walk-start triangle
 * as its center without projecting it again, including an original corner.
 * Direction-aware walk starts use the captured source triangle and live adjacency.
 * Exp-map projection failure instead returns ParameterizationFailed without
 * fallback, because the projection can leave incomplete topology.
 * @pre The mesh is a closed, orientable triangular manifold with nondegenerate faces.
 * This is required by exp-map and is not checked before calling it.
 * Mesh positions, polygon points, and surface_point contain finite values.
 * face_idx names an active face. direction is finite and nonzero, with a nonzero
 * tangent component on that face. Placement setup does not validate these inputs.
 * @return An empty success value, or a SurfaceFailure for an operational failure.
 * Inspect the result before continuing; to_string() provides a diagnostic.
 * @warning No chart-wide validation is performed before or after SLIM.
 * All preparation and post-selection stages are nontransactional: even failed
 * preparation can leave mutations, including subdivisions outside the patch.
 * The public exp-map fields change mesh layout/ABI and require dependents to
 * rebuild. They are derived caches recomputed before each preferred attempt;
 * later projection, subdivision, or relief edits can invalidate them.
 * The unchanged SLIM solver can write uv_new.obj during preparation. Failures after
 * selection do not retry another geometry path. Diagnostic-file failures
 * are also returned, including those after the completed mesh update.
 * Allocation failures and third-party exceptions are not converted to errors.
 */
[[nodiscard]] std::expected<void, SurfaceFailure> fit_polygon_on_surface(
    fit_on_surface::Mesh& mesh,
    const std::vector<std::array<double, 2>>& polygon_points,
    const std::vector<std::vector<std::vector<std::size_t>>>& polygons,
    const std::array<double, 3>& surface_point,
    const gpf::FaceId face_idx,
    const std::array<double, 3>& direction);
}

namespace image_relief {
struct VertexProp {
    std::array<double, 3> pt;
    double angle_sum {};
};

struct FaceProp {
    gpf::FaceId parent;
};

struct GridFaceIndex {
    gpf::FaceId combined_face;
    std::size_t grid_face_index;
    std::size_t grid_row;
    std::size_t grid_column;
    std::size_t triangle_index;
};

using HalfedgeProp = fit_on_surface::HalfedgeProp;
using EdgeProp = fit_on_surface::EdgeProp;
using Mesh = gpf::ManifoldMesh<VertexProp, HalfedgeProp, EdgeProp, FaceProp>;

/**
 * @brief Build image relief on the surface mesh in place.
 * Uses exp-map initialization followed by SLIM, uniformly shrinking the centered
 * grid placement to fit if necessary. Preparation mutates this mesh directly.
 * Recoverable preferred rejections run walk/harmonic plus SLIM fallback on the
 * same mesh, retaining its subdivisions. Legacy reuses a nearby vertex of the
 * resolved walk-start triangle as its center without projecting it again,
 * including an original corner. Direction-aware walk starts use the captured
 * source triangle and live adjacency. Exp-map projection failure returns
 * ParameterizationFailed without fallback because the projection can leave
 * incomplete topology. The projection of direction onto the source-face tangent
 * plane defines local +X (increasing columns); local +Y follows the face
 * orientation (increasing rows). The nominal grid side is 2 * norm(direction),
 * using the original 3D magnitude even when direction has a normal component.
 * Domain coordinate (0.5, 0.5) remains the placement center.
 * @pre The mesh is a closed, orientable triangular manifold with nondegenerate faces.
 * This is required by exp-map and is not checked before calling it.
 * Mesh positions and start_pt contain finite values.
 * face_idx names an active face. direction is finite and nonzero, with a nonzero
 * tangent component on that face. Placement setup does not validate these inputs.
 * @return On success, mappings from generated grid triangles to combined mesh
 * faces; otherwise, a fit_on_surface::SurfaceFailure. Inspect the result before
 * continuing; fit_on_surface::to_string() provides a diagnostic.
 * @warning No chart-wide validation is performed before or after SLIM.
 * All preparation and post-selection stages are nontransactional: even failed
 * preparation can leave mutations, including subdivisions outside the patch.
 * The public exp-map fields change mesh layout/ABI and require dependents to
 * rebuild. They are derived caches recomputed before each preferred attempt;
 * later projection, subdivision, or relief edits can invalidate them.
 * The unchanged SLIM solver can write uv_new.obj during preparation. Failures after
 * selection do not retry another geometry path. Diagnostic-file failures
 * are also returned, including those after the completed mesh is installed.
 * Allocation failures and third-party exceptions are not converted to errors.
 */
[[nodiscard]] std::expected<std::vector<GridFaceIndex>, fit_on_surface::SurfaceFailure> make_relief_on_surface(
    Mesh& mesh,
    const std::span<const double> heights,
    const std::size_t width,
    const gpf::FaceId face_idx,
    const std::array<double, 3>& start_pt,
    const std::array<double, 3>& direction);
}
