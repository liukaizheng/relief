# Local surface patch preparation

`fit_polygon_on_surface` and `make_relief_on_surface` share the private pipeline in
`src/fit_on_surface.cpp`. Patch preparation, polygon fitting, relief assembly,
mesh helpers, and diagnostics all stay in that one implementation file. Their
public signatures and mesh property layouts remain in `include/fit_on_surface.h`.

Both public entry points require a closed, orientable triangular manifold with
nondegenerate faces, as required by upstream `gpf::exp_map`. There is no topology
eligibility scan before the preferred attempt. Open or nontriangular meshes are
outside this contract; they are not automatically routed to the legacy path and
are not guaranteed to return a failure safely. The supplied face ID must name an
active face, and the direction must be finite and nonzero with a nonzero tangent
component on that face. Placement setup assumes these inputs are valid; it does
not check the source face, direction, footprint, or computed support radius.

Geometry, polygon points, placement coordinates, and computed charts are assumed
finite. There are no matrix/vector-wide finiteness checks, so NaN/Inf geometry or
solver output is not guaranteed to produce a `SurfaceFailure`. Existing scalar
checks on heights and selected downstream arithmetic results remain.

## Selection and optimization

1. Capture the original source triangle, its normal, the normalized tangent
   direction, the original direction magnitude, and the domain footprint without
   placement validation. Polygon fitting still validates polygon rings and point
   indices; relief validates grid dimensions and finite heights.
2. Prepare directly on the caller's public surface mesh, without a full-mesh
   copy, success-only transfer, or replacement. Recompute edge lengths, corner
   angles, vertex angle sums, signposts, and halfedge vectors before every
   preferred attempt, including attempts after earlier geometry/topology edits.
3. Run `gpf::exp_map` once with support radius
   `1.2 * sqrt(2) * norm(direction)`. Put `ExpMapResult::boundary_vertex_ids` first
   in the patch's vertex arrays, preserving the supplied loop order; append the
   interior vertices in source order. Copy each vertex's position and actual
   exp-map UV into its new row. A capacity-indexed source-to-local lookup remaps
   triangle indices and the center, while the boundary becomes `[0, boundary_count)`.
   `InitializedExpMapPatch` retains the three original source vertices' local
   indices from this same lookup, in captured source-triangle order, alongside
   the `InitialPatch`. Invalid, out-of-range, or absent source IDs leave invalid
   index sentinels; they do not reject initialization. The full lookup is then
   discarded: no second table, dynamic allocation, or mapping reorder is added.
   Source-mesh IDs and face ordering are unchanged. Use the center returned by
   exp-map without a separate source-triangle compatibility check.
4. Normalize the initializer's orientation using its first triangle, reflecting
   the V coordinate if that triangle has negative UV orientation. Consistent face
   winding, nondegenerate geometry, valid dimensions, indices, and mappings, and a
   simple boundary are assumed rather than checked by the common SLIM stage.
5. Pass the **actual exp-map coordinates** to `FlattenSurface`. Both providers
   use the existing `slim_solve(5, 15)` and `n_bnd_points = 1`. Both supply
   boundary-first rows, so local vertex zero is a boundary translation anchor.
   The center and all remaining boundary vertices stay free; this does not pin
   the whole boundary. `flatten_surface.cpp` and its public header are left
   unchanged. The solver returns `void`, does not expose Eigen factorization or
   solve status, and always writes its best-effort `uv_new.obj` dump. The patch
   layer consumes `solver.uv` directly, with no chart-wide validation before or
   after SLIM. It cannot detect failures hidden by the solver or suppress that
   intermediate dump.
6. Fit placement using the optimized coordinates and the three cached indices.
   These indices belong to the same patch ordering and captured `PlacementInput`;
   SLIM preserves row ordering. Before using each row, placement checks its bound
   against `uv_to_surface_vertex` and verifies the corresponding source ID. This
   replaces the three mapping-wide searches with constant-time checks. Missing
   or mismatched source vertices still reject placement **after SLIM**, preserving
   the solver dump, retained mesh mutations, and preferred-to-legacy fallback.
   Select a successful preferred result without doing any legacy preparation.
   Preferred mutations remain in place whether or not the result is selected.

Internal stages rely on established contracts rather than repeating the same
checks: successful `exp_map` results supply live triangular faces and aligned
vertex/UV arrays; placement consumes caller-supplied input and an optimized chart;
and patch mappings refer directly to live caller-mesh IDs. Geometric checks on
optimized placement, including availability of the original source vertices' UVs,
remain in place, but malformed initial or optimized charts are not guaranteed to
produce a `SurfaceFailure`.

A preferred disk-validation or optimized-placement rejection invokes
walk/anchor projection, extraction, and harmonic initialization on the **same
mutated mesh**. A retained center on the resolved walk-start triangle can be
reused along with its surrounding subdivision; a subdivision on an unrelated
nearby sheet also remains, even if outside the selected patch. There is no topology
rollback. The exception is `gpf::ExpMapFailure::ProjectionFailed`: projection may have left incomplete
topology, so preparation returns the existing
`SurfaceFailure::ParameterizationFailed` immediately, without legacy fallback.
This distinction stays private; no public error enumerator is added.

Fallback keeps the original `PlacementInput`: source vertex IDs and positions,
normal, direction, and point are not recaptured from a split face. The public
input defines the local +X direction `x`, the normalized source tangent, with
`y = normal × x`. The four internal **corner rays** follow normalized `x-y`,
`x+y`, `-x+y`, and `-x-y`, in that order. Each anchor is walked to distance
`sqrt(2) * norm(direction)`; outer support corners retain the factor `1.2`.
Thus, on an undistorted plane, the corner offsets are `(m,-m)`, `(m,m)`,
`(-m,m)`, and `(-m,-m)` in local X/Y, where `m = norm(direction)`.
For each of the four rays, a local resolver searches live adjacency within the
captured source triangle to find a containing child. The saved source-face ID is
only a search seed, not necessarily a safe walk start; `FaceProp::parent` is never
used for this search. At a subdivision edge or center, different rays can start
in different triangles. The resolver crosses zero-distance incident edges as needed,
transporting the tangent across noncoplanar edges while preserving the direction
within coplanar subdivisions. Edge-aligned rays use deterministic incident-edge
ordering and a forward-exit check compatible with GPF's walk-start search. Visited
faces and directed edges bound traversal; invalid, degenerate, boundary, or
unresolved starts return existing walk failures. Only the resolved face, point,
and tangent are passed to the unchanged GPF walker.

Legacy initialization looks for an existing center among the vertices of the
first resolved walk-start triangle. It chooses the nearest vertex within the
projection tolerance, breaking distance ties by vertex ID; an original triangle
corner is eligible too. There is no additional original-triangle neighborhood,
barycentric, or requested-point compatibility check on that vertex. This local
recovery also works when exp-map disk validation failed without returning the
retained center ID; it does not search unrelated sheets or assume the last vertex
is the center. When a nearby vertex exists on the resolved start triangle, its
live ID is reused directly: only the four outer corners and four anchor corners
are projected. Otherwise, as in standalone legacy preparation or a nearby-sheet
rejection with no nearby source-side vertex, the center point is still projected
between those groups. The boundary loop keeps
its original order, and extraction explicitly assembles anchors as the center
followed by the four corners in both cases.

The legacy initializer finds boundary halfedges in its local UV mesh, starts at the smallest
boundary vertex, and follows `prev()` until the loop closes. It checks the expected
boundary size, coverage of all boundary halfedges, referenced vertex count, and
disk Euler characteristic before harmonic initialization. This harmonic helper
itself does not require boundary-first numbering, but the legacy extractor already
supplies boundary-first rows and consistently mapped walk anchors for SLIM. No
additional legacy permutation is needed. Failed initialization returns no partial
boundary data.

The legacy provider uses the same unchecked SLIM stage but keeps its original
post-SLIM anchor selection and sizing formulas. Its containment and frame
reconstruction still use 45-degree rotations and square-root-of-two factors to
convert actual corner vectors into edge axes; those diagonals are internal, not
the public direction convention. Extracted vertex ordering and walk anchors
remain private to the legacy provider. Exp-map supplies its own disk
topology and ordered boundary; the shared optimizer does not analyze connectivity
or vertex links for either provider.

`SurfaceFailure::ParameterizationFailed` reports detected initialization or
placement-fitting failure on the legacy path, and the terminal exp-map projection
failure described above. Recoverable preferred rejections are not returned in
place of the legacy path's own result.

## Optimized placement

The direction defines **local +X** after projection onto the source-face tangent
plane, not a diagonal or a fixed world axis. Its original 3D magnitude sets the
nominal unit-square side `2 * norm(direction)`, even with a normal component;
only its tangent component sets orientation. Arbitrary and reversed tangent
directions are supported. With the optimized center row `c`, the normalized
source tangent mapped through the source triangle's **optimized** UVs `t`, its
positive perpendicular `b = (-t.y, t.x)`, and `m = norm(direction)`:

```
xaxis  = 2 * m * t
yaxis  = 2 * m * b
origin = c - 0.5 * (xaxis + yaxis)
```

Domain coordinate `(0.5, 0.5)` remains the placement center. Polygon coordinate
`u` and relief columns advance along `xaxis`; polygon coordinate `v` and relief
rows advance along `yaxis`. The square's circumradius is still `sqrt(2) * m`, so
the exp-map support radius remains `1.2 * sqrt(2) * m`. This removes the previous
45-degree offset; callers that compensated for it should remove that rotation.
On curved or distorted charts, placement still follows the provider's local
mapping rather than promising global world-axis alignment. Changing the footprint
orientation can change boundary contact and the final fitted size, even though
the nominal side length is unchanged.

A centered convex support rectangle covers the entire grid or the polygon
points/segments (including unreferenced points, which the projector also visits).
The limiting uniform scale is computed against **every optimized boundary
segment**, including concave intrusions between valid corners. In support
coordinates this minimizes a piecewise-linear rectangle-distance function over
each segment. It reserves the projector's distance tolerance without an
additional shrink factor. A request that fits with this clearance is unchanged;
otherwise both axes contract equally about the same center. No enlargement,
translation, anisotropic scaling, radius retry, or second solve is performed.
A numerically collapsed or tolerance-incompatible fit rejects the preferred path.

## Public mesh properties and mutation

Both public surface meshes now store zero-initialized vertex `angle_sum`, halfedge
`angle`, `signpost_angle`, and two-component `vector`, and edge `len` fields.
Halfedge and edge property definitions are shared. Position access stays `.pt`;
the distinct face properties are unchanged, with polygon labels only on the
polygon-fitting mesh. The separate 2D UV mesh layout is unchanged.

These extra fields change public layout/ABI and increase persistent per-element
storage, so dependents must rebuild. They are derived caches, not guaranteed-valid
output metadata: every preferred attempt recomputes them, but later projection,
subdivision, or relief construction can invalidate them again. Removal of temporary
mesh copies alone does not establish lower memory use or a benchmark improvement.

All preparation is nontransactional. Failed preferred preparation leaves its
mutations visible even when fallback also fails or projection failure stops
selection altogether. Fallback topology, IDs, and UV ordering can differ from
legacy preparation on a pristine mesh. The original face provenance and polygon
labels still inherit through subdivision; they do not identify live source
triangles for walking.

## Diagnostics

Once a chart is selected, diagnostic errors and downstream polygon/relief errors
never trigger another geometry attempt or whole-operation rollback. Allocation
failures and third-party exceptions are not converted into operational errors.

Selected diagnostics run exactly once after successful selection. The selected
patch writes `fit_polygon_on_surface_uv.off`,
`fit_polygon_on_surface.off`, and rewrites the auxiliary `uv_new.obj`. After
successful diagnostics, both UV files contain the selected optimized coordinates
used by the consumers. The existing OFF diagnostics remain checked and are never
written for a rejected preparation. The auxiliary OBJ retains the solver's original
best-effort behavior: tentative solves can write it, and a later preparation or
diagnostic failure can leave that intermediate dump in place.

Relief's normal cache is indexed by live source-face IDs, not by the inherited
`parent` metadata. At projected grid corners, a snapped `A -> B -> A` spur is
cancelled before triangulation, matching boundary flood-fill cancellation and
avoiding a non-simple input to the upstream triangulator. Neither change alters
public `GridFaceIndex` semantics.

## Tests and sanitizer builds

The reusable `relief_surface` library is linked by `relief_app`. The three
white-box GoogleTest binaries include `src/fit_on_surface.cpp` directly and
compile the SLIM solver alongside it, keeping private stages testable without
splitting the implementation or exposing internal headers. They do not also
link a second copy of the surface implementation from the library. CTest assigns
separate build-tree working directories; each fixture further isolates its
diagnostics by test name.

```sh
rtk cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
rtk cmake --build build/debug --target relief_app test_surface_patch test_fit_on_surface test_image_relief
rtk ctest --test-dir build/debug --output-on-failure --timeout 120

rtk cmake -S . -B build/asan -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON -DRELIEF_ENABLE_SANITIZERS=ON
rtk cmake --build build/asan --target test_surface_patch test_fit_on_surface test_image_relief
rtk ctest --test-dir build/asan --output-on-failure --timeout 120
```

Tests call the existing private stages directly and compare returned charts,
mesh topology, properties, and diagnostic output; production preparation has no
test callbacks, strategy tags, or global test state. Optimizer tests cover the
one-boundary-vertex gauge, orientation normalization, and consumption of the solver's
actual output. Exp-map tests verify boundary-first rows and consistent UV, triangle,
center, and source mappings, including sparse IDs and a lowest-ID interior center.
The three cached placement indices are compared with a test-only reference search,
including all source-triangle permutations and absent, invalid, and out-of-range
source IDs in every slot. Placement rejects out-of-range local indices and stale
source correspondences with `ParameterizationFailed` before using their optimized
UV rows.
Boundary-walk tests include an interior vertex zero, mismatched boundary sizes,
and multiple loops. Existing fallback expectations use actual topology and
placement failures, comparing preferred rejection followed by legacy preparation
on the same reference mesh. Placement and nearby-sheet rejection expectations
retain preferred mutations; diagnostic error checks still ensure no retry. The
nearby-sheet case rejects missing source-vertex UVs during placement after SLIM,
so it expects the solver's auxiliary dump rather than a pre-SLIM rejection.
The sparse/edit scenario uses native mesh properties rather than testing the
removed transfer mechanism. Fixtures include a three-times-subdivided closed cube,
a curved spherical projection of that cube, sparse edited topology, and the coarse
tetrahedron that mutates before exp-map rejects its patch. Open-plane fixtures
exercise the private legacy stages directly, not the public entry points.

Orientation regressions check signed X/Y axes for arbitrary and reversed inputs,
normal-component sizing, tilted source planes, rotated optimized charts, and
uniform shrinking. Legacy tests verify the ordered corner walks and reconstruct
all four anchor choices. Asymmetric polygon landmarks and an affine relief ramp
check both public APIs against independently calculated planar positions through
preferred and fallback preparation, including grid-cell mappings and closed,
consistent relief orientation. Fallback position checks allow small residual
legacy SLIM distortion; the frame and corner-walk checks remain tight.

The cached-index regressions do not add dedicated automated coverage for the
direction-aware start resolver's edge/vertex behavior or the projection-failure
exception; existing tests and code review do not eliminate those residual
edge-case risks.
