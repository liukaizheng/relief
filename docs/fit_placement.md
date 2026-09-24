# Mathematical explanation of `fit_placement`

`relief::surface::fit_placement` in
[`src/fit_on_surface.cpp`](../src/fit_on_surface.cpp) fits a requested 2D footprint
inside an optimized UV patch. It keeps the center and shape fixed and, if
necessary, **shrinks both placement axes by the same factor**.

The main idea is to transform the footprint into a square, measure the first
contact with **every boundary segment**, reserve space for projection tolerance,
and stop slightly before contact. This works for concave patch boundaries, where
checking only the footprint's corners is insufficient.

![Normalize the footprint, find the limiting boundary, and shrink about the center.](fit_placement.svg)

The diagram magnifies the clearance for readability. The derivation below uses
the actual implementation constants.

## 1. Placement frame and footprint

Write the requested frame as

```math
o = \texttt{frame.origin},\qquad
x = \texttt{frame.xaxis},\qquad
y = \texttt{frame.yaxis}.
```

A domain point with coordinates `(u, v)` maps to UV space by

```math
\Phi(u,v) = o + ux + vy.
```

The axes are displacement vectors, not necessarily unit vectors. The placement
center is the image of domain coordinate `(0.5, 0.5)`:

```math
c = o + \frac12(x+y).
```

Let `(h_x, h_y) = footprint.half_extent`, with positive half-extents. The domain
support rectangle is

```math
D_h = \left\{(u,v):
  \left|u-\frac12\right|\le h_x,\quad
  \left|v-\frac12\right|\le h_y
\right\}.
```

For the relief grid, `h_x = h_y = 0.5`, so this is the unit square. For polygons,
`polygon_footprint` computes

```math
h_x = \max_k \left|u_k-\frac12\right|,\qquad
h_y = \max_k \left|v_k-\frac12\right|
```

from all supplied points, including unreferenced ones visited by the projector.
The rectangle also contains every segment joining these points by convexity.
Its center is always `(0.5, 0.5)`, not the polygon centroid or bounding-box center.

Define the **support matrix**

```math
S = \begin{bmatrix}h_x x & h_y y\end{bmatrix}.
```

The requested UV footprint and a uniformly scaled version are

```math
R_1 = \left\{c+Sz : \|z\|_\infty\le 1\right\},\qquad
R_s = \left\{c+sSz : \|z\|_\infty\le 1\right\}.
```

Here `0 < s <= 1`. In UV space this is a rectangle when `x` and `y` are
orthogonal, or a parallelogram when they are skew. Fitting this convex support is
conservative: it may require more shrinking than fitting only the actual polygon.

## 2. Require an interior center and an invertible support

Let `Omega` be the patch interior, with the ordered boundary vertices supplied
by `coordinates` and `boundary`. The last vertex connects back to the first.
The caller supplies a valid simple boundary and finite geometry; this function
is not a general chart validator.

First, `strictly_inside` requires

```math
c\in\Omega.
```

It explicitly rejects a center on any boundary segment, then uses a nonzero
winding-number test. Either clockwise or counterclockwise loop order works.

This check is essential: a boundary-distance calculation alone does not tell us
which side of the boundary contains the center. Since every `R_s` contains `c`,
shrinking cannot repair a center outside or on the patch boundary.

The support matrix must also pass the relative-area test

```math
|\det S| > \tau\|S\|_F^2,\qquad
\tau = 64\,\epsilon_{\mathrm{mach}},
```

where `epsilon_mach` is `std::numeric_limits<double>::epsilon()` and the squared
Frobenius norm is `support.squaredNorm()`. The determinant measures the area
spanned by the two support columns. This test rejects singular or numerically
near-singular supports, including sufficiently extreme aspect ratios. Because
both sides scale quadratically, the test is invariant under uniform resizing.

## 3. Normalize the footprint to a square

For any UV point `p`, introduce support coordinates

```math
q = S^{-1}(p-c).
```

In these coordinates,

```math
p\in R_s \quad\Longleftrightarrow\quad
\|q\|_\infty = \max(|q_x|,|q_y|)\le s.
```

Without projection clearance, the first boundary contact would occur at

```math
\rho_0 = \min_{p\in\partial\Omega}\|S^{-1}(p-c)\|_\infty.
```

Thus the relevant distance is the rectangle's **Minkowski norm**, not ordinary
Euclidean distance. Growing a centered square in support coordinates is exactly
the same as uniformly growing the requested footprint in UV space.

## 4. Convert projection tolerance into support-coordinate clearance

The code uses the UV-space distance tolerance

```math
\varepsilon = \texttt{kProjectionTolerance} = 10^{-3}.
```

Let `r_x` and `r_y` be the rows of `S^{-1}`. A UV displacement `e` satisfying
`||e||_2 <= epsilon` has normalized coordinate displacements bounded by
Cauchy-Schwarz:

```math
|(S^{-1}e)_i|
= |r_i e|
\le \|r_i\|_2\|e\|_2
\le \varepsilon\|r_i\|_2.
```

Consequently, the implementation reserves

```math
\delta_x = \varepsilon\|r_x\|_2,\qquad
\delta_y = \varepsilon\|r_y\|_2.
```

These are dimensionless quantities, computed once from the **requested**
support. They remain fixed while searching for `s`: the UV tolerance itself does
not shrink with the placement.

A conservative padded footprint is

```math
B_s = \left\{c+Sq:
  |q_x|\le s+\delta_x,\quad
  |q_y|\le s+\delta_y
\right\}.
```

Every point within Euclidean distance `epsilon` of `R_s` belongs to `B_s`. In
other words, `B_s` contains the Minkowski sum of `R_s` and the radius-`epsilon`
Euclidean disk. The coordinate bounds remain valid for skew frames because they
use inverse-row norms, not simply the reciprocals of axis lengths.

This padding is conservative: the transformed Euclidean disk is an ellipse,
and the code bounds it by a coordinate-aligned rectangle. It does not compute an
exact Euclidean offset of the footprint.

## 5. Find first contact with the entire boundary

For a boundary point in support coordinates, define

```math
f(q) = \max\left(|q_x|-\delta_x,\ |q_y|-\delta_y\right).
```

Then

```math
c+Sq\in B_s \quad\Longleftrightarrow\quad f(q)\le s.
```

The implementation's `limiting_scale` is therefore

```math
\rho = \min_{p\in\partial\Omega}
  f\left(S^{-1}(p-c)\right).
```

For any `0 <= s < rho`, the padded footprint contains no boundary point. It is
convex and contains the interior center `c`, so it cannot reach outside `Omega`
without crossing the boundary. Therefore `B_s` lies inside the patch, and the
actual footprint retains its projection clearance. At `s = rho`, the padded
footprint may touch the boundary, while the actual footprint still retains the
reserved clearance in exact arithmetic.

When `rho <= 0`, even the reserved padding at the center cannot fit strictly
inside under this conservative model, so no positive scale is accepted.

Unlike `||q||_infinity`, the clearance-adjusted function `f` is **not a norm**:
it can be negative and is not homogeneous. It is, however, convex and
piecewise linear, which makes its minimum on a segment inexpensive to compute.

Checking the entire boundary is what detects a concave notch entering between
otherwise valid footprint corners.

## 6. Minimize exactly along each boundary segment

Transform a boundary segment's endpoints to support coordinates:

```math
a = S^{-1}(p_i-c),\qquad
b = S^{-1}(p_{i+1}-c),\qquad
d = b-a.
```

Parameterize the segment with a scalar `t`:

```math
q(t)=a+td,\qquad 0\le t\le 1.
```

The segment objective is the maximum of four affine functions:

```math
g(t) = \max\left(
  a_x+td_x-\delta_x,
  -a_x-td_x-\delta_x,
  a_y+td_y-\delta_y,
  -a_y-td_y-\delta_y
\right).
```

A minimum of this continuous piecewise-linear function is attained at an
endpoint or a breakpoint. If the minimum occupies a flat interval, an endpoint
of that interval suffices. All possible breakpoints are covered by the following
candidate parameters.

### Segment endpoints

```math
t=0,\qquad t=1.
```

### Absolute-value breakpoints

These are the crossings of the positive and negative pieces for one coordinate:

```math
t=-\frac{a_x}{d_x}\quad(d_x\ne 0),\qquad
t=-\frac{a_y}{d_y}\quad(d_y\ne 0).
```

### Equalities between an x-piece and a y-piece

For each `sigma_x, sigma_y` in `{-1, +1}`, solve

```math
\sigma_x(a_x+td_x)-\delta_x
=\sigma_y(a_y+td_y)-\delta_y.
```

Writing `n = (sigma_x, -sigma_y)` gives

```math
t = \frac{\delta_x-\delta_y-n\cdot a}{n\cdot d},
\qquad n\cdot d\ne 0.
```

This is the formula used by the nested sign loops in the implementation.
Candidates outside `[0, 1]` are ignored. Zero denominators are skipped: parallel
pieces have no isolated crossing, and coincident pieces require no additional
candidate.

The code evaluates the original `g(t)` at every retained candidate. Some
crossings are not active breakpoints of the maximum, but including these extra
points is harmless. There are at most **eight evaluations per segment**: two
endpoints, two absolute-value breakpoints, and four cross-coordinate equalities.

Taking the smallest value over every segment produces `rho`. The minimization
is exact in real arithmetic, up to floating-point evaluation in the code; there
is no boundary sampling, binary search, or iterative scale optimization. For a
boundary with `m` segments, the fitting calculation takes `O(m)` time and `O(1)`
extra storage.

## 7. Cap the scale and reconstruct the frame

The chosen scale is

```math
s = \min(1,\rho).
```

The cap prevents enlargement. Projection clearance is already included in
`rho`; no additional shrink factor is applied. A request is unchanged whenever
it fits with that clearance:

```math
s=1 \quad\Longleftrightarrow\quad \rho\ge 1.
```

When `0 < rho < 1`, the padded footprint reaches the limiting boundary while
the actual footprint retains the reserved projection clearance.

Both axes are scaled, and the origin is reconstructed around the original
center:

```math
x'=sx,\qquad y'=sy,\qquad
o'=c-\frac12(x'+y').
```

Equivalently, every domain point transforms by

```math
\Phi'(u,v)=c+s\bigl(\Phi(u,v)-c\bigr).
```

This identity proves that the operation preserves the center, axis directions,
angle between axes, and aspect ratio. The stored origin usually changes, but
there is no independent translation of the placement. Positive scaling also
preserves orientation because the frame determinant is multiplied by `s^2`.

Finally, the code requires the shorter fitted support half-axis to exceed the
projection tolerance:

```math
\min\left(h_x\|x'\|_2,\ h_y\|y'\|_2\right)>\varepsilon.
```

This is a minimum-size rejection, separate from boundary containment and the
relative-area test. It prevents returning a footprint too small relative to the
projector's tolerance.

## 8. Worked example: a concave notch

The regression test
`SurfacePatchTest.ShrinkingUsesEntireConcaveBoundaryAndKeepsCenterAndAspect`
uses the boundary

```text
(-2,-2), (2,-2), (2,2), (0.2,2),
(0.2,0.4), (-0.2,0.4), (-0.2,2), (-2,2).
```

The requested frame is

```math
o=(-1,-1),\qquad x=(2,0),\qquad y=(0,2),\qquad
h_x=h_y=\frac12.
```

Thus `c = (0,0)`, `S = I`, and `delta_x = delta_y = 0.001`. All four requested
corners are inside the patch, but the notch crosses the requested top edge.
The bottom of the notch is the segment from `(0.2,0.4)` to `(-0.2,0.4)`.
Every point on that segment satisfies

```math
f(q)=\max(|q_x|-0.001,\ 0.4-0.001)=0.399.
```

No other boundary segment has a smaller value. Therefore

```math
\rho=0.399,\qquad
s=\min(1,0.399)=0.399.
```

The fitted frame is

```math
x'=(0.798,0),\qquad
y'=(0,0.798),\qquad
o'=(-0.399,-0.399).
```

Its top edge is at `y = 0.399`, leaving exactly the `0.001` projection clearance
below the notch. Its center remains exactly `(0,0)` in the mathematical model.

## 9. Failure conditions and pipeline scope

`fit_placement` returns `SurfaceFailure::ParameterizationFailed` when:

1. The center is on or outside the boundary.
2. The support fails the relative determinant test.
3. The computed final scale is nonfinite or nonpositive.
4. A fitted support half-axis is no longer larger than the projection tolerance.

These targeted checks are not general input validation. The routine assumes
valid indices, finite geometry, positive footprint dimensions, and a simple
ordered patch boundary; it performs no chart-wide finiteness or topology check.
The containment argument relies on these assumptions, and floating-point
arithmetic is not a formal geometric certificate.

In the current pipeline, `exp_map_placement` constructs the requested frame
from the optimized center and tangent, then calls `fit_placement` with the
**optimized** coordinates. The mapped tangent defines local +X, with nominal
axis lengths `2 * norm(direction)` using the original 3D magnitude. Domain
coordinate `(0.5, 0.5)` stays at the optimized center. The fitting derivation is
orientation-independent: uniform shrinking preserves these axis directions as
well as the center and aspect ratio. The function changes only the returned
frame; it does not move UV vertices, rerun SLIM, enlarge the patch, or search for
another center.
A preferred placement rejection can trigger the legacy provider, whose separate
anchor-based sizing does not call `fit_placement`.

For the surrounding workflow, see
[`surface_patch_pipeline.md`](surface_patch_pipeline.md). The relevant regression
tests in [`test_surface_patch.cpp`](../tests/surface/test_surface_patch.cpp) include:

- `ShrinkingUsesEntireConcaveBoundaryAndKeepsCenterAndAspect`: the notch example
  above and preservation of center and shape.
- `PlacementKeepsSizeWhenSafeAndUsesOptimizedNotInitialBoundary`: unchanged safe
  requests, shrinking against updated coordinates, and rejection of a collapsed
  patch or an outside center.
- `PlacementKeepsSizeWhenProjectionClearanceFits`: a request close to the
  boundary stays unchanged when the projection clearance fits.
- `LegacyAnchorFrameUsesProvidedScaleWithoutExtraShrink`: legacy anchor sizing
  uses the supplied scale without an additional shrink factor.
- `PreferredUnusablePlacementFallsBackWithRetainedSubdivision`: rejection of an
  extremely elongated support and recovery through the legacy path.
