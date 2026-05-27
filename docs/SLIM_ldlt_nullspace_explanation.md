# SLIM Normal Equation Nullspace and SimplicialLDLT NumericalIssue

This note explains why `FlattenSurface::slim_solve()` can produce

```text
SimplicialLDLT::info() == Eigen::NumericalIssue
```

when no UV vertex is fixed, even though all entries of the assembled matrix are
finite and the diagonal entries of the normal matrix are positive.

The short version:

```text
The free SLIM global step is translation-invariant in UV.
Therefore the normal matrix L = A^T M A is positive semidefinite, not positive
definite. Eigen's SimplicialLDLT can hit an exact zero Schur-complement pivot.
Fixing one vertex removes the two translation modes and makes the solve succeed.
```

## The Global Step as a Least-Squares Problem

In one local/global SLIM iteration, the rotations and weights are fixed from the
current UV map. The global step solves a weighted least-squares problem of the
form

```math
\min_x \frac{1}{2} \| A x - b \|_M^2
```

where

```math
\| y \|_M^2 = y^T M y.
```

In the code:

- `x` is the stacked free UV vector.
- The first block of `x` stores all free `u` coordinates.
- The second block of `x` stores all free `v` coordinates.
- `A` is the weighted gradient/Jacobian system assembled by
  `fill_system_values()`.
- `M` is the diagonal area-weight vector derived from `DDA`.
- `b` is the current local target assembled into `rhs`.

The normal equation is

```math
L x = g
```

with

```math
L = A^T M A,
\qquad
g = A^T M b.
```

This is exactly what the code computes:

```cpp
normal_cache.assign(A, DDA, L);
multiply_normal_rhs(A, DDA, rhs, real_rhs);
```

So `real_rhs` is `g`.

## Why Free UV Translation Is a Nullspace

Each row of `A` is built from derivatives of triangle basis functions. For a
triangle with barycentric basis functions

```math
\phi_0,\ \phi_1,\ \phi_2,
```

we always have

```math
\phi_0 + \phi_1 + \phi_2 = 1.
```

Taking gradients gives

```math
\nabla \phi_0 + \nabla \phi_1 + \nabla \phi_2 = 0.
```

Therefore, if every vertex UV is translated by a constant vector

```math
(u_i, v_i) \mapsto (u_i + c_u, v_i + c_v),
```

then every triangle Jacobian is unchanged:

```math
\nabla(u + c_u) = \nabla u,
\qquad
\nabla(v + c_v) = \nabla v.
```

Equivalently, let

```math
e_u =
\begin{bmatrix}
1 \\
\vdots \\
1 \\
0 \\
\vdots \\
0
\end{bmatrix},
\qquad
e_v =
\begin{bmatrix}
0 \\
\vdots \\
0 \\
1 \\
\vdots \\
1
\end{bmatrix}.
```

Here `e_u` translates all free `u` values, and `e_v` translates all free `v`
values. If no vertices are fixed, then

```math
A e_u = 0,
\qquad
A e_v = 0.
```

So

```math
L e_u = A^T M A e_u = 0,
\qquad
L e_v = A^T M A e_v = 0.
```

Thus `L` has at least two zero eigenvalues.

For a connected mesh, these are the expected two UV translation null modes. For
a mesh with multiple disconnected components, there are two translation modes
per connected component unless each component is anchored.

## Why the RHS Is Still Compatible

Even though `L` is singular, the normal equation is still consistent because

```math
g = A^T M b.
```

For any null vector `z` satisfying `A z = 0`,

```math
z^T g
= z^T A^T M b
= (A z)^T M b
= 0.
```

So the RHS lies in the range of `L`. This means solutions exist, but they are not
unique. If `x` is one solution, then

```math
x + \alpha e_u + \beta e_v
```

is also a solution for any scalars `alpha` and `beta`.

Your debug output confirms this:

```text
||L*u_translation|| = 1.32883e-13
||L*v_translation|| = 1.46417e-13
rhs_dot_u_translation = -1.77636e-15
rhs_dot_v_translation = 1.63758e-15
```

Those values are effectively zero at this scale. Therefore the matrix and RHS
are behaving like the mathematically expected singular compatible system.

## Why Positive Diagonal Entries Do Not Prove Positive Definiteness

The debug output also shows

```text
L diagonal: min=5.44896 max=1020.44 negative=0 near_zero=0
```

This does not contradict singularity.

A symmetric matrix can have strictly positive diagonal entries and still be
singular. A simple example is

```math
\begin{bmatrix}
1 & -1 \\
-1 & 1
\end{bmatrix}.
```

Both diagonal entries are positive, but

```math
\begin{bmatrix}
1 & -1 \\
-1 & 1
\end{bmatrix}
\begin{bmatrix}
1 \\
1
\end{bmatrix}
=
\begin{bmatrix}
0 \\
0
\end{bmatrix}.
```

This is the same structure as a gradient Laplacian: constants are in the
nullspace even though each vertex has positive diagonal weight.

## What the LDLT Failure Means

Eigen's sparse LDLT factorization computes a decomposition of a permuted matrix:

```math
P L P^T = \tilde{L}
```

and factors it as

```math
\tilde{L} = L_f D L_f^T.
```

During factorization, each diagonal entry of `D` is a Schur-complement pivot.
Eigen reports `NumericalIssue` when an LDLT pivot is exactly zero.

Your debug output says

```text
AMD LDLT D: first_exact_zero_pivot=1142
AMD failing pivot: permuted_col=1142 D=0 original_from_Pinv=917 v(vertex=345)
```

This does not mean that the original matrix entry

```math
L_{917,917}
```

is zero. In fact the original row is normal:

```text
candidate row: row=917 v(vertex=345)
diagonal=13.6337
row_sum=-8.88178e-16
v_block_sum=-8.88178e-16
```

The exact zero occurs after previous variables have been eliminated. In other
words, `v(vertex=345)` is where the AMD-ordered factorization finally exposes
the translation nullspace in the Schur complement.

## Why Natural Ordering Succeeds

The same run reported

```text
natural-order LDLT compute info: Success
```

This does not mean the matrix is nonsingular. It only means that this particular
ordering did not produce an exact zero pivot according to Eigen's floating-point
test.

For a semidefinite matrix, different elimination orderings can produce different
roundoff behavior. One ordering may hit an exact zero pivot, while another may
produce a tiny nonzero pivot due to floating-point perturbation.

So using `NaturalOrdering` as a fallback can be useful for debugging, but it is
not a principled fix. The mathematical system is still singular.

## Why Fixing the First Vertex Works

If vertex 0 is fixed, then the free variable vector no longer contains both
translation directions.

A global translation would require changing vertex 0:

```math
(u_0, v_0) \mapsto (u_0 + c_u, v_0 + c_v).
```

But fixed vertices are removed from the unknown vector and moved to the RHS.
Therefore the vectors `e_u` and `e_v` are no longer valid free-variable
directions.

For a connected, nondegenerate mesh, this removes the two translation null modes.
Then `A` has full column rank in the free space, and

```math
L = A^T M A
```

becomes positive definite enough for `SimplicialLDLT`.

This matches your observation:

```text
Fixing the first vertex makes the solve succeed.
```

If the mesh has multiple connected components, one fixed vertex per component is
needed to remove all component-wise translation modes.

## Recommended Fixes

### Preferred: Explicit Gauge Constraint

For a connected patch, fix one UV vertex:

```text
u_0 and v_0 fixed
```

This removes exactly the two translation degrees of freedom.

In the current code structure, this fits naturally with `n_bnd_points` because
fixed vertices are assumed to be at the front of the vertex array and are
excluded from the free unknown vector.

### Alternative: Zero-Mean Gauge

Instead of pinning a vertex, enforce

```math
\sum_i u_i = 0,
\qquad
\sum_i v_i = 0.
```

This keeps the solution centered and avoids privileging a particular vertex.
It requires adding equality constraints or projecting the solve into the
orthogonal complement of the two translation modes.

## Interpretation of the Current Debug Output

The important lines are:

```text
L symmetry: mismatches=0
||L*u_translation||=1.32883e-13
||L*v_translation||=1.46417e-13
rhs_dot_u_translation=-1.77636e-15
rhs_dot_v_translation=1.63758e-15
AMD failing pivot: ... v(vertex=345)
natural-order LDLT compute info: Success
```

These show:

1. The assembled matrix is symmetric.
2. The translation modes are real null modes.
3. The RHS is compatible with those null modes.
4. The AMD failure is an exact zero Schur-complement pivot, not a bad original
   matrix entry.
5. Fixing one vertex is the correct reason the problem becomes solvable by
   `SimplicialLDLT`.
