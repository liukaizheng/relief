# SLIM Formula (17): Why the Weight Matrix Stays on the Left

Formula (17) in `SLIM2017.pdf` defines the weighted local proxy

$$
P_R^W(J) = \|W(J - R)\|_F^2 .
$$

Here:

- $J$ is the candidate Jacobian.
- $R$ is the closest ideal Jacobian, usually the closest rotation.
- $W$ is a $2 \times 2$ weight matrix.
- $\|\cdot\|_F^2$ is the squared Frobenius norm.

Let

$$
A = J - R .
$$

Then formula (17) is

$$
P_R^W(J) = \|WA\|_F^2 .
$$

## Can We Move \(W\) to the Right?

In general, no:

$$
\|W(J - R)\|_F^2 \neq \|(J - R)W\|_F^2 .
$$

Expanding the left-weighted version:

$$
\|WA\|_F^2
=
\operatorname{tr}((WA)^T(WA))
=
\operatorname{tr}(A^T W^T W A).
$$

Expanding the right-weighted version:

$$
\|AW\|_F^2
=
\operatorname{tr}((AW)^T(AW))
=
\operatorname{tr}(W^T A^T A W).
$$

These two expressions are generally different.

## Counterexample

Take

$$
A =
\begin{bmatrix}
0 & 1 \\
0 & 0
\end{bmatrix},
\qquad
W =
\begin{bmatrix}
2 & 0 \\
0 & 1
\end{bmatrix}.
$$

Then

$$
WA =
\begin{bmatrix}
0 & 2 \\
0 & 0
\end{bmatrix},
\qquad
\|WA\|_F^2 = 4.
$$

But

$$
AW =
\begin{bmatrix}
0 & 1 \\
0 & 0
\end{bmatrix},
\qquad
\|AW\|_F^2 = 1.
$$

So left multiplication and right multiplication do not define the same proxy.

## Why the Left Side Matters in SLIM

For formula (17),

$$
P_R^W(J) = \|W(J - R)\|_F^2,
$$

the gradient with respect to $J$ is

$$
\nabla_J P_R^W(J)
=
2W^T W(J - R).
$$

SLIM chooses $W_f^k$ so that this proxy gradient matches the true distortion-gradient at the current Jacobian:

$$
2(W_f^k)^T W_f^k(J_f^k - R_f^k)
=
\nabla_J D(J_f^k).
$$

If we instead used

$$
\|(J - R)W\|_F^2,
$$

the gradient would be

$$
2(J - R)WW^T,
$$

which is a different expression. That would require a different derivation and would not be formula (17) from the paper.

So although a right-weighted proxy is dimensionally possible for $2 \times 2$ matrices, it is not equivalent to the SLIM proxy and does not match the paper's construction of $W$.
