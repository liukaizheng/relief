# Right-Weighted SLIM Proxy Derivation

This note derives the weight matrix for the alternative proxy

$$
P_R^W(J) = \|(J - R)W\|_F^2 .
$$

This differs from the proxy in `SLIM2017.pdf`, formula (17), which uses left multiplication:

$$
P_R^W(J) = \|W(J - R)\|_F^2 .
$$

## Proxy Gradient

Let

$$
A = J - R .
$$

Then

$$
P_R^W(J)
=
\|AW\|_F^2
=
\operatorname{tr}((AW)^T(AW))
=
\operatorname{tr}(W^T A^T A W).
$$

Since $A = J - R$, differentiating with respect to $J$ gives

$$
\nabla_J P_R^W(J)
=
2(J - R)WW^T.
$$

To match the true distortion gradient, require

$$
2(J - R)WW^T
=
\nabla_J D(J).
$$

Therefore,

$$
WW^T
=
\frac{1}{2}(J - R)^{-1}\nabla_J D(J).
$$

One symmetric principal-root choice is

$$
W =
\sqrt{
\frac{1}{2}(J - R)^{-1}\nabla_J D(J)
}.
$$

If $J - R$ is singular, use the pseudoinverse or the singular-value limit.

## SVD Form

Let

$$
J = US_JV^T,
$$

where

$$
S_J =
\begin{bmatrix}
\sigma_1 & 0 \\
0 & \sigma_2
\end{bmatrix}.
$$

For isometric energies, the closest rotation is

$$
R = UV^T.
$$

Then

$$
J - R
=
US_JV^T - UV^T
=
U(S_J - I)V^T.
$$

So

$$
(J - R)^{-1}
=
V(S_J - I)^{-1}U^T.
$$

For rotation-invariant distortion energies,

$$
\nabla_J D(J)
=
U\nabla_{S_J}D(S_J)V^T.
$$

Substitute into the gradient-matching equation:

$$
WW^T
=
\frac{1}{2}
V(S_J - I)^{-1}U^T
U\nabla_{S_J}D(S_J)V^T.
$$

Because

$$
U^TU = I,
$$

we get

$$
WW^T
=
V
\left[
\frac{1}{2}
(S_J - I)^{-1}\nabla_{S_J}D(S_J)
\right]
V^T.
$$

Taking the matrix square root:

$$
W =
V
\sqrt{
\frac{1}{2}
(S_J - I)^{-1}\nabla_{S_J}D(S_J)
}
V^T.
$$

Equivalently,

$$
W = VS_WV^T,
$$

where

$$
S_W
=
\sqrt{
\frac{1}{2}
(S_J - I)^{-1}\nabla_{S_J}D(S_J)
}.
$$

## Per-Singular-Value Form

For separable distortion energies,

$$
D(S_J)=\sum_i f_i(\sigma_i),
$$

we have

$$
\nabla_{S_J}D(S_J)
=
\operatorname{diag}(f_1'(\sigma_1), f_2'(\sigma_2)).
$$

Therefore,

$$
(S_W)_{ii}
=
\sqrt{
\frac{f_i'(\sigma_i)}
{2(\sigma_i - 1)}
}.
$$

At $\sigma_i = 1$, use the limit

$$
(S_W)_{ii}
=
\sqrt{
\frac{f_i''(1)}{2}
}.
$$

## Comparison With Original SLIM

Original left-weighted SLIM proxy:

$$
P_R^W(J)=\|W(J-R)\|_F^2
\quad\Rightarrow\quad
W = US_WU^T.
$$

Alternative right-weighted proxy:

$$
P_R^W(J)=\|(J-R)W\|_F^2
\quad\Rightarrow\quad
W = VS_WV^T.
$$

So the scalar singular-value weights are the same, but the basis changes from the left singular vectors $U$ to the right singular vectors $V$.
