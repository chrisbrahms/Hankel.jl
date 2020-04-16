# Derivations
## ``L^2`` norm integrations
The ``L^2`` norm in the radially symmetric case is
```math
\left\lVert f(r) \right\rVert = 2\pi \int_0^\infty \left\vert f(r)\right\vert^2 r\,\mathrm{d}r\,.
```
In the case where ``f(r)`` is an electric field, this can be used to calculate the total power ``P``:
```math
P(t) = \frac{\epsilon_0 c}{2} \left\lVert E(t, r) \right\rVert\,.
```
When using a `QDHT`, this integral can be approximated by a simple dot product, which is implemented in [`integrateR`](@ref) and [`integrateK`](@ref).

Following eqs. (21)-(24) in Yu et al. we take the integral, expand ``f(r)`` in a Fourier-Bessel series, and use the Hankel transform to obtain the series coefficients:
```math
f(r) = \frac{1}{\pi R^2}\sum_{m=1}^M \tilde{f}\left(\frac{j_m}{2\pi R}\right) J_1^{-2}(j_m)J_0\left(\frac{j_m r}{R}\right)\,,
```
which turns the integral into
```math
\int_0^R \left\vert f(r)\right\vert^2 r\,\mathrm{d}r = \frac{1}{\pi^2 R^4} \sum_{nm}\tilde{f}\left(\frac{j_n}{2\pi R}\right) \tilde{f}^*\left(\frac{j_m}{2\pi R}\right) J_1^{-2}(j_n) J_1^{-2}(j_m) \int_0^R J_0\left(\frac{j_n r}{R}\right) J_0\left(\frac{j_m r}{R}\right)r\,\mathrm{d}r\,.
```
The integral expression here is
```math
\int_0^R J_0\left(\frac{j_n r}{R}\right) J_0\left(\frac{j_m r}{R}\right)r\,\mathrm{d}r = \frac{1}{2}R^2 J_1^2(j_m) \delta_{mn}\,,
```
where ``\delta_{mn}`` is the Kronecker delta, which reduces the sum back to one variable:
```math
\int_0^R \left\vert f(r)\right\vert^2 r\,\mathrm{d}r = \frac{1}{2\pi^2 R^2} \sum_{m=1}^M\left\vert\tilde{f}\left(\frac{j_m}{2\pi R}\right)\right\vert^2 J_1^{-2}(j_m)\,.
```
If we now define ``\tilde{F}`` (this is ``F_2`` in Yu et al.):
```math
\tilde{F}(m) = \tilde{f}\left(\frac{j_m}{2\pi R}\right) J_1^{-1}(j_m) \frac{K}{2\pi}\,,
```
we arrive at
```math
\int_0^R \left\vert f(r)\right\vert^2 r\,\mathrm{d}r = \frac{2}{K^2R^2}\sum_{m=1}^M \vert\tilde{F}(m)\vert^2
```
From Parseval's theorem we know that
```math
\int_0^R \left\vert f(r)\right\vert^2 r\,\mathrm{d}r = \int_0^K \vert \tilde{f}(k)\vert^2 k\,\mathrm{d}k\,.
```
Following the same procedure as above for the ``k`` integral we find
```math
\int_0^K \left\vert \tilde{f}(k)\right\vert^2 k\,\mathrm{d}k = \frac{2}{K^2R^2}\sum_{n=1}^N \vert F(n)\vert^2\,,
```
and hence
```math
\int_0^R \left\vert f(r)\right\vert^2 r\,\mathrm{d}r = \frac{2}{K^2R^2}\sum_{n=1}^N \vert F(n)\vert^2 = \frac{2}{K^2} \sum_{n=1}^N \left\vert f\left(\frac{j_n}{K}\right)\right\vert^2 J_1^{-2}(j_n)\,.
```
This is just the dot product between ``\vert f(r_n) \vert^2`` and a scaling vector ``S_R`` (called `scaleR` in `QDHT`):
```math
\int_0^R \left\vert f(r)\right\vert^2 r\,\mathrm{d}r = S_R \cdot \vert f(r_n) \vert^2
```
with
```math
S_R(n) = 2K^{-2} J_1^{-2}(j_n)
```
The same derivation holds for the ``k`` integral and results in
```math
S_K(n) = 2R^{-2} J_1^{-2}(j_n)\,.
```

## Integration of functions
For integration of the form
```math
\int_0^R f(r) r\,\mathrm{d}r\,,
```
we cannot use Parseval's theorem. Instead, following a similar approach we arrive at
```math
\int_0^R f(r) r\,\mathrm{d}r = \frac{1}{\pi R^2} \sum_{m=1}^M \tilde{f}\left(\frac{j_m}{2\pi R}\right)J_1^{-2}(j_m) \int_0^R J_0\left(\frac{j_m r}{R}\right)r\,\mathrm{d}r\,.
```
Using
```math
\int_0^x x' J_0(x')\,\mathrm{d}x' = xJ_1(x)\,,
```
the integral evaluates to
```math
\int_0^R J_0\left(\frac{j_m r}{R}\right)r\,\mathrm{d}r = \frac{R^2}{j_m} J_1(j_m)\,,
```
and so
```math
\int_0^R f(r) r\,\mathrm{d}r = \frac{1}{\pi} \sum_{m=1}^M \frac{1}{j_m}\tilde{f}\left(\frac{j_m}{2\pi R}\right) J_1^{-1}(j_m) = \frac{2}{K} \sum_{m=1}^M \frac{1}{j_m} \tilde{F}(m)\,.
```
If we now use the QDHT as defined in Yu et al.,
```math
\tilde{F}(m) = \sum_{n=1}^N C_{mn}F(n)
```
with
```math
C_{mn} = \frac{2}{S} J_0\left(\frac{j_n j_m}{S}\right)J_1^{-1}(j_n)J_1^{-1}(j_m)\,,
```
after some algebra we arrive at
```math
\int_0^R f(r) r\,\mathrm{d}r = \frac{4}{K^2} \sum_{n=1}^N J_1^{-2}(j_n)f(r_n) \sum_{m=1}^M \frac{1}{j_m}J_0\left(\frac{j_n j_m}{S}\right)J_1^{-1}(j_m)\,.
```
When taking into account that ``S = j_{N+1}``, the sum over ``m`` is approximately ``1/2`` for all values of ``n`` (this was verified numerically). This leaves us with
```math
\int_0^R f(r) r\,\mathrm{d}r \approx \frac{2}{K^2} \sum_{n=1}^N J_1^{-2}(j_n)f(r_n)
```
as in the case of integrals over ``\vert f(r)\vert^2``.
