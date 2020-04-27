# Hankel

`Hankel` implements the quasi-discrete Hankel transform (QDHT) as introduced in [H. F. Johnson, Comput. Phys. Commun., 43, 2 (1987)](https://doi.org/10.1016/0010-4655(87)90204-9), laid out for 0-order in [L. Yu, *et al.*, Optics Letters 23 (1998)](https://www.osapublishing.org/ol/abstract.cfm?uri=ol-23-6-409), and extended to higher orders in [M. Guizar-Sicairos and J. C. Gutiérrez-Vega, JOSA A 21, 53 (2004)](https://www.osapublishing.org/abstract.cfm?URI=josaa-21-1-53).
We generalized the cylindrical QDHT in the above publications to its (hyper)spherical form.
The forward ``p^{\mathrm{th}}``-order (hyper)spherical Hankel transform of the radially symmetric function ``f(r)`` is defined as
```math
\tilde{f}(k) = c_n^{-1} \int_0^\infty f(r) j_p^n(kr) r^n\,\mathrm{d}r\,,
```
where ``c_n`` is a constant (see [`sphbesselj_scale`](@ref Hankel.sphbesselj_scale)), and ``j_p^n(x)`` is the ``p^{\mathrm{th}}``-order (hyper)spherical Bessel function of the first kind with spherical dimension ``n`` (see [`sphbesselj`](@ref Hankel.sphbesselj)).
Correspondingly, the inverse transform is
```math
f(r) = c_n^{-1} \int_0^\infty \tilde{f}(k) j_p^n(kr) k^n\,\mathrm{d}k\,.
```
Note that here, ``k`` includes the factor of ``2\pi``, i.e. it is spatial angular frequency.

The remainder of the documentation considers without loss of generalization only the cylindrical case (``n=1``, with ``j_p^n(x) = J_p(x)``).

The QDHT approximates these transforms under the assumption that ``f(r) = 0`` for ``r > R`` where ``R`` is the aperture size. By expanding ``f(r)`` as a Fourier-Bessel series and choosing to sample ``f(r)`` at coordinates ``r_n = j_nR/j_{N+1}``, where ``j_n`` is the ``n^{\mathrm{th}}`` zero of ``J_p(x)`` and ``N`` is the number of samples, the Hankel transform turns into a matrix-vector multiplication.

`Hankel` follows the [`AbstractFFTs`](https://juliamath.github.io/AbstractFFTs.jl/stable/) approach of planning a transform in advance by creating a [`QDHT`](@ref) object, which can then be used to transform an array using multiplication ([`*`](@ref) or [`mul!`](@ref)) and to transform back by left-division ([`\`](@ref) or [`ldiv!`](@ref)). In contrast to `AbstractFFTs`, however, **pre-planning a transform is required** since the transform only works properly if the sampling points are chosen for a particular combination of sample number ``N`` and aperture size ``R``. The workflow to transform a function ``f(r)`` is therefore as follows:

```@example
using Hankel # hide
import Hankel: besselj_zero, besselj # hide
j01 = besselj_zero(0, 1)
R = 1
N = 8
q = QDHT(0, R, N)
f(r) = besselj(0, r*j01/R)
fr = f.(q.r)
fk = q * fr # 0th-order QDHT => fk should have only the first entry non-zero
```

## Normalisation

The transform as implemented here is unitary, i.e.
```@example
using Hankel # hide
import Hankel: besselj_zero, besselj # hide
j01 = besselj_zero(0, 1) # hide
R = 1 # hide
N = 512 # hide
q = QDHT(R, N) # hide
f(r) = besselj(0, r*j01/R) # hide
fr = f.(q.r) # hide
q \ (q * fr) ≈ fr
```

The transform satisfies [Parseval's Theorem](https://en.wikipedia.org/wiki/Parseval%27s_theorem). To calculate the ``L^2`` norm (e.g. energy in a signal) with correct scaling, use [`integrateR`](@ref) in real (``r``) space and [`integrateK`](@ref) in reciprocal (``k``) space. For an explanation of how these functions work, see the [Derivations](@ref) page.

## On-axis and symmetric samples
The QDHT does not contain a sample on axis, i.e. at ``r=0``, but for the ``0^{\mathrm{th}}``-order QDHT, it can be obtained using [`onaxis`](@ref), which takes the *transformed* array as its input. This is because the on-axis sample is obtained from

```math
f(r=0) = \int_0^\infty \tilde{f}(k) J_0(0) k\,\mathrm{d}k\,.
```

Note here why this does not work for other values of ``p``: ``J_p(0) = 0 \; \forall \; p>0``.

Given an array `A`, the convenience method [`symmetric`](@ref) produces the symmetric array, i.e. given `A`, sampled at ``[r₁, r₂, r₃, ...]``, it generates the array sampled at ``[...-r₃, -r₂, -r₁, 0, r₁, r₂, r₃...]``. This works also for higher-dimensional arrays. The corresponding spatial coordinates can be obtained with [`Rsymmetric`](@ref).

## Function Reference
```@autodocs
Modules = [Hankel]
```
