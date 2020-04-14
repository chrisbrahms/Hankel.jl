# Hankel

`Hankel` implements the ``0^{\mathrm{th}}``-order quasi-discrete Hankel transform (QDHT) as first laid out in [L. Yu, *et al.*, Optics Letters 23 (1998)](https://www.osapublishing.org/ol/abstract.cfm?uri=ol-23-6-409). The forward ``0^{\mathrm{th}}``-order Hankel transform of the radially symmetric function ``f(r)`` is defined as
```math
\tilde{f}(k) = \int_0^\infty f(r) J_0(kr) r\,\mathrm{d}r\,,
```
where ``J_0(x)`` is the ``0^{\mathrm{th}}``-order Bessel function of the first kind. Correspondingly, the inverse transform is
```math
f(r) = \int_0^\infty \tilde{f}(k) J_0(kr) k\,\mathrm{d}k\,.
```
Note that here, ``k`` includes the factor of ``2\pi``, i.e. it is spatial angular frequency.

The QDHT approximates these transforms under the assumption that ``f(r) = 0`` for ``r > R`` where ``R`` is the aperture size. By expanding ``f(r)`` as a Fourier-Bessel series and choosing to sample ``f(r)`` at coordinates ``r_n = j_nR/j_{N+1}``, where ``j_n`` is the ``n^{\mathrm{th}}`` zero of ``J_0(x)`` and ``N`` is the number of samples, the Hankel transform turns into a matrix-vector multiplication.

`Hankel` follows the [`AbstractFFTs`](https://juliamath.github.io/AbstractFFTs.jl/stable/) approach of planning a transform in advance by creating a [`QDHT`](@ref) object, which can then be used to transform an array using multiplication ([`*`](@ref) or [`mul!`](@ref)) and to transform back by left-division ([`\`](@ref) or [`ldiv!`](@ref)). In contrast to `AbstractFFTs`, however, **pre-planning a transform is required** since the transform only works properly if the sampling points are chosen for a particular combination of sample number ``N`` and aperture size ``R``. The workflow to transform a function ``f(r)`` is therefore as follows:

```@example
using Hankel # hide
import Hankel: besselj_zero, besselj # hide
j01 = besselj_zero(0, 1)
R = 1
N = 8
q = QDHT(R, N)
f(r) = besselj(0, r*j01/R)
fr = f.(q.r)
fk = q * fr # fk should have only the first entry non-zero
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

To avoid unnecessary multiplications, the transform matrix (``C`` in Yu *et al.*, here it's called `T`) is altered slightly (see [`QDHT`](@ref)), and as a consequence the transformation itself does not conserve energy (i.e. satisfy [Parseval's Theorem](https://en.wikipedia.org/wiki/Parseval%27s_theorem)). To calculate the ``L^2`` norm (e.g. energy in a signal) with correct scaling, use [`integrateR`](@ref) in real (``r``) space and [`integrateK`](@ref) in reciprocal (``k``) space.

## On-axis and symmetric samples
The QDHT does not contain a sample on axis, i.e. at ``r=0``, but it can be obtained using [`onaxis`](@ref), which takes the *transformed* array as its input. This is because the on-axis sample is obtained from

```math
f(r=0) = \int_0^\infty \tilde{f}(k) J_0(0) k\,\mathrm{d}k\,.
```

Given an array `A`, the convenience method [`symmetric`](@ref) produces the symmetric array, i.e. given `A`, sampled at ``[r₁, r₂, r₃, ...]``, it generates the array sampled at ``[...-r₃, -r₂, -r₁, 0, r₁, r₂, r₃...]``. This works also for higher-dimensional arrays. The corresponding spatial coordinates can be obtained with [`Rsymmetric`](@ref).

## Function Reference
```@autodocs
Modules = [Hankel]
```