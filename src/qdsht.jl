"""
    QDSHT(p, [n, ]R, N; dim=1)
    QDSHT([p, ]R, N; dim=1)

`p`-th order quasi-discrete spherical Hankel transform over aperture radius `R` with `N`
samples which transforms along dimension `dim`. If not given, `p` defaults to 0, and `n`,
the spherical dimension, defaults to 2.

The QDSHT appears in the Fourier transform of radially symmetric functions in ``ℝ^{n+1}``
dimensions. For `n=1`, the QDSHT is identical to the (cylindrical) [`QDHT`](@ref), and for
all other `n`, it uses a reparameterized QDHT to perform the discrete transform.

Follows [`AbstractFFT`](https://github.com/JuliaMath/AbstractFFTs.jl) approach of applying
fwd and inv transform with `mul` and `ldiv`.

To calculate radial integrals of functions sampled using `QDSHT`, use [`integrateR`](@ref)
and [`integrateK`](@ref).

The type of the coefficients is inferred from the type of `R` (but is promoted to be at
least `Float`), so for arbitrary precision use `QDSHT([p, [n, ]] BigFloat(R), ...)`.
"""
struct QDSHT{nT<:Real,pT<:Real} <: AbstractQDHT{nT}
    p::pT # Order of the transform
    n::Int # Spherical dimension
    N::Int # Number of samples
    T::Array{nT,2} # Transform matrix
    j1sq::Array{nT,1} # j₁² factors
    K::nT # Highest spatial frequency
    k::Vector{nT} # Spatial frequency grid
    R::nT # Aperture size (largest real-space coordinate)
    r::Vector{nT} # Real-space grid
    scaleR::Vector{nT} # Scale factor for real-space integration
    scaleK::Vector{nT} # Scale factor for frequency-space integration
    scaleRK::nT # Scalar factor multiplied/divided during fwd/rev transform
    dim::Int # Dimension along which to transform
end

function QDSHT(p, n, R, N; dim = 1)
    T = float(typeof(R))
    p = convert(T, p)
    cn = convert(T, sphbesselj_scale(n))
    roots = sphbesselj_zero.(p, n, 1:N) # type of sphbesselj_zero is inferred from first argument
    S = sphbesselj_zero(p, n, N + 1)
    r = roots .* R / S # real-space vector
    K = S / R # Highest spatial frequency
    k = roots .* K / S # Spatial frequency vector
    j₁sq = abs2.(sphbesselj.(p + 1, n, roots))
    T = 2 * cn * S^(-(n + 1) / 2) * sphbesselj.(p, n, (roots .* roots') ./ S) ./ j₁sq' # Transform matrix

    K, R = promote(K, R) # deal with R::Int

    scaleR = 2 * cn^2 / K^(n + 1) ./ j₁sq # scale factor for real-space integration
    scaleK = 2 * cn^2 / R^(n + 1) ./ j₁sq # scale factor for reciprocal-space integration
    scaleRK = (R / K) ^ ((n + 1) / 2)
    return QDSHT(p, n, N, T, j₁sq, K, k, R, r, scaleR, scaleK, scaleRK, dim)
end

QDSHT(R, N; dim = 1) = QDSHT(0, R, N; dim = dim)
QDSHT(p, R, N; dim = 1) = QDSHT(p, 2, R, N; dim = dim)

"
    mul!(Y, Q::QSDHT, A)

Calculate the forward quasi-discrete spherical Hankel transform of array `A` using the
QDSHT `Q` and store the result in `Y`.

# Examples
```jldoctest
julia> q = QDSHT(1e-2, 8); A = exp.(-q.r.^2/(1e-3*q.R)); Y = similar(A);
julia> mul!(Y, q, A)
8-element Array{Float64,1}:
  8.73574297951054e-9
  4.166950654425497e-9
  1.2135269740167775e-9
  2.1570485709886677e-10
  2.3441730449103528e-11
  1.5335062911366657e-12
  7.37934569592307e-14
 -3.450933124215762e-15
```
"
mul!(::Any, ::QDSHT, ::Any)

"
    ldiv!(Y, Q::QDSHT, A)

Calculate the inverse quasi-discrete spherical Hankel transform of array `A` using the QDSHT
`Q` and store the result in `Y`.

# Examples
```jldoctest
julia> q = QDSHT(1e-2, 8); A = exp.(-q.r.^2/(1e-3*q.R)); Y = similar(A);
julia> mul!(Y, q, A);
julia> YY = similar(Y); ldiv!(YY, q, Y);
julia> YY ≈ A
true
```
"
ldiv!(::Any, ::QDSHT, ::Any)

"""
    integrateR(A, Q::QDSHT; dim=1)

Radial integral of `A`, over the aperture of `Q` in ``n+1``-dimensional real space.

Assuming `A` contains samples of a function `f(r)` at sample points `Q.r`, then
`integrateR(A, Q)` approximates ∫f(r)rⁿ dr from r=0 to r=∞.

!!! note
    `integrateR` and `integrateK` fulfill Parseval's theorem, i.e. for some array `A`,
    `integrateR(abs2.(A), q)` and `integrateK(abs2.(q*A), q)` are equal, **but**
    `integrateR(A, q)` and `integrateK(q*A, q)` are **not** equal.

!!! warning
    using `integrateR` to integrate a function (i.e. `A` rather than `abs2(A)`) is only
    supported for the 0th-order QDSHT. For more details see [Derivations](@ref).

# Examples
```jldoctest
julia> q = QDSHT(10, 128); A = exp.(-q.r.^2/2);
julia> integrateR(abs2.(A), q) ≈ √π/4 # analytical solution of ∫exp(-r²)rⁿ dr from 0 to ∞
true
```
"""
integrateR(::Any, ::QDSHT)

"""
    integrateK(Ak, Q::QDSHT; dim=1)

Radial integral of `A`, over the aperture of `Q` in ``n+1``-dimensional reciprocal space.

Assuming `A` contains samples of a function `f(k)` at sample points `Q.k`, then
`integrateR(A, Q)` approximates ∫f(k)kⁿ dk from k=0 to k=∞.

!!! note
    `integrateR` and `integrateK` fulfill Parseval's theorem, i.e. for some array `A`,
    `integrateR(abs2.(A), q)` and `integrateK(abs2.(q*A), q)` are equal, **but**
    `integrateR(A, q)` and `integrateK(q*A, q)` are **not** equal.

# Examples
```jldoctest
julia> q = QDSHT(10, 128); A = exp.(-q.r.^2/2);
julia> integrateR(abs2.(A), q) ≈ √π/4 # analytical solution of ∫exp(-r²)rⁿ dr from 0 to ∞
true
julia> Ak = q*A;
julia> integrateK(abs2.(Ak), q) ≈ √π/4 # Same result
true
```
"""
integrateK(::Any, ::QDSHT)

"""
    onaxis(Ak, Q::QDSHT; dim=Q.dim)

Calculate on-axis sample in space (i.e. at r=0) from transformed array `Ak`.

# Examples
```jldoctest
julia> q = QDSHT(10, 128); A = exp.(-q.r.^2/2);
julia> onaxis(q*A, q) ≈ 1 # should be exp(0) = 1
true
```
"""
function onaxis(Ak, Q::QDSHT; dim = Q.dim)
    if Q.p != 0
        throw(DomainError("on-axis samples can only be obtained for 0th-order transforms"))
    end
    j00ocn = inv(gamma((Q.n + 1) / 2) * 2^((Q.n - 1) / 2))
    return j00ocn .* integrateK(Ak, Q; dim = dim)
end

function oversample(Q::QDSHT; factor::Int = 4)
    factor == 1 && return Q
    return QDSHT(Q.p, Q.n, Q.R, factor * Q.N, dim = Q.dim)
end
