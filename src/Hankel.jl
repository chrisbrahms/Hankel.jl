module Hankel
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import LinearAlgebra: mul!, ldiv!, dot
import Base: *, \

export QDHT, integrateK, integrateR, onaxis, symmetric, Rsymmetric

const J₀₀ = besselj(0, 0)

"""
    QDHT([p, ] R, N; dim=1)

`p`-th order quasi-discrete Hankel transform over aperture radius `R` with `N` samples
which transforms along dimension `dim`. If not given, `p` defaults to 0.

After:

[1] L. Yu, M. Huang, M. Chen, W. Chen, W. Huang, and Z. Zhu, Optics Letters 23 (1998)

[2] M. Guizar-Sicairos and J. C. Gutiérrez-Vega, JOSA A 21, 53 (2004)

but with some alterations:

The transform matrix T is not the same as C/T defined in [1, 2].
Instead of dividing by J₁(αₚₙ)J₁(αₚₘ) we divide by J₁(αₚₙ)^2. This cancels out
the factor between f and F so we do not have to mutltiply (divide) by J₁(αₚₙ) (J₁(αₚₘ)) before
and after applying the transform matrix.

Follows [`AbstractFFT`](https://github.com/JuliaMath/AbstractFFTs.jl) approach of applying
fwd and inv transform with `mul` and `ldiv`.

To calculate radial integrals of functions sampled using `QDHT`, use [`integrateR`](@ref)
and [`integrateK`](@ref).

The type of the coefficients is inferred from the type of `R` (but is promoted to be at
least `Float`), so for arbitrary precision use `QDHT([p, ] BigFloat(R), ...)`.
"""
struct QDHT{nT<:Real, pT<:Real}
    p::pT # Order of the transform
    N::Int # Number of samples
    T::Array{nT, 2} # Transform matrix
    J1sq::Array{nT, 1} # J₁² factors
    K::nT # Highest spatial frequency
    k::Vector{nT} # Spatial frequency grid
    R::nT # Aperture size (largest real-space coordinate)
    r::Vector{nT} # Real-space grid
    scaleR::Vector{nT} # Scale factor for real-space integration
    scaleK::Vector{nT} # Scale factor for frequency-space integration
    dim::Int # Dimension along which to transform
end

function QDHT(p, R, N; dim=1)
    p = convert(typeof(R), p)
    roots = besselj_zero.(p, 1:N) # type of besselj_zero is inferred from first argument
    S = besselj_zero(p, N+1)
    r = roots .* R/S # real-space vector
    K = S/R # Highest spatial frequency
    k = roots .* K/S # Spatial frequency vector
    J₁ = abs.(besselj.(p+1, roots))
    J₁sq = J₁ .* J₁
    T = 2/S * besselj.(p, (roots * roots')./S)./J₁sq' # Transform matrix

    K, R = promote(K, R) # deal with R::Int

    scaleR = 2/K^2 ./ J₁sq # scale factor for real-space integration
    scaleK = 2/R^2 ./ J₁sq # scale factor for reciprocal-space integration
    QDHT(p, N, T, J₁sq, K, k, R, r, scaleR, scaleK, dim)
end

QDHT(R, N; dim=1) = QDHT(0, R, N; dim=dim)

"
    mul!(Y, Q::QDHT, A)

Calculate the forward quasi-discrete Hankel transform of array `A` using the QDHT `Q`
and store the result in `Y`.

# Examples
```jldoctest
julia> q = QDHT(1e-2, 8); A = exp.(-q.r.^2/(1e-3*q.R)); Y = similar(A);
julia> mul!(Y, q, A)
8-element Array{Float64,1}:
  4.326937831591551e-6
  2.3341589529175126e-6
  7.689558743828849e-7
  1.546419420523699e-7
  1.8999259906096856e-8
  1.4159642663129888e-9
  7.013670190083954e-11
 -6.07681871673291e-13
```
"
function mul!(Y, Q::QDHT, A)
    dot!(Y, Q.T, A, dim=Q.dim)
    Y .*= Q.R/Q.K
end

"
    ldiv!(Y, Q::QDHT, A)

Calculate the inverse quasi-discrete Hankel transform of array `A` using the QDHT `Q`
and store the result in `Y`.

# Examples
```jldoctest
julia> q = QDHT(1e-2, 8); A = exp.(-q.r.^2/(1e-3*q.R)); Y = similar(A);
julia> mul!(Y, q, A);
julia> YY = similar(Y); ldiv!(YY, q, Y);
julia> YY ≈ A
true
```
"
function ldiv!(Y, Q::QDHT, A)
    dot!(Y, Q.T, A, dim=Q.dim)
    Y .*= Q.K/Q.R
end

"""
    *(Q::QDHT, A)

Calculate the forward quasi-discrete Hankel transform of array `A` using the QDHT `Q`.

# Examples
```jldoctest
julia> q = QDHT(1e-2, 8); A = exp.(-q.r.^2/(1e-3*q.R));
julia> q*A
8-element Array{Float64,1}:
  4.326937831591551e-6
  2.3341589529175126e-6
  7.689558743828849e-7
  1.546419420523699e-7
  1.8999259906096856e-8
  1.4159642663129888e-9
  7.013670190083954e-11
 -6.07681871673291e-13
```
"""
function *(Q::QDHT, A)
    out = similar(A)
    mul!(out, Q, A)
end

"""
    \\(Q::QDHT, A)

Calculate the inverse quasi-discrete Hankel transform of array `A` using the QDHT `Q`.

# Examples
```jldoctest
julia> q = QDHT(1e-2, 8); A = exp.(-q.r.^2/(1e-3*q.R));
julia> Ak = q*A;
julia> q \\ Ak ≈ A
true
```
"""
function \(Q::QDHT, A)
    out = similar(A)
    ldiv!(out, Q, A)
end

"""
    integrateR(A, Q::QDHT; dim=1)

Radial integral of `A`, over the aperture of `Q` in real space.

Assuming `A` contains samples of a function `f(r)` at sample points `Q.r`, then
`integrateR(A, Q)` approximates ∫f(r)r dr from r=0 to r=∞.

!!! note
    `integrateR` and `integrateK` fulfill Parseval's theorem, i.e. for some array `A`,
    `integrateR(abs2.(A), q)` and `integrateK(abs2.(q*A), q)` are equal, **but** 
    `integrateR(A, q)` and `integrateK(q*A, q)` are **not** equal.

!!! warning
    using `integrateR` to integrate a function (i.e. `A` rather than `abs2(A)`) is only
    supported for the 0th-order QDHT. For more details see [Derivations](@ref).

# Examples
```jldoctest
julia> q = QDHT(10, 128); A = exp.(-q.r.^2/2);
julia> integrateR(abs2.(A), q) ≈ 0.5 # analytical solution of ∫exp(-r²)r dr from 0 to ∞
true
```
"""
integrateR(A, Q::QDHT; dim=1) = dimdot(Q.scaleR, A; dim=dim)

"""
    integrateK(Ak, Q::QDHT; dim=1)

Radial integral of `A`, over the aperture of `Q` in reciprocal space.

Assuming `A` contains samples of a function `f(k)` at sample points `Q.k`, then
`integrateR(A, Q)` approximates ∫f(k)k dk from k=0 to k=∞.

!!! note
    `integrateR` and `integrateK` fulfill Parseval's theorem, i.e. for some array `A`,
    `integrateR(abs2.(A), q)` and `integrateK(abs2.(q*A), q)` are equal, **but** 
    `integrateR(A, q)` and `integrateK(q*A, q)` are **not** equal.

# Examples
```jldoctest
julia> q = QDHT(10, 128); A = exp.(-q.r.^2/2);
julia> integrateR(abs2.(A), q) ≈ 0.5 # analytical solution of ∫exp(-r²)r dr from 0 to ∞
true
julia> Ak = q*A;
julia> integrateK(abs2.(Ak), q) ≈ 0.5 # Same result
true

```
"""
integrateK(Ak, Q::QDHT; dim=1) = dimdot(Q.scaleK, Ak; dim=dim)

"""
    onaxis(Ak, Q::QDHT; dim=Q.dim)

Calculate on-axis sample in space (i.e. at r=0) from transformed array `Ak`.

# Examples
```jldoctest
julia> q = QDHT(10, 128); A = exp.(-q.r.^2/2);
julia> onaxis(q*A, q) ≈ 1 # should be exp(0) = 1
true
```
"""
function onaxis(Ak, Q::QDHT; dim=Q.dim) 
    Q.p == 0 || throw(
        DomainError("on-axis samples can only be obtained for 0th-order transforms"))
    J₀₀ .* integrateK(Ak, Q; dim=dim)
end

"""
    symmetric(A, Q::QDHT)

Create symmetric array from samples in `A`, including on-axis sample.

Given `A`, sampled at `[r₁, r₂, r₃, ...]`, generates array sampled at 
`[...-r₃, -r₂, -r₁, 0, r₁, r₂, r₃...]`

# Examples
```jldoctest
julia> q = QDHT(10, 128); A = exp.(-q.r.^2);
julia> As = symmetric(A, q);
julia> size(As)
(257,)
julia> As[1:128] == A[128:-1:1]
true
julia> As[129] ≈ 1 # should be exp(0) = 1
true
julia> As[130:end] == A
true
```
"""
function symmetric(A, Q::QDHT; dim=Q.dim)
    s = collect(size(A))
    N = s[Q.dim]
    s[Q.dim] = 2N + 1
    out = Array{eltype(A)}(undef, Tuple(s))
    idxlo = CartesianIndices(size(A)[1:Q.dim-1])
    idxhi = CartesianIndices(size(A)[Q.dim+1:end])
    out[idxlo, 1:N, idxhi] .= A[idxlo, N:-1:1, idxhi]
    out[idxlo, N+1, idxhi] .= squeeze(onaxis(Q*A, Q), dims=Q.dim)
    out[idxlo, (N+2):(2N+1), idxhi] .= A[idxlo, :, idxhi]
    return out
end

"""
    squeeze(A; dims)

Wrapper around `dropdims` to handle both numbers (return just the number) and arrays
(return `dropdims(A; dims)`).
"""
squeeze(A::Number; dims) = A
squeeze(A::AbstractArray; dims) = dropdims(A, dims=dims)

"""
    Rsymmetric(Q::QDHT)

Create radial coordinate array to go along with `symmetric(A, Q::QDHT)`.

# Examples
```jldoctest
julia> q = QDHT(10, 4);
julia> q.r
4-element Array{Float64,1}:
 1.6106347946239767
 3.697078919099734
 5.795844623798052
 7.8973942990196395
julia> Rsymmetric(q)
9-element Array{Float64,1}:
 -7.8973942990196395
 -5.795844623798052
 -3.697078919099734
 -1.6106347946239767
  0.0
  1.6106347946239767
  3.697078919099734
  5.795844623798052
  7.8973942990196395
```
"""
Rsymmetric(Q::QDHT) = vcat(-Q.r[end:-1:1], 0, Q.r)

"""
    oversample(A, Q::QDHT; factor::Int=4)

Oversample (smooth) the array `A`, which is sampled with the `QDHT` `Q`, by a `factor`.

This works like Fourier-domain zero-padding: a new `QDHT` is created with the same radius,
but `factor` times more points. The existing array is transformed and placed onto this
new spatial frequency grid, and the rest filled with zeros. Transforming back yields the
same shape in space but with more samples.

!!! note
    Unlike in zero-padding using FFTs, the old and oversampled **spatial** grids do not
    have any sampling points in common.
"""
function oversample(A, Q::QDHT; factor::Int=4)
    factor == 1 && (return A, Q)
    QNew = QDHT(Q.R, factor*Q.N, dim=Q.dim)
    @assert all(QNew.k[1:Q.N] .≈ Q.k)
    Ak = Q * A
    shape = collect(size(A))
    shape[Q.dim] = QNew.N
    Ako = zeros(eltype(A), Tuple(shape))
    idxlo = CartesianIndices(size(Ako)[1:Q.dim - 1])
    idxhi = CartesianIndices(size(Ako)[Q.dim + 1:end])
    Ako[idxlo, 1:Q.N, idxhi] .= Ak
    return QNew \ Ako, QNew
end

"""
    dot!(out, M, V; dim=1)

Matrix-vector multiplication along specific dimension of array `V`, storing result in `out`.

This is equivalent to iterating over all dimensions of `V` other than `dim` and applying the
matrix-vector multiplication `M * v[..., :, ...]`, but works by reshaping the array if
necessary to take advantage of faster matrix-matrix multiplication. If `dim==1`, `dot!` is
fastest and allocation-free.
"""
function dot!(out, M, V::AbstractVector; dim=1)
    size(V, dim) == size(M, 1) || throw(DomainError(
        "Size of V along dim must be same as size of M"))
    size(out) == size(V) || throw(DomainError("Input and output arrays must have same size"))
    dim == 1 || throw(DomainError("Cannot multiply along dimension $dim of 1-D array"))
    mul!(out, M, V)
end

function dot!(out, M, V::AbstractArray{T, 2}; dim=1) where T
    size(V, dim) == size(M, 1) || throw(DomainError(
        "Size of V along dim must be same as size of M"))
    size(out) == size(V) || throw(DomainError("Input and output arrays must have same size"))
    dim <= 2 || throw(DomainError("Cannot multiply along dimension $dim of 2-D array"))
    if dim == 1
        mul!(out, M, V)
    else
        Vtmp = permutedims(V, (2, 1))
        tmp = M * Vtmp
        permutedims!(out, tmp, (2, 1))
    end
end

function dot!(out, M, V::AbstractArray; dim=1) where T
    size(V, dim) == size(M, 1) || throw(DomainError(
        "Size of V along dim must be same as size of M"))
    size(out) == size(V) || throw(DomainError("Input and output arrays must have same size"))
    dim <= ndims(V) || throw(DomainError(
        "Cannot multiply along dimension $dim of $(ndims(V))-D array"))
    if dim == 1
        idxhi = CartesianIndices(size(V)[3:end])
        _dot!(out, M, V, idxhi)
    else
        dims = collect(1:ndims(V)) # [1, 2, ..., ndims(V)]
        otherdims = filter(d -> d ≠ dim, dims) # dims but without working dimension
        #= sort dimensions by their size so largest other dimension is part of matrix-matrix
            multiplication, for speed. other dimensions are iterated over in _dot! =#
        sidcs = sortperm(collect(size(V)[otherdims]), rev=true)
        perm = (dim, otherdims[sidcs]...)
        iperm = invperm(perm) # permutation to get back to original size
        Vtmp = permutedims(V, perm)
        tmp = similar(Vtmp)
        idxhi = CartesianIndices(size(Vtmp)[3:end])
        _dot!(tmp, M, Vtmp, idxhi)
        permutedims!(out, tmp, iperm)
    end
end

function _dot!(out, M, V, idxhi)
    for hi in idxhi
        mul!(view(out, :, :, hi), M, view(V, :, :, hi))
    end
end

"""
    dimdot(v, A; dim=1)

Calculate the dot product between vector `v` and one dimension of array `A`, iterating over
all other dimensions.
"""
function dimdot(v, A; dim=1)
    dims = collect(size(A))
    dims[dim] = 1
    out = Array{eltype(A)}(undef, Tuple(dims))
    dimdot!(out, v, A; dim=dim)
    return out
end

function dimdot(v, A::Vector; dim=1)
    return dot(v, A)
end

function dimdot!(out, v, A; dim=1)
    idxlo = CartesianIndices(size(A)[1:dim-1])
    idxhi = CartesianIndices(size(A)[dim+1:end])
    _dimdot!(out, v, A, idxlo, idxhi)
end

function _dimdot!(out, v, A, idxlo, idxhi)
    for lo in idxlo
        for hi in idxhi
            out[lo, 1, hi] = dot(v, view(A, lo, :, hi))
        end
    end
end

end