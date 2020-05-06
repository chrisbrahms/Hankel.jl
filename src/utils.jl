"""
    padzeros(A, N; dim = 1)

Pad array `A` with zeros to length `N` along dimension `dim`.
"""
function padzeros(A, N; dim = 1)
    shape = size(A)
    Nold = shape[dim]
    shape = Base.setindex(shape, N - Nold, dim)
    zs = zeros(eltype(A), shape)
    return cat(A, zs; dims = dim)
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

"""
    sphbesselj_scale(n)

Return the normalization factor for the (hyper)spherical Bessel function
([`sphbesselj`](@ref)) with spherical dimension ``n``, given as
``c_n = \\sqrt{\\frac{π}{2}}`` for even ``n`` and 1 otherwise.

After:

[1] J. S. Avery, J. E. Avery. Hyperspherical Harmonics And Their Physical Applications.
    Singapore: World Scientific Publishing Company, 2017.
"""
sphbesselj_scale(n) = isodd(n) ? 1.0 : √(π / 2)

"""
    sphbesselj(p, n, x)

(Hyper)spherical Bessel function of order ``p`` and spherical dimension ``n``.
The hyperspherical Bessel function generalizes the cylindrical and spherical Bessel
functions to the ``n``-sphere (embedded in ``ℝ^{n+1}``). It is given as

```math
j_p^{n}(x) = c_n x^{-(n-1)/2} J_{p + (n-1)/2}(x),
```

where ``c_n`` is a normalization factor defined by [`sphbesselj_scale`](@ref). Note that
``n`` is not an exponent here.

It has as its special cases:
- Cylindrical Bessel function (``n=1``): ``j_p^{1}(x) = J_p(x)``
- Spherical Bessel function (``n=2``): ``j_p^{2}(x) = j_p(x)``

After:

[1] J. S. Avery, J. E. Avery. Hyperspherical Harmonics And Their Physical Applications.
    Singapore: World Scientific Publishing Company, 2017.
"""
function sphbesselj(p, n, x)
    n == 1 && return besselj(p, x)
    α = (n - 1) / 2
    Jppa = besselj(p + α, x)
    cn = sphbesselj_scale(n)
    if abs(x) ≤ sqrt(eps(real(zero(Jppa))))
        if p == 0
            J0pa = cn / gamma(α + 1) / 2^α
            return oftype(Jppa, J0pa)
        else
            return zero(Jppa)
        end
    end
    jp = cn * Jppa / x^α
    return oftype(Jppa, jp)
end

"""
    sphbesselj_zero(p, n, m)

Get the ``m``th zero of the (hyper)spherical Bessel function of order ``p`` and spherical
dimension ``n``.

See [`sphbesselj`](@ref).
"""
sphbesselj_zero(p, n, m) = besselj_zero(p + (n - 1) / 2, m)
