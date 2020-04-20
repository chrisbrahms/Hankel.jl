"""
    Plan{nT<:Real}

Base class for discrete Hankel transform plans.
"""
abstract type Plan end

function *(Q::Plan, A)
    out = similar(A)
    return mul!(out, Q, A)
end

function \(Q::Plan, A)
    out = similar(A)
    return ldiv!(out, Q, A)
end

abstract type AbstractQDHT{nT<:Real} <: Plan end

function mul!(Y, Q::AbstractQDHT, A)
    dot!(Y, Q.T, A, dim=Q.dim)
    Y .*= Q.scaleRK
end

function ldiv!(Y, Q::AbstractQDHT, A)
    dot!(Y, Q.T, A, dim=Q.dim)
    Y ./= Q.scaleRK
end

integrateR(A, Q::AbstractQDHT; dim = 1) = dimdot(Q.scaleR, A; dim = dim)

integrateK(Ak, Q::AbstractQDHT; dim = 1) = dimdot(Q.scaleK, Ak; dim = dim)

function symmetric(A, Q::AbstractQDHT; dim = Q.dim)
    s = collect(size(A))
    N = s[dim]
    s[dim] = 2N + 1
    out = Array{eltype(A)}(undef, Tuple(s))
    idxlo = CartesianIndices(size(A)[1:(dim - 1)])
    idxhi = CartesianIndices(size(A)[(dim + 1):end])
    out[idxlo, 1:N, idxhi] .= A[idxlo, N:-1:1, idxhi]
    out[idxlo, N + 1, idxhi] .= squeeze(onaxis(Q * A, Q), dims = dim)
    out[idxlo, (N + 2):(2N + 1), idxhi] .= A[idxlo, :, idxhi]
    return out
end

Rsymmetric(Q::AbstractQDHT) = vcat(-Q.r[end:-1:1], 0, Q.r)

function oversample(A, Q::AbstractQDHT; factor::Int = 4)
    factor == 1 && (return A, Q)
    QNew = oversample(Q; factor = factor)
    @assert all(QNew.k[1:(Q.N)] .≈ Q.k)
    Ak = Q * A
    shape = collect(size(A))
    shape[Q.dim] = QNew.N
    Ako = zeros(eltype(A), Tuple(shape))
    idxlo = CartesianIndices(size(Ako)[1:(Q.dim - 1)])
    idxhi = CartesianIndices(size(Ako)[(Q.dim + 1):end])
    Ako[idxlo, 1:(Q.N), idxhi] .= Ak
    return QNew \ Ako, QNew
end
