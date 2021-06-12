# Rules for automatic differentiation

## Constructor
### These rules designate QDHT as non-differentiable
function ChainRulesCore.frule(Δargs, ::Type{T}, args...; kwargs...) where {T<:QDHT}
    return T(args...; kwargs...), NoTangent()
end

function ChainRulesCore.rrule(::Type{T}, args...; kwargs...) where {T<:QDHT}
    function QDHT_pullback(ΔQ)
        return (NoTangent(), map(_ -> NoTangent(), args)...)
    end
    return T(args...; kwargs...), QDHT_pullback
end

## rules for fwd/rev transform
ChainRulesCore.frule((_, _, ΔA), ::typeof(*), Q::QDHT, A) = (Q * A, Q * ΔA)
ChainRulesCore.frule((_, _, ΔA), ::typeof(\), Q::QDHT, A) = (Q \ A, Q \ ΔA)
function ChainRulesCore.frule((_, ΔY, _, ΔA), ::typeof(mul!), Y, Q::QDHT, A)
    return mul!(Y, Q, A), mul!(ΔY, Q, ΔA)
end
function ChainRulesCore.frule((_, ΔY, _, ΔA), ::typeof(ldiv!), Y, Q::QDHT, A)
    return ldiv!(Y, Q, A), ldiv!(ΔY, Q, ΔA)
end

function ChainRulesCore.rrule(::typeof(*), Q::QDHT, A)
    Y = Q * A
    function mul_pullback(ΔY)
        ∂Q = NoTangent()
        ∂A = @thunk _mul_back(ΔY, Q, A, Q.scaleRK)
        return NoTangent(), ∂Q, ∂A
    end
    return Y, mul_pullback
end

function ChainRulesCore.rrule(::typeof(\), Q::QDHT, A)
    Y = Q \ A
    function ldiv_pullback(ΔY)
        ∂Q = NoTangent()
        ∂A = @thunk _mul_back(ΔY, Q, A, inv(Q.scaleRK))
        return NoTangent(), ∂Q, ∂A
    end
    return Y, ldiv_pullback
end

function _mul_back(ΔY, Q, A, s)
    T = typeof(one(eltype(ΔY)) * one(eltype(Q.T)) * one(eltype(s)))
    ∂A = Array{T}(undef, size(A))
    dot!(∂A, Q.T', ΔY, dim = Q.dim)
    ∂A .*= s
    return ∂A
end

## rules for integrateR/integrateK
function ChainRulesCore.frule((_, ΔA, _), ::typeof(integrateR), A, Q::QDHT; kwargs...)
    return integrateR(A, Q; kwargs...), integrateR(ΔA, Q; kwargs...)
end

function ChainRulesCore.frule((_, ΔA, _), ::typeof(integrateK), A, Q::QDHT; kwargs...)
    return integrateK(A, Q; kwargs...), integrateK(ΔA, Q; kwargs...)
end

function ChainRulesCore.rrule(::typeof(integrateR), A, Q::QDHT; dim = 1)
    function integrateR_pullback(ΔΩ)
        ∂A = @thunk _integrateRK_back(ΔΩ, A, Q.scaleR; dim = dim)
        return NoTangent(), ∂A, NoTangent()
    end
    return integrateR(A, Q; dim = dim), integrateR_pullback
end

function ChainRulesCore.rrule(::typeof(integrateK), A, Q::QDHT; dim = 1)
    function integrateK_pullback(ΔΩ)
        ∂A = @thunk _integrateRK_back(ΔΩ, A, Q.scaleK; dim = dim)
        return NoTangent(), ∂A, NoTangent()
    end
    return integrateK(A, Q; dim = dim), integrateK_pullback
end

_integrateRK_back(ΔΩ, A::AbstractVector, scale; dim = 1) = ΔΩ .* scale
function _integrateRK_back(ΔΩ, A::AbstractMatrix, scale; dim = 1)
    return dim == 1 ? scale .* ΔΩ : ΔΩ * scale'
end
function _integrateRK_back(ΔΩ, A, scale; dim = 1)
    N = size(A, dim)
    sz = ntuple(_ -> 1, ndims(A))
    sz = Base.setindex(sz, N, dim)
    scalearray = reshape(scale, sz)
    ∂A = ΔΩ .* scalearray
    return ∂A
end
