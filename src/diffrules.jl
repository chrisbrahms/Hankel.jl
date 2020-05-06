# Rules for automatic differentiation

## Constructor
function ChainRulesCore.rrule(::Type{T}, args...; kwargs...) where {T<:QDHT}
    function QDHT_pullback(ΔQ)
        return (NO_FIELDS, map(_ -> DoesNotExist(), args)...)
    end
    return T(args...; kwargs...), QDHT_pullback
end

## rules for fwd/rev transform
ChainRulesCore.frule((_, ΔA), ::typeof(*), Q::QDHT, A) = (Q * A, Q * ΔA)
ChainRulesCore.frule((_, ΔA), ::typeof(\), Q::QDHT, A) = (Q \ A, Q \ ΔA)
function ChainRulesCore.frule((ΔY, _, ΔA), ::typeof(mul!), Y, Q::QDHT, A)
    return mul!(Y, Q, A), mul!(ΔY, Q, ΔA)
end
function ChainRulesCore.frule((ΔY, _, ΔA), ::typeof(ldiv!), Y, Q::QDHT, A)
    return ldiv!(Y, Q, A), ldiv!(ΔY, Q, A)
end

function ChainRulesCore.rrule(::typeof(*), Q::QDHT, A)
    Y = Q * A
    function mul_pullback(ΔY)
        ∂Q = DoesNotExist()
        ∂A = @thunk _mul_back(ΔY, Q, A, Q.scaleRK)
        return NO_FIELDS, ∂Q, ∂A
    end
    return Y, mul_pullback
end

function ChainRulesCore.rrule(::typeof(\), Q::QDHT, A)
    Y = Q \ A
    function ldiv_pullback(ΔY)
        ∂Q = DoesNotExist()
        ∂A = @thunk _mul_back(ΔY, Q, A, inv(Q.scaleRK))
        return NO_FIELDS, ∂Q, ∂A
    end
    return Y, ldiv_pullback
end

function _mul_back(ΔY, Q, A, s)
    ∂A = similar(ΔY)
    dot!(∂A, Q.T', ΔY, dim = Q.dim)
    ∂A .*= s
    return ∂A
end

### mutating pullbacks need to undo any changes they make to the inputs
function ChainRulesCore.rrule(::typeof(mul!), Y, Q::QDHT, A)
   Ycopy = copy(Y)
   function mul!_pullback(ΔY)
       copyto!(Y, Ycopy)
       ∂Y = DoesNotExist()
       ∂Q = DoesNotExist()
       ∂A = @thunk _mul_back(ΔY, Q, A, Q.scaleRK)
       return NO_FIELDS, ∂Y, ∂Q, ∂A
   end
   return mul!(Y, Q, A), mul!_pullback
end

function ChainRulesCore.rrule(::typeof(ldiv!), Y, Q::QDHT, A)
    Ycopy = copy(Y)
    function ldiv!_pullback(ΔY)
        copyto!(Y, Ycopy)
        ∂Y = DoesNotExist()
        ∂Q = DoesNotExist()
        ∂A = @thunk _mul_back(ΔY, Q, A, inv(Q.scaleRK))
        return NO_FIELDS, ∂Y, ∂Q, ∂A
    end
    return ldiv!(Y, Q, A), ldiv!_pullback
end

## rules for integrateR/integrateK
function ChainRulesCore.frule((ΔA, _), ::typeof(integrateR), A, Q::QDHT; kwargs...)
    return integrateR(A, Q; kwargs...), integrateR(ΔA, Q; kwargs...)
end

function ChainRulesCore.frule((ΔA, _), ::typeof(integrateK), A, Q::QDHT; kwargs...)
    return integrateK(A, Q; kwargs...), integrateK(ΔA, Q; kwargs...)
end

function ChainRulesCore.rrule(::typeof(integrateR), A, Q::QDHT; dim = 1)
    function integrateR_pullback(ΔΩ)
        ∂A = @thunk _integrateRK_back(ΔΩ, A, Q.scaleR; dim = dim)
        return NO_FIELDS, ∂A, DoesNotExist()
    end
    return integrateR(A, Q; dim = dim), integrateR_pullback
end

function ChainRulesCore.rrule(::typeof(integrateK), A, Q::QDHT; dim = 1)
    function integrateK_pullback(ΔΩ)
        ∂A = @thunk _integrateRK_back(ΔΩ, A, Q.scaleK; dim = dim)
        return NO_FIELDS, ∂A, DoesNotExist()
    end
    return integrateK(A, Q; dim = dim), integrateK_pullback
end

_integrateRK_back(ΔΩ, A::AbstractVector, scale; dim = 1) = ΔΩ .* conj.(scale)
function _integrateRK_back(ΔΩ, A::AbstractMatrix, scale; dim = 1)
    return dim == 1 ? conj.(scale) .* ΔΩ : ΔΩ * scale'
end
function _integrateRK_back(ΔΩ, A, scale; dim = 1)
    dims = Tuple(collect(size(A)))
    n = ndims(A)
    scalearray = reshape(
        scale,
        ntuple(_ -> 1, dim - 1)...,
        dims[dim],
        ntuple(_ -> 1, n - dim)...,
    )
    ∂A = ΔΩ .* conj.(scalearray)
    return ∂A
end
