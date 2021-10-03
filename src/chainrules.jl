# Rules for automatic differentiation

## Constructor
function ChainRulesCore.frule((_, ΔR, _), ::Type{QT}, R0, N; kwargs...) where {QT<:QDHT}
    Q = QT(R0, N; kwargs...)
    n = sphericaldim(Q)
    R = Q.R
    K = Q.K

    ∂c = ΔR / R
    ∂d = (n + 1) * ∂c
    ∂Q = Tangent{typeof(Q)}(;
        K = -K * ∂c,
        k = @thunk(Q.k .* -∂c),
        R = ΔR,
        r = @thunk(Q.r .* ∂c),
        scaleR = @thunk(Q.scaleR .* ∂d),
        scaleK = @thunk(Q.scaleK .* -∂d),
        scaleRK = scaleRK * ∂d,
    )
    return Q, ∂Q
end

function ChainRulesCore.rrule(::Type{QT}, R0, N; kwargs...) where {QT<:QDHT}
    Q = QT(R0, N; kwargs...)
    project_R0 = ProjectTo(R0)
    function QDHT_pullback(ΔQ)
        n = sphericaldim(Q)
        ∂Q = unthunk(ΔQ)
        ∂d = muladd(Q.scaleRK, ∂Q.scaleRK, dot(Q.scaleR, ∂Q.scaleR) - dot(Q.scaleK, ∂Q.scaleK))
        ∂c = muladd(-Q.K, ∂Q.K, (n + 1) * ∂d - dot(Q.k, ∂Q.k) + dot(Q.r, ∂Q.r))
        ∂R0 = project_R0(∂c / Q.R + ∂Q.R)
        return (NoTangent(), ∂R0, NoTangent())
    end
    return Q, QDHT_pullback
end

## rules for fwd/rev transform
function ChainRulesCore.frule((_, ΔY, ΔQ, ΔA), ::typeof(mul!), Y, Q::QDHT, A)
    mul!(Y, Q, A)
    mul!(ΔY, Q, unthunk(ΔA))
    axpy!(ΔQ.∂scaleRK / Q.scaleRK, Y, ΔY)
    return Y, ΔY
end
function ChainRulesCore.frule((_, ΔY, ΔQ, ΔA), ::typeof(ldiv!), Y, Q::QDHT, A)
    ldiv!(Y, Q, A)
    ldiv!(ΔY, Q, unthunk(ΔA))
    axpy!(-ΔQ.∂scaleRK / Q.scaleRK, Y, ΔY)
    return Y, ΔY
end

function ChainRulesCore.rrule(::typeof(*), Q::QDHT, A)
    Y = Q * A
    project_A = ProjectTo(A)
    project_scaleRK = ProjectTo(Q.scaleRK)
    function mul_pullback(ΔY)
        ∂Y = unthunk(ΔY)
        ∂A = @thunk _mul_back(∂Y, Q, A, Q.scaleRK)
        ∂Q = Tangent{typeof(Q)}(; scaleRK = @thunk(project_scaleRK(dot(Y, ∂Y) / Q.scaleRK)))
        return NoTangent(), ∂Q, project_A(∂A)
    end
    return Y, mul_pullback
end

function ChainRulesCore.rrule(::typeof(\), Q::QDHT, A)
    Y = Q \ A
    project_A = ProjectTo(A)
    project_scaleRK = ProjectTo(Q.scaleRK)
    function ldiv_pullback(ΔY)
        ∂Y = unthunk(ΔY)
        c = inv(Q.scaleRK)
        ∂A = _mul_back(∂Y, Q, A, c)
        ∂Q = Tangent{typeof(Q)}(; scaleRK = @thunk(project_scaleRK(dot(Y, ∂Y) * -c)))
        return NoTangent(), ∂Q, project_A(∂A)
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
    return integrateR(A, Q; kwargs...), integrateR(unthunk(ΔA), Q; kwargs...)
end

function ChainRulesCore.frule((_, ΔA, _), ::typeof(integrateK), A, Q::QDHT; kwargs...)
    return integrateK(A, Q; kwargs...), integrateK(unthunk(ΔA), Q; kwargs...)
end

function ChainRulesCore.rrule(::typeof(integrateR), A, Q::QDHT; dim = 1)
    function integrateR_pullback(ΔΩ)
        ∂A = _integrateRK_back(unthunk(ΔΩ), A, Q.scaleR; dim = dim)
        return NoTangent(), ∂A, NoTangent()
    end
    return integrateR(A, Q; dim = dim), integrateR_pullback
end

function ChainRulesCore.rrule(::typeof(integrateK), A, Q::QDHT; dim = 1)
    function integrateK_pullback(ΔΩ)
        ∂A = _integrateRK_back(unthunk(ΔΩ), A, Q.scaleK; dim = dim)
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
