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

## rules for dimdot
function ChainRulesCore.frule((_, Δv, ΔA), ::typeof(dimdot), v, A::AbstractVector; dim = 1)
    return dot(v, A), dot(v, ΔA) + dot(Δv, A)
end
function ChainRulesCore.frule((_, Δv, ΔA), ::typeof(dimdot), v, A; dim = 1)
    out = dimdot(v, A; dim = dim)
    ∂out = zero(out)
    ∂v, ∂A = unthunk(Δv), unthunk(ΔA)
    ∂v isa AbstractZero || dimdot!(∂out, ∂v, A; dim = dim)
    ∂A isa AbstractZero || dimdot!(∂out, v, ∂A; dim = dim)
    return out, ∂out
end

function ChainRulesCore.rrule(::typeof(dimdot), v, A; dim = 1)
    project_A = ProjectTo(A)
    project_v = ProjectTo(v)
    function dimdot_pullback(Δout)
        ∂out = unthunk(Δout)
        ∂v = @thunk project_v(_dimdot_v_back(∂out, v, A, dim))
        ∂A = @thunk project_A(_dimdot_A_back(∂out, v, A, dim))
        return NoTangent(), ∂v, ∂A
    end
    return dimdot(v, A; dim = dim), dimdot_pullback
end

_dimdot_A_back(Δout, v, A::AbstractVector, dim) = v .* Δout
function _dimdot_A_back(Δout, v, A::AbstractMatrix, dim)
    return dim == 1 ? v * Δout : Δout * transpose(v)
end
function _dimdot_A_back(Δout, v, A, dim)
    N = length(v)
    sz = ntuple(i -> ifelse(i == dim, N, 1), ndims(A))
    v_arr = reshape(v, sz)
    return v_arr .* Δout
end

_dimdot_v_back(Δout, v, A::AbstractVecOrMat, dim) = vec(dim == 1 ? A * Δout' : Δout' * A)
function _dimdot_v_back(Δout, v, A, dim)
    N = size(A, dim)
    perm = ((1:(dim-1))..., ((dim+1):ndims(A))..., dim)
    A_mat = reshape(permutedims(A, perm), (:, N))
    ∂out_mat = vec(permutedims(Δout, perm))
    return vec(∂out_mat' * A_mat)
end
