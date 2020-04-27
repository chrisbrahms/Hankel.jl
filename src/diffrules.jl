# Rules for automatic differentiation

## rules for fwd/rev QDHT
function _mul_back(ΔY, Q, A)
    ∂A = similar(ΔY)
    dot!(∂A, Q.T', ΔY, dim = Q.dim)
    ∂A .*= Q.scaleRK
    return ∂A
end

function _ldiv_back(ΔY, Q, A)
    ∂A = similar(ΔY)
    dot!(∂A, Q.T', ΔY, dim = Q.dim)
    ∂A ./= Q.scaleRK
    return ∂A
end

ZygoteRules.@adjoint function *(Q::QDHT, A)
    Y = Q * A
    return Y, Δ -> (nothing, _mul_back(Δ, Q, A))
end

ZygoteRules.@adjoint function \(Q::QDHT, A)
    Y = Q \ A
    return Y, Δ -> (nothing, _ldiv_back(Δ, Q, A))
end

## rules for dimdot, makes integrateR/K autodiffable
function _dimdot_back(ΔΩ, v, A; dim = 1, dims = Tuple(collect(size(A))))
    T = Base.promote_eltype(v, ΔΩ)
    ∂A = similar(A, T, dims)
    idxlo = CartesianIndices(dims[1:(dim - 1)])
    idxhi = CartesianIndices(dims[(dim + 1):end])
    for lo in idxlo, hi in idxhi
        ∂A[lo, :, hi] .= ΔΩ[lo, 1, hi] .* v
    end
    return ∂A
end
_dimdot_back(ΔΩ, v, A::AbstractVector; dim = 1) = ΔΩ .* v

ZygoteRules.@adjoint function dimdot(v, A; dim = 1)
    return dimdot(v, A; dim = dim), Δ -> (nothing, _dimdot_back(Δ, v, A; dim = dim))
end
