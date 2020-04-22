module Hankel
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj, gamma
import LinearAlgebra: mul!, ldiv!, dot
import Base: *, \

export QDHT, integrateK, integrateR, onaxis, symmetric, Rsymmetric

const J₀₀ = besselj(0, 0)

include("utils.jl")
include("qdht.jl")

end
