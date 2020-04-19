import Test: @test, @test_throws, @testset
using Hankel
import LinearAlgebra: diagm, mul!, ldiv!
import SpecialFunctions: besseli, besselix, besselj
import HCubature: hquadrature

# Brute-force equivalent of Hankel.dot! - slow but certain to be correct
function slowdot!(out, M, V; dim=1)
    idxlo = CartesianIndices(size(V)[1:dim-1])
    idxhi = CartesianIndices(size(V)[dim+1:end])
    _slowdot!(out, M, V, idxlo, idxhi)
end

function _slowdot!(out, M, V, idxlo, idxhi)
    for lo in idxlo
        for hi in idxhi
            view(out, lo, :, hi) .= M * view(V, lo, :, hi)
        end
    end
end

@testset "multiplication" begin
    M = diagm(0 => [1, 2, 3])
    V = 2 .* ones((3, 3, 2))
    out = similar(V)
    Hankel.dot!(out, M, V)
    @test all(out[1, :, :] .== 2)
    @test all(out[2, :, :] .== 4)
    @test all(out[3, :, :] .== 6)
    Hankel.dot!(out, M, V, dim=2)
    @test all(out[:, 1, :] .== 2)
    @test all(out[:, 2, :] .== 4)
    @test all(out[:, 3, :] .== 6)
    @test_throws DomainError Hankel.dot!(out, M, V, dim=3)

    M = rand(32, 32)
    V = rand(32)
    for N = 1:5
        for n = 1:N
            shape = 16*ones(Int64, N)
            shape[n] = 32
            V = rand(shape...)
            out = similar(V)
            out2 = similar(out)
            Hankel.dot!(out, M, V, dim=n)
            slowdot!(out2, M, V, dim=n)
            @test all(out2 .≈ out)
        end
    end
end

include("qdht.jl")
include("qdsht.jl")
