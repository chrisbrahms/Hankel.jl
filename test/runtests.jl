import Test: @test, @test_throws, @testset
using Hankel
import LinearAlgebra: diagm, mul!, ldiv!
import SpecialFunctions: besseli, besselix, besselj
import HCubature: hquadrature
using FiniteDifferences
using Random
using ChainRulesCore
using ChainRulesTestUtils

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

@testset "sphbesselj functions" begin
    @testset "sphbessel_scale" begin
        @test Hankel.sphbesselj_scale(1) == 1
        @test Hankel.sphbesselj_scale(3) == 1
        @test Hankel.sphbesselj_scale(5) == 1
        @test Hankel.sphbesselj_scale(2) == √(π/2)
        @test Hankel.sphbesselj_scale(4) == √(π/2)
        @test Hankel.sphbesselj_scale(6) == √(π/2)
    end

    @testset "sphbesselj" begin
        @test Hankel.sphbesselj(0, 1, 0) ≈ 1
        @test Hankel.sphbesselj(0, 2, 0) ≈ 1
        @test Hankel.sphbesselj(0, 3, 0) ≈ 1 / 2
        @test Hankel.sphbesselj(1, 1, 0) ≈ 0
        @test Hankel.sphbesselj(1, 2, 0) ≈ 0
        @test Hankel.sphbesselj(2, 1, 0) ≈ 0
        @test Hankel.sphbesselj(2, 2, 0) ≈ 0
        @test Hankel.sphbesselj(0, 1, 0.2) == besselj(0, 0.2)
        @test Hankel.sphbesselj(1, 1, 0.2) == besselj(1, 0.2)
        @test Hankel.sphbesselj(2, 1, 0.2) == besselj(2, 0.2)
        @test Hankel.sphbesselj(0, 2, 0.2) ≈ besselj(0.5, 0.2) * √(π / 2 / 0.2)
        @test Hankel.sphbesselj(1, 2, 0.2) ≈ besselj(1.5, 0.2) * √(π / 2 / 0.2)
        @test Hankel.sphbesselj(2, 2, 0.2) ≈ besselj(2.5, 0.2) * √(π / 2 / 0.2)
        @test Hankel.sphbesselj(0, 3, 0.2) == besselj(1, 0.2) / 0.2
        @test Hankel.sphbesselj(1, 3, 0.2) == besselj(2, 0.2) / 0.2
        @test Hankel.sphbesselj(2, 3, 0.2) == besselj(3, 0.2) / 0.2
        @test Hankel.sphbesselj(0, 4, 0.2) ≈ besselj(1.5, 0.2) * √(π / 2 / 0.2^3)
        @test Hankel.sphbesselj(1, 4, 0.2) ≈ besselj(2.5, 0.2) * √(π / 2 / 0.2^3)
        @test Hankel.sphbesselj(2, 4, 0.2) ≈ besselj(3.5, 0.2) * √(π / 2 / 0.2^3)
    end

    @testset "sphbesselj_zero($p, $n, m) for m = 1:200" for p in [1:4; 10; 20; 50; 100; 500], n in 1:4
        zs = Hankel.sphbesselj_zero.(p, n, 1:200)
        # test sequence is monotonic
        @test all(>(0), zs[2:end] .- zs[1:end-1])
        # test are zeros
        @test all(x -> isapprox(x, 0; atol=1e-12), Hankel.sphbesselj.(p, n, zs))
    end
end

include("qdht.jl")
include("diffrules.jl")
