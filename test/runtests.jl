import Test: @test, @test_throws, @testset
using Hankel
import LinearAlgebra: diagm, mul!
import SpecialFunctions: besselj
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

@testset "transform" begin
    R = 4e-2
    N = 256
    w0 = 1e-3
    a = 2/w0
    q = Hankel.QDHT(R, N)
    @test all(isreal.(q.T))
    f(r) = exp(-1//2 * a^2 * r^2)
    fk(k) = 1/a^2 * exp(-k^2/(2*a^2))
    v = f.(q.r)
    global vv = copy(v)
    for _ = 1:1000
        tmp = q * vv
        global vv = q \ tmp
    end
    @test all(isapprox.(v, vv, atol=2e-12))
    vk = q * v
    vka = fk.(q.k)
    fki(k) = hquadrature(r -> r.*f(r).*besselj(0, k.*r), 0, R)[1]
    @test all(isapprox.(vka, vk, atol=5e-22))
    @test fki(q.k[1]) ≈ vk[1] # doing all of them takes too long
    @test fki(q.k[128]) ≈ vk[128]
    Er = Hankel.integrateR(v.^2, q)
    Ek = Hankel.integrateK(vk.^2, q)
    @test Er ≈ Ek
    Er_c = hquadrature(r -> r.*f(r).^2, 0, 1)
    @test Er_c[1] ≈ Er
    # Test that in-place transform works
    vk2 = similar(vk)
    vk3 = copy(v)
    mul!(vk2, q, v)
    @test all(vk2 .≈ vk)

    v2d = repeat(v, outer=(1, 16))'
    q2d = Hankel.QDHT(R, N, dim=2)
    v2dk = q2d * v2d
    @test all([all(v2dk[ii, :] ≈ vk) for ii = 1:size(v2dk, 1)])

    f0 = f(0)
    f0q = Hankel.onaxis(vk, q)
    @test f0 ≈ f0q

    f2(r) = sinc(100*r)^2
    v = f2.(q.r);
    vk = q * v
    f0 = f2(0)
    f0q = Hankel.onaxis(vk, q)
    @test f0 ≈ f0q

    q = QDHT(10, 128); A = exp.(-q.r.^2)
    As = symmetric(A, q)
    @test length(As) == 2*length(A)+1
    @test As[1:128] == A[128:-1:1]
    @test As[129] ≈ 1
    @test As[130:end] == A

    AA = hcat(A, A)
    AAs = symmetric(AA, q; dim=1)
    @test size(AAs, 1) == 2*length(A)+1
    for i=1:2
        @test AAs[1:128, i] == A[128:-1:1]
        @test AAs[129, i] ≈ 1
        @test AAs[130:end, i] == A
    end
end

@testset "big floats" begin
    R = 4e-2
    N = 256
    w0 = 4e-3
    a = 2/w0
    q = QDHT(BigFloat(R), 128)
    f(r) = exp(-1//2 * a^2 * r^2)
    fk(k) = 1/a^2 * exp(-k^2/(2*a^2))
    v = f.(q.r)
    global vv = copy(v)
    for _ = 1:100
        tmp = q * vv
        global vv = q \ tmp
    end
    @test all(isapprox.(v, vv, atol=2e-12))
    vk = q * v
    vka = fk.(q.k)
    @test all(isapprox.(vka, vk, atol=5e-22))
end

@testset "Gaussian divergence" begin
    q = Hankel.QDHT(12.7e-3, 512)
    λ = 800e-9
    k = 2π/λ
    kz = @. sqrt(k^2 - q.k^2)
    z = 2 # propagation distance
    prop = @. exp(1im*kz*z)
    w0 = 200e-6 # start at focus
    w1 = w0*sqrt(1+(z*λ/(π*w0^2))^2)
    Ir0 = exp.(-2*(q.r/w0).^2)
    Ir1 = exp.(-2*(q.r/w1).^2)*(w0/w1)^2 # analytical solution (in paraxial approx)
    Er0 = sqrt.(Ir0)
    Ek0 = q * Er0
    Ek1 = prop .* Ek0
    Er1 = q \ Ek1
    @test isapprox(abs2.(Er1), Ir1, rtol=1e-6)
    energy = π/2*w0^2 # analytical energy for gaussian beam
    @test energy ≈ 2π*Hankel.integrateR(Ir0, q)
    @test energy ≈ 2π*Hankel.integrateR(abs2.(Er1), q)
    @test energy ≈ 2π*Hankel.integrateK(abs2.(Ek1), q)
end

@testset "Oversampling" begin
    R = 4e-2
    N = 256
    w0 = 1e-2
    a = 2/w0
    q = Hankel.QDHT(R, N)
    f(r) = exp(-1//2 * a^2 * r^2)
    v = f.(q.r)
    vo, qo = Hankel.oversample(v, q, factor=4)
    @test qo.N == 4*q.N
    @test all(isapprox(vo, f.(qo.r), rtol=1e-13))
    @test integrateR(abs2.(v), q) ≈ integrateR(abs2.(vo), qo)
    @test f(0) ≈ onaxis(qo*vo, qo)

    v2 = hcat(v, v)
    vo, qo = Hankel.oversample(v2, q, factor=4)
    @test qo.N == 4*q.N
    @test all(isapprox(vo[:, 1], f.(qo.r), rtol=1e-13))
    @test all(isapprox(vo[:, 2], f.(qo.r), rtol=1e-13))
    @test integrateR(abs2.(v2), q) ≈ integrateR(abs2.(vo), qo)
end
