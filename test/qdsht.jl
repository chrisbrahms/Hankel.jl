function quad_transform(q::QDSHT, f; rmin = 0, rmax = q.R)
    return hquadrature(rmin, rmax) do r
        return @. r^q.n * f(r) * Hankel.sphbesselj(q.p, q.n, q.k * q.r) /
                  Hankel.sphbesselj_scale(q.n)
    end[1]
end

function quad_integrate(q::QDSHT, f; rmin = 0, rmax = q.R)
    return hquadrature(rmin, rmax) do r
        return @. r^q.n * f(r)
    end[1]
end

function test_transform(
    q::QDSHT,
    f;
    fk = nothing,
    quad = true,
    nroundtrip = 1000,
    inplace = true,
    kwargs...,
)
    v = f.(q.r)
    vk = q * v
    @test eltype(vk) === eltype(v)

    inplace && @testset "inplace" begin
        vk2 = similar(vk)
        mul!(vk2, q, v)
        @test all(isapprox.(vk2, vk; kwargs...))

        v2 = similar(v)
        ldiv!(v2, q, vk)
        @test all(isapprox.(v2, v; kwargs...))
    end

    @testset "roundtrip" begin
        vv = copy(v)
        tmp = q * vv
        for _ in 1:nroundtrip
            vv = q \ tmp
            tmp = q * vv
        end
        @test all(isapprox.(v, vv; kwargs...))
    end

    fk !== nothing && @testset "analytical" begin
        vka = fk.(q.k)
        @test all(isapprox.(vk, vka; kwargs...))
    end

    quad && @testset "quadrature" begin
        vki = quad_transform(q, f)
        @test all(isapprox.(vk, vki; kwargs...))
    end
    return nothing
end

function test_l2norm(q::QDSHT, f; Efr = nothing, quad = true, kwargs...)
    v = f.(q.r)
    vk = q * v
    Er = integrateR(v .^ 2, q)
    Ek = integrateK(vk .^ 2, q)
    @test all(isapprox.(Er, Ek; kwargs...))

    Efr !== nothing && @testset "analytical" begin
        Era = Efr.(q.r)
        @test all(isapprox.(Er, Era; kwargs...))
    end

    quad && @testset "quadrature" begin
        Eri = quad_integrate(q, r -> f(r)^2)
        @test all(isapprox.(Er, Eri; kwargs...))
    end
    return nothing
end

function test_integrate(q::QDSHT, f; Ifr = nothing, quad = true, kwargs...)
    v = f.(q.r)
    Ir = integrateR(v, q)

    Ifr !== nothing && @testset "analytical" begin
        Ira = Ifr.(q.r)
        @test all(isapprox.(Ir, Ira; kwargs...))
    end

    quad && @testset "quadrature" begin
        Iri = quad_integrate(q, f)
        @test all(isapprox.(Ir, Iri; kwargs...))
    end
    return nothing
end

dynε(ext, est) = 20 * log10.(abs.(ext .- est) ./ maximum(abs.(est)))

@testset "QDSHT" begin
    @testset "transform" begin
        @testset "f(r) = exp(-a²r²/2)" begin
            R = 4e-2
            N = 256
            q = Hankel.QDSHT(R, N)
            w0 = 1e-3
            a = 2 / w0
            @test all(isreal.(q.T))
            f(r) = exp(-1 // 2 * a^2 * r^2)
            fk(k) = 1 / a^3 * exp(-k^2 / (2 * a^2))
            test_transform(q, f; fk = fk, atol = 1e-9)
            test_l2norm(q, f; atol = 1e-20)
            test_integrate(q, f; atol = 1e-20)

            v = f.(q.r)
            vk = q * v
            @testset "2d" begin
                v2d = repeat(v, outer = (1, 16))'
                q2d = Hankel.QDSHT(R, N, dim = 2)
                v2dk = q2d * v2d
                @test all([all(v2dk[ii, :] ≈ vk) for ii in 1:size(v2dk, 1)])
            end
        end

        @testset "f(r) = exp(-a²r²/2)cos(16ar)" begin
            R = 4e-2
            N = 256
            q = Hankel.QDSHT(R, N)
            w0 = 5e-3
            a = 2 / w0
            f(r) = exp(-1 // 2 * a^2 * r^2) * cos(16 * a * r)
            test_transform(q, f; atol = 1e-10, quad = false)
            test_l2norm(q, f; atol = 1e-20)
            # quadrature takes too long here
            # test_integrate(q, f; quad = false, atol=1e-20)
        end
    end

    @testset "non-zero orders" begin
        #= Test case from Guizar-Sicairos and Gutierrez-Vega,
          "Computation of quasi-discrete Hankel transforms of integer order for propagating
          optical wave fields" =#
        # Adapted from Gradshteyn and Ryzhik 6.671.1
        @testset "p=$p, n=$n" for p in (1, 1 / 2, 2, 3), n in (1, 2, 3)
            γ = 5
            f(r) = r^(-(n + 1) / 2) * sin(2π * γ * r)
            function fk(k)
                s = p + (n - 1) / 2
                return if k < 2π * γ
                    return k^s * cos(s * π / 2) / sqrt(4 * π^2 * γ^2 - k^2) /
                           (2π * γ + sqrt(4 * π^2 * γ^2 - k^2))^s / k^((n - 1) / 2)
                else
                    return sin(s * asin(2π * γ / k)) / sqrt(k^2 - 4 * π^2 * γ^2) /
                           k^((n - 1) / 2)
                end
            end

            R = 1
            N = 256
            q1 = QDSHT(p, n, R, N)
            v = f.(q1.r)
            vk = q1 * v
            vka = fk.(q1.k)
            err = dynε(vka, vk)
            i = findlast(ki -> ki < 2π * γ, q1.k) # find point flanking discontinuity
            @test maximum(dynε(vka, vk)[[1:(i - 1); (i + 2):end]]) < -10
            # @test_throws DomainError onaxis(vk, q1)
            @test integrateR(abs2.(v), q1) ≈ integrateK(abs2.(vk), q1)
            # curve is too steep for quadrature, so compare with integral computed with
            # Mathematica
            @test isapprox(integrateR(abs2.(v), q1), 2.358965372, rtol = 1e-2)

            R = 4e-2
            N = 256
            w0 = 1e-2
            a = 2 / w0
            q4 = QDSHT(p, n, R, N)
            g(r) = exp(-1 // 2 * a^2 * r^2)
            v = g.(q4.r)
            vk = q4 * v
            @test integrateR(abs2.(v), q4) ≈ integrateK(abs2.(vk), q4)
            # note the high tolerance here is due to the lack of samples close to the axis
            @test isapprox(
                integrateR(abs2.(v), q4),
                quad_integrate(q4, abs2 ∘ g),
                rtol = 1e-2,
            )
        end
    end

    @testset "big floats" begin
        # SpecialFunctions currently does not allow besselj(::BigFloat, ::Complex{BigFloat})
        @test_throws MethodError QDSHT(BigFloat(0), 128)
    end
end
