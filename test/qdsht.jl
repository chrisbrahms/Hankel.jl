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

logbesseli(ν, x) = x > one(x) ? log(besselix(ν, x)) + x : log(besseli(ν, x))

# log(I_ν(x) / x^ν), approaches -log(2^ν * γ(ν + 1)) as x → 0
logbesselirat(ν, x) = logbesseli(ν, x) - ν * log(x)

# pdf of non-central chi distribution reparameterized with location μ and scale σ
function pdf_ncchi(r, μ, σ, n)
    ν = n + 1
    x = r / σ
    λ = μ / σ
    logJ = -log(σ) # log derivative of f: z → r
    logpdf = -(x^2 + λ^2) / 2 + logbesselirat(ν / 2 - 1, λ * x) + logJ
    return exp(logpdf)
end

# transform of pdf_ncchi. Adapted from Gradshteyn and Ryzhik 6.633.2.
function trans_pdf_ncchi(k, μ, σ, n, p)
    cninv = 1 / Hankel.sphbesselj_scale(n)
    return cninv * exp(-σ^2 / 2 * k^2) * Hankel.sphbesselj(p, n, μ * k)
end

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
            f0 = f(0)
            f0q = Hankel.onaxis(q * f.(q.r), q)
            @test f0 ≈ f0q

            v = f.(q.r)
            vk = q * v
            @testset "2d" begin
                v2d = repeat(v, outer = (1, 16))'
                q2d = Hankel.QDSHT(R, N, dim = 2)
                v2dk = q2d * v2d
                @test all([all(v2dk[ii, :] ≈ vk) for ii in 1:size(v2dk, 1)])
            end
        end

        @testset "f(r) = sinc(100r)²" begin
            R = 4e-2
            N = 256
            q = Hankel.QDHT(R, N)
            f2(r) = sinc(100 * r)^2
            v = f2.(q.r)
            vk = q * v
            f0 = f2(0)
            f0q = Hankel.onaxis(vk, q)
            @test f0 ≈ f0q
        end

        @testset "f(r) = exp(-a²r²/2)cos(16ar)" begin
            R = 1e-2
            N = 256
            q = Hankel.QDSHT(R, N)
            w0 = 5e-3
            a = 2 / w0
            f(r) = exp(-1 // 2 * a^2 * r^2) * cos(16 * a * r)
            test_transform(q, f; atol = 1e-10, quad = false)
            test_l2norm(q, f; atol = 1e-15)
            test_integrate(q, f; atol = 1e-10)
        end

        @testset "pdf non-central chi" begin
            R = 30
            N = 256
            q = Hankel.QDSHT(R, N)
            f(r) = pdf_ncchi(r, 10, 1, q.n)
            fk(k) = trans_pdf_ncchi(k, 10, 1, q.n, q.p)
            test_transform(q, f; fk = fk, atol = 2e-12, quad = false)
            test_l2norm(q, f; quad = true, atol = 1e-15)
            test_integrate(q, f; Ifr = _ -> 1, quad = false, atol = 1e-15)
        end
    end

    @testset "symmetric" begin
        q = QDSHT(10, 128)
        A = exp.(-q.r .^ 2)
        As = symmetric(A, q)
        @test length(As) == 2 * length(A) + 1
        @test As[1:128] == A[128:-1:1]
        @test As[129] ≈ 1
        @test As[130:end] == A

        AA = hcat(A, A)
        AAs = symmetric(AA, q; dim = 1)
        @test size(AAs, 1) == 2 * length(A) + 1
        for i in 1:2
            @test AAs[1:128, i] == A[128:-1:1]
            @test AAs[129, i] ≈ 1
            @test AAs[130:end, i] == A
        end
    end

    @testset "multiple orders/spherical dimensions" begin
        @testset "p=$p, n=$n" for p in (0, 1 / 2, 1, 2, 3), n in (1, 2, 3)
            #= Test case from Guizar-Sicairos and Gutierrez-Vega,
              "Computation of quasi-discrete Hankel transforms of integer order for propagating
              optical wave fields" =#
            # Adapted from Gradshteyn and Ryzhik 6.671.1
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
            if p != 0
                @test_throws DomainError onaxis(vk, q1)
            end
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

            @testset "pdf non-central chi" begin
                R = 30
                N = 256
                q5 = Hankel.QDSHT(p, n, R, N)
                h(r) = pdf_ncchi(r, 10, 1, n)
                hk(k) = trans_pdf_ncchi(k, 10, 1, n, p)

                v = h.(q5.r)
                vk = q5 * v
                vka = hk.(q5.k)
                @test maximum(dynε(vka, vk)) < -10
                @test integrateR(abs2.(v), q5) ≈ integrateK(abs2.(vk), q5)
            end
        end
    end

    @testset "QDSHT === QDHT when n = 1" begin
        @testset "p=$p" for p in (0, 1 / 2, 1, 2, 3, 4)
            R = 1.0
            N = 256
            q = QDSHT(p, 1, R, N)
            qht = QDHT(p, R, N)
            f(r) = exp(-100 / 2 * r^2)
            @test q.r ≈ qht.r
            @test q.k ≈ qht.k
            @test q.T ≈ qht.T
            @test q.scaleR ≈ qht.scaleR
            @test q.scaleK ≈ qht.scaleK
            @test q.j1sq ≈ qht.J1sq
            v = f.(q.r)
            vk = q * v
            @test vk ≈ qht * v
            @test q \ vk ≈ qht \ vk
            @test integrateR(abs2.(v), q) ≈ integrateR(abs2.(v), qht)
            @test integrateK(abs2.(vk), q) ≈ integrateK(abs2.(vk), qht)
            if p == 0
                @test integrateR(v, q) ≈ integrateR(v, qht)
                @test integrateK(vk, q) ≈ integrateK(vk, qht)
                @test Hankel.onaxis(vk, q) ≈ Hankel.onaxis(vk, qht) ≈ f(0)
            end
            qos = Hankel.oversample(q)
            qhtos = Hankel.oversample(qht)
            @test qos.N == qhtos.N
            @test qos.r ≈ qhtos.r
            @test qos.k ≈ qhtos.k
        end
    end

    @testset "big floats" begin
        # SpecialFunctions currently does not allow besselj(::BigFloat, ::Complex{BigFloat})
        @test_throws MethodError QDSHT(BigFloat(0), 128)
    end

    @testset "Oversampling" begin
        R = 4e-2
        N = 256
        w0 = 1e-2
        a = 2 / w0
        q = Hankel.QDSHT(R, N)
        f(r) = exp(-1 // 2 * a^2 * r^2)
        v = f.(q.r)
        vo, qo = Hankel.oversample(v, q, factor = 4)
        @test qo isa QDSHT
        @test qo.N == 4 * q.N
        @test all(isapprox(vo, f.(qo.r), rtol = 1e-13))
        @test integrateR(abs2.(v), q) ≈ integrateR(abs2.(vo), qo)
        @test f(0) ≈ onaxis(qo * vo, qo)

        v2 = hcat(v, v)
        vo, qo = Hankel.oversample(v2, q, factor = 4)
        @test qo isa QDSHT
        @test qo.N == 4 * q.N
        @test all(isapprox(vo[:, 1], f.(qo.r), rtol = 1e-13))
        @test all(isapprox(vo[:, 2], f.(qo.r), rtol = 1e-13))
        @test integrateR(abs2.(v2), q) ≈ integrateR(abs2.(vo), qo)
    end
end
