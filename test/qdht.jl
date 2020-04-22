function quad_transform(q::QDHT{p,n}, f; rmin = 0, rmax = q.R) where {p,n}
    return hquadrature(rmin, rmax) do r
        return @. r^n * f(r) * Hankel.sphbesselj(p, n, q.k * q.r) /
                  Hankel.sphbesselj_scale(n)
    end[1]
end

function quad_integrate(q::QDHT{p,n}, f; rmin = 0, rmax = q.R) where {p,n}
    return hquadrature(rmin, rmax) do r
        return @. r^n * f(r)
    end[1]
end

function test_transform(
    q::QDHT,
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

function test_l2norm(q::QDHT, f; Efr = nothing, quad = true, kwargs...)
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

function test_integrate(q::QDHT, f; Ifr = nothing, quad = true, kwargs...)
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

@testset "QDHT" begin
    @testset "constructors" begin
        q = QDHT(1.0, 10)
        @test q isa QDHT{0,1,Float64}
        q = QDHT{1}(1.0, 10)
        @test q isa QDHT{1,1,Float64}
        q = QDHT(2, 1.0, 10)
        @test q isa QDHT{2,1,Float64}
        q = QDHT{1, 2}(1.0, 10)
        @test q isa QDHT{1,2,Float64}
        q = QDHT(1, 2, 1.0, 10)
        @test q isa QDHT{1,2,Float64}
    end

    @testset "cylindrical" begin
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
            # Test integral of function without squaring
            ir = Hankel.integrateR(v, q)
            ir_c = hquadrature(r -> r*f(r), 0, 1)
            @test ir_c[1] ≈ ir
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

            # test also with a different function which isn't positive definite
            w0 = 5e-3
            a = 2/w0
            f(r) = exp(-1//2 * a^2 * r^2) * cos(16*a*r)
            v = f.(q.r);
            vk = q * v;
            Ir = Hankel.integrateR(v.^2, q)
            Ik = Hankel.integrateK(vk.^2, q)
            Ir_c = hquadrature(r -> r*f(r)^2, 0, 1)
            @test Ir_c[1] ≈ Ir ≈ Ik
            # Test integral of function without squaring
            ir = Hankel.integrateR(v, q)
            ir_c = hquadrature(r -> r*f(r), 0, 1)
            @test ir_c[1] ≈ ir

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

        @testset "non-zero orders" begin
            #= Test case from Guizar-Sicairos and Gutierrez-Vega,
              "Computation of quasi-discrete Hankel transforms of integer order for propagating
              optical wave fields" =#
            γ = 5
            f(r) = sinc(2γ*r)
            function fk(ν, p)
                if ν < γ
                    return ν^p * cos(p*π/2) / (2π*γ * sqrt(γ^2 - ν^2)*(γ + sqrt(γ^2 - ν^2))^p)
                else
                    return sin(p*asin(γ/ν))/(2π*γ*sqrt(ν^2 - γ^2))
                end
            end

            R = 3
            N = 256
            q1 = QDHT(1, R, N)
            v = f.(q1.r);
            vk = q1*v;
            vka = fk.(q1.k/2π, 1)/2π
            dynε(ext, est) = 20*log10.(abs.(ext.-est)./maximum(est))
            @test maximum(dynε(vka, vk)) < -10
            @test_throws MethodError onaxis(vk, q1)
            @test integrateR(abs2.(v), q1) ≈ integrateK(abs2.(vk), q1)
            @test isapprox(integrateR(abs2.(v), q1), hquadrature(r -> r*f(r)^2, 0, 3)[1], rtol=1e-2)


            q4 = QDHT(4, R, N)
            v = f.(q4.r);
            vk = q4*v;
            vka = fk.(q4.k/2π, 4)/2π
            dynε(ext, est) = 20*log10.(abs.(ext.-est)./maximum(est))
            @test maximum(dynε(vka, vk)) < -10
            @test integrateR(abs2.(v), q4) ≈ integrateK(abs2.(vk), q4)

            R = 4e-2
            N = 256
            w0 = 1e-2
            a = 2/w0
            q4 = QDHT(4, R, N)
            f(r) = exp(-1//2 * a^2 * r^2)
            fk(k) = 1/a^2 * exp(-k^2/(2*a^2))
            v = f.(q4.r)
            vk = q4 * v
            @test integrateR(abs2.(v), q4) ≈ integrateK(abs2.(vk), q4)
            # note the high tolerance here is due to the lack of samples close to the axis
            @test isapprox(integrateR(abs2.(v), q4), hquadrature(r -> r*f(r)^2, 0, R)[1], rtol=1e-2)

            R = 4e-2
            N = 256
            w0 = 1e-2
            a = 2/w0
            q12 = QDHT(1/2, R, N)
            f(r) = exp(-1//2 * a^2 * r^2)
            fk(k) = 1/a^2 * exp(-k^2/(2*a^2))
            v = f.(q12.r)
            vk = q12 * v
            @test integrateR(abs2.(v), q12) ≈ integrateK(abs2.(vk), q12)
            @test isapprox(integrateR(abs2.(v), q12), hquadrature(r -> r*f(r)^2, 0, R)[1], rtol=1e-3)
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
    end

    @testset "spherical" begin
        @testset "transform" begin
            @testset "f(r) = exp(-a²r²/2)" begin
                R = 4e-2
                N = 256
                q = Hankel.QDHT{0,2}(R, N)
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
                    q2d = Hankel.QDHT{0,2}(R, N, dim = 2)
                    v2dk = q2d * v2d
                    @test all([all(v2dk[ii, :] ≈ vk) for ii in 1:size(v2dk, 1)])
                end
            end

            # @testset "f(r) = sinc(100r)²" begin
            #     R = 4e-2
            #     N = 256
            #     q = Hankel.QDHT{0,2}(R, N)
            #     f(r) = sinc(100 * r)^2
            #     v = f.(q.r)
            #     vk = q * v
            #     f0 = f(0)
            #     f0q = Hankel.onaxis(vk, q)
            #     @test f0 ≈ f0q
            # end

            @testset "f(r) = exp(-a²r²/2)cos(16ar)" begin
                R = 1e-2
                N = 256
                q = Hankel.QDHT{0,2}(R, N)
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
                q = Hankel.QDHT{0,2}(R, N)
                f(r) = pdf_ncchi(r, 10, 1, 2)
                fk(k) = trans_pdf_ncchi(k, 10, 1, 2, 0)
                test_transform(q, f; fk = fk, atol = 2e-12, quad = false)
                test_l2norm(q, f; quad = true, atol = 1e-15)
                test_integrate(q, f; Ifr = _ -> 1, quad = false, atol = 1e-15)
            end
        end

        @testset "symmetric" begin
            q = QDHT{0,2}(10, 128)
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
                q1 = QDHT{p,n}(R, N)
                v = f.(q1.r)
                vk = q1 * v
                vka = fk.(q1.k)
                err = dynε(vka, vk)
                i = findlast(ki -> ki < 2π * γ, q1.k) # find point flanking discontinuity
                @test maximum(dynε(vka, vk)[[1:(i - 1); (i + 2):end]]) < -10
                if p != 0
                    @test_throws MethodError onaxis(vk, q1)
                end
                @test integrateR(abs2.(v), q1) ≈ integrateK(abs2.(vk), q1)
                # curve is too steep for quadrature, so compare with integral computed with
                # Mathematica
                @test isapprox(integrateR(abs2.(v), q1), 2.358965372, rtol = 1e-2)

                R = 4e-2
                N = 256
                w0 = 1e-2
                a = 2 / w0
                q4 = QDHT{p,n}(R, N)
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
                    q5 = Hankel.QDHT{p,n}(R, N)
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

        @testset "big floats" begin
            # SpecialFunctions currently does not allow besselj(::BigFloat, ::Complex{BigFloat})
            @test_throws MethodError QDHT{0,2}(BigFloat(0), 128)
        end

        @testset "Oversampling" begin
            R = 4e-2
            N = 256
            w0 = 1e-2
            a = 2 / w0
            q = Hankel.QDHT{0,2}(R, N)
            f(r) = exp(-1 // 2 * a^2 * r^2)
            v = f.(q.r)
            vo, qo = Hankel.oversample(v, q, factor = 4)
            @test qo isa QDHT{0,2}
            @test qo.N == 4 * q.N
            @test all(isapprox(vo, f.(qo.r), rtol = 1e-13))
            @test integrateR(abs2.(v), q) ≈ integrateR(abs2.(vo), qo)
            @test f(0) ≈ onaxis(qo * vo, qo)

            v2 = hcat(v, v)
            vo, qo = Hankel.oversample(v2, q, factor = 4)
            @test qo isa QDHT{0,2}
            @test qo.N == 4 * q.N
            @test all(isapprox(vo[:, 1], f.(qo.r), rtol = 1e-13))
            @test all(isapprox(vo[:, 2], f.(qo.r), rtol = 1e-13))
            @test integrateR(abs2.(v2), q) ≈ integrateR(abs2.(vo), qo)
        end
    end
end
