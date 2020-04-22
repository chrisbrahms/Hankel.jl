@testset "QDHT" begin
    @testset "constructors" begin
        q = QDHT(1.0, 10)
        @test q isa QDHT{0,Float64}
        q = QDHT{1}(1.0, 10)
        @test q isa QDHT{1,Float64}
        q = QDHT(2, 1.0, 10)
        @test q isa QDHT{2,Float64}
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
        @test_throws DomainError onaxis(vk, q1)
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
