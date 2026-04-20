@testset "Automatic differentiation rules" begin
    @testset "QDHT" begin
        rng = MersenneTwister(68)
        N = 64
        R = 10.0
        q = Hankel.QDHT{1,2}(R, N)
        @testset "rrule" begin
            q_rev, back = rrule(Hankel.QDHT{1,2}, R, N)
            @test typeof(q_rev) === typeof(q)
            for p in propertynames(q)
                @test getproperty(q, p) == getproperty(q_rev, p)
            end
            ∂QDHT, ∂R, ∂N = back(true)
            @test ∂QDHT === NoTangent()
            @test ∂R isa AbstractZero
            @test ∂N isa AbstractZero
        end
        @testset "frule" begin
            ΔQDHT, ΔR, ΔN = NoTangent(), true, true
            q_fwd, ∂q = frule((ΔQDHT, ΔR, ΔN), Hankel.QDHT{1,2}, R, N)
            @test typeof(q_fwd) === typeof(q)
            for p in propertynames(q)
                @test getproperty(q, p) == getproperty(q_fwd, p)
            end
            @test ∂q isa AbstractZero
        end
    end

    @testset "$f" for f in (*, \)
        rng = MersenneTwister(86)
        N, M, K = 64, 5, 10
        @testset "$f(::QDHT, ::Vector{$T})" for T in (Float64,)
            q = Hankel.QDHT{1,2}(10.0, N)
            fr = randn(rng, N)
            test_rrule(f, q, fr)
        end

        @testset "$f(::QDHT, ::Matrix{$T}; dim=$dim)" for T in (Float64,), dim in (1, 2)
            q = Hankel.QDHT{1,2}(10.0, N; dim = dim)
            s = dim == 1 ? (N, M) : (M, N)
            fr = randn(rng, T, s)
            test_rrule(f, q, fr)
        end

        @testset "$f(::QDHT, ::Array{$T,3}; dim=$dim)" for T in (Float64,), dim in (1, 2, 3)
            q = Hankel.QDHT{1,2}(10.0, N; dim = dim)
            s = dim == 1 ? (N, M, K) : (dim == 2 ? (M, N, K) : (M, K, N))
            fr = randn(rng, s)
            test_rrule(f, q, fr)
        end
    end

    @testset "$f" for f in (mul!, ldiv!)
        rng = MersenneTwister(86)
        N, M, K = 64, 5, 10
        @testset "$f(::Vector{$T}, ::QDHT, ::Vector{$T})" for T in (Float64,)
            q = Hankel.QDHT{1,2}(10.0, N)
            fr, fk = ntuple(_ -> randn(rng, T, N), 2)
            test_frule(f, fk, q, fr)
        end

        @testset "$f(::Matrix{$T}, ::QDHT, ::Matrix{$T})" for T in (Float64,), dim in (1, 2)
            q = Hankel.QDHT{1,2}(10.0, N; dim = dim)
            s = dim == 1 ? (N, M) : (M, N)
            fr, fk = ntuple(_ -> randn(rng, T, s), 2)
            test_frule(f, fk, q, fr)
        end

        @testset "$f(::Array{$T,3}, ::QDHT, ::Array{$T,3})" for T in (Float64,), dim in (1, 2, 3)
            q = Hankel.QDHT{1,2}(10.0, N; dim = dim)
            s = dim == 1 ? (N, M, K) : (dim == 2 ? (M, N, K) : (M, K, N))
            fr, fk = ntuple(_ -> randn(rng, T, s), 2)
            test_frule(f, fk, q, fr)
        end
    end

    @testset "dimdot" begin
        rng = MersenneTwister(27)
        N, M, K = 64, 5, 10
        v = randn(rng, N)
        @testset "dimdot(v, A::Vector{$T})" for T in (Float64,)
            A = randn(rng, T, N)
            test_rrule(Hankel.dimdot, v, A; fkwargs = (dim = 1,))
            test_frule(Hankel.dimdot, v, A; fkwargs = (dim = 1,))
        end

        @testset "dimdot(v, A::Matrix{$T}; dim=$dim)" for T in (Float64,), dim in (1, 2)
            s = dim == 1 ? (N, M) : (M, N)
            A = randn(rng, T, s)
            test_rrule(Hankel.dimdot, v, A; fkwargs = (dim = dim,))
            test_frule(Hankel.dimdot, v, A; fkwargs = (dim = dim,))
        end

        @testset "dimdot(v, A::Array{$T,3}; dim=$dim)" for T in (Float64,), dim in (1, 2, 3)
            s = dim == 1 ? (N, M, K) : (dim == 2 ? (M, N, K) : (M, K, N))
            A = randn(rng, T, s)
            test_rrule(Hankel.dimdot, v, A; fkwargs = (dim = dim,))
            test_frule(Hankel.dimdot, v, A; fkwargs = (dim = dim,))
        end
    end
end
