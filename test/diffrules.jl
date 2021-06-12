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

    @testset "$f(::QDHT, ::Array{<:Real})" for f in (*, \)
        rng = MersenneTwister(86)
        N, M, K = 64, 5, 10
        @testset "Vector" begin
            q = Hankel.QDHT{1,2}(10, N)
            fr = randn(rng, N)
            test_rrule(f, q ⊢ NoTangent(), fr)
            test_frule(f, q ⊢ NoTangent(), fr)
        end

        @testset "Matrix" begin
            @testset for dim in (1, 2)
                q = Hankel.QDHT{1,2}(10, N; dim = dim)
                s = dim == 1 ? (N, M) : (M, N)
                fr = randn(rng, s)
                test_rrule(f, q ⊢ NoTangent(), fr)
                test_frule(f, q ⊢ NoTangent(), fr)
            end
        end

        @testset "Array{<:Real,3}" begin
            q = Hankel.QDHT{1,2}(10, N; dim = 2)
            s = (M, N, K)
            fr = randn(rng, s)
            test_rrule(f, q ⊢ NoTangent(), fr)
            test_frule(f, q ⊢ NoTangent(), fr)
        end
    end

    @testset "$f(::Array{<:Real}, ::QDHT, ::Array{<:Real})" for f in (mul!, ldiv!)
        rng = MersenneTwister(86)
        N, M, K = 64, 5, 10
        @testset "Vector" begin
            q = Hankel.QDHT{1,2}(10, N)
            fr, fk = ntuple(_ -> randn(rng, N), 2)
            test_frule(f, fk, q ⊢ NoTangent(), fr)
        end

        @testset "Matrix" begin
            @testset for dim in (1, 2)
                q = Hankel.QDHT{1,2}(10, N; dim = dim)
                s = dim == 1 ? (N, M) : (M, N)
                fr, fk = ntuple(_ -> randn(rng, s), 2)
                test_frule(f, fk, q ⊢ NoTangent(), fr)
            end
        end

        @testset "Array{<:Real,3}" begin
            q = Hankel.QDHT{1,2}(10, N; dim = 2)
            s = (M, N, K)
            fr, fk = ntuple(_ -> randn(rng, s), 2)
            test_frule(f, fk, q ⊢ NoTangent(), fr)
        end
    end

    @testset "$f(::Array{<:Real}, ::QDHT)" for f in (integrateR, integrateK)
        rng = MersenneTwister(27)
        N, M, K = 64, 5, 10
        q = Hankel.QDHT{1,2}(10, N)
        @testset "Vector" begin
            A = randn(rng, N)
            test_rrule(f, A, q ⊢ NoTangent())
            test_frule(f, A, q ⊢ NoTangent())
        end

        @testset "Matrix" begin
            @testset for dim in (1, 2)
                s = dim == 1 ? (N, M) : (M, N)
                sy = dim == 1 ? (1, M) : (M, 1)
                A = randn(rng, s)
                test_rrule(f, A, q ⊢ NoTangent(); fkwargs = (dim = dim,))
                test_frule(f, A, q ⊢ NoTangent(); fkwargs = (dim = dim,))
            end
        end

        @testset "Array{<:Real,3}" begin
            s = (M, N, K)
            sy = (M, 1, K)
            dim = 2
            A = randn(rng, s)
            test_rrule(f, A, q ⊢ NoTangent(); fkwargs = (dim = dim,))
            test_frule(f, A, q ⊢ NoTangent(); fkwargs = (dim = dim,))
        end
    end
end
