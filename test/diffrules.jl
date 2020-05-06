# adapted from ChainRulesTestUtils.pullback_test
function pullback_test(
    f,
    ȳ,
    xx̄s::Tuple{Any,Any}...;
    rtol = 1e-9,
    atol = 1e-9,
    fkwargs = NamedTuple(),
    fdm = central_fdm(5, 1),
    kwargs...,
)
    # Check correctness of evaluation.
    xs, x̄s = collect(zip(xx̄s...))
    y, pullback = Zygote.pullback((xs...) -> f(xs...; fkwargs...), xs...)
    @test isapprox(y, f(xs...; fkwargs...); rtol = rtol, atol = atol, kwargs...)

    x̄s_ad = pullback(ȳ)

    # Correctness testing via finite differencing.
    return for (i, x̄_ad) in enumerate(x̄s_ad)
        if x̄s[i] === nothing
            @test x̄_ad === nothing
        else
            x̄_fd = j′vp(
                fdm,
                x -> f(xs[1:(i - 1)]..., x, xs[(i + 1):end]...; fkwargs...),
                ȳ,
                xs[i],
            )[1]
            @test isapprox(x̄_ad, x̄_fd; rtol = rtol, atol = atol, kwargs...)
        end
    end
end

@testset "Zygote rules" begin
    @testset for T in [Float64, ComplexF64]
        @testset "*(::AbstractQDHT, ::Array{$T})" begin
            rng = MersenneTwister(86)
            @testset "Vector" begin
                N = 64
                q = Hankel.QDHT{1,2}(10, N)
                fr, f̄r, f̄k = randn(rng, T, N), randn(rng, T, N), randn(rng, T, N)
                pullback_test(*, f̄k, (q, nothing), (fr, f̄r))
            end

            @testset "Matrix" begin
                N = 64
                M = 5
                @testset for dim in (1, 2)
                    q = Hankel.QDHT{1,2}(10, N; dim = dim)
                    s = dim == 1 ? (N, M) : (M, N)
                    fr, f̄r, f̄k = randn(rng, T, s), randn(rng, T, s), randn(rng, T, s)
                    pullback_test(*, f̄k, (q, nothing), (fr, f̄r))
                end
            end

            @testset "Array{$T,3}" begin
                N = 64
                M = 5
                K = 10
                q = Hankel.QDHT{1,2}(10, N; dim = 2)
                s = (M, N, K)
                fr, f̄r, f̄k = randn(rng, T, s), randn(rng, T, s), randn(rng, T, s)
                pullback_test(*, f̄k, (q, nothing), (fr, f̄r))
            end
        end

        @testset "\\(::AbstractQDHT, ::Array{$T})" begin
            rng = MersenneTwister(62)
            @testset "Vector" begin
                N = 64
                q = Hankel.QDHT{1,2}(10, N)
                fr, f̄r, f̄k = randn(rng, T, N), randn(rng, T, N), randn(rng, T, N)
                pullback_test(\, f̄k, (q, nothing), (fr, f̄r))
            end

            @testset "Matrix" begin
                N = 64
                M = 5
                @testset for dim in (1, 2)
                    q = Hankel.QDHT{1,2}(10, N; dim = dim)
                    s = dim == 1 ? (N, M) : (M, N)
                    fr, f̄r, f̄k = randn(rng, T, s), randn(rng, T, s), randn(rng, T, s)
                    pullback_test(\, f̄k, (q, nothing), (fr, f̄r))
                end
            end

            @testset "Array{$T,3}" begin
                N = 64
                M = 5
                K = 10
                q = Hankel.QDHT{1,2}(10, N; dim = 2)
                s = (M, N, K)
                fr, f̄r, f̄k = randn(rng, T, s), randn(rng, T, s), randn(rng, T, s)
                pullback_test(\, f̄k, (q, nothing), (fr, f̄r))
            end
        end

        @testset "integrateR/K(::AbstractMatrix, ::Array{$T})" begin
            rng = MersenneTwister(27)
            @testset "Vector" begin
                N = 64
                q = Hankel.QDHT{1,2}(10, N)
                A, Ā, ȳ = randn(rng, T, N), randn(rng, T, N), randn(rng, T)
                pullback_test(integrateR, ȳ, (A, Ā), (q, nothing))
                pullback_test(integrateK, ȳ, (A, Ā), (q, nothing))
            end

            @testset "Matrix" begin
                N = 64
                M = 5
                @testset for dim in (1, 2)
                    q = Hankel.QDHT{1,2}(10, N)
                    s = dim == 1 ? (N, M) : (M, N)
                    sy = dim == 1 ? (1, M) : (M, 1)
                    A, Ā, ȳ = randn(rng, T, s), randn(rng, T, s), randn(rng, T, sy)
                    pullback_test(integrateR, ȳ, (A, Ā), (q, nothing); fkwargs = (; dim = dim))
                    pullback_test(integrateK, ȳ, (A, Ā), (q, nothing); fkwargs = (; dim = dim))
                end
            end

            @testset "Array{$T,3}" begin
                N = 64
                M = 5
                K = 10
                q = Hankel.QDHT{1,2}(10, N)
                s = (M, N, K)
                sy = (M, 1, K)
                A, Ā, ȳ = randn(rng, T, s), randn(rng, T, s), randn(rng, T, sy)
                pullback_test(integrateR, ȳ, (A, Ā), (q, nothing); fkwargs = (; dim = 2))
                pullback_test(integrateK, ȳ, (A, Ā), (q, nothing); fkwargs = (; dim = 2))
            end
        end
    end
end
