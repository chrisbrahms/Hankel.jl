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
    xscopy = deepcopy(xs)
    y = f(deepcopy(xs)...; fkwargs...)
    y_ad, pullback = Zygote.pullback((xs...) -> f(xs...; fkwargs...), xs...)
    @test isapprox(y_ad, y; rtol = rtol, atol = atol, kwargs...)

    x̄s_ad = pullback(ȳ)

    # Correctness testing via finite differencing.
    for (i, x̄_ad) in enumerate(x̄s_ad)
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
        # test that pullbacks of mutating functions undo the mutation
        if xs[i] isa Number || xs[i] isa AbstractVector
            @test isapprox(xs[i], xscopy[i]; rtol = rtol, atol = atol, kwargs...)
        end
    end
    return nothing
end

@testset "Zygote rules" begin
    @testset "$f(::QDHT, ::Array{$T})" for f in (*, \), T in (Float64, ComplexF64)
        rng = MersenneTwister(86)
        N, M, K = 64, 5, 10
        @testset "Vector" begin
            q = Hankel.QDHT{1,2}(10, N)
            fr, f̄r, f̄k = ntuple(_ -> randn(rng, T, N), 3)
            pullback_test(f, f̄k, (q, nothing), (fr, f̄r))
        end

        @testset "Matrix" begin
            @testset for dim in (1, 2)
                q = Hankel.QDHT{1,2}(10, N; dim = dim)
                s = dim == 1 ? (N, M) : (M, N)
                fr, f̄r, f̄k = ntuple(_ -> randn(rng, T, s), 3)
                pullback_test(f, f̄k, (q, nothing), (fr, f̄r))
            end
        end

        @testset "Array{$T,3}" begin
            q = Hankel.QDHT{1,2}(10, N; dim = 2)
            s = (M, N, K)
            fr, f̄r, f̄k = ntuple(_ -> randn(rng, T, s), 3)
            pullback_test(f, f̄k, (q, nothing), (fr, f̄r))
        end
    end

    @testset "$f(::Array{$T}, ::QDHT, ::Array{$T})" for f in (mul!, ldiv!), T in (Float64, ComplexF64)
        rng = MersenneTwister(86)
        N, M, K = 64, 5, 10
        @testset "Vector" begin
            q = Hankel.QDHT{1,2}(10, N)
            fr, fk, f̄r, f̄k = ntuple(_ -> randn(rng, T, N), 4)
            pullback_test(f, f̄k, (fk, nothing), (q, nothing), (fr, f̄r))
        end

        @testset "Matrix" begin
            @testset for dim in (1, 2)
                q = Hankel.QDHT{1,2}(10, N; dim = dim)
                s = dim == 1 ? (N, M) : (M, N)
                fr, fk, f̄r, f̄k = ntuple(_ -> randn(rng, T, s), 4)
                pullback_test(f, f̄k, (fk, nothing), (q, nothing), (fr, f̄r))
            end
        end

        @testset "Array{$T,3}" begin
            q = Hankel.QDHT{1,2}(10, N; dim = 2)
            s = (M, N, K)
            fr, fk, f̄r, f̄k = ntuple(_ -> randn(rng, T, s), 4)
            pullback_test(f, f̄k, (fk, nothing), (q, nothing), (fr, f̄r))
        end
    end

    @testset "$f(::Array{$T}, ::QDHT)" for f in (integrateR, integrateK), T in (Float64, ComplexF64)
        rng = MersenneTwister(27)
        N, M, K = 64, 5, 10
        @testset "Vector" begin
            q = Hankel.QDHT{1,2}(10, N)
            A, Ā, ȳ = randn(rng, T, N), randn(rng, T, N), randn(rng, T)
            pullback_test(f, ȳ, (A, Ā), (q, nothing))
        end

        @testset "Matrix" begin
            @testset for dim in (1, 2)
                q = Hankel.QDHT{1,2}(10, N; dim = dim)
                s = dim == 1 ? (N, M) : (M, N)
                sy = dim == 1 ? (1, M) : (M, 1)
                A, Ā, ȳ = randn(rng, T, s), randn(rng, T, s), randn(rng, T, sy)
                pullback_test(f, ȳ, (A, Ā), (q, nothing); fkwargs = (; dim = dim))
            end
        end

        @testset "Array{$T,3}" begin
            q = Hankel.QDHT{1,2}(10, N; dim = 2)
            s = (M, N, K)
            sy = (M, 1, K)
            A, Ā, ȳ = randn(rng, T, s), randn(rng, T, s), randn(rng, T, sy)
            pullback_test(f, ȳ, (A, Ā), (q, nothing); fkwargs = (; dim = 2))
        end
    end
end
