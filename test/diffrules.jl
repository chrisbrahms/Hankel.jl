# adapted from ChainRulesTestUtils.frule_test
# allows non-numeric/array arguments and respects `nothing` tangents
function frule_test(
    f,
    xẋs::Tuple{Any,Any}...;
    rtol = 1e-9,
    atol = 1e-9,
    fkwargs = NamedTuple(),
    fdm = central_fdm(5, 1),
    kwargs...,
)
    xs, ẋs = first.(xẋs), last.(xẋs)

    y = f(deepcopy(xs)...; fkwargs...)
    y_ad, dy_ad = frule((NoTangent(), deepcopy(ẋs)...), f, deepcopy(xs)...; fkwargs...)
    if y isa Number || y isa AbstractVector
        @test isapprox(y_ad, y; rtol = rtol, atol = atol, kwargs...)
    end

    function f2(args...; kwargs...)
        allargs = ()
        j = 1
        for i in eachindex(xs)
            if ẋs[i] === nothing
                allargs = (allargs..., xs[i])
            else
                allargs = (allargs..., args[j])
                j += 1
            end
        end
        @assert j == length(args) + 1
        @assert length(allargs) == length(xs)
        return f(allargs...; kwargs...)
    end

    arginds = findall(ẋs .!== nothing)
    length(arginds) > 0 || return nothing
    xsargs = deepcopy(xs[arginds])
    ẋsargs = deepcopy(ẋs[arginds])

    dy_fd = jvp(fdm, xs -> f2(xs...; fkwargs...), (xsargs, ẋsargs))
    @test isapprox(
        collect(unthunk.(dy_ad)),  # Use collect so can use vector equality
        collect(dy_fd);
        rtol = rtol,
        atol = atol,
        kwargs...,
    )
    return nothing
end

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
            ∂QDHT, ∂R, ∂N = back(One())
            @test ∂QDHT === NoTangent()
            @test ∂R isa AbstractZero
            @test ∂N isa AbstractZero
        end
        @testset "frule" begin
            ΔQDHT, ΔR, ΔN = NoTangent(), One(), One()
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
            fr, ḟr, f̄r, f̄k = ntuple(_ -> randn(rng, N), 4)
            rrule_test(f, f̄k, (q, nothing), (fr, f̄r))
            frule_test(f, (q, nothing), (fr, ḟr))
        end

        @testset "Matrix" begin
            @testset for dim in (1, 2)
                q = Hankel.QDHT{1,2}(10, N; dim = dim)
                s = dim == 1 ? (N, M) : (M, N)
                fr, ḟr, f̄r, f̄k = ntuple(_ -> randn(rng, s), 4)
                rrule_test(f, f̄k, (q, nothing), (fr, f̄r))
                frule_test(f, (q, nothing), (fr, ḟr))
            end
        end

        @testset "Array{<:Real,3}" begin
            q = Hankel.QDHT{1,2}(10, N; dim = 2)
            s = (M, N, K)
            fr, ḟr, f̄r, f̄k = ntuple(_ -> randn(rng, s), 4)
            rrule_test(f, f̄k, (q, nothing), (fr, f̄r))
            frule_test(f, (q, nothing), (fr, ḟr))
        end
    end

    @testset "$f(::Array{<:Real}, ::QDHT, ::Array{<:Real})" for f in (mul!, ldiv!)
        rng = MersenneTwister(86)
        N, M, K = 64, 5, 10
        @testset "Vector" begin
            q = Hankel.QDHT{1,2}(10, N)
            fr, fk, ḟr, ḟk, f̄r, f̄k = ntuple(_ -> randn(rng, N), 6)
            frule_test(f, (fk, ḟk), (q, nothing), (fr, ḟr))
        end

        @testset "Matrix" begin
            @testset for dim in (1, 2)
                q = Hankel.QDHT{1,2}(10, N; dim = dim)
                s = dim == 1 ? (N, M) : (M, N)
                fr, fk, ḟr, ḟk, f̄r, f̄k = ntuple(_ -> randn(rng, s), 6)
                frule_test(f, (fk, ḟk), (q, nothing), (fr, ḟr))
            end
        end

        @testset "Array{<:Real,3}" begin
            q = Hankel.QDHT{1,2}(10, N; dim = 2)
            s = (M, N, K)
            fr, fk, ḟr, ḟk, f̄r, f̄k = ntuple(_ -> randn(rng, s), 6)
            frule_test(f, (fk, ḟk), (q, nothing), (fr, ḟr))
        end
    end

    @testset "$f(::Array{<:Real}, ::QDHT)" for f in (integrateR, integrateK)
        rng = MersenneTwister(27)
        N, M, K = 64, 5, 10
        q = Hankel.QDHT{1,2}(10, N)
        @testset "Vector" begin
            A, Ȧ, Ā = ntuple(_ -> randn(rng, N), 3)
            ȳ = randn(rng)
            rrule_test(f, ȳ, (A, Ā), (q, nothing))
            frule_test(f, (A, Ȧ), (q, nothing))
        end

        @testset "Matrix" begin
            @testset for dim in (1, 2)
                s = dim == 1 ? (N, M) : (M, N)
                sy = dim == 1 ? (1, M) : (M, 1)
                A, Ȧ, Ā = ntuple(_ -> randn(rng, s), 3)
                ȳ = randn(rng, sy)
                rrule_test(f, ȳ, (A, Ā), (q, nothing); fkwargs = (dim = dim,))
                frule_test(f, (A, Ȧ), (q, nothing); fkwargs = (dim = dim,))
            end
        end

        @testset "Array{<:Real,3}" begin
            s = (M, N, K)
            sy = (M, 1, K)
            dim = 2
            A, Ȧ, Ā = ntuple(_ -> randn(rng, s), 3)
            ȳ = randn(rng, sy)
            rrule_test(f, ȳ, (A, Ā), (q, nothing); fkwargs = (dim = dim,))
            frule_test(f, (A, Ȧ), (q, nothing); fkwargs = (dim = dim,))
        end
    end
end
