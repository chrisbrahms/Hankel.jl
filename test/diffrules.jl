# adapted from ChainRulesTestUtils.rrule_test with a few changes:
# - Allows kwargs to be passed to function
# - Allows the output of the rrule to be only approximately equal to the output of the usual
#   function
# - Checks that pullbacks of mutating functions reverse the mutation
function rrule_test(
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
    y_ad, pullback = ChainRulesCore.rrule(f, xs...; fkwargs...)
    if y isa Number || y isa AbstractVector
        @test isapprox(y_ad, y; rtol = rtol, atol = atol, kwargs...)
    end

    @assert !(isa(ȳ, Thunk))
    ∂s = pullback(ȳ)
    ∂self = ∂s[1]
    x̄s_ad = ∂s[2:end]
    @test ∂self === NO_FIELDS

    # Correctness testing via finite differencing.
    for (i, x̄_ad) in enumerate(x̄s_ad)
        if x̄s[i] === nothing
            @test x̄_ad isa DoesNotExist
        else
            xscopy2 = deepcopy(xscopy)
            x̄_fd = j′vp(
                fdm,
                x -> f(xscopy2[1:(i - 1)]..., x, xscopy2[(i + 1):end]...; fkwargs...),
                ȳ,
                xscopy2[i],
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

# adapted from ChainRulesTestUtils.frule_test
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
    y_ad, dy_ad = frule((NO_FIELDS, deepcopy(ẋs)...), f, deepcopy(xs)...; fkwargs...)
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
        collect(extern.(dy_ad)),  # Use collect so can use vector equality
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
        rrule_test(Hankel.QDHT{1,2}, nothing, (R, nothing), (N, nothing))
        frule_test(Hankel.QDHT{1,2}, (R, nothing), (N, nothing))
    end

    @testset "$f(::QDHT, ::Array{$T})" for f in (*, \), T in (Float64, ComplexF64)
        rng = MersenneTwister(86)
        N, M, K = 64, 5, 10
        @testset "Vector" begin
            q = Hankel.QDHT{1,2}(10, N)
            fr, ḟr, f̄r, f̄k = ntuple(_ -> randn(rng, T, N), 4)
            rrule_test(f, f̄k, (q, nothing), (fr, f̄r))
            frule_test(f, (q, nothing), (fr, ḟr))
        end

        @testset "Matrix" begin
            @testset for dim in (1, 2)
                q = Hankel.QDHT{1,2}(10, N; dim = dim)
                s = dim == 1 ? (N, M) : (M, N)
                fr, ḟr, f̄r, f̄k = ntuple(_ -> randn(rng, T, s), 4)
                rrule_test(f, f̄k, (q, nothing), (fr, f̄r))
                frule_test(f, (q, nothing), (fr, ḟr))
            end
        end

        @testset "Array{$T,3}" begin
            q = Hankel.QDHT{1,2}(10, N; dim = 2)
            s = (M, N, K)
            fr, ḟr, f̄r, f̄k = ntuple(_ -> randn(rng, T, s), 4)
            rrule_test(f, f̄k, (q, nothing), (fr, f̄r))
            frule_test(f, (q, nothing), (fr, ḟr))
        end
    end

    @testset "$f(::Array{$T}, ::QDHT, ::Array{$T})" for f in (mul!, ldiv!),
        T in (Float64, ComplexF64)

        rng = MersenneTwister(86)
        N, M, K = 64, 5, 10
        @testset "Vector" begin
            q = Hankel.QDHT{1,2}(10, N)
            fr, fk, ḟr, ḟk, f̄r, f̄k = ntuple(_ -> randn(rng, T, N), 6)
            frule_test(f, (fk, ḟk), (q, nothing), (fr, ḟr))
        end

        @testset "Matrix" begin
            @testset for dim in (1, 2)
                q = Hankel.QDHT{1,2}(10, N; dim = dim)
                s = dim == 1 ? (N, M) : (M, N)
                fr, fk, ḟr, ḟk, f̄r, f̄k = ntuple(_ -> randn(rng, T, s), 6)
                frule_test(f, (fk, ḟk), (q, nothing), (fr, ḟr))
            end
        end

        @testset "Array{$T,3}" begin
            q = Hankel.QDHT{1,2}(10, N; dim = 2)
            s = (M, N, K)
            fr, fk, ḟr, ḟk, f̄r, f̄k = ntuple(_ -> randn(rng, T, s), 6)
            frule_test(f, (fk, ḟk), (q, nothing), (fr, ḟr))
        end
    end

    @testset "$f(::Array{$T}, ::QDHT)" for f in (integrateR, integrateK),
        T in (Float64, ComplexF64)

        rng = MersenneTwister(27)
        N, M, K = 64, 5, 10
        q = Hankel.QDHT{1,2}(10, N)
        @testset "Vector" begin
            A, Ȧ, Ā = ntuple(_ -> randn(rng, T, N), 3)
            ȳ = randn(rng, T)
            rrule_test(f, ȳ, (A, Ā), (q, nothing))
            frule_test(f, (A, Ȧ), (q, nothing))
        end

        @testset "Matrix" begin
            @testset for dim in (1, 2)
                s = dim == 1 ? (N, M) : (M, N)
                sy = dim == 1 ? (1, M) : (M, 1)
                A, Ȧ, Ā = ntuple(_ -> randn(rng, T, s), 3)
                ȳ = randn(rng, T, sy)
                rrule_test(f, ȳ, (A, Ā), (q, nothing); fkwargs = (; dim = dim))
                frule_test(f, (A, Ȧ), (q, nothing); fkwargs = (; dim = dim))
            end
        end

        @testset "Array{$T,3}" begin
            s = (M, N, K)
            sy = (M, 1, K)
            dim = 2
            A, Ȧ, Ā = ntuple(_ -> randn(rng, T, s), 3)
            ȳ = randn(rng, T, sy)
            rrule_test(f, ȳ, (A, Ā), (q, nothing); fkwargs = (; dim = dim))
            frule_test(f, (A, Ȧ), (q, nothing); fkwargs = (; dim = dim))
        end
    end
end
