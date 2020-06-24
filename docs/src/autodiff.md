# Automatic Differentiation

`Hankel` implements the primitives defined by [`ChainRules`](https://github.com/JuliaDiff/ChainRules.jl) for automatic differentiation (AD).
These enables all AD packages that use `ChainRules`' rules to differentiate the exported functions.

Here is an example of reverse-mode automatic differentiation using [`Zygote`](https://github.com/FluxML/Zygote.jl).
To run this example, first call

```julia
julia> using Pkg

julia> Pkg.add(Zygote)
```

Then call the following:

```@repl
using Hankel, Zygote
R, N = 10.0, 10
q = QDHT{0,1}(R, N);
f(r) = exp(-r^2 / 2);
fk = q * f.(q.r)
# Compute the function and a pullback function for computing the gradient
I, back = Zygote.pullback(fk -> integrateR(q \ fk, q), fk);
I
Igrad = only(back(1)) # Compute the gradient
```

This example computes the gradient of the real space integral of the function `f` with respect to each sampled point in the reciprocal space.

## Pushforwards and Pullbacks

For a summary of `ChainRules`' primitives and an introduction to the terminology used here, see the [`ChainRules` docs](http://www.juliadiff.org/ChainRulesCore.jl/stable/).
We define custom rules for 2 reasons:
1. Many AD packages in Julia do not completely support mutating arrays. Since our internal functions mutate, we need custom rules.
2. While for vector samples, the functional form of the pushforwards and pullbacks are simple (see below), they are more complicated for multi-dimensional samples. Providing our own rules helps the AD system sidestep this difficulty.

### The `QDHT` constructor

The `QDHT` objects are intended to be used like `Plan`s in `AbstractFFTs`,
defined once and then potentially used many times.
Consequently, we define its constructor as non-differentiable with respect to its inputs.

### The transform

The quasi-discrete Hankel transform of a vector can be written in component form as

```math
\tilde{f}(k_i) = s \sum_{j=1}^N T_{ij} f(r_j),
```

where ``s = \left(\frac{R}{K}\right)^{(n + 1) / 2}``

The pushforward of the transform is written as

```math
\dot{\tilde{f}}(k_i) = s \sum_{j=1}^N T_{ij} \dot{f}(r_j).
```

That is, the pushforward is just the transform itself.
The transform's pullback is

```math
\overline{f}(r_j) = s \sum_{i=1}^N T_{ij} \overline{\tilde{f}}(k_i).
```

The pushforwards and pullbacks of the inverse transform are similar, where the scalar ``s`` is inverted.

### Integration

Integration using the QDHT can be written as

```math
I = \sum_{i=1}^N v_i f(i),
```

where the elements of ``v_i``, which are precomputed, are given in [Integration of functions](@ref).
The pushforward is then

```math
\dot{I} = \sum_{i=1}^N v_i \dot{f}(r_i),
```

and the pullback is

```math
\overline{f}(r_i) = \overline{I} v_i.
```
