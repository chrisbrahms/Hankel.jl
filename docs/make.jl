using Documenter
using Hankel
import LinearAlgebra: mul!, ldiv!

makedocs(
    sitename="Hankel.jl",
    pages = [
        "Documentation" => "index.md",
        "Derivations" => "derivations.md",
        "Automatic Differentiation" => "autodiff.md"
    ]
)

deploydocs(
    repo = "github.com/chrisbrahms/Hankel.jl.git",
)
