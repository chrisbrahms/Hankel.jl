using Documenter
using Hankel
import LinearAlgebra: mul!, ldiv!

makedocs(
    sitename="Hankel.jl",
    pages = [
        "Documentation" => "index.md",
        "Derivations" => "derivations.md"
    ]
)

deploydocs(
    repo = "github.com/chrisbrahms/Hankel.jl.git",
)