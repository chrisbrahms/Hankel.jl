using Documenter
using Hankel
import LinearAlgebra: mul!, ldiv!

makedocs(
    sitename="Hankel.jl",
)

deploydocs(
    repo = "github.com/chrisbrahms/Hankel.jl.git",
)