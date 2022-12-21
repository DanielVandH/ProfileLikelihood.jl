using Documenter, ProfileLikelihood
using CairoMakie, LaTeXStrings

makedocs(sitename = "ProfileLikelihood.jl",
modules = [ProfileLikelihood],
pages = [
    "Home" => "index.md"
    "Interface" => "interface.md"
    "Docstrings" => "docstrings.md"
    "Example I: Multiple linear regression" => "regression.md"
    "Example II: Logistic ordinary differential equation" => "logistic.md"
    "Example III: Linear exponential ODE and grid searching" => "exponential.md"
    "Example IV: Diffusion equation on a square plate" => "heat.md"
    "Mathematical and Implementation Details" => "math.md"
])

deploydocs(;
    repo="github.com/DanielVandH/ProfileLikelihood.jl",
    devbranch="main"
)