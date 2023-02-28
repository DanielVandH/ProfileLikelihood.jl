using ProfileLikelihood
using Documenter

DocMeta.setdocmeta!(ProfileLikelihood, :DocTestSetup, :(using ProfileLikelihood);
    recursive=true)

makedocs(sitename="ProfileLikelihood.jl",
    modules=[ProfileLikelihood],
    authors="Daniel VandenHeuvel <danj.vandenheuvel@gmail.com>",
    repo="https://github.com/DanielVandH/ProfileLikelihood.jl/blob/{commit}{path}#{line}",
    sitename="ProfileLikelihood.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://DanielVandH.github.io/ProfileLikelihood.jl",
        edit_link="main",
        assets=String[]),
    pages=[
        "Home" => "index.md"
        "Interface" => "interface.md"
        "Docstrings" => "docstrings.md"
        "Example I: Multiple linear regression" => "regression.md"
        "Example II: Logistic ordinary differential equation" => "logistic.md"
        "Example III: Linear exponential ODE and grid searching" => "exponential.md"
        "Example IV: Diffusion equation on a square plate" => "heat.md"
        "Example V: Lotka-Volterra ODE, GeneralLazyBufferCache, and computing bivarate profile likelihoods" => "lotka.md"
        "Mathematical and Implementation Details" => "math.md"
    ])

deploydocs(;
    repo="github.com/DanielVandH/ProfileLikelihood.jl",
    devbranch="main"
)