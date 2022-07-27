using Test
using CairoMakie
using LinearAlgebra
using Random

using ProfileLikelihood

const bounds = ProfileLikelihood.bounds

include("template_functions.jl")
@testset "Regression" begin
    include("regression.jl")
end
@testset "Linear exponential ODE" begin
    include("linear_exponential_ode.jl")
end
@testset "Logistic ODE" begin
    include("logistic_ode.jl")
end
@testset "Transforming results" begin
    include("transforms.jl")
end
@testset "General tests" begin
    include("general.jl")
end
@testset "Refinement" begin
    include("refinement.jl")
end
@testset "Parameter scaling" begin 
    include("parameter_scaling.jl")
end
@testset "Objective scaling" begin 
    include("objective_scaling.jl")
end
@testset "Grid search" begin 
    include("grid_search.jl")
end