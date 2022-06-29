using Test
using ProfileLikelihood

@testset "Regression" begin
    include("regression.jl")
end
@testset "Linear exponential ODE" begin
    include("linear_exponential_ode.jl")
end
@testset "Logistic ODE" begin
    include("logistic_ode.jl")
end