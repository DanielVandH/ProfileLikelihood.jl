using ProfileLikelihood
using Test
using FiniteDiff
using Optimization
using OrdinaryDiffEq
using OptimizationNLopt
using OptimizationBBO
using Optimization: OptimizationProblem
using FunctionWrappers
using LinearAlgebra
using PreallocationTools
using Interpolations
using InvertedIndices
using Random
using Distributions
using OptimizationOptimJL
using CairoMakie
using LaTeXStrings
using LatinHypercubeSampling
using FiniteVolumeMethod
using DelaunayTriangulation
using LinearSolve
using SciMLBase
using MuladdMacro
using Base.Threads
using LoopVectorization
const PL = ProfileLikelihood
global SAVE_FIGURE = false

include("templates.jl")
@testset "Utilities" begin
    include("utils.jl")
end
@testset "Problem updates" begin
    include("problem_updates.jl")
end
@testset "LikelihoodProblem" begin
    include("likelihood_problem.jl")
end
@testset "MLE" begin
    include("mle.jl")
end
@testset "RegularGrid" begin
    include("regular_grid.jl")
end
@testset "IrregularGrid" begin
    include("irregular_grid.jl")
end
@testset "GridSearch" begin
    include("grid_search.jl")
end
@testset "ConfidenceInterval" begin
    include("confidence_interval.jl")
end
@testset "ProfileLikelihood" begin
    include("profile_likelihood.jl")
end
@testset "Prediction Intervals" begin
    include("prediction_intervals.jl")
end
@testset "Example I: Regression" begin
    include("regression_example.jl")
end
@testset "Example II: Logistic ODE" begin
    include("logistic.jl")
end
@testset "Example III: Linear Exponential" begin
    include("linear_exponential_example.jl")
end
@testset "Example IV: Heat Equation" begin
    include("heat_equation_example.jl")
end
