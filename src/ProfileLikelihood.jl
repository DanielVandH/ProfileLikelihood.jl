module ProfileLikelihood 
## Load packages 
using FiniteDiff
using Optimization 
using OptimizationEvolutionary
using OptimizationPolyalgorithms 
using OptimizationNLopt
using OptimizationOptimJL
using Random 
using Distributions 
using LaTeXStrings 
using DifferentialEquations 
using LinearAlgebra 
using LoopVectorization 
using InvertedIndices 
using CairoMakie 
using Dierckx 
using PreallocationTools
using Printf 

## Include some code
include("problems.jl");               export LikelihoodProblem, setup_integrator
include("confidence_intervals.jl");   export ConfidenceInterval, bounds
include("solutions.jl");              export LikelihoodSolution, ProfileLikelihoodSolution
include("utils.jl");                  export gaussian_loglikelihood
include("mle.jl");                    export mle
include("step_profile.jl");           ###
include("update_optimiser.jl");       ###    
include("profile.jl");                export profile, confidence_intervals
include("plotting.jl");               export plot_profiles
include("display.jl");                ###
end
# new line