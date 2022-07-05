module ProfileLikelihood 
## Load packages 
using FiniteDiff
using Optimization 
using OptimizationEvolutionary
using OptimizationPolyalgorithms 
using OptimizationNLopt
using OptimizationOptimJL
using OptimizationMultistartOptimization
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
using LatinHypercubeSampling

## Include some code
include("problems.jl");               export LikelihoodProblem, setup_integrator, data, num_params
include("confidence_intervals.jl");   export ConfidenceInterval, bounds
include("solutions.jl");              export LikelihoodSolution, ProfileLikelihoodSolution
include("utils.jl");                  export gaussian_loglikelihood
include("mle.jl");                    export mle, refine, refine_tiktak, refine_lhc
include("step_profile.jl");           ###
include("update_optimiser.jl");       ###    
include("profile.jl");                export profile, confidence_intervals
include("plotting.jl");               export plot_profiles
include("display.jl");                ###
include("transform_results.jl");      export transform_result
end
# new line