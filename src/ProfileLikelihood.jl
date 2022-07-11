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
using Dierckx 
using PreallocationTools
using Printf 
using LatinHypercubeSampling
using Requires

## Include some code
include("constructor_utils.jl");      ###
include("problems.jl");               export LikelihoodProblem, setup_integrator, data, num_params
include("confidence_intervals.jl");   export ConfidenceInterval, bounds
include("solutions.jl");              export LikelihoodSolution, ProfileLikelihoodSolution
include("utils.jl");                  export gaussian_loglikelihood
include("mle.jl");                    export mle, refine, refine_tiktak, refine_lhc
include("step_profile.jl");           ###
include("update_optimiser.jl");       ###    
include("profile.jl");                export profile, confidence_intervals
function __init__()
    @require CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0" begin 
        include("plotting.jl")
        export plot_profiles
    end
end
include("display.jl");                ###
include("transform_results.jl");      export transform_result
end
# new line