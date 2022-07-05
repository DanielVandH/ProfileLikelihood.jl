using LaTeXStrings
using Random
using Distributions
using OptimizationNLopt
using DifferentialEquations
using Test
using PreallocationTools
using LinearAlgebra
using Optimization
using Dierckx
using LoopVectorization

## Regression 
nprob, loglikk, θ, dat = MultipleLinearRegression()
nsol = mle(nprob)
@test_throws "Problem must have finite" refine(nsol; local_method=NLopt.LD_LBFGS())

prob, loglikk, θ, dat = MultipleLinearRegressionBounded()
sol = mle(prob, NLopt.LD_LBFGS(), maxiters=5)
Random.seed!(282881)
refined_sol = refine(sol)
@test mle(refined_sol) ≈ mle(nsol)
@test maximum(refined_sol) ≈ maximum(nsol)

refined_sol = refine(sol; n=250)
@test mle(refined_sol) ≈ mle(nsol)
@test maximum(refined_sol) ≈ maximum(nsol)

refined_sol = refine(sol; local_method=NLopt.LN_NELDERMEAD())
@test mle(refined_sol) ≈ mle(nsol)
@test maximum(refined_sol) ≈ maximum(nsol)

#=
## Linear exponential ODE 
prob, loglikk, θ, yᵒ, n = LogisticODE()
λ, σ, y₀ = θ
sol = mle(prob, NLopt.LN_NELDERMEAD())
=#