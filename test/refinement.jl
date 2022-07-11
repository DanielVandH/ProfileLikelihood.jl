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
using OptimizationMultistartOptimization
using OptimizationPolyalgorithms

## Latin grids 
Random.seed!(92992881)
prob, loglikk, θ, dat = MultipleLinearRegressionBounded()
latin_grid = ProfileLikelihood.get_lhc_params(prob, 25, 1000)'
@test latin_grid ≈ [7.5 1.66667 -5.83333 10.0 -0.833333
    8.33333 -2.5 10.0 2.5 -4.16667
    0.416667 2.5 -2.5 1.66667 -9.16667
    7.08333 4.16667 3.33333 -10.0 1.66667
    1.0e-12 -5.83333 6.66667 -5.0 -3.33333
    7.91667 7.5 1.66667 -1.66667 -10.0
    2.91667 -9.16667 -1.66667 -2.5 -6.66667
    4.16667 -6.66667 -10.0 0.0 5.0
    5.41667 9.16667 -8.33333 -0.833333 7.5
    6.25 -5.0 0.0 -8.33333 10.0
    3.33333 10.0 9.16667 0.833333 -2.5
    0.833333 -4.16667 -3.33333 6.66667 5.83333
    5.83333 -8.33333 4.16667 5.0 8.33333
    2.5 -10.0 5.83333 8.33333 -1.66667
    3.75 3.33333 7.5 9.16667 3.33333
    4.58333 -1.66667 -9.16667 7.5 -8.33333
    1.66667 0.833333 -5.0 -9.16667 2.5
    9.58333 -0.833333 -0.833333 -4.16667 -7.5
    1.25 5.0 5.0 -3.33333 6.66667
    10.0 -3.33333 -6.66667 -6.66667 4.16667
    6.66667 -7.5 8.33333 -7.5 -5.83333
    9.16667 0.0 2.5 3.33333 9.16667
    5.0 6.66667 -7.5 -5.83333 -5.0
    2.08333 8.33333 -4.16667 5.83333 0.833333
    8.75 5.83333 0.833333 4.16667 0.0] atol = 1e-4
for j in 1:5 
    for i in 1:25 
        @test ProfileLikelihood.lower_bounds(prob, j) ≤ latin_grid[i, j] ≤ ProfileLikelihood.upper_bounds(prob, j)
    end
end

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
@test refined_sol.alg == (TikTak(250, 25, 0.1, 0.995, 0.5), NLopt.LN_NELDERMEAD)

refined_sol = refine(sol; local_method=NLopt.LN_NELDERMEAD())
@test mle(refined_sol) ≈ mle(nsol)
@test maximum(refined_sol) ≈ maximum(nsol)
@test refined_sol.alg[1] == MultistartOptimization.TikTak(100, 10, 0.1, 0.995, 0.5)
@test refined_sol.alg[2] == NLopt.LN_NELDERMEAD

## Linear exponential ODE 
Random.seed!(29999988)
prob, loglikk, θ, yᵒ, n = LinearExponentialODE()
nsol = mle(prob, NLopt.LN_NELDERMEAD())
refined_sol = refine(nsol)
@test mle(refined_sol) ≈ mle(nsol) atol=1e-3
@test maximum(refined_sol) ≈ maximum(nsol)
@test refined_sol.alg == (TikTak(100, 10, 0.1, 0.995, 0.5), NLopt.LN_NELDERMEAD)

Random.seed!(28881777)
refined_sol = refine(nsol; method = :lhc)
@test mle(refined_sol) ≈ mle(nsol) atol=1e-3
@test maximum(refined_sol) ≈ maximum(nsol)
@test refined_sol.alg == NLopt.LN_NELDERMEAD()

refined_sol = refine(nsol, NLopt.LN_BOBYQA(); gens = 500, method = :lhc)
@test mle(refined_sol) ≈ mle(nsol) atol=1e-3
@test maximum(refined_sol) ≈ maximum(nsol) atol=1e-3
@test refined_sol.alg == NLopt.LN_BOBYQA()

nsol = mle(prob, NLopt.LN_NELDERMEAD(); maxiters = 2)
refined_sol = refine(nsol; method = :lhc)
@test maximum(refined_sol) > maximum(nsol)
@test mle(refined_sol) ≈ mle(prob, NLopt.LN_NELDERMEAD()).θ atol=1e-3

## Logistic ODE 
Random.seed!(9999)
prob, loglikk, θ, yᵒ, n = LogisticODE()
nsol = mle(prob, NLopt.LN_NELDERMEAD())
refined_sol = refine(nsol)
@test mle(refined_sol) ≈ mle(nsol) atol=1e-3
@test maximum(refined_sol) ≈ maximum(nsol) atol=1e-3
@test refined_sol.alg == (TikTak(100, 10, 0.1, 0.995, 0.5), NLopt.LN_NELDERMEAD)

Random.seed!(28881777)
refined_sol = refine(nsol; method = :lhc)
@test mle(refined_sol) ≈ mle(nsol) atol=1e-3
@test maximum(refined_sol) ≈ maximum(nsol) atol=1e-3
@test refined_sol.alg == NLopt.LN_NELDERMEAD()

refined_sol = refine(nsol, NLopt.LN_BOBYQA(); gens = 500, method = :lhc)
@test mle(refined_sol) ≈ mle(nsol) atol=1e-3
@test maximum(refined_sol) ≈ maximum(nsol) atol=1e-3
@test refined_sol.alg == NLopt.LN_BOBYQA()

nsol = mle(prob, NLopt.LN_NELDERMEAD(); maxiters = 2)
refined_sol = refine(nsol; method = :lhc)
@test maximum(refined_sol) > maximum(nsol)
@test mle(refined_sol) ≈ mle(prob, NLopt.LN_NELDERMEAD()).θ atol=1e-3


