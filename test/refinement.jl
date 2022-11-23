using LaTeXStrings
using Random
using Distributions
using OptimizationNLopt
using DifferentialEquations
using Test
using PreallocationTools
using LinearAlgebra
using Optimization
using Interpolations
using LoopVectorization
using OptimizationMultistartOptimization
using OptimizationPolyalgorithms
using OptimizationOptimJL
using OptimizationBBO

## Multiple algorithms 
prob, loglikk, θ, dat = MultipleLinearRegressionBounded()
sol = mle(prob, (BBO_adaptive_de_rand_1_bin_radiuslimited(), NLopt.LN_NELDERMEAD))
@test maximum(sol) ≈ loglikk(reduce(vcat, θ), data(prob)) rtol = 1e-2
@test mle(sol) ≈ reduce(vcat, θ) rtol = 1e-2
@test ProfileLikelihood.algorithm_name(sol) == "Nelder-Mead simplex algorithm (local, no-derivative)"

prob, loglikk, θ, dat = MultipleLinearRegressionBounded()
_sol = mle(prob, (BBO_adaptive_de_rand_1_bin_radiuslimited(), NLopt.LN_NELDERMEAD, LBFGS(), Optim.LBFGS()))
@test maximum(_sol) ≈ loglikk(reduce(vcat, θ), data(prob)) rtol = 1e-2
@test mle(_sol) ≈ reduce(vcat, θ) rtol = 1e-2
@test algorithm_name(_sol) == :LBFGS

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
nsol = mle(nprob, LBFGS())
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

prof = profile(prob, refined_sol; alg=NLopt.LN_NELDERMEAD, abstol=1e-1, resolution=10, min_steps=6)
oldprof = deepcopy(prof)
fig = plot_profiles(prof; spline=false)
profile!(prof, 4; alg=NLopt.LD_LBFGS(), abstol=1e-13)
fig = plot_profiles(prof; spline=false)
@test prof.θ[4] ≠ oldprof.θ[4]
@test prof.profile[4] ≠ oldprof.profile[4]
@test prof.confidence_intervals[4] ≠ oldprof.confidence_intervals[4]

## Linear exponential ODE 
Random.seed!(29999988)
prob, loglikk, θ, yᵒ, n = LinearExponentialODE()
nsol = mle(prob, NLopt.LN_NELDERMEAD())
refined_sol = refine(nsol)
@test mle(refined_sol) ≈ mle(nsol) atol = 1e-3
@test maximum(refined_sol) ≈ maximum(nsol)
@test refined_sol.alg == (TikTak(100, 10, 0.1, 0.995, 0.5), NLopt.LN_NELDERMEAD)

Random.seed!(28881777)
refined_sol = refine(nsol; method=:lhc)
@test mle(refined_sol) ≈ mle(nsol) atol = 1e-3
@test maximum(refined_sol) ≈ maximum(nsol)
@test refined_sol.alg == NLopt.LN_NELDERMEAD()

refined_sol = refine(nsol, NLopt.LN_BOBYQA(); gens=500, method=:lhc)
@test mle(refined_sol) ≈ mle(nsol) atol = 1e-3
@test maximum(refined_sol) ≈ maximum(nsol) atol = 1e-3
@test refined_sol.alg == NLopt.LN_BOBYQA()

nsol = mle(prob, NLopt.LN_NELDERMEAD(); maxiters=2)
refined_sol = refine(nsol; method=:lhc)
@test maximum(refined_sol) > maximum(nsol)
@test mle(refined_sol) ≈ mle(prob, NLopt.LN_NELDERMEAD()).θ atol = 1e-3

## Logistic ODE 
Random.seed!(9999)
prob, loglikk, θ, yᵒ, n = LogisticODE()
nsol = mle(prob, NLopt.LN_NELDERMEAD())
refined_sol = refine(nsol)
@test mle(refined_sol) ≈ mle(nsol) atol = 1e-3
@test maximum(refined_sol) ≈ maximum(nsol) atol = 1e-3
@test refined_sol.alg == (TikTak(100, 10, 0.1, 0.995, 0.5), NLopt.LN_NELDERMEAD)

Random.seed!(28881777)
refined_sol = refine(nsol; method=:lhc)
@test mle(refined_sol) ≈ mle(nsol) atol = 1e-3
@test maximum(refined_sol) ≈ maximum(nsol) atol = 1e-3
@test refined_sol.alg == NLopt.LN_NELDERMEAD()

refined_sol = refine(nsol, NLopt.LN_BOBYQA(); gens=500, method=:lhc)
@test mle(refined_sol) ≈ mle(nsol) atol = 1e-3
@test maximum(refined_sol) ≈ maximum(nsol) atol = 1e-3
@test refined_sol.alg == NLopt.LN_BOBYQA()

nsol = mle(prob, NLopt.LN_NELDERMEAD(); maxiters=2)
refined_sol = refine(nsol; method=:lhc)
@test maximum(refined_sol) > maximum(nsol)
@test mle(refined_sol) ≈ mle(prob, NLopt.LN_NELDERMEAD()).θ atol = 1e-3

# Refining a profile 
prof = profile(prob, refined_sol; resolution=100)
oldprof = deepcopy(prof)
profile!(prof)
for i in 1:4 # didn't change anything, so nothing should really change 
    @test prof.θ[i] == oldprof.θ[i]
    @test norm(prof.profile[i] - oldprof.profile[i]) < 1e-5
    @test norm(prof.other_mles[i] - oldprof.other_mles[i]) < 1e-4
end

prof = deepcopy(oldprof)
profile!(prof; n=1)
@test prof.θ[1] == oldprof.θ[1]
@test norm(prof.profile[1] - oldprof.profile[1]) < 1e-5
@test norm(prof.other_mles[1] - oldprof.other_mles[1]) < 1e-4

prof = deepcopy(oldprof)
profile!(prof; n=[1, 4])
for i in [1, 4]
    @test prof.θ[i] == oldprof.θ[i]
    @test norm(prof.profile[i] - oldprof.profile[i]) < 1e-5
    @test norm(prof.other_mles[i] - oldprof.other_mles[i]) < 1e-4
end

prof = deepcopy(oldprof)
profile!(prof; n=1, alg=NLopt.LN_BOBYQA)
@test prof.θ[1] == oldprof.θ[1]
@test norm(prof.profile[1] - oldprof.profile[1]) < 1e-5
@test norm(prof.other_mles[1] - oldprof.other_mles[1]) < 1e-5

prof = deepcopy(oldprof)
profile!(prof; n=3, alg=NLopt.LN_BOBYQA)
@test prof.θ[3] == oldprof.θ[3]
@test norm(prof.profile[3] - oldprof.profile[3]) < 1e-5
@test norm(prof.other_mles[3] - oldprof.other_mles[3]) < 1e-5

prof = profile(prob, refined_sol; resolution=100, maxtime=0.0001)
oldprof = deepcopy(prof)
profile!(prof)
#@test confidence_intervals(prof, 1) |> bounds |> collect ≈ [0.5046605487788631, 1.1229415665758413] rtol = 1e-3
#@test confidence_intervals(oldprof, 2) |> bounds |> collect ≠ confidence_intervals(prof, 2) |> bounds |> collect
@test prof.spline[1] ≠ oldprof.spline[1]
@test sum(prof.spline[1](prof.θ[1])) ≈ sum(oldprof.spline[1](oldprof.θ[1])) rtol=1e-1
@test length(prof.θ[3]) == length(oldprof.θ[3])
