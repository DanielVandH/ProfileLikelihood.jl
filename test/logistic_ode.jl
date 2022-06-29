using LaTeXStrings
using Random
using Distributions
using OptimizationNLopt
using DifferentialEquations
using Test
using Optimization
using Dierckx

################################################################################
## Define the data
################################################################################
Random.seed!(2929911002)
u₀ = 0.5
λ = 1.0
K = 1.0
n = 100
T = 10.0
t = LinRange(0, T, n)
u = @. K * u₀ * exp(λ * t) / (K - u₀ + u₀ * exp(λ * t))
σ = 0.1
uᵒ = u .+ [0.0, σ * randn(length(u) - 1)...]

################################################################################
## Define the ODE and the log-likelihood
################################################################################
function ode_fnc(u, p, t)
    λ, K = p
    du = λ * u * (1 - u / K)
    return du
end
function loglik(θ, data, integrator)
    ## Extract the parameters
    uᵒ, n = data
    λ, K, σ, u0 = θ
    ## What do you want to do with the integrator?
    integrator.p[1] = λ
    integrator.p[2] = K
    ## Now solve the problem 
    reinit!(integrator, u0)
    solve!(integrator)
    return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, n)
end

################################################################################
## Maximum likelihood estimation 
################################################################################
θ₀ = [0.7, 2.0, 0.15, 0.4]
lb = [0.0, 1e-6, 1e-6, 0.0]
ub = [10.0, 10.0, 10.0, 10.0]
param_names = [L"\lambda", L"K", L"\sigma", L"u_0"]
prob = LikelihoodProblem(loglik, 4, ode_fnc, u₀, (0.0, T), [1.0, 1.0], t;
    data=(uᵒ, n), θ₀, lb, ub, ode_kwargs=(verbose=false,),
    names=param_names, syms=[:λ, :K, :σ, :u₀])
sol = mle(prob, NLopt.LN_NELDERMEAD())

## Test
@testset "Problem configuration" begin
    @test prob.prob isa SciMLBase.OptimizationProblem
    @test prob.prob.u0 ≈ θ₀
    @test prob.prob.f.adtype isa Optimization.AutoFiniteDiff
    @test prob.prob.p == (uᵒ, n)
    @test prob.prob.lb == lb
    @test prob.prob.ub == ub
    @test all(isnothing, [prob.prob.lcons, prob.prob.ucons, prob.prob.sense])
    @test prob.names == param_names
    @test prob.θ₀ == θ₀
end

@testset "Problem solution" begin
    @test sol.θ ≈ [0.77514040769514169770815215088077820837497711181640625
        1.0214258997572567277956068210187368094921112060546875
        0.1018316618807033335780687366423080675303936004638671875
        0.5354131612786698912742622269433923065662384033203125]
    @test sol.prob == prob
    @test sol.maximum ≈ 86.54963187499551
    @test sol.retcode == :XTOL_REACHED
end

@testset "New methods" begin
    ## Problem
    @test ProfileLikelihood.num_params(prob) == 4
    @test ProfileLikelihood.data(prob) == (uᵒ, n)
    @test names(prob) == param_names
    @test ProfileLikelihood.lower_bounds(prob) == lb
    @test ProfileLikelihood.upper_bounds(prob) == ub
    @test ProfileLikelihood.sym_names(sol) == [:λ, :K, :σ, :u₀]

    ## Solution
    @test ProfileLikelihood.num_params(sol) == 4
    @test ProfileLikelihood.data(sol) == (uᵒ, n)
    @test names(sol) == param_names
    @test ProfileLikelihood.lower_bounds(sol) == lb
    @test ProfileLikelihood.upper_bounds(sol) == ub
    @test maximum(sol) ≈ 86.54963187499551
    @test ProfileLikelihood.mle(sol) ≈ [0.77514040769514169770815215088077820837497711181640625
        1.0214258997572567277956068210187368094921112060546875
        0.1018316618807033335780687366423080675303936004638671875
        0.5354131612786698912742622269433923065662384033203125]
    @test ProfileLikelihood.algorithm_name(sol) == "Nelder-Mead simplex algorithm (local, no-derivative)"
end

################################################################################
## Profile likelihood 
################################################################################
a1, b1 = profile(prob, sol, 1; alg=NLopt.LN_NELDERMEAD(), threshold=-0.5quantile(Chisq(1), 0.95))
a2, b2 = profile(prob, sol, 2; alg=NLopt.LN_NELDERMEAD(), threshold=-0.5quantile(Chisq(1), 0.95))
a3, b3 = profile(prob, sol, 3; alg=NLopt.LN_NELDERMEAD(), threshold=-0.5quantile(Chisq(1), 0.95))
a4, b4 = profile(prob, sol, 4; alg=NLopt.LN_NELDERMEAD(), threshold=-0.5quantile(Chisq(1), 0.95))
prof = profile(prob, sol; alg=NLopt.LN_NELDERMEAD(), conf_level=0.95)
fig = plot_profiles(prof; fontsize=20, resolution=(1600, 800))

@testset "Problem configuration" begin
    ## Parameter values
    @test prof.θ isa Dict{Int64,Vector{Float64}}
    @test prof.θ[1] == b1
    @test prof.θ[2] == b2
    @test prof.θ[3] == b3
    @test prof.θ[4] == b4

    ## Profile values
    @test prof.profile isa Dict{Int64,Vector{Float64}}
    @test prof.profile[1] == a1
    @test prof.profile[2] == a2
    @test prof.profile[3] == a3
    @test prof.profile[4] == a4

    ## Problem and MLE structure
    @test prof.prob == prob
    @test prof.mle.θ ≈ sol.θ

    ## Spline and calling the structure
    @test prof.spline isa Dict{Int64,T} where {T<:Spline1D}
    @test prof.spline[1](b1) ≈ a1
    @test prof.spline[2](b2) ≈ a2
    @test prof.spline[3](b3) ≈ a3
    @test prof.spline[4](b4) ≈ a4
    @test prof(b1, 1) ≈ a1
    @test prof(b2, 2) ≈ a2
    @test prof(b3, 3) ≈ a3
    @test prof(b4, 4) ≈ a4

    ## Confidence intervals
    @test prof.confidence_intervals isa Dict{Int64,ProfileLikelihood.ConfidenceInterval{Float64,Float64}}
    conf_lowers = [0.50466049826454539850573155490565113723278045654296875
        0.9911362219771640003074253399972803890705108642578125
        0.0891970022187113242839728854960412718355655670166015625
        0.44852361543771446239503575270646251738071441650390625]
    conf_uppers = [1.123055949738116066072279863874427974224090576171875
        1.0590950616037979603589747057412751019001007080078125
        0.11775663655200267754263876440745661966502666473388671875
        0.6204495782497130296206933053326793015003204345703125]
    for i in 1:4
        @test prof.confidence_intervals[i].lower ≈ conf_lowers[i]
        @test prof.confidence_intervals[i].upper ≈ conf_uppers[i]
        @test prof.confidence_intervals[i].level == 0.95
    end
end

@testset "Problem solution" begin
    ## Confidence intervals 
    @test prof.confidence_intervals[1][1] ≤ λ ≤ prof.confidence_intervals[1][2]
    @test prof.confidence_intervals[2][1] ≤ K ≤ prof.confidence_intervals[2][2]
    @test prof.confidence_intervals[3][1] ≤ σ ≤ prof.confidence_intervals[3][2]
    @test prof.confidence_intervals[4][1] ≤ u₀ ≤ prof.confidence_intervals[4][2]
end

@testset "New methods" begin
    @test ProfileLikelihood.num_params(prof) == 4
    @test ProfileLikelihood.data(prof) == (uᵒ, n)
    @test names(prof) == param_names
    @test ProfileLikelihood.lower_bounds(prof) == lb
    @test ProfileLikelihood.upper_bounds(prof) == ub
    @test maximum(prof) ≈ 86.54963187499551
    @test ProfileLikelihood.mle(prof) ≈ [0.77514040769514169770815215088077820837497711181640625
    1.0214258997572567277956068210187368094921112060546875
    0.1018316618807033335780687366423080675303936004638671875
    0.5354131612786698912742622269433923065662384033203125]
    @test confidence_intervals(prof) == prof.confidence_intervals
    for i in 1:4
        @test confidence_intervals(prof, i) == prof.confidence_intervals[i]
        @test ProfileLikelihood.lower(confidence_intervals(prof, i)) == prof.confidence_intervals[i].lower == confidence_intervals(prof, i)[1]
        @test ProfileLikelihood.upper(confidence_intervals(prof, i)) == prof.confidence_intervals[i].upper == confidence_intervals(prof, i)[2]
        @test ProfileLikelihood.level(confidence_intervals(prof, i)) == prof.confidence_intervals[i].level
        @test bounds(confidence_intervals(prof, i)) == (prof.confidence_intervals[i].lower, prof.confidence_intervals[i].upper)
        ℓ, u = confidence_intervals(prof, i)
        @test ℓ == prof.confidence_intervals[i].lower
        @test u == prof.confidence_intervals[i].upper
        @test eltype(confidence_intervals(prof, i)) == Float64
        @test length(confidence_intervals(prof, i)) == 2
    end
    @test λ ∈ prof.confidence_intervals[1]
    @test K ∈ prof.confidence_intervals[2]
    @test σ ∈ prof.confidence_intervals[3]
    @test u₀ ∈ prof.confidence_intervals[4]
    @test 1000.0 ∉ prof.confidence_intervals[2]
    @test ProfileLikelihood.algorithm_name(prof) == "Nelder-Mead simplex algorithm (local, no-derivative)"
    @test ProfileLikelihood.sym_names(prof) == [:λ, :K, :σ, :u₀]
end
