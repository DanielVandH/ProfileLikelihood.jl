using LaTeXStrings
using Random
using Distributions
using OptimizationNLopt
using DifferentialEquations
using Test
using Optimization
using Dierckx

prob, loglikk, θ, yᵒ, n = LinearExponentialODE()
λ, σ, y₀ = θ
sol = mle(prob, NLopt.LN_NELDERMEAD())

θ₀ = [-1.0, 0.5, 19.73]

## Test
@testset "Problem configuration" begin
    @test prob.prob isa SciMLBase.OptimizationProblem
    @test prob.prob.u0 ≈ θ₀
    @test prob.prob.f.adtype isa Optimization.AutoFiniteDiff
    @test prob.prob.p == (yᵒ, n)
    @test prob.prob.lb == [-10.0, 1e-6, 0.5]
    @test prob.prob.ub == [10.0, 10.0, 25.0]
    @test all(isnothing, [prob.prob.lcons, prob.prob.ucons, prob.prob.sense])
    @test prob.names == [L"\lambda", L"\sigma", L"y_0"]
    @test prob.θ₀ == θ₀
end

@testset "Problem solution" begin
    @test sol.θ ≈ [-0.50056892573506506227687395949033088982105255126953125
        0.09708073332894538720605481785241863690316677093505859375
        15.040984400433927703488734550774097442626953125]
    @test sol.prob == prob
    @test sol.maximum ≈ 182.65476373481573091339669190347194671630859375
    @test sol.retcode == :XTOL_REACHED
end

@testset "New methods" begin
    ## Problem
    @test ProfileLikelihood.num_params(prob) == 3
    @test ProfileLikelihood.data(prob) == (yᵒ, n)
    @test names(prob) == [L"\lambda", L"\sigma", L"y_0"]
    @test ProfileLikelihood.lower_bounds(prob) == [-10.0, 1e-6, 0.5]
    @test ProfileLikelihood.upper_bounds(prob) == [10.0, 10.0, 25.0]

    ## Solution
    @test ProfileLikelihood.num_params(sol) == 3
    @test ProfileLikelihood.data(sol) == (yᵒ, n)
    @test names(sol) == [L"\lambda", L"\sigma", L"y_0"]
    @test ProfileLikelihood.lower_bounds(sol) == [-10.0, 1e-6, 0.5]
    @test ProfileLikelihood.upper_bounds(sol) == [10.0, 10.0, 25.0]
    @test maximum(sol) ≈ 182.65476373481573091339669190347194671630859375
    @test ProfileLikelihood.mle(sol) ≈ [-0.50056892573506506227687395949033088982105255126953125
        0.09708073332894538720605481785241863690316677093505859375
        15.040984400433927703488734550774097442626953125]
end

################################################################################
## Profile likelihood 
################################################################################
a1, b1 = profile(prob, sol, 1; alg=NLopt.LN_NELDERMEAD(), threshold=-0.5quantile(Chisq(1), 0.95))
a2, b2 = profile(prob, sol, 2; alg=NLopt.LN_NELDERMEAD(), threshold=-0.5quantile(Chisq(1), 0.95))
a3, b3 = profile(prob, sol, 3; alg=NLopt.LN_NELDERMEAD(), threshold=-0.5quantile(Chisq(1), 0.95))
prof = profile(prob, sol; alg=NLopt.LN_NELDERMEAD(), conf_level=0.95)
fig = plot_profiles(prof; fontsize=20, resolution=(1600, 800))

@testset "Problem configuration" begin
    ## Parameter values
    @test prof.θ isa Dict{Int64,Vector{Float64}}
    @test prof.θ[1] == b1
    @test prof.θ[2] == b2
    @test prof.θ[3] == b3

    ## Profile values
    @test prof.profile isa Dict{Int64,Vector{Float64}}
    @test prof.profile[1] == a1
    @test prof.profile[2] == a2
    @test prof.profile[3] == a3

    ## Problem and MLE structure
    @test prof.prob == prob
    @test prof.mle.θ ≈ sol.θ

    ## Spline and calling the structure
    @test prof.spline isa Dict{Int64,T} where {T<:Spline1D}
    @test prof.spline[1](b1) ≈ a1
    @test prof.spline[2](b2) ≈ a2
    @test prof.spline[3](b3) ≈ a3
    @test prof(b1, 1) ≈ a1
    @test prof(b2, 2) ≈ a2
    @test prof(b3, 3) ≈ a3

    ## Confidence intervals
    @test prof.confidence_intervals isa Dict{Int64,ProfileLikelihood.ConfidenceInterval{Float64,Float64}}
    lower_confs = [-0.50277320890830135002858014559024013578891754150390625
        0.08829152027685986670046958124657976441085338592529296875
        14.9973585632818018353873412706889212131500244140625]
    upper_confs = [-0.4983713388062707139170015580020844936370849609375
        0.1074310958975970564655000316633959300816059112548828125
        15.084663612381799424611017457209527492523193359375]
    for i in 1:3
        @test prof.confidence_intervals[i].lower ≈ lower_confs[i]
        @test prof.confidence_intervals[i].upper ≈ upper_confs[i]
        @test prof.confidence_intervals[i].level == 0.95
    end
end

@testset "Problem solution" begin
    ## Confidence intervals 
    @test prof.confidence_intervals[1][1] ≤ λ ≤ prof.confidence_intervals[1][2]
    @test prof.confidence_intervals[2][1] ≤ σ ≤ prof.confidence_intervals[2][2]
    @test prof.confidence_intervals[3][1] ≤ y₀ ≤ prof.confidence_intervals[3][2]
end

@testset "New methods" begin
    @test ProfileLikelihood.num_params(prof) == 3
    @test ProfileLikelihood.data(prof) == (yᵒ, n)
    @test names(prof) == [L"\lambda", L"\sigma", L"y_0"]
    @test ProfileLikelihood.lower_bounds(prof) == [-10.0, 1e-6, 0.5]
    @test ProfileLikelihood.upper_bounds(prof) == [10.0, 10.0, 25.0]
    @test maximum(prof) ≈ 182.65476373481573091339669190347194671630859375
    @test ProfileLikelihood.mle(prof) ≈ [-0.50056892573506506227687395949033088982105255126953125
        0.09708073332894538720605481785241863690316677093505859375
        15.040984400433927703488734550774097442626953125]
    @test confidence_intervals(prof) == prof.confidence_intervals
    for i in 1:3
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
    @test σ ∈ prof.confidence_intervals[2]
    @test y₀ ∈ prof.confidence_intervals[3]
    @test 1000.0 ∉ prof.confidence_intervals[2]
end
