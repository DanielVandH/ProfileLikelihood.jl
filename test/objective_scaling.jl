using LaTeXStrings
using Random
using Distributions
using OptimizationNLopt
using DifferentialEquations
using Test
using Optimization
using Dierckx
using PreallocationTools

prob, loglikk, θ, uᵒ, n = LogisticODE();
λ, K, σ, u₀ = θ
p = data(prob)
sol = mle(prob, NLopt.LN_NELDERMEAD())

@testset "Scaling problems" begin
    scale = true

    # Scaled OptimizationFunction 
    optprob = prob.prob
    new_f = ProfileLikelihood.scaled_f(optprob, scale)
    @test new_f(θ, p) == optprob.f(θ, p)
    @inferred new_f(θ, p)

    # Scaled OptimizationProblem 
    newoptprob = ProfileLikelihood.scale_prob(optprob, scale)
    @test newoptprob.f(θ, p) == optprob.f(θ, p)
    @inferred newoptprob.f(θ, p)

    # Scaled LikelihoodProblem
    newprob = ProfileLikelihood.scale_prob(prob, scale)
    @test newprob.prob.f(θ, p) == prob.prob.f(θ, p)
    @inferred newprob.prob.f(θ, p)

    for _ in 1:100
        scale = rand()
        # Scaled OptimizationFunction 
        optprob = prob.prob
        new_f = ProfileLikelihood.scaled_f(optprob, scale)
        @test new_f(θ, p) ≈ optprob.f(θ, p) / scale
        @inferred new_f(θ, p)

        # Scaled OptimizationProblem 
        newoptprob = ProfileLikelihood.scale_prob(optprob, scale)
        @test newoptprob.f(θ, p) ≈ optprob.f(θ, p) / scale
        @inferred newoptprob.f(θ, p)

        # Scaled LikelihoodProblem
        newprob = ProfileLikelihood.scale_prob(prob, scale)
        @test newprob.prob.f(θ, p) ≈ prob.prob.f(θ, p) / scale
        @inferred newprob.prob.f(θ, p)
    end
end

@testset "Scaling LikelihoodSolutions" begin
    scale = true
    sol = mle(prob, NLopt.LN_NELDERMEAD; scale)
    @test mle(sol) ≈ [0.7751404076923547, 1.021425899810363, 0.10183166192237934, 0.5354131613054318]
    @test maximum(sol) ≈ 86.54963187499544

    scale = 1.891298
    sol = mle(prob, NLopt.LN_NELDERMEAD; scale)
    @test mle(sol) ≈ [0.7751404076923547, 1.021425899810363, 0.10183166192237934, 0.5354131613054318]
    @test maximum(sol) ≈ 86.54963187499544

    scale = 0.0
    sol = mle(prob, NLopt.LN_NELDERMEAD; scale)
    @test isnan(maximum(sol))

    scale = 1e36
    sol = mle(prob, NLopt.LN_NELDERMEAD; scale)
    @test mle(sol) ≈ [0.7751404076923547, 1.021425899810363, 0.10183166192237934, 0.5354131613054318]
    @test maximum(sol) ≈ 86.54963187499544

    scale = 2.9837171
    sol = mle(prob, (NLopt.LN_BOBYQA, NLopt.LN_NELDERMEAD); scale)
    @test mle(sol) ≈ [0.7751404076923547, 1.021425899810363, 0.10183166192237934, 0.5354131613054318]
    @test maximum(sol) ≈ 86.54963187499544
end

prob, loglikk, θ, uᵒ, n = LogisticODE();
λ, K, σ, u₀ = θ
p = data(prob)
sol = mle(prob, NLopt.LN_NELDERMEAD())
resolution = 100
param_ranges = ProfileLikelihood.construct_profile_ranges(prob, sol, resolution)
new_prob = ProfileLikelihood.scale_prob(prob, 1.0071)
a1, b1, c1 = profile(new_prob, sol, 1, NLopt.LN_NELDERMEAD(), -0.5quantile(Chisq(1), 0.95), param_ranges[1], 30,1.0071,flag=false)
a2, b2, c2 = profile(new_prob, sol, 2, NLopt.LN_NELDERMEAD(), -0.5quantile(Chisq(1), 0.95), param_ranges[2], 30,1.0071,flag=false)
a3, b3, c3 = profile(new_prob, sol, 3, NLopt.LN_NELDERMEAD(), -0.5quantile(Chisq(1), 0.95), param_ranges[3], 30,1.0071,flag=false)
a4, b4, c4 = profile(new_prob, sol, 4, NLopt.LN_NELDERMEAD(), -0.5quantile(Chisq(1), 0.95), param_ranges[4], 30,1.0071,flag=false)
prof = profile(prob, sol; alg=NLopt.LN_NELDERMEAD(), conf_level=0.95, param_ranges, min_steps=30,scale=1.0071)
fig = plot_profiles(prof; fig_kwargs=(fontsize=20, resolution=(1600, 800)))

@testset "Scaling ProfileLikelihoodSolution" begin
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
            @test prof.confidence_intervals[i].lower ≈ conf_lowers[i] rtol = 1e-3
            @test prof.confidence_intervals[i].upper ≈ conf_uppers[i] rtol = 1e-3
            @test prof.confidence_intervals[i].level == 0.95
        end
    
        ## Other MLEs 
         @test length(c1) == length(a1) == length(b1)
         @test length(c2) == length(a2) == length(b2)
         @test length(c3) == length(a3) == length(b3)
         @test length(c4) == length(a4) == length(b4)
         @test prof.other_mles[1] == c1 
         @test prof.other_mles[2] == c2 
         @test prof.other_mles[3] == c3
         @test prof.other_mles[4] == c4
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
    
        a, b = [a1, a2, a3, a4], [b1, b2, b3, b4]
        for i in 1:4
            prof_view = prof[i]
            @test prof[i].parent === prof
            @test prof[i].θ == prof.θ[i]
            @test prof[i].profile == prof.profile[i]
            @test prof[i].mle == prof.mle[i]
            @test prof[i].spline == prof.spline[i]
            @test prof[i].confidence_intervals == prof.confidence_intervals[i]
            @test prof[i](b[i]) ≈ a[i]
        end
        @test prof[1] === prof[:λ]
        @test prof[2] === prof[:K]
        @test prof[3] === prof[:σ]
        @test prof[4] === prof[:u₀]
    end
end