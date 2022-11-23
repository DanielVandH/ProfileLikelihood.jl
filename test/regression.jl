using LaTeXStrings
using Random
using Distributions
using OptimizationNLopt
using OptimizationOptimJL
using DifferentialEquations
using Test
using PreallocationTools
using LinearAlgebra
using Optimization
using Interpolations
using LoopVectorization

prob, loglikk, θ, dat = MultipleLinearRegression()
σ, β = θ
sol = mle(prob, Optim.LBFGS())

## Test 
@testset "Problem configuration" begin
    @test prob.loglik == loglikk
    @test prob.prob isa SciMLBase.OptimizationProblem
    @test prob.prob.u0 ≈ ones(5)
    @test prob.prob.f.adtype isa Optimization.AutoForwardDiff
    @test prob.prob.p == dat
    @test prob.prob.lb == [0.0, -Inf * ones(4)...]
    @test prob.prob.ub == Inf * ones(5)
    @test all(isnothing, [prob.prob.lcons, prob.prob.ucons, prob.prob.sense])
    @test prob.names == [L"\sigma", L"\beta_0", L"\beta_1", L"\beta_2", L"\beta_3"]
    @test prob.θ₀ ≈ ones(5)
    @test !ProfileLikelihood.finite_bounds(prob)
end

@testset "Problem solution" begin
    @test sol.θ ≈ [0.045846608919739233189982741123458254151046276092529296875
        -0.99574347488690573282354989714804105460643768310546875
        0.99805007135858925249038975380244664847850799560546875
        0.497044010369973865426374004528042860329151153564453125
        3.00093210914198404992703217430971562862396240234375] 
    @test sol.prob == prob
    @test sol.maximum ≈ 499.0546530363559440957033075392246246337890625 
    @test sol.retcode == :Success
end

@testset "New methods" begin
    ## Problem
    @test ProfileLikelihood.num_params(prob) == 5
    @test ProfileLikelihood.num_params(prob.prob) == 5
    @test ProfileLikelihood.data(prob) == dat
    @test names(prob) == [L"\sigma", L"\beta_0", L"\beta_1", L"\beta_2", L"\beta_3"]
    @test ProfileLikelihood.lower_bounds(prob) == [0.0, -Inf * ones(4)...]
    @test ProfileLikelihood.upper_bounds(prob) == Inf * ones(5)
    @test ProfileLikelihood.lower_bounds(prob, 1) == 0.0
    @test ProfileLikelihood.upper_bounds(prob, 3) == Inf
    @test ProfileLikelihood.bounds(prob, 1) == (0.0, Inf)
    @test ProfileLikelihood.bounds(prob, 2) == (-Inf, Inf)
    @test ProfileLikelihood.bounds(prob, 3) == (-Inf, Inf)
    @test ProfileLikelihood.bounds(prob, 4) == (-Inf, Inf)
    @test ProfileLikelihood.bounds(prob, 5) == (-Inf, Inf)
    @test ProfileLikelihood.bounds(prob) == [(0.0, Inf), (-Inf, Inf), (-Inf, Inf), (-Inf, Inf), (-Inf, Inf)]
    @test ProfileLikelihood.sym_names(prob) == Symbol.(["θ₁", "θ₂", "θ₃", "θ₄", "θ₅"])

    @test ProfileLikelihood.lower_bounds(prob; make_open=true) == [ProfileLikelihood.OPEN_EXT, -Inf * ones(4)...]
    @test ProfileLikelihood.upper_bounds(prob; make_open=true) == Inf * ones(5)
    @test ProfileLikelihood.lower_bounds(prob, 1; make_open=true) == ProfileLikelihood.OPEN_EXT
    @test ProfileLikelihood.upper_bounds(prob, 3; make_open=true) == Inf
    @test ProfileLikelihood.bounds(prob, 1; make_open=true) == (ProfileLikelihood.OPEN_EXT, Inf)
    @test ProfileLikelihood.bounds(prob, 2; make_open=true) == (-Inf, Inf)
    @test ProfileLikelihood.bounds(prob, 3; make_open=true) == (-Inf, Inf)
    @test ProfileLikelihood.bounds(prob, 4; make_open=true) == (-Inf, Inf)
    @test ProfileLikelihood.bounds(prob, 5; make_open=true) == (-Inf, Inf)
    @test ProfileLikelihood.bounds(prob; make_open=true) == [(ProfileLikelihood.OPEN_EXT, Inf), (-Inf, Inf), (-Inf, Inf), (-Inf, Inf), (-Inf, Inf)]

    ## Solution
    @test ProfileLikelihood.num_params(sol) == 5
    @test ProfileLikelihood.data(sol) == dat
    @test names(sol) == [L"\sigma", L"\beta_0", L"\beta_1", L"\beta_2", L"\beta_3"]
    @test ProfileLikelihood.lower_bounds(sol) == [0.0, -Inf * ones(4)...]
    @test ProfileLikelihood.upper_bounds(sol) == Inf * ones(5)
    @test maximum(sol) ≈ 499.0546530363559440957033075392246246337890625
    @test mle(sol) ≈ [0.045846608919739233189982741123458254151046276092529296875
        -0.99574347488690573282354989714804105460643768310546875
        0.99805007135858925249038975380244664847850799560546875
        0.497044010369973865426374004528042860329151153564453125
        3.00093210914198404992703217430971562862396240234375]
    @test ProfileLikelihood.algorithm_name(sol) == :LBFGS
    @test sol[1] == sol.θ[1]
    @test sol[2] == sol.θ[2]
    @test sol[3] == sol.θ[3]
    @test sol[4] == sol.θ[4]
    @test sol[5] == sol.θ[5]
end

################################################################################
## Profile likelihood 
################################################################################
resolution = 1000
@test_throws "The provided parameter bounds must be finite." ProfileLikelihood.construct_profile_ranges(prob, sol, resolution)
param_bounds = [
    (0.001, 0.1),
    (-1.2, -0.8),
    (0.8, 1.2),
    (0.3, 0.7),
    (2.5, 3.5)
]
param_ranges = ProfileLikelihood.construct_profile_ranges(prob, sol, resolution; param_bounds)
mles = mle(sol)
for i in 1:num_params(prob)
    @test param_ranges[i] == (LinRange(mles[i], param_bounds[i][1], resolution), LinRange(mles[i], param_bounds[i][2], resolution))
end
resolution = [17, 20, 13, 5, 10]
param_ranges = ProfileLikelihood.construct_profile_ranges(prob, sol, resolution; param_bounds)
for i in 1:num_params(prob)
    @test param_ranges[i] == (LinRange(mles[i], param_bounds[i][1], resolution[i]), LinRange(mles[i], param_bounds[i][2], resolution[i]))
end

resolution = 1000
param_ranges = ProfileLikelihood.construct_profile_ranges(prob, sol, resolution; param_bounds)
a1, b1, c1 = profile(prob, sol, 1, sol.alg, -0.5quantile(Chisq(1), 0.99), param_ranges[1], 10, false)
a2, b2, c2 = profile(prob, sol, 2, sol.alg, -0.5quantile(Chisq(1), 0.99), param_ranges[2], 10, false)
a3, b3, c3 = profile(prob, sol, 3, sol.alg, -0.5quantile(Chisq(1), 0.99), param_ranges[3], 10, false)
a4, b4, c4 = profile(prob, sol, 4, sol.alg, -0.5quantile(Chisq(1), 0.99), param_ranges[4], 10, false)
a5, b5, c5 = profile(prob, sol, 5, sol.alg, -0.5quantile(Chisq(1), 0.99), param_ranges[5], 10, false)
prof = profile(prob, sol; conf_level=0.99, param_ranges, spline=false, min_steps=10, normalise=false)
@test length(prof[1].θ) == length(prof[1].profile) == 198
@test length(prof[2].θ) == length(prof[2].profile) == 156
@test length(prof[3].θ) == length(prof[3].profile) == 264
@test length(prof[4].θ) == length(prof[4].profile) == 139
@test length(prof[5].θ) == length(prof[5].profile) === 95
fig = plot_profiles(prof; fig_kwargs=(fontsize=20, resolution=(1600, 800)), axis_kwargs=(width=700, height=350))
resize_to_layout!(fig)
fig

fig = plot_profiles(prof, [1, 3, 5]; fig_kwargs=(fontsize=20, resolution=(1600, 800)), axis_kwargs=(width=700, height=350))
resize_to_layout!(fig)
fig
@testset "Problem configuration" begin
    ## Parameter values
    @test prof.θ isa Dict{Int64,Vector{Float64}}
    @test prof.θ[1] == b1
    @test prof.θ[2] == b2
    @test prof.θ[3] == b3
    @test prof.θ[4] == b4
    @test prof.θ[5] == b5

    ## Profile values
    @test prof.profile isa Dict{Int64,Vector{Float64}}
    @test prof.profile[1] == a1
    @test prof.profile[2] == a2
    @test prof.profile[3] == a3
    @test prof.profile[4] == a4
    @test prof.profile[5] == a5

    ## Problem and MLE structure
    @test prof.prob == prob
    @test prof.mle == sol

    ## Spline and calling the structure
    @test prof.spline isa Dict{Int64,T} where {T<:AbstractExtrapolation}
    @test prof.spline[1](b1) ≈ a1
    @test prof.spline[2](b2) ≈ a2
    @test prof.spline[3](b3) ≈ a3
    @test prof.spline[4](b4) ≈ a4
    @test prof.spline[5](b5) ≈ a5
    @test prof(b1, 1) ≈ a1
    @test prof(b2, 2) ≈ a2
    @test prof(b3, 3) ≈ a3
    @test prof(b4, 4) ≈ a4
    @test prof(b5, 5) ≈ a5

    ## Confidence intervals
    @test prof.confidence_intervals isa Dict{Int64,ProfileLikelihood.ConfidenceInterval{Float64,Float64}}
    @test prof.confidence_intervals[1].lower ≈ 0.04141751574983325301371195337196695618331432342529296875 rtol = 1e-3
    @test prof.confidence_intervals[1].upper ≈ 0.05112522181324398451440771395937190391123294830322265625 rtol = 1e-3
    @test prof.confidence_intervals[2].lower ≈ -1.0112190325565231230342533308430574834346771240234375 rtol = 1e-3
    @test prof.confidence_intervals[2].upper ≈ -0.9802679172172743538027361864806152880191802978515625 rtol = 1e-3
    @test prof.confidence_intervals[3].lower ≈ 0.97177367883809229187619393997010774910449981689453125 rtol = 1e-3
    @test prof.confidence_intervals[3].upper ≈ 1.0243264638789992826417574178776703774929046630859375 rtol = 1e-3
    @test prof.confidence_intervals[4].lower ≈ 0.48334717021014939053458192574908025562763214111328125 rtol = 1e-3
    @test prof.confidence_intervals[4].upper ≈ 0.5107408505297854617310804314911365509033203125 rtol = 1e-3
    @test prof.confidence_intervals[5].lower ≈ 2.97766403961401238120743073523044586181640625 rtol = 1e-3
    @test prof.confidence_intervals[5].upper ≈ 3.024200178670172878270250294008292257785797119140625 rtol = 1e-3
    for i in 1:5
        @test prof.confidence_intervals[i].level == 0.99
    end

    ## Other MLEs 
    @test sum(c1) ≈ [-197.15720802760532
        197.61391412899405
        98.41471405325326
        594.1845576101321]
    @test sum(c2) ≈ [7.179095145173143
        155.69493130146472
        77.52228199445916
        468.1451686301226]
    @test sum(c3) ≈ [12.148970073165131
        -262.87561689281057
        131.21985435889783
        792.282239857978]
    @test sum(c4) ≈ [6.3970931478274
        -138.39464499635085
        138.72921917771941
        417.1285024670855]
    @test sum(c5) ≈ [4.372313669611411
        -94.59561941220197
        94.81690137691011
        47.21912381678868]
    @test length(c1) == length(a1) == length(b1)
    @test length(c2) == length(a2) == length(b2)
    @test length(c3) == length(a3) == length(b3)
    @test length(c4) == length(a4) == length(b4)
    @test length(c5) == length(a5) == length(b5)
    @test prof.other_mles[1] == c1
    @test prof.other_mles[2] == c2
    @test prof.other_mles[3] == c3
    @test prof.other_mles[4] == c4
    @test prof.other_mles[5] == c5
end

@testset "Problem solution" begin
    ## Confidence intervals 
    @test prof.confidence_intervals[1][1] ≤ σ ≤ prof.confidence_intervals[1][2]
    @test prof.confidence_intervals[2][1] ≤ β[1] ≤ prof.confidence_intervals[2][2]
    @test prof.confidence_intervals[3][1] ≤ β[2] ≤ prof.confidence_intervals[3][2]
    @test prof.confidence_intervals[4][1] ≤ β[3] ≤ prof.confidence_intervals[4][2]
    @test prof.confidence_intervals[5][1] ≤ β[4] ≤ prof.confidence_intervals[5][2]
end

@testset "New methods" begin
    @test ProfileLikelihood.num_params(prof) == 5
    @test ProfileLikelihood.data(prof) == dat
    @test names(prof) == [L"\sigma", L"\beta_0", L"\beta_1", L"\beta_2", L"\beta_3"]
    @test ProfileLikelihood.lower_bounds(prof) == [0.0, -Inf * ones(4)...]
    @test ProfileLikelihood.upper_bounds(prof) == Inf * ones(5)
    @test maximum(prof) ≈ 499.0546530363559440957033075392246246337890625
    @test mle(prof) ≈ [0.045846608919739233189982741123458254151046276092529296875
        -0.99574347488690573282354989714804105460643768310546875
        0.99805007135858925249038975380244664847850799560546875
        0.497044010369973865426374004528042860329151153564453125
        3.00093210914198404992703217430971562862396240234375]
    @test confidence_intervals(prof) == prof.confidence_intervals
    for i in 1:5
        @test confidence_intervals(prof, i) == prof.confidence_intervals[i]
        @test ProfileLikelihood.lower(confidence_intervals(prof, i)) == prof.confidence_intervals[i].lower == confidence_intervals(prof, i)[1]
        @test ProfileLikelihood.upper(confidence_intervals(prof, i)) == prof.confidence_intervals[i].upper == confidence_intervals(prof, i)[2]
        @test ProfileLikelihood.level(confidence_intervals(prof, i)) == prof.confidence_intervals[i].level
        @test ProfileLikelihood.bounds(confidence_intervals(prof, i)) == (prof.confidence_intervals[i].lower, prof.confidence_intervals[i].upper)
        ℓ, u = confidence_intervals(prof, i)
        @test ℓ == prof.confidence_intervals[i].lower
        @test u == prof.confidence_intervals[i].upper
        @test eltype(confidence_intervals(prof, i)) == Float64
        @test length(confidence_intervals(prof, i)) == 2
    end
    @test σ ∈ prof.confidence_intervals[1]
    @test β[1] ∈ prof.confidence_intervals[2]
    @test β[2] ∈ prof.confidence_intervals[3]
    @test β[3] ∈ prof.confidence_intervals[4]
    @test β[4] ∈ prof.confidence_intervals[5]
    @test 1000.0 ∉ prof.confidence_intervals[5]

    a, b = [a1, a2, a3, a4, a5], [b1, b2, b3, b4, b5]
    for i in 1:5
        prof_view = prof[i]
        @test prof[i].parent === prof
        @test prof[i].θ == prof.θ[i]
        @test prof[i].profile == prof.profile[i]
        @test prof[i].prob === prob
        @test prof[i].mle == prof.mle[i]
        @test prof[i].spline == prof.spline[i]
        @test prof[i].confidence_intervals == prof.confidence_intervals[i]
        @test prof[i](b[i]) ≈ a[i]
    end
end

@testset "Checking minimum steps" begin
    prof = profile(prob, sol; conf_level=0.99, param_ranges, spline=false, min_steps=500)
    for i in 1:num_params(prob)
        @test length(prof[i].θ) === 999
        @test length(prof[i].profile) === 999
    end
    @test σ ∈ prof.confidence_intervals[1]
    @test β[1] ∈ prof.confidence_intervals[2]
    @test β[2] ∈ prof.confidence_intervals[3]
    @test β[3] ∈ prof.confidence_intervals[4]
    @test β[4] ∈ prof.confidence_intervals[5]
    @test 1000.0 ∉ prof.confidence_intervals[5]
    @test prof.confidence_intervals isa Dict{Int64,ProfileLikelihood.ConfidenceInterval{Float64,Float64}}
    @test prof.confidence_intervals[1].lower ≈ 0.04141751574983325301371195337196695618331432342529296875 rtol = 1e-3
    @test prof.confidence_intervals[1].upper ≈ 0.05112522181324398451440771395937190391123294830322265625 rtol = 1e-3
    @test prof.confidence_intervals[2].lower ≈ -1.0112190325565231230342533308430574834346771240234375 rtol = 1e-3
    @test prof.confidence_intervals[2].upper ≈ -0.9802679172172743538027361864806152880191802978515625 rtol = 1e-3
    @test prof.confidence_intervals[3].lower ≈ 0.97177367883809229187619393997010774910449981689453125 rtol = 1e-3
    @test prof.confidence_intervals[3].upper ≈ 1.0243264638789992826417574178776703774929046630859375 rtol = 1e-3
    @test prof.confidence_intervals[4].lower ≈ 0.48334717021014939053458192574908025562763214111328125 rtol = 1e-3
    @test prof.confidence_intervals[4].upper ≈ 0.5107408505297854617310804314911365509033203125 rtol = 1e-3
    @test prof.confidence_intervals[5].lower ≈ 2.97766403961401238120743073523044586181640625 rtol = 1e-3
    @test prof.confidence_intervals[5].upper ≈ 3.024200178670172878270250294008292257785797119140625 rtol = 1e-3
    for i in 1:5
        @test prof.confidence_intervals[i].level == 0.99
    end
end
