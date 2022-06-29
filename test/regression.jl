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

################################################################################
## Define the data
################################################################################
Random.seed!(98871)
n = 300
β = [-1.0, 1.0, 0.5, 3.0]
σ = 0.05
x₁ = rand(Uniform(-1, 1), n)
x₂ = rand(Normal(1.0, 0.5), n)
X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
ε = rand(Normal(0.0, σ), n)
y = X * β + ε
sse = dualcache(zeros(n))
β_cache = dualcache(similar(β))
dat = (y, X, sse, n, β_cache)
θ₀ = ones(5)

################################################################################
## Maximum likelihood estimation 
################################################################################
@inline function loglik(θ, data)
    σ, β₀, β₁, β₂, β₃ = θ
    y, X, sse, n, β = data
    sse = get_tmp(sse, θ)
    β = get_tmp(β, θ)
    β[1] = β₀
    β[2] = β₁
    β[3] = β₂
    β[4] = β₃
    ℓℓ = -0.5n * log(2π * σ^2)
    mul!(sse, X, β)
    @turbo for i in 1:n
        ℓℓ = ℓℓ - 0.5 / σ^2 * (y[i] - sse[i])^2
    end
    return ℓℓ
end

## Now define the likelihood problem 
prob = LikelihoodProblem(loglik, 5;
    θ₀,
    data=dat,
    adtype=Optimization.AutoForwardDiff(),
    lb=[0.0, -Inf, -Inf, -Inf, -Inf],
    ub=Inf * ones(5),
    names=[L"\sigma", L"\beta_0", L"\beta_1", L"\beta_2", L"\beta_3"])
sol = mle(prob)

## Test 
@testset "Problem configuration" begin
    @test prob.loglik == loglik
    @test prob.prob isa SciMLBase.OptimizationProblem
    @test prob.prob.u0 ≈ θ₀
    @test prob.prob.f.adtype isa Optimization.AutoForwardDiff
    @test prob.prob.p == dat
    @test prob.prob.lb == [0.0, -Inf * ones(4)...]
    @test prob.prob.ub == Inf * ones(5)
    @test all(isnothing, [prob.prob.lcons, prob.prob.ucons, prob.prob.sense])
    @test prob.names == [L"\sigma", L"\beta_0", L"\beta_1", L"\beta_2", L"\beta_3"]
    @test prob.θ₀ == θ₀
end

@testset "Problem solution" begin
    @test sol.θ ≈ [0.045846608919739233189982741123458254151046276092529296875
        -0.99574347488690573282354989714804105460643768310546875
        0.99805007135858925249038975380244664847850799560546875
        0.497044010369973865426374004528042860329151153564453125
        3.00093210914198404992703217430971562862396240234375]
    @test sol.prob == prob
    @test sol.maximum ≈ 499.0546530363559440957033075392246246337890625
    @test sol.retcode == Symbol("true")
end

@testset "New methods" begin
    ## Problem
    @test ProfileLikelihood.num_params(prob) == 5
    @test ProfileLikelihood.data(prob) == dat
    @test names(prob) == [L"\sigma", L"\beta_0", L"\beta_1", L"\beta_2", L"\beta_3"]
    @test ProfileLikelihood.lower_bounds(prob) == [0.0, -Inf * ones(4)...]
    @test ProfileLikelihood.upper_bounds(prob) == Inf * ones(5)
    @test ProfileLikelihood.sym_names(prob) == ["θ₁", "θ₂", "θ₃", "θ₄", "θ₅"]

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
    @test ProfileLikelihood.algorithm_name(sol) == :PolyOpt
end

################################################################################
## Profile likelihood 
################################################################################
a1, b1 = profile(prob, sol, 1)
a2, b2 = profile(prob, sol, 2)
a3, b3 = profile(prob, sol, 3)
a4, b4 = profile(prob, sol, 4)
a5, b5 = profile(prob, sol, 5)
prof = profile(prob, sol)
fig = plot_profiles(prof; fontsize=20, resolution=(1600, 800))

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
    @test prof.spline isa Dict{Int64,T} where {T<:Spline1D}
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
    @test prof.confidence_intervals[1].lower ≈ 0.04141751574983325301371195337196695618331432342529296875
    @test prof.confidence_intervals[1].upper ≈ 0.05112522181324398451440771395937190391123294830322265625
    @test prof.confidence_intervals[2].lower ≈ -1.0112190325565231230342533308430574834346771240234375
    @test prof.confidence_intervals[2].upper ≈ -0.9802679172172743538027361864806152880191802978515625
    @test prof.confidence_intervals[3].lower ≈ 0.97177367883809229187619393997010774910449981689453125
    @test prof.confidence_intervals[3].upper ≈ 1.0243264638789992826417574178776703774929046630859375
    @test prof.confidence_intervals[4].lower ≈ 0.48334717021014939053458192574908025562763214111328125
    @test prof.confidence_intervals[4].upper ≈ 0.5107408505297854617310804314911365509033203125
    @test prof.confidence_intervals[5].lower ≈ 2.97766403961401238120743073523044586181640625
    @test prof.confidence_intervals[5].upper ≈ 3.024200178670172878270250294008292257785797119140625
    for i in 1:5
        @test prof.confidence_intervals[i].level == 0.99
    end
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
        @test bounds(confidence_intervals(prof, i)) == (prof.confidence_intervals[i].lower, prof.confidence_intervals[i].upper)
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
end

