using ..ProfileLikelihood
using Random, Distributions, StableRNGs
using PreallocationTools, LinearAlgebra
using Optimization 
using OptimizationNLopt 

function _sprint(prob)
    return sprint() do io 
        show(IOContext(io, :color => true), MIME"text/plain"(), prob)
    end
end

rng = StableRNG(98871)
n = 600
β = [-1.0, 1.0, 0.5, 3.0, 1.0, 1.0]
σ = 0.05
x₁ = rand(rng, Normal(0, 0.2), n)
x₂ = rand(rng, Uniform(-1, 1), n)
x₃ = rand(rng, Normal(0, 1), n)
x₄ = rand(rng, Exponential(1), n)
ε = rand(rng, Normal(0, σ), n)
X = hcat(ones(n), x₁, x₂, x₁ .* x₃, x₄)
βcombined = [β[1], β[2], β[3], β[4], β[5] * β[6]] 
y = X * βcombined + ε
sse = DiffCache(zeros(n))
βcache = DiffCache(similar(β, length(β) - 1), 10) # -1 because we combine β[5] and β[6]
data = (y, X, sse, n, βcache)
function loglik(θ, data)
    σ, β₀, β₁, β₂, β₃, β₄, β₅ = θ
    β₄β₅ = β₄ * β₅
    y, X, sse, n, β = data
    _sse = get_tmp(sse, θ)
    _β = get_tmp(β, θ)
    _β .= (β₀, β₁, β₂, β₃, β₄β₅)
    ℓℓ = -0.5n * log(2π * σ^2)
    mul!(_sse, X, _β)
    for (yᵢ, sseᵢ) in zip(y, _sse)
        ℓℓ -= 0.5 * (yᵢ - sseᵢ)^2 / σ^2
    end
    return ℓℓ
end
θ₀ = [1.3,2.3,4.5,1.1,2.2,3.3,10.0] # initial guess 
prob = LikelihoodProblem(loglik, θ₀; data,
    f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
    prob_kwargs=(
        lb=[0.0, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf],
        ub=fill(Inf, 7),
    ),
    syms=[:σ, :β₀, :β₁, :β₂, :β₃, :β₄, :β₅])
prob_no_syms = LikelihoodProblem(loglik, θ₀; data,
    f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
    prob_kwargs=(
        lb=[0.0, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf],
        ub=fill(Inf, 7),
    ))
prob_sprint = _sprint(prob)
prob_no_syms_sprint = _sprint(prob_no_syms)
@test prob_sprint == "\e[38;2;86;182;194mLikelihoodProblem\e[0m. In-place: \e[38;2;86;182;194mtrue\n\e[0mθ₀: 7-element Vector{Float64}\n\e[0m     σ: 1.3\n\e[0m     β₀: 2.3\n\e[0m     β₁: 4.5\n\e[0m     β₂: 1.1\n\e[0m     β₃: 2.2\n\e[0m     β₄: 3.3\n\e[0m     β₅: 10.0"
@test prob_no_syms_sprint == "\e[38;2;86;182;194mLikelihoodProblem\e[0m. In-place: \e[38;2;86;182;194mtrue\n\e[0mθ₀: 7-element Vector{Float64}\n\e[0m     θ₁: 1.3\n\e[0m     θ₂: 2.3\n\e[0m     θ₃: 4.5\n\e[0m     θ₄: 1.1\n\e[0m     θ₅: 2.2\n\e[0m     θ₆: 3.3\n\e[0m     θ₇: 10.0"

sol = mle(prob, (NLopt.LN_NELDERMEAD(), NLopt.LD_LBFGS()))
sol_no_syms = mle(prob_no_syms, (NLopt.LN_NELDERMEAD(), NLopt.LD_LBFGS()))
sol_sprint = _sprint(sol)
sol_no_syms_sprint = _sprint(sol_no_syms)
@test sol_sprint == "\e[38;2;86;182;194mLikelihoodSolution\e[0m. retcode: \e[38;2;86;182;194mSuccess\n\e[0mMaximum likelihood: 930.3177800683162\n\e[0mMaximum likelihood estimates: 7-element Vector{Float64}\n\e[0m     σ: 0.051330602449744085\n\e[0m     β₀: -0.9959998065120137\n\e[0m     β₁: 1.005739244836975\n\e[0m     β₂: 0.4951059914675016\n\e[0m     β₃: 3.0066628757589697\n\e[0m     β₄: -2.128650740776393\n\e[0m     β₅: -0.4695089473776722"
@test sol_no_syms_sprint == "\e[38;2;86;182;194mLikelihoodSolution\e[0m. retcode: \e[38;2;86;182;194mSuccess\n\e[0mMaximum likelihood: 930.3177800683162\n\e[0mMaximum likelihood estimates: 7-element Vector{Float64}\n\e[0m     θ₁: 0.051330602449744085\n\e[0m     θ₂: -0.9959998065120137\n\e[0m     θ₃: 1.005739244836975\n\e[0m     θ₄: 0.4951059914675016\n\e[0m     θ₅: 3.0066628757589697\n\e[0m     θ₆: -2.128650740776393\n\e[0m     θ₇: -0.4695089473776722"

prof_lb = [1e-12, -5.0, -5.0, -5.0, -2.0, -5.0, -5.0]
prof_ub = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
resolutions = [1200, 200, 200, 200, 200, 200, 200] 
param_ranges = construct_profile_ranges(sol, prof_lb, prof_ub, resolutions)
prof = profile(prob, sol; param_ranges, parallel=true)
prof_no_syms = profile(prob_no_syms, sol_no_syms; param_ranges, parallel=true)
prof_sprint = _sprint(prof)
prof_no_syms_sprint = _sprint(prof_no_syms)
@test prof_sprint == "\e[38;2;86;182;194mProfileLikelihoodSolution\e[0m. MLE retcode: \e[38;2;86;182;194mSuccess\n\e[0mConfidence intervals: \n\e[0m     95.0% CI for σ: (0.048557863157896265, 0.054378929873026595)\n\e[0m     95.0% CI for β₀: (-1.0018489708901752, -0.9901815710112007)\n\e[0m     95.0% CI for β₁: (0.9851645388754124, 1.0263131430575183)\n\e[0m     95.0% CI for β₂: (0.4881124597706954, 0.5020910392238356)\n\e[0m     95.0% CI for β₃: (2.984798581472277, 3.028527445212413)\n\e[0m     95.0% CI for β₄: (-4.99999999999985, -2.1286507407760933)\n\e[0m     95.0% CI for β₅: (-0.49227523407426677, -0.4695089473776722)"
@test prof_no_syms_sprint == "\e[38;2;86;182;194mProfileLikelihoodSolution\e[0m. MLE retcode: \e[38;2;86;182;194mSuccess\n\e[0mConfidence intervals: \n\e[0m     95.0% CI for θ₁: (0.048557863157896265, 0.054378929873026595)\n\e[0m     95.0% CI for θ₂: (-1.0018489708901752, -0.9901815710112007)\n\e[0m     95.0% CI for θ₃: (0.9851645388754124, 1.0263131430575183)\n\e[0m     95.0% CI for θ₄: (0.4881124597706954, 0.5020910392238356)\n\e[0m     95.0% CI for θ₅: (2.984798581472277, 3.028527445212413)\n\e[0m     95.0% CI for θ₆: (-4.99999999999985, -2.1286507407760933)\n\e[0m     95.0% CI for θ₇: (-0.49227523407426677, -0.4695089473776722)"

prof_view1 = prof[1]
prof_view4 = prof[:β₂]
prof_view1_no_syms = prof_no_syms[1]
prof_view4_no_syms = prof_no_syms[4]
prof_view1_sprint = _sprint(prof_view1)
prof_view4_sprint = _sprint(prof_view4)
prof_view1_no_syms_sprint = _sprint(prof_view1_no_syms)
prof_view4_no_syms_sprint = _sprint(prof_view4_no_syms)
@test prof_view1_sprint == "\e[38;2;86;182;194mProfile likelihood\e[0m for parameter\e[38;2;86;182;194m σ\e[0m. MLE retcode: \e[38;2;86;182;194mSuccess\n\e[0mMLE: 0.051330602449744085\n\e[0m95.0% CI for σ: (0.048557863157896265, 0.054378929873026595)"
@test prof_view4_sprint == "\e[38;2;86;182;194mProfile likelihood\e[0m for parameter\e[38;2;86;182;194m β₂\e[0m. MLE retcode: \e[38;2;86;182;194mSuccess\n\e[0mMLE: 0.4951059914675016\n\e[0m95.0% CI for β₂: (0.4881124597706954, 0.5020910392238356)"
@test prof_view1_no_syms_sprint == "\e[38;2;86;182;194mProfile likelihood\e[0m for parameter\e[38;2;86;182;194m θ₁\e[0m. MLE retcode: \e[38;2;86;182;194mSuccess\n\e[0mMLE: 0.051330602449744085\n\e[0m95.0% CI for θ₁: (0.048557863157896265, 0.054378929873026595)"
@test prof_view4_no_syms_sprint == "\e[38;2;86;182;194mProfile likelihood\e[0m for parameter\e[38;2;86;182;194m θ₄\e[0m. MLE retcode: \e[38;2;86;182;194mSuccess\n\e[0mMLE: 0.4951059914675016\n\e[0m95.0% CI for θ₄: (0.4881124597706954, 0.5020910392238356)"

param_pairs = ((:σ, :β₁), (:β₃, :β₄), (:σ, :β₅))
param_pairs_no_syms = ((1, 3), (5, 6), (1, 7))
grids = construct_profile_grids(param_pairs, sol, prof_lb, prof_ub, [12, 20, 20, 20, 20, 20, 20])
bivariate_prof = bivariate_profile(prob, sol, param_pairs; grids, parallel=true, resolution=5, outer_layers=3) 
grids_no_syms = construct_profile_grids(param_pairs_no_syms, sol_no_syms, prof_lb, prof_ub, [12, 20, 20, 20, 20, 20, 20])
bivariate_prof_no_syms = bivariate_profile(prob_no_syms, sol_no_syms, param_pairs_no_syms; grids = grids_no_syms, parallel=true, resolution=5, outer_layers=3)
bivariate_prof_sprint = _sprint(bivariate_prof)
bivariate_prof_no_syms_sprint = _sprint(bivariate_prof_no_syms)
@test bivariate_prof_sprint == "\e[38;2;86;182;194mBivariateProfileLikelihoodSolution\e[0m. MLE retcode: \e[38;2;86;182;194mSuccess\n\e[0mProfile info: \n\e[0m     (σ, β₅): 20 layers. Bbox for 95.0% CR: [0.04813093883468115, 0.052991533084078314] × [-5.0, 15.0]\n\e[0m     (σ, β₁): 10 layers. Bbox for 95.0% CR: [0.048130938834682474, 0.05299153308407461] × [1.0035475779110081, 1.0066798146817615]\n\e[0m     (β₃, β₄): 10 layers. Bbox for 95.0% CR: [3.002570528192633, 3.010058900277469] × [-3.1346450861299227, 5.579785236772706]"
@test bivariate_prof_no_syms_sprint == "\e[38;2;86;182;194mBivariateProfileLikelihoodSolution\e[0m. MLE retcode: \e[38;2;86;182;194mSuccess\n\e[0mProfile info: \n\e[0m     (θ₁, θ₇): 20 layers. Bbox for 95.0% CR: [0.04813093883468115, 0.052991533084078314] × [-5.0, 15.0]\n\e[0m     (θ₁, θ₃): 10 layers. Bbox for 95.0% CR: [0.048130938834682474, 0.05299153308407461] × [1.0035475779110081, 1.0066798146817615]\n\e[0m     (θ₅, θ₆): 10 layers. Bbox for 95.0% CR: [3.002570528192633, 3.010058900277469] × [-3.1346450861299227, 5.579785236772706]"

bivariate_prof_view13 = bivariate_prof[:σ, :β₁]
bivariate_prof_view13_no_syms = bivariate_prof_no_syms[1, 3]
bivariate_prof_view56 = bivariate_prof[5, 6]
bivariate_prof_view56_no_syms = bivariate_prof_no_syms[5, 6]
@test_throws BoundsError bivariate_prof_no_syms[:θ₅, :θ₆]
bivariate_prof_view13_sprint = _sprint(bivariate_prof_view13)
bivariate_prof_view13_no_syms_sprint = _sprint(bivariate_prof_view13_no_syms)
bivariate_prof_view56_sprint = _sprint(bivariate_prof_view56)
bivariate_prof_view56_no_syms_sprint = _sprint(bivariate_prof_view56_no_syms)
@test bivariate_prof_view13_sprint == "\e[38;2;86;182;194mBivariate profile likelihood\e[0m for parameters\e[38;2;86;182;194m (σ, β₁)\e[0m. MLE retcode: \e[38;2;86;182;194mSuccess\n\e[0mMLEs: (0.051330602449744085, 1.005739244836975)\n\e[0m95.0% bounding box for (σ, β₁): [0.048130938834682474, 0.05299153308407461] × [1.0035475779110081, 1.0066798146817615]" 
@test bivariate_prof_view13_no_syms_sprint == "\e[38;2;86;182;194mBivariate profile likelihood\e[0m for parameters\e[38;2;86;182;194m (θ₁, θ₃)\e[0m. MLE retcode: \e[38;2;86;182;194mSuccess\n\e[0mMLEs: (0.051330602449744085, 1.005739244836975)\n\e[0m95.0% bounding box for (θ₁, θ₃): [0.048130938834682474, 0.05299153308407461] × [1.0035475779110081, 1.0066798146817615]"
@test bivariate_prof_view56_sprint == "\e[38;2;86;182;194mBivariate profile likelihood\e[0m for parameters\e[38;2;86;182;194m (β₃, β₄)\e[0m. MLE retcode: \e[38;2;86;182;194mSuccess\n\e[0mMLEs: (3.0066628757589697, -2.128650740776393)\n\e[0m95.0% bounding box for (β₃, β₄): [3.002570528192633, 3.010058900277469] × [-3.1346450861299227, 5.579785236772706]"
@test bivariate_prof_view56_no_syms_sprint == "\e[38;2;86;182;194mBivariate profile likelihood\e[0m for parameters\e[38;2;86;182;194m (θ₅, θ₆)\e[0m. MLE retcode: \e[38;2;86;182;194mSuccess\n\e[0mMLEs: (3.0066628757589697, -2.128650740776393)\n\e[0m95.0% bounding box for (θ₅, θ₆): [3.002570528192633, 3.010058900277469] × [-3.1346450861299227, 5.579785236772706]"