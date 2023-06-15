using ..ProfileLikelihood
using LinearAlgebra
using PreallocationTools
using Random
using Distributions
using LinearSolve
using OrdinaryDiffEq
using Optimization
using OptimizationOptimJL
using OptimizationNLopt
using FiniteVolumeMethod
using InteractiveUtils
using DelaunayTriangulation
using Interpolations
using CairoMakie
using LaTeXStrings
include("templates.jl")
const test_f(grid1, grid2) = length(grid1) == length(grid2) ? grid1 ≈ grid2 : grid1[begin:end-1] ≈ grid2

@testset "Test that we can correctly construct the parameter ranges" begin
    lb = -2.0
    ub = 2.0
    mpt = 0.0
    res = 50
    lr, ur = ProfileLikelihood.construct_profile_ranges(lb, ub, mpt, res)
    @test lr == LinRange(0.0, -2.0, 50)
    @test ur == LinRange(0.0, 2.0, 50)
end

@testset "Test that we can construct the parameter ranges from a solution" begin
    prob, loglikk, θ, dat = multiple_linear_regression()
    true_ℓ = loglikk(reduce(vcat, θ), dat)
    lb = (1e-12, -3.0, 0.0, 0.0, -6.0)
    ub = (0.2, 0.0, 3.0, 3.0, 6.0)
    res = 27
    ug = RegularGrid(lb, ub, res)
    sol = grid_search(prob, ug; save_vals=Val(false))

    res2 = 50
    ranges = ProfileLikelihood.construct_profile_ranges(sol, lb, ub, res2)
    @test all(i -> ranges[i] == (LinRange(sol.mle[i], lb[i], res2), LinRange(sol.mle[i], ub[i], res2)), eachindex(lb))

    res2 = (50, 102, 50, 671, 123)
    ranges = ProfileLikelihood.construct_profile_ranges(sol, lb, ub, res2)
    @test all(i -> ranges[i] == (LinRange(sol.mle[i], lb[i], res2[i]), LinRange(sol.mle[i], ub[i], res2[i])), eachindex(lb))
end

@testset "Test that we can extract a problem and solution" begin
    prob, loglikk, θ, dat = multiple_linear_regression()
    true_ℓ = loglikk(reduce(vcat, θ), dat)
    lb = (1e-12, -3.0, 0.0, 0.0, -6.0)
    ub = (0.2, 0.0, 3.0, 3.0, 6.0)
    res = 27
    ug = RegularGrid(lb, ub, res)
    sol = grid_search(prob, ug; save_vals=Val(false))
    _opt_prob, _mles, _ℓmax = ProfileLikelihood.extract_problem_and_solution(prob, sol)
    @test _opt_prob === sol.problem.problem
    @test _mles == sol.mle
    @test !(_mles === sol.mle)
    @test _ℓmax == sol.maximum
end

@testset "Test that we can prepare the profile results correctly" begin
    N = 5
    T = Float64
    F = Float64
    _θ, _prof, _other_mles, _splines, _confidence_intervals = ProfileLikelihood.prepare_profile_results(N, T, F)
    @test _θ == Dict{Int,Vector{T}}([])
    @test _prof == Dict{Int,Vector{T}}([])
    @test _other_mles == Dict{Int,Vector{Vector{T}}}([])
    @test _confidence_intervals == Dict{Int,ProfileLikelihood.ConfidenceInterval{T,F}}([])
end

@testset "Test that we can correctly normalise the objective function" begin
    prob, loglikk, θ, dat = multiple_linear_regression()
    true_ℓ = loglikk(reduce(vcat, θ), dat)
    lb = (1e-12, -3.0, 0.0, 0.0, -6.0)
    ub = (0.2, 0.0, 3.0, 3.0, 6.0)
    res = 27
    ug = RegularGrid(lb, ub, res)
    sol = grid_search(prob, ug; save_vals=Val(false))
    _opt_prob, _mles, _ℓmax = ProfileLikelihood.extract_problem_and_solution(prob, sol)
    shifted_opt_prob = ProfileLikelihood.normalise_objective_function(_opt_prob, _ℓmax, false)
    @test shifted_opt_prob.f.f.shift == 0.0
    @test shifted_opt_prob.f.f.original_f == _opt_prob.f
    @test shifted_opt_prob.f(reduce(vcat, θ), dat) ≈ -loglikk(reduce(vcat, θ), dat)
    @inferred shifted_opt_prob.f(reduce(vcat, θ), dat)
    shifted_opt_prob = ProfileLikelihood.normalise_objective_function(_opt_prob, _ℓmax, true)
    @test shifted_opt_prob.f(reduce(vcat, θ), dat) ≈ -(loglikk(reduce(vcat, θ), dat) - _ℓmax)
    @inferred shifted_opt_prob.f(reduce(vcat, θ), dat)
end

@testset "Test that we can prepare the cache" begin
    prob, loglikk, θ, dat = multiple_linear_regression()
    true_ℓ = loglikk(reduce(vcat, θ), dat)
    lb = (1e-12, -3.0, 0.0, 0.0, -6.0)
    ub = (0.2, 0.0, 3.0, 3.0, 6.0)
    res = 27
    ug = RegularGrid(lb, ub, res)
    sol = grid_search(prob, ug; save_vals=Val(false))
    _opt_prob, _mles, _ℓmax = ProfileLikelihood.extract_problem_and_solution(prob, sol)
    N = 5
    n = 2
    res2 = (50, 102, 50, 671, 123)
    ranges = ProfileLikelihood.construct_profile_ranges(sol, lb, ub, res2)
    _left_profile_vals, _right_profile_vals,
    _left_param_vals, _right_param_vals,
    _left_other_mles, _right_other_mles,
    _combined_profiles, _combined_param_vals, _combined_other_mles,
    _cache, _sub_cache = ProfileLikelihood.prepare_cache_vectors(n, N, ranges[n], _mles)
    T = Float64
    F = Float64
    @test _left_profile_vals == Vector{T}([])
    @test _right_profile_vals == Vector{T}([])
    @test _left_param_vals == Vector{T}([])
    @test _right_param_vals == Vector{T}([])
    @test _left_other_mles == Vector{Vector{T}}([])
    @test _right_other_mles == Vector{Vector{T}}([])
    @test _combined_profiles == Vector{T}([])
    @test _combined_param_vals == Vector{T}([])
    @test _combined_other_mles == Vector{Vector{T}}([])
    @test _cache.dual_du == DiffCache(zeros(T, N)).dual_du
    @test _cache.du == DiffCache(zeros(T, N)).du
    @test _sub_cache == _mles[[1, 3, 4, 5]]
end

@testset "Test the construction of the ProfileLikelihoodSolution" begin
    Random.seed!(98871)
    n = 300
    β = [-1.0, 1.0, 0.5, 3.0]
    σ = 0.05
    x₁ = rand(Uniform(-1, 1), n)
    x₂ = rand(Normal(1.0, 0.5), n)
    X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
    ε = rand(Normal(0.0, σ), n)
    y = X * β + ε
    sse = DiffCache(zeros(n))
    β_cache = DiffCache(similar(β), 10)
    dat = (y, X, sse, n, β_cache)
    @inline function loglik_fnc(θ, data)
        σ, β₀, β₁, β₂, β₃ = θ
        y, X, sse, n, β = data
        _sse = get_tmp(sse, θ)
        _β = get_tmp(β, θ)
        _β[1] = β₀
        _β[2] = β₁
        _β[3] = β₂
        _β[4] = β₃
        ℓℓ = -0.5n * log(2π * σ^2)
        mul!(_sse, X, _β)
        for i in eachindex(y)
            ℓℓ = ℓℓ - 0.5 / σ^2 * (y[i] - _sse[i])^2
        end
        return ℓℓ
    end
    θ₀ = ones(5)
    prob = LikelihoodProblem(loglik_fnc, θ₀;
        data=dat,
        f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
        prob_kwargs=(lb=[0.0, -5.0, -5.0, -5.0, -5.0],
            ub=[15.0, 15.0, 15.0, 15.0, 15.0]),
        syms=[:σ, :β₀, :β₁, :β₂, :β₃])
    sol = mle(prob, Optim.LBFGS())
    prof = profile(prob, sol, [1, 3])
    @test ProfileLikelihood.get_parameter_values(prof) == prof.parameter_values
    @test ProfileLikelihood.get_parameter_values(prof, 1) == prof.parameter_values[1]
    @test ProfileLikelihood.get_parameter_values(prof, :σ) == prof.parameter_values[1]
    @test ProfileLikelihood.get_parameter_values(prof, 3) == prof.parameter_values[3]
    @test ProfileLikelihood.get_parameter_values(prof, :β₁) == prof.parameter_values[3]
    @test ProfileLikelihood.get_profile_values(prof) == prof.profile_values
    @test ProfileLikelihood.get_profile_values(prof, 3) == prof.profile_values[3]
    @test ProfileLikelihood.get_profile_values(prof, :σ) == prof.profile_values[1]
    @test ProfileLikelihood.get_likelihood_problem(prof) == prof.likelihood_problem == prob
    @test ProfileLikelihood.get_likelihood_solution(prof) == prof.likelihood_solution == sol
    @test ProfileLikelihood.get_splines(prof) == prof.splines
    @test ProfileLikelihood.get_splines(prof, 3) == prof.splines[3]
    @test ProfileLikelihood.get_splines(prof, :σ) == prof.splines[1]
    @test ProfileLikelihood.get_confidence_intervals(prof) == prof.confidence_intervals
    @test ProfileLikelihood.get_confidence_intervals(prof, 1) == prof.confidence_intervals[1]
    @test ProfileLikelihood.get_confidence_intervals(prof, :β₁) == prof.confidence_intervals[3]
    @test ProfileLikelihood.get_other_mles(prof) == prof.other_mles
    @test ProfileLikelihood.get_other_mles(prof, 3) == prof.other_mles[3]
    @test ProfileLikelihood.get_syms(prof) == prob.syms == [:σ, :β₀, :β₁, :β₂, :β₃]
    @test ProfileLikelihood.get_syms(prof, 4) == :β₂
    @test SciMLBase.sym_to_index(:σ, prof) == 1
    @test SciMLBase.sym_to_index(:β₀, prof) == 2
    @test SciMLBase.sym_to_index(:β₁, prof) == 3
    @test SciMLBase.sym_to_index(:β₂, prof) == 4
    @test SciMLBase.sym_to_index(:β₃, prof) == 5
    @test ProfileLikelihood.profiled_parameters(prof) == [1, 3]
    @test ProfileLikelihood.number_of_profiled_parameters(prof) == 2
end

@testset "Check that the parameter values are correct" begin
    Random.seed!(98871)
    n = 300
    β = [-1.0, 1.0, 0.5, 3.0]
    σ = 0.05
    x₁ = rand(Uniform(-1, 1), n)
    x₂ = rand(Normal(1.0, 0.5), n)
    X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
    ε = rand(Normal(0.0, σ), n)
    y = X * β + ε
    sse = DiffCache(zeros(n))
    β_cache = DiffCache(similar(β), 10)
    dat = (y, X, sse, n, β_cache)
    @inline function loglik_fnc(θ, data)
        σ, β₀, β₁, β₂, β₃ = θ
        y, X, sse, n, β = data
        _sse = get_tmp(sse, θ)
        _β = get_tmp(β, θ)
        _β[1] = β₀
        _β[2] = β₁
        _β[3] = β₂
        _β[4] = β₃
        ℓℓ = -0.5n * log(2π * σ^2)
        mul!(_sse, X, _β)
        for i in eachindex(y)
            ℓℓ = ℓℓ - 0.5 / σ^2 * (y[i] - _sse[i])^2
        end
        return ℓℓ
    end
    θ₀ = ones(5)
    prob = LikelihoodProblem(loglik_fnc, θ₀;
        data=dat,
        f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
        prob_kwargs=(lb=[0.0, -5.0, -5.0, -5.0, -5.0],
            ub=[15.0, 15.0, 15.0, 15.0, 15.0]),
        syms=[:σ, :β₀, :β₁, :β₂, :β₃])
    sol = mle(prob, Optim.LBFGS())
    prof = profile(prob, sol, [1, 3])
    xmin, xmax = extrema(get_parameter_values(prof, :σ))
    m = length(get_parameter_values(prof, :σ))
    Δleft = (sol[:σ] - get_lower_bounds(prob)[1]) / (200 - 1)
    Δright = (xmax - sol[:σ]) / (10 - 1) # reached min_steps
    left_grid = xmin:Δleft:sol[:σ]
    right_grid = sol[:σ]:Δright:xmax
    full_grid = [left_grid..., right_grid[2:end]...]
    @test test_f(get_parameter_values(prof, :σ), full_grid)

    xmin, xmax = extrema(get_parameter_values(prof, :β₁))
    m = length(get_parameter_values(prof, :β₁))
    Δleft = (sol[:β₁] - xmin) / (10 - 1)
    Δright = (xmax - sol[:β₁]) / (10 - 1) # reached min_steps
    left_grid = xmin:Δleft:sol[:β₁]
    right_grid = sol[:β₁]:Δright:xmax
    full_grid = [left_grid..., right_grid[2:end]...]
    @test test_f(get_parameter_values(prof, :β₁), full_grid)
end

@testset "Test that other_mles and parameter_values line up" begin
    Random.seed!(98871)
    n = 300
    β = [-1.0, 1.0, 0.5, 3.0]
    σ = 0.05
    x₁ = rand(Uniform(-1, 1), n)
    x₂ = rand(Normal(1.0, 0.5), n)
    X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
    ε = rand(Normal(0.0, σ), n)
    y = X * β + ε
    sse = DiffCache(zeros(n))
    β_cache = DiffCache(similar(β), 10)
    dat = (y, X, sse, n, β_cache)
    @inline function loglik_fnc(θ, data)
        σ, β₀, β₁, β₂, β₃ = θ
        y, X, sse, n, β = data
        _sse = get_tmp(sse, θ)
        _β = get_tmp(β, θ)
        _β[1] = β₀
        _β[2] = β₁
        _β[3] = β₂
        _β[4] = β₃
        ℓℓ = -0.5n * log(2π * σ^2)
        mul!(_sse, X, _β)
        for i in eachindex(y)
            ℓℓ = ℓℓ - 0.5 / σ^2 * (y[i] - _sse[i])^2
        end
        return ℓℓ
    end
    θ₀ = ones(5)
    prob = LikelihoodProblem(loglik_fnc, θ₀;
        data=dat,
        f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
        prob_kwargs=(lb=[0.0, -5.0, -5.0, -5.0, -5.0],
            ub=[15.0, 15.0, 15.0, 15.0, 15.0]),
        syms=[:σ, :β₀, :β₁, :β₂, :β₃])
    sol = mle(prob, Optim.LBFGS())
    prof = profile(prob, sol, [1, 3])
    for i in eachindex(prof.parameter_values[1])
        _σ = get_parameter_values(prof, :σ)[i]
        b0, b1, b2, b3 = ProfileLikelihood.get_other_mles(prof, :σ)[i]
        θ = [_σ, b0, b1, b2, b3]
        ℓ = prob.log_likelihood_function(θ, prob.data) - get_maximum(sol)
        @test ℓ ≈ get_profile_values(prof, :σ)[i]
        @test ℓ ≈ prof[:σ](_σ) atol = 1e-12
    end
    for i in eachindex(prof.parameter_values[3])
        _β₁ = get_parameter_values(prof, :β₁)[i]
        s, b0, b2, b3 = ProfileLikelihood.get_other_mles(prof, :β₁)[i]
        θ = [s, b0, _β₁, b2, b3]
        ℓ = prob.log_likelihood_function(θ, prob.data) - get_maximum(sol)
        @test ℓ ≈ get_profile_values(prof, :β₁)[i]
        @test ℓ ≈ prof[:β₁](_β₁) atol = 1e-12
    end
end

@testset "Test that views are working correctly on the ProfileLikelihoodSolution" begin
    Random.seed!(98871)
    n = 300
    β = [-1.0, 1.0, 0.5, 3.0]
    σ = 0.05
    x₁ = rand(Uniform(-1, 1), n)
    x₂ = rand(Normal(1.0, 0.5), n)
    X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
    ε = rand(Normal(0.0, σ), n)
    y = X * β + ε
    sse = DiffCache(zeros(n))
    β_cache = DiffCache(similar(β), 10)
    dat = (y, X, sse, n, β_cache)
    @inline function loglik_fnc(θ, data)
        σ, β₀, β₁, β₂, β₃ = θ
        y, X, sse, n, β = data
        _sse = get_tmp(sse, θ)
        _β = get_tmp(β, θ)
        _β[1] = β₀
        _β[2] = β₁
        _β[3] = β₂
        _β[4] = β₃
        ℓℓ = -0.5n * log(2π * σ^2)
        mul!(_sse, X, _β)
        for i in eachindex(y)
            ℓℓ = ℓℓ - 0.5 / σ^2 * (y[i] - _sse[i])^2
        end
        return ℓℓ
    end
    θ₀ = ones(5)
    prob = LikelihoodProblem(loglik_fnc, θ₀;
        data=dat,
        f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
        prob_kwargs=(lb=[0.0, -5.0, -5.0, -5.0, -5.0],
            ub=[15.0, 15.0, 15.0, 15.0, 15.0]),
        syms=[:σ, :β₀, :β₁, :β₂, :β₃])
    sol = mle(prob, Optim.LBFGS())
    prof = profile(prob, sol, [1, 3])
    i = 1
    prof_view = prof[i]
    @test ProfileLikelihood.get_parent(prof_view) == prof
    @test ProfileLikelihood.get_index(prof_view) == i
    @test ProfileLikelihood.get_parameter_values(prof_view) == prof.parameter_values[i]
    @test ProfileLikelihood.get_parameter_values(prof_view, 1) == prof.parameter_values[i][1]
    @test ProfileLikelihood.get_parameter_values(prof_view, 3) == prof.parameter_values[i][3]
    @test ProfileLikelihood.get_profile_values(prof_view) == prof.profile_values[i]
    @test ProfileLikelihood.get_profile_values(prof_view, 3) == prof.profile_values[i][3]
    @test ProfileLikelihood.get_likelihood_problem(prof_view) == prof.likelihood_problem == prob
    @test ProfileLikelihood.get_likelihood_solution(prof_view) == prof.likelihood_solution == sol
    @test ProfileLikelihood.get_splines(prof_view) == prof.splines[i]
    @test ProfileLikelihood.get_confidence_intervals(prof_view) == prof.confidence_intervals[i]
    @test ProfileLikelihood.get_confidence_intervals(prof_view, 1) == prof.confidence_intervals[i][1]
    @test ProfileLikelihood.get_other_mles(prof_view) == prof.other_mles[i]
    @test ProfileLikelihood.get_other_mles(prof_view, 3) == prof.other_mles[i][3]
    @test ProfileLikelihood.get_syms(prof_view) == :σ
    @test prof[:β₁] == prof[3]
end

@testset "Test that we can correctly call the profiles" begin
    Random.seed!(98871)
    n = 300
    β = [-1.0, 1.0, 0.5, 3.0]
    σ = 0.05
    x₁ = rand(Uniform(-1, 1), n)
    x₂ = rand(Normal(1.0, 0.5), n)
    X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
    ε = rand(Normal(0.0, σ), n)
    y = X * β + ε
    sse = DiffCache(zeros(n))
    β_cache = DiffCache(similar(β), 10)
    dat = (y, X, sse, n, β_cache)
    @inline function loglik_fnc(θ, data)
        σ, β₀, β₁, β₂, β₃ = θ
        y, X, sse, n, β = data
        _sse = get_tmp(sse, θ)
        _β = get_tmp(β, θ)
        _β[1] = β₀
        _β[2] = β₁
        _β[3] = β₂
        _β[4] = β₃
        ℓℓ = -0.5n * log(2π * σ^2)
        mul!(_sse, X, _β)
        for i in eachindex(y)
            ℓℓ = ℓℓ - 0.5 / σ^2 * (y[i] - _sse[i])^2
        end
        return ℓℓ
    end
    θ₀ = ones(5)
    prob = LikelihoodProblem(loglik_fnc, θ₀;
        data=dat,
        f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
        prob_kwargs=(lb=[0.0, -5.0, -5.0, -5.0, -5.0],
            ub=[15.0, 15.0, 15.0, 15.0, 15.0]),
        syms=[:σ, :β₀, :β₁, :β₂, :β₃])
    sol = mle(prob, Optim.LBFGS())
    prof = profile(prob, sol, [1, 3])
    i = 1
    prof_view = prof[i]
    x = prof.splines[i].itp.knots
    @test prof_view(x) == prof(x, i) == prof.splines[i](x) ≈ prof_view.parent.profile_values[i]
end

@testset "Test that resolution is being correctly applied" begin
    Random.seed!(98871)
    n = 300
    β = [-1.0, 1.0, 0.5, 3.0]
    σ = 0.05
    x₁ = rand(Uniform(-1, 1), n)
    x₂ = rand(Normal(1.0, 0.5), n)
    X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
    ε = rand(Normal(0.0, σ), n)
    y = X * β + ε
    sse = DiffCache(zeros(n))
    β_cache = DiffCache(similar(β), 10)
    dat = (y, X, sse, n, β_cache)
    @inline function loglik_fnc(θ, data)
        σ, β₀, β₁, β₂, β₃ = θ
        y, X, sse, n, β = data
        _sse = get_tmp(sse, θ)
        _β = get_tmp(β, θ)
        _β[1] = β₀
        _β[2] = β₁
        _β[3] = β₂
        _β[4] = β₃
        ℓℓ = -0.5n * log(2π * σ^2)
        mul!(_sse, X, _β)
        for i in eachindex(y)
            ℓℓ = ℓℓ - 0.5 / σ^2 * (y[i] - _sse[i])^2
        end
        return ℓℓ
    end
    θ₀ = ones(5)
    prob = LikelihoodProblem(loglik_fnc, θ₀;
        data=dat,
        f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
        prob_kwargs=(lb=[0.0, -5.0, -5.0, -5.0, -5.0],
            ub=[15.0, 15.0, 15.0, 15.0, 15.0]),
        syms=[:σ, :β₀, :β₁, :β₂, :β₃])
    sol = mle(prob, Optim.LBFGS())
    prof = profile(prob, sol, [1, 3])
    prof = profile(prob, sol, [1, 3]; resolution=[50000, 18000, 75000, 5000, 5000], min_steps=0)

    xmin, xmax = extrema(get_parameter_values(prof, :σ))
    m = length(get_parameter_values(prof, :σ))
    Δleft = (sol[:σ] - get_lower_bounds(prob)[1]) / (50000 - 1)
    Δright = (get_upper_bounds(prob)[1] - sol[:σ]) / (50000 - 1)
    left_grid = xmin:Δleft:sol[:σ]
    right_grid = sol[:σ]:Δright:xmax
    full_grid = [left_grid..., right_grid[2:end]...]
    @test test_f(get_parameter_values(prof, :σ), full_grid)

    xmin, xmax = extrema(get_parameter_values(prof, :β₁))
    m = length(get_parameter_values(prof, :β₁))
    Δleft = (sol[:β₁] - get_lower_bounds(prob)[3]) / (75000 - 1)
    Δright = (get_upper_bounds(prob)[3] - sol[:β₁]) / (75000 - 1)
    left_grid = xmin:Δleft:sol[:β₁]
    right_grid = sol[:β₁]:Δright:xmax
    full_grid = [left_grid..., right_grid[2:end]...]
    @test test_f(get_parameter_values(prof, :β₁), full_grid)
end

@testset "Threads" begin
    Random.seed!(98871)
    n = 30
    β = [-1.0, 1.0, 0.5, 3.0]
    σ = 0.4
    x₁ = rand(Uniform(-1, 1), n)
    x₂ = rand(Normal(1.0, 0.5), n)
    X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
    ε = rand(Normal(0.0, σ), n)
    y = X * β + ε
    sse = DiffCache(zeros(n))
    β_cache = DiffCache(similar(β), 10)
    dat = (y, X, sse, n, β_cache)
    @inline function loglik_fnc(θ, data)
        σ, β₀, β₁, β₂, β₃ = θ
        y, X, sse, n, β = data
        _sse = get_tmp(sse, θ)
        _β = get_tmp(β, θ)
        _β[1] = β₀
        _β[2] = β₁
        _β[3] = β₂
        _β[4] = β₃
        ℓℓ = -0.5n * log(2π * σ^2)
        mul!(_sse, X, _β)
        for i in eachindex(y)
            ℓℓ = ℓℓ - 0.5 / σ^2 * (y[i] - _sse[i])^2
        end
        return ℓℓ
    end
    θ₀ = ones(5)
    prob = LikelihoodProblem(loglik_fnc, θ₀;
        data=dat,
        f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
        prob_kwargs=(lb=[0.0, -5.0, -5.0, -5.0, -5.0],
            ub=[15.0, 15.0, 15.0, 15.0, 15.0]),
        syms=[:σ, :β₀, :β₁, :β₂, :β₃])
    sol = mle(prob, Optim.LBFGS())
    prof_serial = profile(prob, sol)
    prof_parallel = profile(prob, sol; parallel=true)
    @test all(i -> abs((prof_parallel.confidence_intervals[i].lower - prof_serial.confidence_intervals[i].lower) / prof_serial.confidence_intervals[i].lower) < 1e-2, 1:5)
    @test all(i -> abs((prof_parallel.confidence_intervals[i].upper - prof_serial.confidence_intervals[i].upper) / prof_serial.confidence_intervals[i].upper) < 1e-2, 1:5)
    @test all(i -> prof_parallel.parameter_values[i] ≈ prof_serial.parameter_values[i], 1:5)
    @test all(i -> prof_parallel.profile_values[i] ≈ prof_serial.profile_values[i], 1:5)
end

@testset "More parallel testing for a problem involving an integrator" begin
    Random.seed!(2929911002)
    u₀, λ, K, n, T = 0.5, 1.0, 1.0, 100, 10.0
    t = LinRange(0, T, n)
    u = @. K * u₀ * exp(λ * t) / (K - u₀ + u₀ * exp(λ * t))
    σ = 0.1
    uᵒ = u .+ [0.0, σ * randn(length(u) - 1)...] # add some noise 
    @inline function ode_fnc(u, p, t)
        local λ, K
        λ, K = p
        du = λ * u * (1 - u / K)
        return du
    end
    @inline function loglik_fnc2(θ, data, integrator)
        local uᵒ, n, λ, K, σ, u0
        uᵒ, n = data
        λ, K, σ, u0 = θ
        integrator.p[1] = λ
        integrator.p[2] = K
        reinit!(integrator, u0)
        solve!(integrator)
        return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, n)
    end
    θ₀ = [0.7, 2.0, 0.15, 0.4]
    lb = [0.0, 1e-6, 1e-6, 0.0]
    ub = [10.0, 10.0, 10.0, 10.0]
    syms = [:λ, :K, :σ, :u₀]
    prob = LikelihoodProblem(
        loglik_fnc2, θ₀, ode_fnc, u₀, (0.0, T); # u₀ is just a placeholder IC in this case
        syms=syms,
        data=(uᵒ, n),
        ode_parameters=[1.0, 1.0], # temp values for [λ, K],
        ode_kwargs=(verbose=false, saveat=t),
        f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
        prob_kwargs=(lb=lb, ub=ub),
        ode_alg=Tsit5()
    )
    sol = mle(prob, NLopt.LN_NELDERMEAD)
    prof1 = profile(prob, sol)
    prof2 = profile(prob, sol; parallel=true)

    @test prof1.confidence_intervals[1].lower ≈ prof2.confidence_intervals[1].lower rtol = 1e-1
    @test prof1.confidence_intervals[2].lower ≈ prof2.confidence_intervals[2].lower rtol = 1e-1
    @test prof1.confidence_intervals[3].lower ≈ prof2.confidence_intervals[3].lower rtol = 1e-1
    @test prof1.confidence_intervals[4].lower ≈ prof2.confidence_intervals[4].lower rtol = 1e-1
    @test prof1.confidence_intervals[1].upper ≈ prof2.confidence_intervals[1].upper rtol = 1e-1
    @test prof1.confidence_intervals[2].upper ≈ prof2.confidence_intervals[2].upper rtol = 1e-1
    @test prof1.confidence_intervals[3].upper ≈ prof2.confidence_intervals[3].upper rtol = 1e-1
    @test prof1.confidence_intervals[4].upper ≈ prof2.confidence_intervals[4].upper rtol = 1e-1
    @test prof1.other_mles[1] ≈ prof2.other_mles[1] rtol = 1e-0 atol = 1e-1
    @test prof1.other_mles[2] ≈ prof2.other_mles[2] rtol = 1e-1 atol = 1e-1
    @test prof1.other_mles[3] ≈ prof2.other_mles[3] rtol = 1e-3 atol = 1e-1
    @test prof1.other_mles[4] ≈ prof2.other_mles[4] rtol = 1e-0 atol = 1e-1
    @test prof1.parameter_values[1] ≈ prof2.parameter_values[1] rtol = 1e-1
    @test prof1.parameter_values[2] ≈ prof2.parameter_values[2] rtol = 1e-3
    @test prof1.parameter_values[3] ≈ prof2.parameter_values[3] rtol = 1e-3
    @test prof1.parameter_values[4] ≈ prof2.parameter_values[4] rtol = 1e-3
    @test issorted(prof1.parameter_values[1])
    @test issorted(prof1.parameter_values[2])
    @test issorted(prof1.parameter_values[3])
    @test issorted(prof1.parameter_values[4])
    @test issorted(prof2.parameter_values[1])
    @test issorted(prof2.parameter_values[2])
    @test issorted(prof2.parameter_values[3])
    @test issorted(prof2.parameter_values[4])
    @test prof1.profile_values[1] ≈ prof2.profile_values[1] rtol = 1e-1
    @test prof1.profile_values[2] ≈ prof2.profile_values[2] rtol = 1e-1
    @test prof1.profile_values[3] ≈ prof2.profile_values[3] rtol = 1e-1
    @test prof1.profile_values[4] ≈ prof2.profile_values[4] rtol = 1e-1
    @test prof1.splines[1].itp.knots ≈ prof2.splines[1].itp.knots
    @test prof1.splines[2].itp.knots ≈ prof2.splines[2].itp.knots
    @test prof1.splines[3].itp.knots ≈ prof2.splines[3].itp.knots
    @test prof1.splines[4].itp.knots ≈ prof2.splines[4].itp.knots

    prof1 = profile(prob, sol; parallel=true)
    prof2 = profile(prob, sol; parallel=true, next_initial_estimate_method=:interp)

    @test prof1.confidence_intervals[1].lower ≈ prof2.confidence_intervals[1].lower rtol = 1e-1
    @test prof1.confidence_intervals[2].lower ≈ prof2.confidence_intervals[2].lower rtol = 1e-1
    @test prof1.confidence_intervals[3].lower ≈ prof2.confidence_intervals[3].lower rtol = 1e-1
    @test prof1.confidence_intervals[4].lower ≈ prof2.confidence_intervals[4].lower rtol = 1e-1
    @test prof1.confidence_intervals[1].upper ≈ prof2.confidence_intervals[1].upper rtol = 1e-1
    @test prof1.confidence_intervals[2].upper ≈ prof2.confidence_intervals[2].upper rtol = 1e-1
    @test prof1.confidence_intervals[3].upper ≈ prof2.confidence_intervals[3].upper rtol = 1e-1
    @test prof1.confidence_intervals[4].upper ≈ prof2.confidence_intervals[4].upper rtol = 1e-1
    @test prof1.other_mles[1] ≈ prof2.other_mles[1] rtol = 1e-0 atol = 1e-1
    @test prof1.other_mles[2] ≈ prof2.other_mles[2] rtol = 1e-1 atol = 1e-1
    @test prof1.other_mles[3] ≈ prof2.other_mles[3] rtol = 1e-3 atol = 1e-1
    @test prof1.other_mles[4] ≈ prof2.other_mles[4] rtol = 1e-0 atol = 1e-1
    @test prof1.parameter_values[1] ≈ prof2.parameter_values[1] rtol = 1e-1
    @test prof1.parameter_values[2] ≈ prof2.parameter_values[2] rtol = 1e-3
    @test prof1.parameter_values[3] ≈ prof2.parameter_values[3] rtol = 1e-3
    @test prof1.parameter_values[4] ≈ prof2.parameter_values[4] rtol = 1e-3
    @test issorted(prof1.parameter_values[1])
    @test issorted(prof1.parameter_values[2])
    @test issorted(prof1.parameter_values[3])
    @test issorted(prof1.parameter_values[4])
    @test issorted(prof2.parameter_values[1])
    @test issorted(prof2.parameter_values[2])
    @test issorted(prof2.parameter_values[3])
    @test issorted(prof2.parameter_values[4])
    @test prof1.profile_values[1] ≈ prof2.profile_values[1] rtol = 1e-1
    @test prof1.profile_values[2] ≈ prof2.profile_values[2] rtol = 1e-1
    @test prof1.profile_values[3] ≈ prof2.profile_values[3] rtol = 1e-1
    @test prof1.profile_values[4] ≈ prof2.profile_values[4] rtol = 1e-1
    @test prof1.splines[1].itp.knots ≈ prof2.splines[1].itp.knots
    @test prof1.splines[2].itp.knots ≈ prof2.splines[2].itp.knots
    @test prof1.splines[3].itp.knots ≈ prof2.splines[3].itp.knots
    @test prof1.splines[4].itp.knots ≈ prof2.splines[4].itp.knots

    prof1 = profile(prob, sol; parallel=false)
    prof2 = profile(prob, sol; parallel=false, next_initial_estimate_method=:interp)

    @test prof1.confidence_intervals[1].lower ≈ prof2.confidence_intervals[1].lower rtol = 1e-1
    @test prof1.confidence_intervals[2].lower ≈ prof2.confidence_intervals[2].lower rtol = 1e-1
    @test prof1.confidence_intervals[3].lower ≈ prof2.confidence_intervals[3].lower rtol = 1e-1
    @test prof1.confidence_intervals[4].lower ≈ prof2.confidence_intervals[4].lower rtol = 1e-1
    @test prof1.confidence_intervals[1].upper ≈ prof2.confidence_intervals[1].upper rtol = 1e-1
    @test prof1.confidence_intervals[2].upper ≈ prof2.confidence_intervals[2].upper rtol = 1e-1
    @test prof1.confidence_intervals[3].upper ≈ prof2.confidence_intervals[3].upper rtol = 1e-1
    @test prof1.confidence_intervals[4].upper ≈ prof2.confidence_intervals[4].upper rtol = 1e-1
    @test prof1.other_mles[1] ≈ prof2.other_mles[1] rtol = 1e-0 atol = 1e-1
    @test prof1.other_mles[2] ≈ prof2.other_mles[2] rtol = 1e-1 atol = 1e-1
    @test prof1.other_mles[3] ≈ prof2.other_mles[3] rtol = 1e-3 atol = 1e-1
    @test prof1.other_mles[4] ≈ prof2.other_mles[4] rtol = 1e-0 atol = 1e-1
    @test prof1.parameter_values[1] ≈ prof2.parameter_values[1] rtol = 1e-1
    @test prof1.parameter_values[2] ≈ prof2.parameter_values[2] rtol = 1e-3
    @test prof1.parameter_values[3] ≈ prof2.parameter_values[3] rtol = 1e-3
    @test prof1.parameter_values[4] ≈ prof2.parameter_values[4] rtol = 1e-3
    @test issorted(prof1.parameter_values[1])
    @test issorted(prof1.parameter_values[2])
    @test issorted(prof1.parameter_values[3])
    @test issorted(prof1.parameter_values[4])
    @test issorted(prof2.parameter_values[1])
    @test issorted(prof2.parameter_values[2])
    @test issorted(prof2.parameter_values[3])
    @test issorted(prof2.parameter_values[4])
    @test prof1.profile_values[1] ≈ prof2.profile_values[1] rtol = 1e-1
    @test prof1.profile_values[2] ≈ prof2.profile_values[2] rtol = 1e-1
    @test prof1.profile_values[3] ≈ prof2.profile_values[3] rtol = 1e-1
    @test prof1.profile_values[4] ≈ prof2.profile_values[4] rtol = 1e-1
    @test prof1.splines[1].itp.knots ≈ prof2.splines[1].itp.knots
    @test prof1.splines[2].itp.knots ≈ prof2.splines[2].itp.knots
    @test prof1.splines[3].itp.knots ≈ prof2.splines[3].itp.knots
    @test prof1.splines[4].itp.knots ≈ prof2.splines[4].itp.knots
end

@testset "Linear interpolation" begin
    Random.seed!(98871)
    n = 300
    β = [-1.0, 1.0, 0.5, 3.0]
    σ = 0.4
    x₁ = rand(Uniform(-1, 1), n)
    x₂ = rand(Normal(1.0, 0.5), n)
    X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
    ε = rand(Normal(0.0, σ), n)
    y = X * β + ε
    sse = DiffCache(zeros(n))
    β_cache = DiffCache(similar(β), 10)
    dat = (y, X, sse, n, β_cache)
    @inline function loglik_fnc(θ, data)
        σ, β₀, β₁, β₂, β₃ = θ
        y, X, sse, n, __β = data
        _sse = get_tmp(sse, θ)
        _β = get_tmp(__β, θ)
        _β[1] = β₀
        _β[2] = β₁
        _β[3] = β₂
        _β[4] = β₃
        ℓℓ = -0.5n * log(2π * σ^2)
        mul!(_sse, X, _β)
        for i in eachindex(y)
            ℓℓ = ℓℓ - 0.5 / σ^2 * (y[i] - _sse[i])^2
        end
        return ℓℓ
    end
    θ₀ = ones(5)
    prob = LikelihoodProblem(loglik_fnc, θ₀;
        data=dat,
        f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
        prob_kwargs=(lb=[0.0, -5.0, -5.0, -5.0, -5.0],
            ub=[15.0, 15.0, 15.0, 15.0, 15.0]),
        syms=[:σ, :β₀, :β₁, :β₂, :β₃])
    sol = mle(prob, Optim.LBFGS())

    @testset "Test set_next_initial_estimate!" begin
        lb = prob.problem.lb
        ub = prob.problem.ub
        sub_cache = zeros(4)
        param_vals = [2.0, 3.0, 4.0]
        other_mles = [[(lb[2] + ub[2]) / 2, (lb[3] + ub[3]) / 2, (lb[4] + ub[4]) / 2, (lb[5] + ub[5]) / 2],
            [4.7, 4.3, 1.0, 5.0], [2.3, 3.3, 3.3, 4.9]]
        _prob = ProfileLikelihood.exclude_parameter(prob.problem, 1)
        θₙ = 4.4
        ProfileLikelihood.set_next_initial_estimate!(sub_cache, param_vals, other_mles, _prob, θₙ; next_initial_estimate_method=Val(:prev))
        @test sub_cache ≈ [2.3, 3.3, 3.3, 4.9]
        ProfileLikelihood.set_next_initial_estimate!(sub_cache, param_vals, other_mles, _prob, θₙ; next_initial_estimate_method=Val(:interp))
        @test sub_cache ≠ other_mles[end]
        @test sum((sub_cache .- other_mles[end]) / (θₙ - param_vals[end]) .- (other_mles[end-1] .- other_mles[end]) / (param_vals[end-1] - param_vals[end])) ≈ 0.0 atol = 1e-12
        @inferred ProfileLikelihood.set_next_initial_estimate!(sub_cache, param_vals, other_mles, _prob, θₙ; next_initial_estimate_method=Val(:interp))
        other_mles = [[(lb[2] + ub[2]) / 2, (lb[3] + ub[3]) / 2, (lb[4] + ub[4]) / 2, (lb[5] + ub[5]) / 2],
            [4.7, 4.3, 1.0, 15.0], [2.3, 3.3, 3.3, 15.9]]
        ProfileLikelihood.set_next_initial_estimate!(sub_cache, param_vals, other_mles, _prob, θₙ; next_initial_estimate_method=Val(:interp))
        @test sub_cache ≈ [2.3, 3.3, 3.3, 15.9]
        @inferred ProfileLikelihood.set_next_initial_estimate!(sub_cache, param_vals, other_mles, _prob, θₙ; next_initial_estimate_method=Val(:interp))
        param_vals = [2.0]
        other_mles = [[(lb[2] + ub[2]) / 2, (lb[3] + ub[3]) / 2, (lb[4] + ub[4]) / 2, (lb[5] + ub[5]) / 2]]
        ProfileLikelihood.set_next_initial_estimate!(sub_cache, param_vals, other_mles, _prob, θₙ; next_initial_estimate_method=Val(:interp))
        @test sub_cache ≈ other_mles[1]
        @inferred ProfileLikelihood.set_next_initial_estimate!(sub_cache, param_vals, other_mles, _prob, θₙ; next_initial_estimate_method=Val(:interp))
    end

    @testset "Compare solutions" begin
        prof_serial_interp = profile(prob, sol; next_initial_estimate_method=Val(:interp))
        prof_parallel_interp = profile(prob, sol; parallel=true, next_initial_estimate_method=Val(:interp))
        prof_serial = profile(prob, sol)
        prof_parallel = profile(prob, sol; parallel=true)
        @test_throws "Invalid initial estimate method provided" profile(prob, sol; parallel=true, next_initial_estimate_method=Val(:linear))
        @test prof_serial_interp.other_mles ≠ prof_serial.other_mles
        @test all(i -> abs((prof_parallel.confidence_intervals[i].lower - prof_parallel_interp.confidence_intervals[i].lower) / prof_serial.confidence_intervals[i].lower) < 1e-2, 1:5)
        @test all(i -> abs((prof_parallel.confidence_intervals[i].upper - prof_parallel_interp.confidence_intervals[i].upper) / prof_serial.confidence_intervals[i].upper) < 1e-2, 1:5)
        @test all(i -> abs((prof_serial.confidence_intervals[i].lower - prof_serial_interp.confidence_intervals[i].lower) / prof_serial.confidence_intervals[i].lower) < 1e-2, 1:5)
        @test all(i -> abs((prof_serial.confidence_intervals[i].upper - prof_serial_interp.confidence_intervals[i].upper) / prof_serial.confidence_intervals[i].upper) < 1e-2, 1:5)
        @test all(i -> prof_parallel.parameter_values[i] ≈ prof_parallel_interp.parameter_values[i], 1:5)
        @test all(i -> prof_parallel.profile_values[i] ≈ prof_parallel_interp.profile_values[i], 1:5)
        @test all(i -> prof_serial.parameter_values[i] ≈ prof_serial_interp.parameter_values[i], 1:5)
        @test all(i -> prof_serial.profile_values[i] ≈ prof_serial_interp.profile_values[i], 1:5)
        vcov_mat = sol[:σ]^2 * inv(X' * X)
        for i in 1:4
            @test prof_parallel_interp.confidence_intervals[i+1][1] ≈ sol.mle[i+1] - 1.96sqrt(vcov_mat[i, i]) atol = 1e-3 rtol = 1e-1
            @test prof_parallel_interp.confidence_intervals[i+1][2] ≈ sol.mle[i+1] + 1.96sqrt(vcov_mat[i, i]) atol = 1e-3 rtol = 1e-1
        end
        df = n - (length(β) + 1)
        resids = y .- X * sol[2:5]
        rss = sum(resids .^ 2)
        χ²_up = quantile(Chisq(df), 0.975)
        χ²_lo = quantile(Chisq(df), 0.025)
        σ_CI_exact = sqrt.(rss ./ (χ²_up, χ²_lo))
        @test get_confidence_intervals(prof_parallel_interp, :σ).lower ≈ σ_CI_exact[1] atol = 1e-3 rtol = 1e-2
        @test ProfileLikelihood.get_upper(get_confidence_intervals(prof_parallel_interp, :σ)) ≈ σ_CI_exact[2] atol = 1e-3 rtol = 1e-1
        vcov_mat = sol[:σ]^2 * inv(X' * X)
        for i in 1:4
            @test prof_serial_interp.confidence_intervals[i+1][1] ≈ sol.mle[i+1] - 1.96sqrt(vcov_mat[i, i]) atol = 1e-3
            @test prof_serial_interp.confidence_intervals[i+1][2] ≈ sol.mle[i+1] + 1.96sqrt(vcov_mat[i, i]) atol = 1e-3
        end
        rss = sum(resids .^ 2)
        χ²_up = quantile(Chisq(df), 0.975)
        χ²_lo = quantile(Chisq(df), 0.025)
        σ_CI_exact = sqrt.(rss ./ (χ²_up, χ²_lo))
        @test get_confidence_intervals(prof_serial_interp, :σ).lower ≈ σ_CI_exact[1] atol = 1e-3 rtol = 1e-1
        @test ProfileLikelihood.get_upper(get_confidence_intervals(prof_serial_interp, :σ)) ≈ σ_CI_exact[2] atol = 1e-3 rtol = 1e-1
    end
end

@testset "Replacing a solution" begin
    λ = 0.01
    K = 100.0
    u₀ = 10.0
    t = 0:100:1000
    σ = 10.0
    @inline function ode_fnc(u, p, t)
        λ, K = p
        du = λ * u * (1 - u / K)
        return du
    end
    tspan = extrema(t)
    p = (λ, K)
    prob = ODEProblem(ode_fnc, u₀, tspan, p)
    sol = solve(prob, Rosenbrock23(), saveat=t)
    Random.seed!(2828)
    uᵒ = sol.u + σ * randn(length(t))
    @inline function loglik_fnc2(θ, data, integrator)
        λ, K, u₀ = θ
        uᵒ, σ = data
        integrator.p[1] = λ
        integrator.p[2] = K
        reinit!(integrator, u₀)
        solve!(integrator)
        return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, length(uᵒ))
    end
    lb = [0.0, 50.0, 0.0]
    ub = [0.05, 150.0, 50.0]
    θ₀ = [λ, K, u₀]
    syms = [:λ, :K, :u₀]
    prob = LikelihoodProblem(
        loglik_fnc2, θ₀, ode_fnc, u₀, maximum(t);
        syms=syms,
        data=(uᵒ, σ),
        ode_parameters=[1.0, 1.0],
        ode_kwargs=(verbose=false, saveat=t),
        f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
        prob_kwargs=(lb=lb, ub=ub),
        ode_alg=Rosenbrock23()
    )
    sol = mle(prob, NLopt.LN_NELDERMEAD)
    prof = profile(prob, sol;
        alg=NLopt.LN_NELDERMEAD, parallel=false, min_steps=2, resolution=[5, 500, 10])
    _prof = deepcopy(prof)
    replace_profile!(prof, 1; min_steps=50)
    @test prof.parameter_values[2] == _prof.parameter_values[2]
    @test prof.profile_values[2] == _prof.profile_values[2]
    @test prof.other_mles[2] == _prof.other_mles[2]
    @test prof.splines[2] == _prof.splines[2]
    @test prof.confidence_intervals[2][1] == _prof.confidence_intervals[2][1]
    @test prof.confidence_intervals[2][2] == _prof.confidence_intervals[2][2]
    @test prof.parameter_values[3] == _prof.parameter_values[3]
    @test prof.profile_values[3] == _prof.profile_values[3]
    @test prof.other_mles[3] == _prof.other_mles[3]
    @test prof.splines[3] == _prof.splines[3]
    @test prof.confidence_intervals[3][1] == _prof.confidence_intervals[3][1]
    @test prof.confidence_intervals[3][2] == _prof.confidence_intervals[3][2]
    @test prof.likelihood_problem.log_likelihood_function.loglik(prof.likelihood_solution.mle, prof.likelihood_problem.data, prof.likelihood_problem.log_likelihood_function.integrator) === _prof.likelihood_problem.log_likelihood_function.loglik(_prof.likelihood_solution.mle, _prof.likelihood_problem.data, _prof.likelihood_problem.log_likelihood_function.integrator)
    @test prof.likelihood_problem.θ₀ == _prof.likelihood_problem.θ₀
    @test prof.likelihood_solution.mle == _prof.likelihood_solution.mle
    @test prof.likelihood_solution.maximum == _prof.likelihood_solution.maximum
    @test length(prof.parameter_values[1]) ≥ 49
    @test length(prof.profile_values[1]) ≥ 49
    @test length(prof.other_mles[1]) ≥ 49
    @test length(prof.splines[1]) ≥ 49
    @test prof.confidence_intervals[1][1] ≈ _prof.confidence_intervals[1][1] rtol = 1e-1
    @test prof.confidence_intervals[1][2] ≈ _prof.confidence_intervals[1][2] rtol = 1e-1
    @test prof.confidence_intervals[1][1] ≠ _prof.confidence_intervals[1][1]
    @test prof.confidence_intervals[1][2] ≠ _prof.confidence_intervals[1][2]
end

@testset "Refining a solution" begin
    @testset "Check that we can properly fill up points" begin
        range_1 = collect(LinRange(0, 5, 32))
        ProfileLikelihood.repopulate_points!(range_1, 40)
        @test length(range_1) == 40
        @test range_1[1:32] == collect(LinRange(0, 5, 32))
        range_2 = collect(LinRange(0, 5, 10))
        @test range_1[33:40] ≈ range_2[2:end-1]

        range_1 = collect(LinRange(-4, 4, 1382))
        ProfileLikelihood.repopulate_points!(range_1, 1666)
        @test length(range_1) == 1666
        @test range_1[1:1382] == collect(LinRange(-4, 4, 1382))
        range_2 = collect(LinRange(-4, 4, 1666 - 1382 + 2))
        @test range_1[1383:1666] ≈ range_2[2:end-1]

        range_1 = collect(LinRange(5, 0, 32))
        ProfileLikelihood.repopulate_points!(range_1, 40)
        @test length(range_1) == 40
        @test range_1[1:32] == collect(LinRange(5, 0, 32))
        range_2 = collect(LinRange(5, 0, 10))
        @test range_1[33:40] ≈ range_2[2:end-1]

        range_1 = collect(LinRange(4, -4, 1382))
        ProfileLikelihood.repopulate_points!(range_1, 1666)
        @test length(range_1) == 1666
        @test range_1[1:1382] == collect(LinRange(4, -4, 1382))
        range_2 = collect(LinRange(4, -4, 1666 - 1382 + 2))
        @test range_1[1383:1666] ≈ range_2[2:end-1]
    end

    @testset "Check that we can properly resize all the data" begin
        noise_dat = rand(32)
        param_vals = LinRange(0, 2, 32)
        other_mle_noise = [[param_vals[i]^2, param_vals[i]^(5 / 2) + 4] for i in 1:32]
        profile_vals = collect(noise_dat)
        other_mles = collect(other_mle_noise)
        reduced_mles = reduce(hcat, other_mles)
        itp1 = interpolate(reduced_mles[1, :], BSpline(Cubic(Line(OnGrid()))))
        itp2 = interpolate(reduced_mles[2, :], BSpline(Cubic(Line(OnGrid()))))
        spl1 = Interpolations.scale(itp1, param_vals)
        spl2 = Interpolations.scale(itp2, param_vals)
        param_vals = collect(param_vals)
        ProfileLikelihood.resize_profile_data!(param_vals, profile_vals, other_mles, 40)
        @test param_vals ≈ [LinRange(0, 2, 32)..., LinRange(0, 2, 10)[2:end-1]...]
        @test profile_vals[1:32] ≈ noise_dat
        @test length(profile_vals) == 40
        @test length(other_mles) == 40
        @test other_mles[1:32] == other_mle_noise
        @test other_mles[33:40] ≈ [[spl1(x), spl2(x)] for x in param_vals[33:40]]
        @test getindex.(other_mles[33:40], 1) ≈ param_vals[33:40] .^ 2 rtol = 0.1
        @test getindex.(other_mles[33:40], 2) ≈ param_vals[33:40] .^ (5 / 2) .+ 4.0 rtol = 0.1

        noise_dat = rand(32)
        param_vals = LinRange(2, 0, 32)
        other_mle_noise = [[param_vals[i]^2, param_vals[i]^(5 / 2) + 4] for i in 1:32]
        profile_vals = collect(noise_dat)
        other_mles = collect(other_mle_noise)
        reduced_mles = reduce(hcat, other_mles)
        itp1 = interpolate(reverse(reduced_mles[1, :]), BSpline(Cubic(Line(OnGrid()))))
        itp2 = interpolate(reverse(reduced_mles[2, :]), BSpline(Cubic(Line(OnGrid()))))
        spl1 = Interpolations.scale(itp1, reverse(param_vals))
        spl2 = Interpolations.scale(itp2, reverse(param_vals))
        param_vals = collect(param_vals)
        ProfileLikelihood.resize_profile_data!(param_vals, profile_vals, other_mles, 40)
        @test param_vals ≈ [LinRange(2, 0, 32)..., LinRange(2, 0, 10)[2:end-1]...]
        @test profile_vals[1:32] ≈ noise_dat
        @test length(profile_vals) == 40
        @test length(other_mles) == 40
        @test other_mles[1:32] == other_mle_noise
        @test other_mles[33:40] ≈ [[spl1(x), spl2(x)] for x in param_vals[33:40]]
        @test getindex.(other_mles[33:40], 1) ≈ param_vals[33:40] .^ 2 rtol = 0.2
        @test getindex.(other_mles[33:40], 2) ≈ param_vals[33:40] .^ (5 / 2) .+ 4.0 rtol = 0.1

        param_vals = [0.010438031266786045, 0.013263886176301327, 0.016089741085816612, 0.018915595995331894]
        other_mles = [[99.5761184310497, 8.094637304374515], [97.77129117887698, 4.610877949211225], [96.53432680720022, 2.6217967940631275], [95.53802793620645, 1.5284755336047502]]
        profile_vals = rand(4)
        min_steps = 10
        ProfileLikelihood.resize_profile_data!(param_vals, profile_vals, other_mles, min_steps) # this used to segfault
    end

    @testset "Profile refinement during algorithm" begin
        λ = 0.01
        K = 100.0
        u₀ = 10.0
        t = 0:100:1000
        σ = 10.0
        @inline function ode_fnc(u, p, t)
            λ, K = p
            du = λ * u * (1 - u / K)
            return du
        end
        tspan = extrema(t)
        p = (λ, K)
        prob = ODEProblem(ode_fnc, u₀, tspan, p)
        sol = solve(prob, Rosenbrock23(), saveat=t)
        Random.seed!(2828)
        uᵒ = sol.u + σ * randn(length(t))
        @inline function loglik_fnc2(θ, data, integrator)
            λ, K, u₀ = θ
            uᵒ, σ = data
            integrator.p[1] = λ
            integrator.p[2] = K
            reinit!(integrator, u₀)
            solve!(integrator)
            return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, length(uᵒ))
        end
        lb = [0.0, 10.0, 0.0]
        ub = [0.25, 550.0, 550.0]
        θ₀ = [λ, K, u₀]
        syms = [:λ, :K, :u₀]
        prob = LikelihoodProblem(
            loglik_fnc2, θ₀, ode_fnc, u₀, maximum(t);
            syms=syms,
            data=(uᵒ, σ),
            ode_parameters=[1.0, 1.0],
            ode_kwargs=(verbose=false, saveat=t),
            f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
            prob_kwargs=(lb=lb, ub=ub),
            ode_alg=Rosenbrock23()
        )
        sol = mle(prob, NLopt.LN_NELDERMEAD)
        @time prof = profile(prob, sol;
            alg=NLopt.LN_NELDERMEAD, parallel=false, min_steps=40, resolution=30,
            min_steps_fallback=:replace)
        @time _prof = profile(prob, sol;
            alg=NLopt.LN_NELDERMEAD, parallel=false, min_steps=40, resolution=30,
            min_steps_fallback=:refine)
        @time __prof = profile(prob, sol;
            alg=NLopt.LN_NELDERMEAD, parallel=false, min_steps=0, resolution=30,
            min_steps_fallback=:refine)
       # F1 = plot_profiles(prof; show_points=true, spline=false)
       # F2 = plot_profiles(_prof; show_points=true, spline=false)
       # F3 = plot_profiles(__prof; show_points=true, spline=false)

        prof1 = prof
        prof2 = _prof

        @test prof1.confidence_intervals[1].lower ≈ prof2.confidence_intervals[1].lower rtol = 1e-1
        @test prof1.confidence_intervals[2].lower ≈ prof2.confidence_intervals[2].lower rtol = 1e-1
        @test prof1.confidence_intervals[3].lower ≈ prof2.confidence_intervals[3].lower rtol = 1e-1
        @test prof1.confidence_intervals[1].upper ≈ prof2.confidence_intervals[1].upper rtol = 1e-1
        @test prof1.confidence_intervals[2].upper ≈ prof2.confidence_intervals[2].upper rtol = 1e-1
        @test prof1.confidence_intervals[3].upper ≈ prof2.confidence_intervals[3].upper rtol = 1e-1
        @test issorted(prof1.parameter_values[1])
        @test issorted(prof1.parameter_values[2])
        @test issorted(prof1.parameter_values[3])
        @test issorted(prof2.parameter_values[1])
        @test issorted(prof2.parameter_values[2])
        @test issorted(prof2.parameter_values[3])
        @test length(get_parameter_values(prof1[1])) == 79
        @test length(get_parameter_values(prof1[2])) == 79
        @test length(get_parameter_values(prof1[3])) == 79
        @test length(get_parameter_values(prof2[2])) == 79
        @test length(get_parameter_values(prof2[3])) == 73
    end

    @testset "refine_profile" begin
        λ = 0.01
        K = 100.0
        u₀ = 10.0
        t = 0:100:1000
        σ = 10.0
        @inline function ode_fnc(u, p, t)
            λ, K = p
            du = λ * u * (1 - u / K)
            return du
        end
        tspan = extrema(t)
        p = (λ, K)
        prob = ODEProblem(ode_fnc, u₀, tspan, p)
        sol = solve(prob, Rosenbrock23(), saveat=t)
        Random.seed!(2828)
        uᵒ = sol.u + σ * randn(length(t))
        @inline function loglik_fnc2(θ, data, integrator)
            λ, K, u₀ = θ
            uᵒ, σ = data
            integrator.p[1] = λ
            integrator.p[2] = K
            reinit!(integrator, u₀)
            solve!(integrator)
            return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, length(uᵒ))
        end
        lb = [0.0, 10.0, 0.0]
        ub = [0.25, 550.0, 550.0]
        θ₀ = [λ, K, u₀]
        syms = [:λ, :K, :u₀]
        prob = LikelihoodProblem(
            loglik_fnc2, θ₀, ode_fnc, u₀, maximum(t);
            syms=syms,
            data=(uᵒ, σ),
            ode_parameters=[1.0, 1.0],
            ode_kwargs=(verbose=false, saveat=t),
            f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
            prob_kwargs=(lb=lb, ub=ub),
            ode_alg=Rosenbrock23()
        )
        sol = mle(prob, NLopt.LN_NELDERMEAD)
        prof = profile(prob, sol;
            alg=NLopt.LN_NELDERMEAD, min_steps=25, resolution=15,
            min_steps_fallback=:replace)
        _prof = deepcopy(prof)
        @time refine_profile!(prof, 1; target_number=250)

        # F1 = plot_profiles(prof; show_points=true, spline=true)

        prof1 = prof
        prof2 = _prof
        @test prof1.confidence_intervals[1].lower ≈ prof2.confidence_intervals[1].lower rtol = 1e-1
        @test prof1.confidence_intervals[2].lower ≈ prof2.confidence_intervals[2].lower rtol = 1e-1
        @test prof1.confidence_intervals[3].lower ≈ prof2.confidence_intervals[3].lower rtol = 1e-1
        @test prof1.confidence_intervals[1].upper ≈ prof2.confidence_intervals[1].upper rtol = 1e-1
        @test prof1.confidence_intervals[2].upper ≈ prof2.confidence_intervals[2].upper rtol = 1e-1
        @test prof1.confidence_intervals[3].upper ≈ prof2.confidence_intervals[3].upper rtol = 1e-1
        @test issorted(prof1.parameter_values[1])
        @test issorted(prof1.parameter_values[2])
        @test issorted(prof1.parameter_values[3])
        @test issorted(prof2.parameter_values[1])
        @test issorted(prof2.parameter_values[2])
        @test issorted(prof2.parameter_values[3])
        @test length(get_parameter_values(prof1[1])) == 250
        @test length(get_parameter_values(prof1[2])) < 250
        @test length(get_parameter_values(prof1[3])) < 250
        @test prof1.parameter_values[2] == prof2.parameter_values[2]
        @test prof1.parameter_values[3] == prof2.parameter_values[3]
        @test prof1.profile_values[2] == prof2.profile_values[2]
        @test prof1.profile_values[3] == prof2.profile_values[3]
        @test prof1.other_mles[2] == prof2.other_mles[2]
        @test prof1.other_mles[3] == prof2.other_mles[3]

        refine_profile!(prof, [1, 2, 3]; target_number=250)

        # F1 = plot_profiles(prof; show_points=true, spline=true)

        prof1 = prof
        prof2 = _prof
        @test prof1.confidence_intervals[1].lower ≈ prof2.confidence_intervals[1].lower rtol = 1e-1
        @test prof1.confidence_intervals[2].lower ≈ prof2.confidence_intervals[2].lower rtol = 1e-1
        @test prof1.confidence_intervals[3].lower ≈ prof2.confidence_intervals[3].lower rtol = 1e-1
        @test prof1.confidence_intervals[1].upper ≈ prof2.confidence_intervals[1].upper rtol = 1e-1
        @test prof1.confidence_intervals[2].upper ≈ prof2.confidence_intervals[2].upper rtol = 1e-1
        @test prof1.confidence_intervals[3].upper ≈ prof2.confidence_intervals[3].upper rtol = 1e-1
        @test issorted(prof1.parameter_values[1])
        @test issorted(prof1.parameter_values[2])
        @test issorted(prof1.parameter_values[3])
        @test issorted(prof2.parameter_values[1])
        @test issorted(prof2.parameter_values[2])
        @test issorted(prof2.parameter_values[3])
        @test length(get_parameter_values(prof1[1])) == 250
        @test length(get_parameter_values(prof1[2])) == 250
        @test length(get_parameter_values(prof1[3])) == 250
    end

    @testset "parallel refinement" begin
        λ = 0.01
        K = 100.0
        u₀ = 10.0
        t = 0:100:1000
        σ = 10.0
        @inline function ode_fnc(u, p, t)
            λ, K = p
            du = λ * u * (1 - u / K)
            return du
        end
        tspan = extrema(t)
        p = (λ, K)
        prob = ODEProblem(ode_fnc, u₀, tspan, p)
        sol = solve(prob, Rosenbrock23(), saveat=t)
        Random.seed!(2828)
        uᵒ = sol.u + σ * randn(length(t))
        @inline function loglik_fnc2(θ, data, integrator)
            λ, K, u₀ = θ
            uᵒ, σ = data
            integrator.p[1] = λ
            integrator.p[2] = K
            reinit!(integrator, u₀)
            solve!(integrator)
            return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, length(uᵒ))
        end
        lb = [0.0, 10.0, 0.0]
        ub = [0.25, 550.0, 550.0]
        θ₀ = [λ, K, u₀]
        syms = [:λ, :K, :u₀]
        prob = LikelihoodProblem(
            loglik_fnc2, θ₀, ode_fnc, u₀, maximum(t);
            syms=syms,
            data=(uᵒ, σ),
            ode_parameters=[1.0, 1.0],
            ode_kwargs=(verbose=false, saveat=t),
            f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
            prob_kwargs=(lb=lb, ub=ub),
            ode_alg=Rosenbrock23()
        )
        sol = mle(prob, NLopt.LN_NELDERMEAD)
        prof1 = profile(deepcopy(prob), deepcopy(sol);
            alg=NLopt.LN_NELDERMEAD, min_steps=25, resolution=15,
            min_steps_fallback=:replace)
        prof2 = profile(deepcopy(prob), deepcopy(sol);
            alg=NLopt.LN_NELDERMEAD, min_steps=25, resolution=15,
            min_steps_fallback=:replace)
        @time refine_profile!(prof1, 1; target_number=250, parallel=false)
        @time refine_profile!(prof2, 1; target_number=250, parallel=true)

        # F1 = plot_profiles(prof2; show_points=true, spline=true)

        @test prof1.confidence_intervals[1].lower ≈ prof2.confidence_intervals[1].lower rtol = 1e-1
        @test prof1.confidence_intervals[2].lower ≈ prof2.confidence_intervals[2].lower rtol = 1e-1
        @test prof1.confidence_intervals[3].lower ≈ prof2.confidence_intervals[3].lower rtol = 1e-1
        @test prof1.confidence_intervals[1].upper ≈ prof2.confidence_intervals[1].upper rtol = 1e-1
        @test prof1.confidence_intervals[2].upper ≈ prof2.confidence_intervals[2].upper rtol = 1e-1
        @test prof1.confidence_intervals[3].upper ≈ prof2.confidence_intervals[3].upper rtol = 1e-1
        @test issorted(prof1.parameter_values[1])
        @test issorted(prof1.parameter_values[2])
        @test issorted(prof1.parameter_values[3])
        @test issorted(prof2.parameter_values[1])
        @test issorted(prof2.parameter_values[2])
        @test issorted(prof2.parameter_values[3])
        @test length(get_parameter_values(prof1[1])) == 250
        @test length(get_parameter_values(prof1[2])) < 250
        @test length(get_parameter_values(prof1[3])) < 250
        @test prof1.parameter_values[2] == prof2.parameter_values[2]
        @test prof1.parameter_values[3] == prof2.parameter_values[3]
        @test prof1.profile_values[2] == prof2.profile_values[2]
        @test prof1.profile_values[3] == prof2.profile_values[3]
        @test prof1.other_mles[2] == prof2.other_mles[2]
        @test prof1.other_mles[3] == prof2.other_mles[3]
        @test prof1.other_mles[1] == prof2.other_mles[1]
        @test prof1.profile_values[1] == prof2.profile_values[1]
        @test prof1.parameter_values[1] == prof2.parameter_values[1]

        @time refine_profile!(prof1, [1, 2, 3]; target_number=250)
        @time refine_profile!(prof2, [1, 2, 3]; target_number=250, parallel=true)

        # F1 = plot_profiles(prof1; show_points=true, spline=true)

        @test prof1.confidence_intervals[1].lower ≈ prof2.confidence_intervals[1].lower rtol = 1e-1
        @test prof1.confidence_intervals[2].lower ≈ prof2.confidence_intervals[2].lower rtol = 1e-1
        @test prof1.confidence_intervals[3].lower ≈ prof2.confidence_intervals[3].lower rtol = 1e-1
        @test prof1.confidence_intervals[1].upper ≈ prof2.confidence_intervals[1].upper rtol = 1e-1
        @test prof1.confidence_intervals[2].upper ≈ prof2.confidence_intervals[2].upper rtol = 1e-1
        @test prof1.confidence_intervals[3].upper ≈ prof2.confidence_intervals[3].upper rtol = 1e-1
        @test issorted(prof1.parameter_values[1])
        @test issorted(prof1.parameter_values[2])
        @test issorted(prof1.parameter_values[3])
        @test issorted(prof2.parameter_values[1])
        @test issorted(prof2.parameter_values[2])
        @test issorted(prof2.parameter_values[3])
        @test length(get_parameter_values(prof1[1])) == 250
        @test length(get_parameter_values(prof1[2])) == 250
        @test length(get_parameter_values(prof1[3])) == 250
        @test prof1.parameter_values[2] == prof2.parameter_values[2]
        @test prof1.parameter_values[3] == prof2.parameter_values[3]
        @test prof1.profile_values[2] == prof2.profile_values[2]
        @test prof1.profile_values[3] == prof2.profile_values[3]
        @test prof1.other_mles[2] == prof2.other_mles[2]
        @test prof1.other_mles[3] == prof2.other_mles[3]
        @test prof1.other_mles[1] == prof2.other_mles[1]
        @test prof1.profile_values[1] == prof2.profile_values[1]
        @test prof1.parameter_values[1] == prof2.parameter_values[1]
    end
end