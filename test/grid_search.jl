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
using CairoMakie

@testset "GridSearch construction" begin
    rb_f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    rb_bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    rb_resolution = 100
    gs = GridSearch(rb_f, rb_bounds, rb_resolution)
    @test gs.f == rb_f
    @test gs.bounds == rb_bounds
    @test gs.resolution == [rb_resolution, rb_resolution]
    _gs = GridSearch{2,typeof(rb_f),Vector{Tuple{Float64,Float64}},Vector{Int64},Float64,Float64}(rb_f, rb_bounds, [rb_resolution, rb_resolution])
    @test gs.f == _gs.f && gs.bounds == _gs.bounds && gs.resolution == _gs.resolution
    @test typeof(gs) == typeof(_gs)
    rb_resolution = [100, 100]
    gs = GridSearch(rb_f, rb_bounds, rb_resolution)
    @test gs.f == rb_f
    @test gs.bounds == rb_bounds
    @test gs.resolution == rb_resolution
    _gs = GridSearch{2,typeof(rb_f),Vector{Tuple{Float64,Float64}},Vector{Int64},Float64,Float64}(rb_f, rb_bounds, rb_resolution)
    @test gs.f == _gs.f && gs.bounds == _gs.bounds && gs.resolution == _gs.resolution
    @test typeof(gs) == typeof(_gs)

    prob, loglikk, θ, dat = MultipleLinearRegression()
    mlr_f(θ) = loglikk(θ, dat)
    mlr_bounds = [(1e-12, 0.2), (-3.0, 0.0), (0.0, 3.0), (0.0, 3.0), (-6.0, 6.0)]
    mlr_resolution = 20
    @test_throws ArgumentError GridSearch(prob)
    gs = GridSearch(prob, mlr_bounds)
    _gs = GridSearch(mlr_f, mlr_bounds, 20)
    @test gs.f([θ[1], θ[2]...]) == _gs.f([θ[1], θ[2]...]) && gs.bounds == _gs.bounds && gs.resolution == _gs.resolution
    gs = GridSearch(prob, mlr_bounds, [27, 50])
    _gs = GridSearch(mlr_f, mlr_bounds, [27, 50])
    @test gs.f([θ[1], θ[2]...]) == _gs.f([θ[1], θ[2]...]) && gs.bounds == _gs.bounds && gs.resolution == _gs.resolution == [27, 50]
end

@testset "Rastrigin function" begin
    n = 4
    A = 10
    rastrigin_f(x) = A * n + (x[1]^2 - A * cos(2π * x[1])) + (x[2]^2 - A * cos(2π * x[2])) + (x[3]^2 - A * cos(2π * x[3])) + (x[4]^2 - A * cos(2π * x[4]))
    @test rastrigin_f(zeros(n)) == 0.0
    gs = GridSearch(x -> -rastrigin_f(x), [(-5.12, 5.12) for i in 1:n], 25)
    f_min, x_min = grid_search(gs)
    @test f_min ≈ 0.0
    @test x_min ≈ zeros(n)
    f_min, x_min = grid_search(rastrigin_f, [(-5.12, 5.12) for i in 1:n], 25; find_max=false)
    @test f_min ≈ 0.0
    @test x_min ≈ zeros(n)
    f_min, x_min = grid_search(x -> -rastrigin_f(x), [(-5.12, 5.12) for i in 1:n], 25)
    @test f_min ≈ 0.0
    @test x_min ≈ zeros(n)
    f_min, x_min, f_res = grid_search(gs; save_res=true)
    @test f_min ≈ 0.0
    @test x_min ≈ zeros(n)
    param_ranges = [LinRange(-5.12, 5.12, 25) for i in 1:n]
    @test f_res ≈ -[rastrigin_f(x) for x in Iterators.product(param_ranges...)]
end

@testset "Ackley function" begin
    n = 2
    ackley_f(x) = -20exp(-0.2sqrt(x[1]^2 + x[2]^2)) - exp(0.5(cos(2π * x[1]) + cos(2π * x[2]))) + exp(1) + 20
    @test ackley_f([0, 0]) == 0
    gs = GridSearch(x -> -ackley_f(x), [(-15.12, 15.12) for i in 1:n], [73, 121])
    f_min, x_min = grid_search(gs)
    @test f_min ≈ 0.0
    @test x_min ≈ zeros(n)
    f_min, x_min = grid_search(ackley_f, [(-15.12, 15.12) for i in 1:n], [73, 121]; find_max=false)
    @test f_min ≈ 0.0
    @test x_min ≈ zeros(n)
    f_min, x_min = grid_search(x -> -ackley_f(x), [(-15.12, 15.12) for i in 1:n], [73, 121])
    @test f_min ≈ 0.0
    @test x_min ≈ zeros(n)
    f_min, x_min, f_res = grid_search(x -> -ackley_f(x), [(-15.12, 15.12) for i in 1:n], [731, 1217]; save_res=true)
    @test f_min ≈ 0.0
    @test x_min ≈ zeros(n)
    param_ranges = [LinRange(-15.12, 15.12, i == 1 ? 731 : 1217) for i in 1:n]
    @test f_res ≈ -[ackley_f(x) for x in Iterators.product(param_ranges...)]
    fig = Figure()
    ax = Axis(fig[1, 1])
    xs = param_ranges[1]
    ys = param_ranges[2]
    zs = f_res
    heatmap!(ax, xs, ys, zs)
    xlims!(ax, -5.12, 5.12)
    ylims!(ax, -5.12, 5.12)
    fig
end

@testset "Sphere function" begin
    n = 5
    sphere_f(x) = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 + x[5]^2
    @test sphere_f(zeros(n)) == 0
    resolutions = [15, 17, 15, 19, 23]
    gs = GridSearch(x -> -sphere_f(x), [(-15.12, 15.12) for i in 1:n], resolutions)
    f_min, x_min = grid_search(gs)
    @test f_min ≈ 0.0
    @test x_min ≈ zeros(n)
    f_min, x_min = grid_search(sphere_f, [(-15.12, 15.12) for i in 1:n], resolutions; find_max=false)
    @test f_min ≈ 0.0
    @test x_min ≈ zeros(n)
    f_min, x_min = grid_search(x -> -sphere_f(x), [(-15.12, 15.12) for i in 1:n], resolutions)
    @test f_min ≈ 0.0
    @test x_min ≈ zeros(n)
    f_min, x_min, f_res = grid_search(x -> -sphere_f(x), [(-15.12, 15.12) for i in 1:n], resolutions; save_res=true)
    @test f_min ≈ 0.0
    @test x_min ≈ zeros(n)
    param_ranges = [LinRange(-15.12, 15.12, resolutions[i]) for i in 1:n]
    @test f_res ≈ -[sphere_f(x) for x in Iterators.product(param_ranges...)]
end

@testset "Regression" begin
    prob, loglikk, θ, dat = MultipleLinearRegression()
    true_ℓ = loglikk(reduce(vcat, θ), dat)
    @test_throws ArgumentError grid_search(prob)
    mlr_bounds = [(1e-12, 0.2), (-3.0, 0.0), (0.0, 3.0), (0.0, 3.0), (-6.0, 6.0)]
    mlr_resolution = 27
    sol = grid_search(prob, mlr_bounds, mlr_resolution)
    @test maximum(sol) ≈ 281.7360323629172
    @test mle(sol) ≈ [0.09230769230823077
        -0.9230769230769231
        0.8076923076923077
        0.46153846153846156
        3.2307692307692317]
end

@testset "Linear exponential ODE" begin
    prob, loglikk, θ, yᵒ, n = LinearExponentialODE()
    λ, σ, y₀ = θ
    true_ℓ = prob.loglik(θ, data(prob))
    sol = grid_search(prob)
    @test sol.maximum ≈ -90.1022780056474
    @test sol.θ ≈ [-0.526315789473685, 0.5263167368421052, 15.973684210526315]
    sol = grid_search(prob, [(-0.6, -0.4), (0.05, 0.15), (14.0, 16.0)], 70)
    @test sol.maximum ≈ 182.1232099637145
    @test sol.θ ≈ [-0.5014492753623189, 0.09782608695652174, 15.043478260869566]
end

@testset "Logistic ODE" begin
    prob, loglikk, θ, uᵒ, n = LogisticODE()
    λ, K, σ, u₀ = θ
    true_ℓ = prob.loglik(θ, data(prob))
    sol = grid_search(prob, [(0.5, 1.2), (0.7, 1.3), (0.05, 0.15), (0.4, 0.6)], 25)
    @test maximum(sol) ≈ 86.4805427143682
    @test mle(sol) ≈ [0.7625, 1.025, 0.1, 0.53333333333]
    @test sol isa LikelihoodSolution
    @test algorithm_name(sol) == "GridSearch"
    @test sol.alg == :GridSearch
    @test sol.original === nothing
    @test sol.prob == prob
end