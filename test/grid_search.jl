using Random
using CairoMakie

@testset "UniformGrid construction and usage" begin
    ## Construction
    xbounds = [(1.0, 3.0), (5.0, 2.0)]
    resolutions = [4, 3]
    steps = [(3.0 - 1.0) / 3, (2.0 - 5.0) / 2]
    ug1 = UniformGrid(xbounds, resolutions)
    @test ug1.bounds == xbounds
    @test ug1.resolution == resolutions
    @test ug1.steps ≈ steps
    @test typeof(ug1) == UniformGrid{2,Vector{Tuple{Float64,Float64}},Vector{Int64},Vector{Float64},Float64}
    @test ug1 isa UniformGrid
    @test typeof(ug1) <: AbstractGrid

    resolutions = 4
    steps = [(3.0 - 1.0) / 3, (2.0 - 5.0) / 3]
    ug2 = UniformGrid(xbounds, resolutions)
    @test ug2.bounds == xbounds
    @test ug2.resolution == resolutions
    @test ug2.steps ≈ steps
    @test typeof(ug2) == UniformGrid{2,Vector{Tuple{Float64,Float64}},Int64,Vector{Float64},Float64}
    @test ug2 isa UniformGrid
    @test typeof(ug2) <: AbstractGrid

    @test_throws ArgumentError UniformGrid([(Inf, 3.0), (5.0, 2.0)], resolutions)
    @test_throws ArgumentError UniformGrid([(1.0, Inf), (5.0, 2.0)], resolutions)
    @test_throws ArgumentError UniformGrid([(1.0, 3.0), (Inf, 2.0)], resolutions)
    @test_throws ArgumentError UniformGrid([(1.0, nothing), (5.0, Inf)], resolutions)
    @test_throws ArgumentError UniformGrid([(NaN, Inf), (5.0, nothing)], resolutions)
    @test_throws ArgumentError UniformGrid([(1.0, Inf), (5.0, NaN)], resolutions)
    @test_throws ArgumentError UniformGrid([(nothing, 3.0), (5.0, NaN)], resolutions)
    @test_throws ArgumentError UniformGrid([(1.0, Inf), (NaN, 2.0)], resolutions)
    @test_throws ArgumentError UniformGrid([(Inf, 3.0), (-Inf, 2.0)], resolutions)
    @test_throws ArgumentError UniformGrid([(NaN, 3.0), (nothing, NaN)], resolutions)
    @test_throws ArgumentError UniformGrid([(Inf, -Inf), (NaN, NaN)], resolutions)

    ## Getters
    @test ProfileLikelihood.resolution(ug1, 1) == 4
    @test ProfileLikelihood.resolution(ug1, 2) == 3
    @test ProfileLikelihood.resolution(ug2, 1) == ProfileLikelihood.resolution(ug2, 2) == ProfileLikelihood.resolution(ug2, 2002099) == 4
    @test bounds(ug1, 1) == (1.0, 3.0)
    @test bounds(ug1, 2) == (5.0, 2.0)
    @test bounds(ug2, 1) == (1.0, 3.0)
    @test bounds(ug2, 2) == (5.0, 2.0)
    @test bounds(ug1, 1, 1) == 1.0
    @test bounds(ug2, 2, 2) == 2.0
    @test bounds(ug1, 1, 2) == 3.0
    @test bounds(ug2, 2, 1) == 5.0
    @test ProfileLikelihood.steps(ug1, 1) ≈ (3.0 - 1.0) / 3
    @test ProfileLikelihood.steps(ug1, 2) ≈ (2.0 - 5.0) / 2
    @test ProfileLikelihood.steps(ug2, 1) ≈ (3.0 - 1.0) / 3
    @test ProfileLikelihood.steps(ug2, 2) ≈ (2.0 - 5.0) / 3

    ## Bounds check 
    @test Base.checkbounds(ug1, 1, 1) === nothing
    @test Base.checkbounds(ug1, 1, 2) === nothing
    @test Base.checkbounds(ug1, 2, 1) === nothing
    @test Base.checkbounds(ug1, 2, 2) === nothing
    @test Base.checkbounds(ug2, 1, 1) === nothing
    @test Base.checkbounds(ug2, 1, 2) === nothing
    @test Base.checkbounds(ug2, 2, 1) === nothing
    @test Base.checkbounds(ug2, 2, 2) === nothing
    @test_throws BoundsError Base.checkbounds(ug1, 0, 1)
    @test_throws BoundsError Base.checkbounds(ug1, 1, 5)
    @test_throws BoundsError Base.checkbounds(ug1, 0, 0)
    @test_throws BoundsError Base.checkbounds(ug1, 2, 4)
    @test_throws BoundsError Base.checkbounds(ug1, -5, 3)
    @test_throws BoundsError Base.checkbounds(ug2, 0, 1)
    @test_throws BoundsError Base.checkbounds(ug2, 1, 5)
    @test_throws BoundsError Base.checkbounds(ug2, 0, 0)
    @test_throws BoundsError Base.checkbounds(ug2, 2, 5)
    @test_throws BoundsError Base.checkbounds(ug2, -5, 3)

    ## Indexing 
    r1_1 = LinRange(1.0, 3.0, 4)
    r2_1 = LinRange(5.0, 2.0, 3)
    r1_2 = LinRange(1.0, 3.0, 4)
    r2_2 = LinRange(5.0, 2.0, 4)
    for i in 1:4
        if i ≤ 3
            @inferred ug1[2, i]
            @test ug1[2, i] ≈ r2_1[i]
        end
        @inferred ug1[1, i]
        @inferred ug2[1, i]
        @inferred ug2[2, i]
        @test ug1[1, i] ≈ r1_1[i]
        @test ug2[1, i] ≈ r1_2[i]
        @test ug2[2, i] ≈ r2_2[i]
    end
end

@testset "LatinGrid construction and usage" begin
    ## Construction 
    xbounds = [(1.0, 3.0), (5.0, 2.0)]
    m = 25
    gens = 500
    Random.seed!(138)
    lg = LatinGrid(xbounds, m, gens)
    @test lg.bounds == xbounds
    Random.seed!(138)
    @test lg.grid == collect(ProfileLikelihood.get_lhc_params(m, 2, gens, xbounds)')
    @test typeof(lg) == LatinGrid{2,25,Vector{Tuple{Float64,Float64}},Matrix{Float64},Float64}
    @test lg isa LatinGrid
    @test typeof(lg) <: AbstractGrid{2,Vector{Tuple{Float64,Float64}},Float64}

    @test_throws ArgumentError LatinGrid([(Inf, 3.0), (5.0, 2.0)], m, gens)
    @test_throws ArgumentError LatinGrid([(1.0, Inf), (5.0, 2.0)], m, gens)
    @test_throws ArgumentError LatinGrid([(1.0, 3.0), (Inf, 2.0)], m, gens)
    @test_throws ArgumentError LatinGrid([(1.0, nothing), (5.0, Inf)], m, gens)
    @test_throws ArgumentError LatinGrid([(NaN, Inf), (5.0, nothing)], m, gens)
    @test_throws ArgumentError LatinGrid([(1.0, Inf), (5.0, NaN)], m, gens)
    @test_throws ArgumentError LatinGrid([(nothing, 3.0), (5.0, NaN)], m, gens)
    @test_throws ArgumentError LatinGrid([(1.0, Inf), (NaN, 2.0)], m, gens)
    @test_throws ArgumentError LatinGrid([(Inf, 3.0), (-Inf, 2.0)], m, gens)
    @test_throws ArgumentError LatinGrid([(NaN, 3.0), (nothing, NaN)], m, gens)
    @test_throws ArgumentError LatinGrid([(Inf, -Inf), (NaN, NaN)], m, gens)

    ## Getters
    @test bounds(lg, 1) == (1.0, 3.0)
    @test bounds(lg, 2) == (5.0, 2.0)
    @test bounds(lg, 1, 1) == 1.0
    @test bounds(lg, 1, 2) == 3.0

    ## Indexing 
    @test lg[1, 1] == lg.grid[1, 1]
    @test lg[:, 1] == lg.grid[:, 1]
    @test lg[:, :] == lg.grid[:, :]
    for i in 1:2, j in 1:m
        @test lg[i, j] == lg.grid[i, j]
    end
end

@testset "GridSearch construction" begin
    @testset "UniformGrid" begin
        xbounds = [(1.0, 3.0), (5.0, 2.0)]
        resolutions = [4, 3]
        steps = [(3.0 - 1.0) / 3, (2.0 - 5.0) / 2]
        ug1 = UniformGrid(xbounds, resolutions)
        f = x -> x[1] + x[2]
        @test GridSearch(f, ug1) == GridSearch{typeof(f),2,typeof(xbounds),Float64,typeof(ug1),Float64}(f, ug1)
        ug2 = GridSearch(f, xbounds, resolutions)
        @test ug2.f == f && ug2.grid.bounds == xbounds && ug2.grid.resolution == resolutions && ug2.grid.steps == ug1.steps

        rb_f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        rb_bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        rb_resolution = 100
        gs = GridSearch(rb_f, rb_bounds, rb_resolution)
        @test gs.f == rb_f
        @test gs.grid.bounds == rb_bounds
        @test gs.grid.resolution == rb_resolution
        @test gs.grid isa UniformGrid
    end

    @testset "LatinGrid" begin
        xbounds = [(1.0, 3.0), (5.0, 2.0)]
        m = 25
        f = x -> x[1] + x[2]
        gens = 500
        Random.seed!(138)
        lg = LatinGrid(xbounds, m, gens)
        @test GridSearch(f, lg) == GridSearch{typeof(f),2,typeof(xbounds),Float64,typeof(lg),Float64}(f, lg)
        Random.seed!(138)
        lg2 = GridSearch(f, xbounds, m, gens)
        @test lg2.grid isa LatinGrid
        @test lg2.f == f && lg2.grid.bounds == xbounds && lg2.grid.grid == lg.grid
    end

    @testset "LikelihoodProblem" begin
        prob, loglikk, θ, dat = MultipleLinearRegression()
        mlr_f(θ) = loglikk(θ, dat)
        mlr_bounds = [(1e-12, 0.2), (-3.0, 0.0), (0.0, 3.0), (0.0, 3.0), (-6.0, 6.0)]
        mlr_resolution = 20
        @test_throws MethodError GridSearch(prob)
        gs = GridSearch(prob, mlr_bounds, [27, 50, 31, 32, 33])
        @test gs.grid isa UniformGrid
        _gs = GridSearch(mlr_f, mlr_bounds, [27, 50, 31, 32, 33])
        @test gs.f([θ[1], θ[2]...]) == _gs.f([θ[1], θ[2]...]) && gs.grid.bounds == _gs.grid.bounds && gs.grid.resolution == _gs.grid.resolution == [27, 50, 31, 32, 33]

        Random.seed!(13899)
        gs = GridSearch(prob, mlr_bounds, 30, 500)
        Random.seed!(13899)
        _gs = GridSearch(mlr_f, mlr_bounds, 30, 500)
        Random.seed!(13899)
        lg = LatinGrid(mlr_bounds, 30, 500)
        @test gs.grid isa LatinGrid
        @test gs.f([θ[1], θ[2]...]) == _gs.f([θ[1], θ[2]...]) && gs.grid.bounds == _gs.grid.bounds && gs.grid.grid == lg.grid
    end
end

@testset "GridSearch grid preparation" begin
    xbounds = [(1.0, 3.0), (5.0, 2.0)]
    resolutions = [4, 3]
    steps = [(3.0 - 1.0) / 3, (2.0 - 5.0) / 2]
    ug1 = UniformGrid(xbounds, resolutions)
    A = ProfileLikelihood.prepare_grid(ug1)
    @test A == zeros(Float64, 4, 3)

    resolutions = 4
    steps = [(3.0 - 1.0) / 3, (2.0 - 5.0) / 3]
    ug2 = UniformGrid(xbounds, resolutions)
    A = ProfileLikelihood.prepare_grid(ug2)
    @test A == zeros(Float64, 4, 4)

    xbounds = [(1.0, 3.0), (5.0, 2.0)]
    m = 25
    gens = 500
    Random.seed!(138)
    lg = LatinGrid(xbounds, m, gens)
    A = ProfileLikelihood.prepare_grid(lg)
    @test A == zeros(Float64, m)
end

@testset "Rastrigin function" begin
    n = 4
    A = 10
    rastrigin_f(x) = @inline @inbounds A * n + (x[1]^2 - A * cos(2π * x[1])) + (x[2]^2 - A * cos(2π * x[2])) + (x[3]^2 - A * cos(2π * x[3])) + (x[4]^2 - A * cos(2π * x[4]))
    @test rastrigin_f(zeros(n)) == 0.0
    gs = GridSearch(x -> (@inline; -rastrigin_f(x)), [(-5.12, 5.12) for i in 1:n], 25)
    f_min, x_min = grid_search(gs)
    @test f_min ≈ 0.0
    @test x_min ≈ zeros(n)
    f_min, x_min = grid_search(rastrigin_f, [(-5.12, 5.12) for i in 1:n], 25; find_max=Val(false))
    @test f_min ≈ 0.0
    @test x_min ≈ zeros(n)
    f_min, x_min, f_res = grid_search(gs; save_res=Val(true))
    @test f_min ≈ 0.0
    @test x_min ≈ zeros(n)
    param_ranges = [LinRange(-5.12, 5.12, 25) for i in 1:n]
    @test f_res ≈ -[rastrigin_f(x) for x in Iterators.product(param_ranges...)]

    Random.seed!(288831)
    gs = GridSearch(x -> -rastrigin_f(x), [(-5.12, 5.12) for i in 1:n], 371, 500)
    f_min, x_min = grid_search(gs)
    @test f_min ≈ -20.42560331820396
    @test x_min ≈ [-0.913297297297297, -2.1587027027027026, -1.8542702702702707, 0.9409729729729728]
    Random.seed!(288831)
    f_min, x_min = grid_search(rastrigin_f, [(-5.12, 5.12) for i in 1:n], 371, 500; find_max=Val(false))
    @test f_min ≈ -20.42560331820396
    @test x_min ≈ [-0.913297297297297, -2.1587027027027026, -1.8542702702702707, 0.9409729729729728]
    f_min, x_min, f_res = grid_search(gs; save_res=Val(true))
    @test f_min ≈ -20.42560331820396
    @test x_min ≈ [-0.913297297297297, -2.1587027027027026, -1.8542702702702707, 0.9409729729729728]
    for i in 1:371 
        @test f_res[i] ≈ -rastrigin_f(gs.grid.grid[:, i])
    end
end

@testset "Ackley function" begin
    n = 2
    ackley_f(x) = -20exp(-0.2sqrt(x[1]^2 + x[2]^2)) - exp(0.5(cos(2π * x[1]) + cos(2π * x[2]))) + exp(1) + 20
    @test ackley_f([0, 0]) == 0
    gs = GridSearch(x -> -ackley_f(x), [(-15.12, 15.12) for i in 1:n], [73, 121])
    f_min, x_min = grid_search(gs)
    @test f_min ≈ 0.0 atol = 1e-7
    @test x_min ≈ zeros(n) atol = 1e-7
    f_min, x_min = grid_search(ackley_f, [(-15.12, 15.12) for i in 1:n], [73, 121]; find_max=Val(false))
    @test f_min ≈ 0.0 atol = 1e-7
    @test x_min ≈ zeros(n) atol = 1e-7
    f_min, x_min = grid_search(x -> -ackley_f(x), [(-15.12, 15.12) for i in 1:n], [73, 121])
    @test f_min ≈ 0.0 atol = 1e-7
    @test x_min ≈ zeros(n) atol = 1e-7
    f_min, x_min, f_res = grid_search(x -> -ackley_f(x), [(-15.12, 15.12) for i in 1:n], [731, 1217]; save_res=Val(true))
    @test f_min ≈ 0.0 atol = 1e-7
    @test x_min ≈ zeros(n) atol = 1e-7
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

    Random.seed!(92881)
    gs = GridSearch(x -> -ackley_f(x), [(-15.12, 15.12) for i in 1:n], 131, 10000)
    f_min, x_min = grid_search(gs)
    @test f_min ≈ -5.28428706685472 atol = 1e-7
    @test x_min ≈ [-0.6978461538461538, 0.4652307692307698] atol = 1e-7
    Random.seed!(92881)
    f_min, x_min = grid_search(ackley_f, [(-15.12, 15.12) for i in 1:n], 131, 10000; find_max=Val(false))
    @test f_min ≈ -5.28428706685472 atol = 1e-7
    @test x_min ≈ [-0.6978461538461538, 0.4652307692307698] atol = 1e-7
    Random.seed!(92881)
    f_min, x_min, f_res = grid_search(x -> -ackley_f(x), [(-15.12, 15.12) for i in 1:n], 131, 10000; save_res=Val(true))
    @test f_min ≈ -5.28428706685472 atol = 1e-7
    @test x_min ≈ [-0.6978461538461538, 0.4652307692307698] atol = 1e-7
    for i in 1:131 
        @test f_res[i] ≈ -ackley_f(gs.grid.grid[:, i])
    end
end

@testset "Sphere function" begin
    n = 5
    sphere_f(x) = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 + x[5]^2
    @test sphere_f(zeros(n)) == 0
    resolutions = [15, 17, 15, 19, 23]
    gs = GridSearch(x -> -sphere_f(x), [(-15.12, 15.12) for i in 1:n], resolutions)
    f_min, x_min = grid_search(gs)
    @test f_min ≈ 0.0 atol = 1e-7
    @test x_min ≈ zeros(n) atol = 1e-7
    f_min, x_min = grid_search(sphere_f, [(-15.12, 15.12) for i in 1:n], resolutions; find_max=Val(false))
    @test f_min ≈ 0.0 atol = 1e-7
    @test x_min ≈ zeros(n) atol = 1e-7
    f_min, x_min = grid_search(x -> -sphere_f(x), [(-15.12, 15.12) for i in 1:n], resolutions)
    @test f_min ≈ 0.0 atol = 1e-7
    @test x_min ≈ zeros(n) atol = 1e-7
    f_min, x_min, f_res = grid_search(x -> -sphere_f(x), [(-15.12, 15.12) for i in 1:n], resolutions; save_res=Val(true))
    @test f_min ≈ 0.0 atol = 1e-7
    @test x_min ≈ zeros(n) atol = 1e-7
    param_ranges = [LinRange(-15.12, 15.12, resolutions[i]) for i in 1:n]
    @test f_res ≈ -[sphere_f(x) for x in Iterators.product(param_ranges...)]
end

@testset "Regression" begin
    prob, loglikk, θ, dat = MultipleLinearRegression()
    true_ℓ = loglikk(reduce(vcat, θ), dat)
    @test_throws MethodError grid_search(prob)
    mlr_bounds = [(1e-12, 0.2), (-3.0, 0.0), (0.0, 3.0), (0.0, 3.0), (-6.0, 6.0)]
    mlr_resolution = 27
    sol = grid_search(prob, mlr_bounds, mlr_resolution)
    @test maximum(sol) ≈ 281.7360323629172
    @test mle(sol) ≈ [0.09230769230823077
        -0.9230769230769231
        0.8076923076923077
        0.46153846153846156
        3.2307692307692317]
    
    Random.seed!(88888)
    lg1 = grid_search(prob, mlr_bounds, 10, 50)
    Random.seed!(88888)
    lg2 = grid_search(θ -> prob.loglik(θ, data(prob)), mlr_bounds, 10, 50; find_max = Val(true))
    @test maximum(lg1) == lg2[1] && mle(lg1) == lg2[2]
end

@testset "Linear exponential ODE" begin
    prob, loglikk, θ, yᵒ, n = LinearExponentialODE()
    λ, σ, y₀ = θ
    true_ℓ = prob.loglik(θ, data(prob))
    sol = grid_search(prob, bounds(prob; make_open=true), 20)
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
    @test algorithm_name(sol) == "UniformGridSearch"
    @test sol.alg == :UniformGridSearch
    @test sol.original === nothing
    @test sol.prob == prob
end