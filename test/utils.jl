using ..ProfileLikelihood
using Distributions

@testset "number_type" begin
    x = 5.0
    @test ProfileLikelihood.number_type(x) == Float64
    x = 5.0f0
    @test ProfileLikelihood.number_type(x) == Float32

    x = [[5.0, 2.0], [2.0], [5.0, 5.0]]
    @test ProfileLikelihood.number_type(x) == Float64
    x = [[[[[[[[[[[[[5.0]]]]]]]]]]]]]
    @test ProfileLikelihood.number_type(x) == Float64
    x = [[2, 3, 4], [2, 3, 5]]
    @test ProfileLikelihood.number_type(x) == Int64

    x = rand(5, 8)
    @test ProfileLikelihood.number_type(x) == Float64

    x = ((5.0, 3.0), (2.0, 3.0), (5.0, 1.0))
    @test ProfileLikelihood.number_type(x) == Float64

    x = ((5, 3), (2, 3), (5, 1), (2, 5))
    @test ProfileLikelihood.number_type(x) == Int64
end

@testset "get_default_extremum" begin
    @test ProfileLikelihood.get_default_extremum(Float64, Val{false}) == typemin(Float64)
    @test ProfileLikelihood.get_default_extremum(Float64, Val{true}) == typemax(Float64)
    @test ProfileLikelihood.get_default_extremum(Float32, Val{false}) == typemin(Float32)
    @test ProfileLikelihood.get_default_extremum(Float32, Val{true}) == typemax(Float32)
end

@testset "update_extrema!" begin
    new_x = zeros(4)
    new_f = 2.0
    old_x = [1.0, 2.0, 3.0, 4.0]
    old_f = 1.0
    new_f = ProfileLikelihood.update_extremum!(new_x, new_f, old_x, old_f; minimise=Val(false))
    @test new_f == 2.0
    @test new_x == old_x

    new_x = zeros(4)
    new_f = 0.5
    old_x = [1.0, 2.0, 3.0, 4.0]
    old_f = 1.0
    new_f = ProfileLikelihood.update_extremum!(new_x, new_f, old_x, old_f; minimise=Val(false))
    @test new_f == 1.0
    @test new_x == zeros(4)

    new_x = zeros(4)
    new_f = 0.5
    old_x = [1.0, 2.0, 3.0, 4.0]
    old_f = 1.0
    new_f = ProfileLikelihood.update_extremum!(new_x, new_f, old_x, old_f; minimise=Val(true))
    @test new_f == 0.5
    @test new_x == old_x

    new_x = zeros(4)
    new_f = 1.5
    old_x = [1.0, 2.0, 3.0, 4.0]
    old_f = 1.0
    new_f = ProfileLikelihood.update_extremum!(new_x, new_f, old_x, old_f; minimise=Val(true))
    @test new_f == 1.0
    @test new_x == zeros(4)
end

@testset "gaussian_loglikelihood" begin
    for _ in 1:250
        local x
        n = rand(1:500)
        x = rand(n)
        μ = rand(n)
        σ = 5rand()
        ℓ = 0.0
        for i in 1:n
            ℓ = ℓ - log(sqrt(2π * σ^2)) - (x[i] - μ[i])^2 / (2σ^2)
        end
        @test ℓ ≈ ProfileLikelihood.gaussian_loglikelihood(x, μ, σ, n)
        @inferred ProfileLikelihood.gaussian_loglikelihood(x, μ, σ, n)
    end
end

@testset "get_chisq_threshold" begin
    @test all(x -> ProfileLikelihood.get_chisq_threshold(x) ≈ -0.5quantile(Chisq(1), x), 0.001:0.001:0.999)
    @test all(x -> ProfileLikelihood.get_chisq_threshold(x, 3) ≈ -0.5quantile(Chisq(3), x), 0.001:0.001:0.999)
end

@testset "subscriptnumber" begin
    @test ProfileLikelihood.subscriptnumber(1) == "₁"
    @test ProfileLikelihood.subscriptnumber(2) == "₂"
    @test ProfileLikelihood.subscriptnumber(3) == "₃"
    @test ProfileLikelihood.subscriptnumber(4) == "₄"
    @test ProfileLikelihood.subscriptnumber(5) == "₅"
    @test ProfileLikelihood.subscriptnumber(6) == "₆"
    @test ProfileLikelihood.subscriptnumber(7) == "₇"
    @test ProfileLikelihood.subscriptnumber(13) == "₁₃"
end

@testset "linear_extrapolation" begin
    x₀, y₀ = 0.23, 0.58
    x₁, y₁ = -0.271, 0.8
    x = 0.22
    y = ProfileLikelihood.linear_extrapolation(x, x₀, y₀, x₁, y₁)
    @test (y - y₁) / (x - x₁) ≈ (y₀ - y₁) / (x₀ - x₁)
    y = ProfileLikelihood.linear_extrapolation(x₀, x₀, y₀, x₁, y₁)
    @test y ≈ y₀
    y = ProfileLikelihood.linear_extrapolation(x₁, x₀, y₀, x₁, y₁)
    @test y ≈ y₁

    x₀ = 0.23
    y₀ = rand(4)
    x₁ = 0.999
    y₁ = rand(4)
    x = -0.29291
    y = zeros(4)
    ProfileLikelihood.linear_extrapolation!(y, x, x₀, y₀, x₁, y₁)
    @test all(i -> y[i] == ProfileLikelihood.linear_extrapolation(x, x₀, y₀[i], x₁, y₁[i]), 1:4)
    ProfileLikelihood.linear_extrapolation!(y, x₀, x₀, y₀, x₁, y₁)
    @test y ≈ y₀
    ProfileLikelihood.linear_extrapolation!(y, x₁, x₀, y₀, x₁, y₁)
    @test y ≈ y₁
end

@testset "_Val" begin
    @test ProfileLikelihood._Val(6) == Val(6)
    @test ProfileLikelihood._Val(true) == Val(true)
    @test ProfileLikelihood._Val(false) == Val(false)
    @test ProfileLikelihood._Val(:interp) == Val(:interp)
    @test ProfileLikelihood._Val(Val(:interp)) == Val(:interp)
    @test ProfileLikelihood._Val(Val(true)) == Val(true)
end

@testset "take_val" begin
    @test ProfileLikelihood.take_val(6) == 6
    @test ProfileLikelihood.take_val(true) == true
    @test ProfileLikelihood.take_val(Val(6)) == 6
    @test ProfileLikelihood.take_val(Val(:interp)) == :interp
end