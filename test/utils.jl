using ..ProfileLikelihood
using Distributions
using ConcreteStructs
using CairoMakie
using SymbolicIndexingInterface

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
    @test ProfileLikelihood.number_type(x) == Int

    x = rand(5, 8)
    @test ProfileLikelihood.number_type(x) == Float64

    x = ((5.0, 3.0), (2.0, 3.0), (5.0, 1.0))
    @test ProfileLikelihood.number_type(x) == Float64

    x = ((5, 3), (2, 3), (5, 1), (2, 5))
    @test ProfileLikelihood.number_type(x) == Int
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

@concrete struct FStruct2
    f
    a
    b
    hess
    grad
    x
    y
    z
    adtype
end
@testset "_to_namedtuple" begin
    obj = FStruct2(1, 2, 3, 4, 5, 6, 7, 8, 9)
    @test ProfileLikelihood._to_namedtuple(obj) == (a=2, b=3, x=6, y=7, z=8)
    @inferred ProfileLikelihood._to_namedtuple(obj)
end

@testset "latexify" begin
    ext = Base.get_extension(ProfileLikelihood, :ProfileLikelihoodMakieExt)
    @test ext.latexify("A₁B₂C₃") == L"A_{1}B_{2}C_{3}"
    @test ext.latexify("XᵃYᵇZᶜ") == L"X^{a}Y^{b}Z^{c}"
    @test ext.latexify("EₐᵍFₓᵥ") == L"E_{a}^{g}F_{xv}"
    @test ext.latexify("H₀₁₂O₄₅⁶⁷") == L"H_{012}O_{45}^{67}"
    @test ext.latexify("CompoundₓFormᵗX₁₂ⁱ") == L"Compound_{x}Form^{t}X_{12}^{i}"
    @test ext.latexify("aₓbₓcᵦdᵘᵡe") == L"a_{x}b_{x}c_{β}d^{uχ}e"
    @test ext.latexify("Hello World") == L"Hello World"
    @test ext.latexify("α₁₂₃θᵧ⁶⁷") == L"α_{123}θ_{γ}^{67}"
    @test ext.latexify("") == L""
    @test ext.latexify("1234!@#") == L"1234!@#"
end

@testset "default_latex_names" begin
    ext = Base.get_extension(ProfileLikelihood, :ProfileLikelihoodMakieExt)
    prof = SymbolCache([:α, :β, :γ, :δ, :θ₁, :γ², :θ₁₂³⁴])
    names = ext.default_latex_names(prof, [1, 2, 3])
    @test names == Dict(1 => L"α", 2 => L"β", 3 => L"γ")
    names = ext.default_latex_names(prof, [1, 4, 5, 7])
    @test names == Dict(1 => L"α", 4 => L"δ", 5 => L"θ_{1}", 7 => L"θ_{12}^{34}")
    names = ext.default_latex_names(prof, [1, 2, 3, 4, 5, 6, 7])
    @test names == Dict(1 => L"α", 2 => L"β", 3 => L"γ", 4 => L"δ", 5 => L"θ_{1}", 6 => L"γ^{2}", 7 => L"θ_{12}^{34}")
    names = ext.default_latex_names(prof, [:α, :β, :γ, :δ, :θ₁, :γ², :θ₁₂³⁴])
    @test names == Dict(:α => L"α", :β => L"β", :γ => L"γ", :δ => L"δ", :θ₁ => L"θ_{1}", :γ² => L"γ^{2}", :θ₁₂³⁴ => L"θ_{12}^{34}")
    names = ext.default_latex_names(prof, [:β, 1, :γ, 4, 5, :γ²])
    @test names == Dict(:β => L"β", 1 => L"α", :γ => L"γ", 4 => L"δ", 5 => L"θ_{1}", :γ² => L"γ^{2}")
end