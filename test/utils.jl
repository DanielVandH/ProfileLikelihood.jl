
######################################################
## Utilities 
######################################################
## number_type 
x = 5.0
@test PL.number_type(x) == Float64
x = 5.0f0
@test PL.number_type(x) == Float32

x = [[5.0, 2.0], [2.0], [5.0, 5.0]]
@test PL.number_type(x) == Float64
x = [[[[[[[[[[[[[5.0]]]]]]]]]]]]]
@test PL.number_type(x) == Float64
x = [[2, 3, 4], [2, 3, 5]]
@test PL.number_type(x) == Int64

x = rand(5, 8)
@test PL.number_type(x) == Float64

x = ((5.0, 3.0), (2.0, 3.0), (5.0, 1.0))
@test PL.number_type(x) == Float64

x = ((5, 3), (2, 3), (5, 1), (2, 5))
@test PL.number_type(x) == Int64

## get_default_extremum
@test PL.get_default_extremum(Float64, Val{false}) == typemin(Float64)
@test PL.get_default_extremum(Float64, Val{true}) == typemax(Float64)
@test PL.get_default_extremum(Float32, Val{false}) == typemin(Float32)
@test PL.get_default_extremum(Float32, Val{true}) == typemax(Float32)

## update_extrema! 
new_x = zeros(4)
new_f = 2.0
old_x = [1.0, 2.0, 3.0, 4.0]
old_f = 1.0
new_f = PL.update_extremum!(new_x, new_f, old_x, old_f; minimise=Val(false))
@test new_f == 2.0
@test new_x == old_x

new_x = zeros(4)
new_f = 0.5
old_x = [1.0, 2.0, 3.0, 4.0]
old_f = 1.0
new_f = PL.update_extremum!(new_x, new_f, old_x, old_f; minimise=Val(false))
@test new_f == 1.0
@test new_x == zeros(4)

new_x = zeros(4)
new_f = 0.5
old_x = [1.0, 2.0, 3.0, 4.0]
old_f = 1.0
new_f = PL.update_extremum!(new_x, new_f, old_x, old_f; minimise=Val(true))
@test new_f == 0.5
@test new_x == old_x

new_x = zeros(4)
new_f = 1.5
old_x = [1.0, 2.0, 3.0, 4.0]
old_f = 1.0
new_f = PL.update_extremum!(new_x, new_f, old_x, old_f; minimise=Val(true))
@test new_f == 1.0
@test new_x == zeros(4)

## gaussian_loglikelihood
for _ in 1:250
    n = rand(1:500)
    x = rand(n)
    μ = rand(n)
    σ = 5rand()
    ℓ = 0.0
    for i in 1:n
        ℓ = ℓ - log(sqrt(2π * σ^2)) - (x[i] - μ[i])^2 / (2σ^2)
    end
    @test ℓ ≈ PL.gaussian_loglikelihood(x, μ, σ, n)
    @inferred PL.gaussian_loglikelihood(x, μ, σ, n)
end

## get_chisq_threshold
@test all(x -> PL.get_chisq_threshold(x) ≈ -0.5quantile(Chisq(1), x), 0.001:0.001:0.999)
@test all(x -> PL.get_chisq_threshold(x, 3) ≈ -0.5quantile(Chisq(3), x), 0.001:0.001:0.999)

## subscriptnumber
@test PL.subscriptnumber(1) == "₁"
@test PL.subscriptnumber(2) == "₂"
@test PL.subscriptnumber(3) == "₃"
@test PL.subscriptnumber(4) == "₄"
@test PL.subscriptnumber(5) == "₅"
@test PL.subscriptnumber(6) == "₆"
@test PL.subscriptnumber(7) == "₇"
@test PL.subscriptnumber(13) == "₁₃"