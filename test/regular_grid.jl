using ..ProfileLikelihood
######################################################
## RegularGrid 
######################################################
## Check that the stepsizes are being computed correctly 
lb = [2.0, 3.0, 5.0]
ub = [5.0, 5.3, 13.2]
res = 50
@test all(i -> ProfileLikelihood.compute_step_size(lb, ub, res, i) == (ub[i] - lb[i]) / (res - 1), eachindex(lb))

lb = [2.7, 13.3, 45.4, 10.0]
ub = [-1.0, -1.0, 5.0, 9.9]
res = [4, 18, 49, 23]
@test all(i -> ProfileLikelihood.compute_step_size(lb, ub, res, i) == (ub[i] - lb[i]) / (res[i] - 1), eachindex(lb))

@test ProfileLikelihood.compute_step_size(2.0, 5.7, 43) == 3.7 / 42

## Test that the grid is constructed correctly 
lb = [2.7, 5.3, 10.0]
ub = [10.0, 7.7, 14.4]
res = 50
ug = ProfileLikelihood.RegularGrid(lb, ub, res)
@test ProfileLikelihood.get_lower_bounds(ug) == ug.lower_bounds == lb
@test ProfileLikelihood.get_upper_bounds(ug) == ug.upper_bounds == ub
@test all(i -> ProfileLikelihood.get_lower_bounds(ug, i) == lb[i], eachindex(lb))
@test all(i -> ProfileLikelihood.get_upper_bounds(ug, i) == ub[i], eachindex(ub))
@test ProfileLikelihood.get_step_sizes(ug) == ug.step_sizes
@test all(i -> ProfileLikelihood.get_step_sizes(ug, i) == (ub[i] - lb[i]) / (res - 1) == ug.step_sizes[i], eachindex(lb))
@test ProfileLikelihood.get_resolutions(ug) == ug.resolution
@test all(i -> ProfileLikelihood.get_resolutions(ug, i) == res == ug.resolution, eachindex(lb))
@test ProfileLikelihood.number_of_parameters(ug) == 3
for i in eachindex(lb)
    for j in 1:res
        @test ProfileLikelihood.get_step(ug, i, j) ≈ (j - 1) * (ub[i] - lb[i]) / (res - 1)
        @test ProfileLikelihood.increment_parameter(ug, i, j) == ug[i, j] ≈ lb[i] + (j - 1) * (ub[i] - lb[i]) / (res - 1)
    end
end
@test ProfileLikelihood.number_type(ug) == Float64

lb = [2.7, 5.3, 10.0, 4.4]
ub = [10.0, 7.7, 14.4, -57.4]
res = [50, 32, 10, 100]
ug = ProfileLikelihood.RegularGrid(lb, ub, res)
@test ProfileLikelihood.get_lower_bounds(ug) == ug.lower_bounds == lb
@test ProfileLikelihood.get_upper_bounds(ug) == ug.upper_bounds == ub
@test all(i -> ProfileLikelihood.get_lower_bounds(ug, i) == lb[i], eachindex(lb))
@test all(i -> ProfileLikelihood.get_upper_bounds(ug, i) == ub[i], eachindex(ub))
@test ProfileLikelihood.get_step_sizes(ug) == ug.step_sizes
@test all(i -> ProfileLikelihood.get_step_sizes(ug, i) == (ub[i] - lb[i]) / (res[i] - 1) == ug.step_sizes[i], eachindex(lb))
@test ProfileLikelihood.get_resolutions(ug) == ug.resolution
@test all(i -> ProfileLikelihood.get_resolutions(ug, i) == res[i] == ug.resolution[i], eachindex(res))
@test ProfileLikelihood.number_of_parameters(ug) == 4
for i in eachindex(lb)
    for j in 1:res[i]
        @test ProfileLikelihood.get_step(ug, i, j) ≈ (j - 1) * (ub[i] - lb[i]) / (res[i] - 1)
        @test ProfileLikelihood.increment_parameter(ug, i, j) == ug[i, j] ≈ lb[i] + (j - 1) * (ub[i] - lb[i]) / (res[i] - 1)
    end
end
@test ProfileLikelihood.number_type(ug) == Float64

## Test that we can get the parameters correctly 
lb = [2.7, 5.3, 10.0, 4.4]
ub = [10.0, 7.7, 14.4, -57.4]
res = [50, 32, 10, 100]
ug = ProfileLikelihood.RegularGrid(lb, ub, res)
I = (2, 3, 4, 10)
θ = ProfileLikelihood.get_parameters(ug, I)
@test θ == [ug[1, 2], ug[2, 3], ug[3, 4], ug[4, 10]]

I = (27, 31, 9, 100)
θ = ProfileLikelihood.get_parameters(ug, I)
@test θ == [ug[1, 27], ug[2, 31], ug[3, 9], ug[4, 100]]

## Test get_range 
lb22 = [2.7, 5.3, 10.0, 4.4]
ub22 = [10.0, 7.7, 14.4, -57.4]
res22 = [50, 32, 10, 100]
ug22 = ProfileLikelihood.RegularGrid(lb22, ub22, res22)
@test ProfileLikelihood.get_range(ug22, 1) == LinRange(2.7, 10.0, 50)
@test ProfileLikelihood.get_range(ug22, 2) == LinRange(5.3, 7.7, 32)
@test ProfileLikelihood.get_range(ug22, 3) == LinRange(10.0, 14.4, 10)
@test ProfileLikelihood.get_range(ug22, 4) == LinRange(4.4, -57.4, 100)