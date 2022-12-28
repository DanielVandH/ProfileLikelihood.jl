using ..ProfileLikelihood
using OffsetArrays

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

######################################################
## FusedRegularGrid 
######################################################
## Getting the resolution tuples 
res = ProfileLikelihood.get_resolution_tuples(5, 3)
@test res == [(5, 5), (5, 5), (5, 5)]
res = ProfileLikelihood.get_resolution_tuples([2, 5, 7, 10], 4)
@test res == [(2, 2), (5, 5), (7, 7), (10, 10)]
res = ProfileLikelihood.get_resolution_tuples([(2, 10), (3, 5), (10, 11)], 3)
@test res == [(2, 10), (3, 5), (10, 11)]
res = ProfileLikelihood.get_resolution_tuples([2, (5, 10), 11, (7, 11)], 4)
@test res == [(2, 2), (5, 10), (11, 11), (7, 11)]
res = ProfileLikelihood.get_resolution_tuples((2, 3, 4, 11), 4)
@test res == [(2, 2), (3, 3), (4, 4), (11, 11)]

## Same resolution
lb = [2.0, 3.0, 1.0, 5.0]
ub = [15.0, 13.0, 27.0, 10.0]
centre = [7.3, 8.3, 2.3, 7.5]
grid_1 = RegularGrid(centre .+ (ub .- centre) / 173, ub, 173)
grid_2 = RegularGrid(centre .- (centre .- lb) / 173, lb, 173)
fused = ProfileLikelihood.FusedRegularGrid(lb, ub, centre, 173)
@test fused.positive_grid.lower_bounds == grid_1.lower_bounds == centre .+ (ub .- centre) ./ 173
@test fused.positive_grid.upper_bounds == grid_1.upper_bounds == ub == get_upper_bounds(fused)
@test fused.negative_grid.lower_bounds == grid_2.lower_bounds == centre .- (centre .- lb) / 173
@test fused.negative_grid.upper_bounds == grid_2.upper_bounds == lb == get_lower_bounds(fused)
@test ProfileLikelihood.get_positive_grid(fused) == fused.positive_grid
@test ProfileLikelihood.get_negative_grid(fused) == fused.negative_grid
@test_throws BoundsError fused[1, 180]
@test_throws BoundsError fused[1, -180]
@test_throws BoundsError fused[5, 0]
@test_throws BoundsError fused[5, 200]
@test_throws BoundsError fused[-5, 200]
@test_throws BoundsError fused[5, -288]
@test ProfileLikelihood.get_centre(fused, 1) == centre[1]
@test ProfileLikelihood.get_centre(fused, 3) == centre[3]
Δpos = (ub .- centre) / 173
Δneg = (centre .- lb) / 173
@test fused[1, 0] == centre[1]
@test fused[2, 0] == centre[2]
@test fused[3, 0] == centre[3]
@test fused[4, 0] == centre[4]
@test fused[1, 1] == centre[1] + Δpos[1]
@test fused[1, 7] ≈ centre[1] + 7Δpos[1]
for i in 1:4
    for j in 1:173
        @test fused[i, j] ≈ centre[i] + j * Δpos[i]
        @test fused[i, -j] ≈ centre[i] - j * Δneg[i]
    end
end
for i in 1:4
    @test fused[i, 173] ≈ ub[i]
    @test fused[i, -173] ≈ lb[i]
end
for i in 1:4
    rng1 = get_range(fused, i)
    @test rng1[begin] ≈ lb[i]
    @test rng1[end] ≈ ub[i]
    left = [centre[i] - j * Δneg[i] for j in 1:173]
    reverse!(left)
    right = [centre[i] + j * Δpos[i] for j in 1:173]
    rng1_true = [left..., centre[i], right...]
    @test rng1 ≈ OffsetVector(rng1_true, -173:173)
end
θ = zeros(4)
ProfileLikelihood.get_parameters!(θ, fused, (2, 0, -7, 17))
@test θ ≈ [fused[1, 2], fused[2, 0], fused[3, -7], fused[4, 17]]
@test θ ≈ ProfileLikelihood.get_parameters(fused, (2, 0, -7, 17))
ProfileLikelihood.get_parameters!(θ, fused, (0, 0, 0, 0))
@test θ ≈ centre
@test θ ≈ ProfileLikelihood.get_parameters(fused, (0, 0, 0, 0))
ProfileLikelihood.get_parameters!(θ, fused, (-173, -173, -173, -173))
@test θ ≈ lb
@test θ ≈ ProfileLikelihood.get_parameters(fused, (-173, -173, -173, -173))
ProfileLikelihood.get_parameters!(θ, fused, (173, 173, 173, 173))
@test θ ≈ ub
@test θ ≈ ProfileLikelihood.get_parameters(fused, (173, 173, 173, 173))
ProfileLikelihood.get_parameters!(θ, fused, [2, 5, 160, -100])
@test θ ≈ [centre[1] + 2Δpos[1], centre[2] + 5Δpos[2], centre[3] + 160Δpos[3], centre[4] - 100Δneg[4]]
@test θ ≈ ProfileLikelihood.get_parameters(fused, (2, 5, 160, -100))
ProfileLikelihood.get_parameters!(θ, fused, [-22, 75, -160, 100])
@test θ ≈ [centre[1] - 22Δneg[1], centre[2] + 75Δpos[2], centre[3] - 160Δneg[3], centre[4] + 100Δpos[4]]
@test θ ≈ ProfileLikelihood.get_parameters(fused, (-22, 75, -160, 100))
@test ProfileLikelihood.finite_bounds(fused)
@test fused.resolutions == ProfileLikelihood.get_resolution_tuples(173, 4)

## Varying resolutons 
lb = [2.0, 3.0, 1.0, 5.0, 4.0]
ub = [15.0, 13.0, 27.0, 10.0, 13.0]
centre = [7.3, 8.3, 2.3, 7.5, 10.0]
res = [(2, 11), (23, 25), (19, 21), (50, 51), (17, 99)]
grid_1 = RegularGrid(centre .+ (ub .- centre) ./ [2, 23, 19, 50, 17], ub, [2, 23, 19, 50, 17])
grid_2 = RegularGrid(centre .- (centre .- lb) ./ [11, 25, 21, 51, 99], lb, [11, 25, 21, 51, 99])
fused = ProfileLikelihood.FusedRegularGrid(lb, ub, centre, res)
@test fused.positive_grid.lower_bounds == grid_1.lower_bounds == centre .+ (ub .- centre) ./ [2, 23, 19, 50, 17]
@test fused.positive_grid.upper_bounds == grid_1.upper_bounds == ub == get_upper_bounds(fused)
@test fused.negative_grid.lower_bounds == grid_2.lower_bounds == centre .- (centre .- lb) ./ [11, 25, 21, 51, 99]
@test fused.negative_grid.upper_bounds == grid_2.upper_bounds == lb == get_lower_bounds(fused)
@test ProfileLikelihood.get_positive_grid(fused) == fused.positive_grid
@test ProfileLikelihood.get_negative_grid(fused) == fused.negative_grid
@test_throws BoundsError fused[1, 180]
@test_throws BoundsError fused[1, -180]
@test_throws BoundsError fused[6, 0]
@test_throws BoundsError fused[5, 200]
@test_throws BoundsError fused[-5, 200]
@test_throws BoundsError fused[5, -288]
@test_throws BoundsError fused[1, 3]
@test_throws BoundsError fused[1, -12]
@test_throws BoundsError fused[2, 24]
@test_throws BoundsError fused[2, -26]
@test_throws BoundsError fused[3, 20]
@test_throws BoundsError fused[3, -22]
@test_throws BoundsError fused[4, 51]
@test_throws BoundsError fused[4, -52]
@test_throws BoundsError fused[5, 18]
@test_throws BoundsError fused[5, -100]
@test ProfileLikelihood.get_centre(fused, 1) == centre[1]
@test ProfileLikelihood.get_centre(fused, 2) == centre[2]
@test ProfileLikelihood.get_centre(fused, 3) == centre[3]
@test ProfileLikelihood.get_centre(fused, 4) == centre[4]
@test ProfileLikelihood.get_centre(fused, 5) == centre[5]
Δpos = (ub .- centre) ./ [2, 23, 19, 50, 17]
Δneg = (centre .- lb) ./ [11, 25, 21, 51, 99]
@test fused[1, 0] == centre[1]
@test fused[2, 0] == centre[2]
@test fused[3, 0] == centre[3]
@test fused[4, 0] == centre[4]
@test fused[1, 1] ≈ centre[1] + Δpos[1]
for i in 1:5
    for j in [2, 23, 19, 50, 17][i]
        for k in 1:j
            @test fused[i, k] ≈ centre[i] + k * Δpos[i]
        end
    end
    for j in [11, 25, 21, 51, 99][i]
        for k in 1:j
            @test fused[i, -k] ≈ centre[i] - k * Δneg[i]
        end
    end
end
for i in 1:5
    @test fused[i, [2, 23, 19, 50, 17][i]] ≈ ub[i]
    @test fused[i, -[11, 25, 21, 51, 99][i]] ≈ lb[i]
end
for i in 1:5
    rng1 = get_range(fused, i)
    @test rng1[begin] ≈ lb[i]
    @test rng1[end] ≈ ub[i]
    left = [centre[i] - j * Δneg[i] for j in 1:[11, 25, 21, 51, 99][i]]
    reverse!(left)
    right = [centre[i] + j * Δpos[i] for j in 1:[2, 23, 19, 50, 17][i]]
    rng1_true = [left..., centre[i], right...]
    @test rng1 ≈ OffsetVector(rng1_true, (-[11, 25, 21, 51, 99][i]):[2, 23, 19, 50, 17][i])
end
θ = zeros(5)
ProfileLikelihood.get_parameters!(θ, fused, (2, 0, -7, 17, 6))
@test θ ≈ [fused[1, 2], fused[2, 0], fused[3, -7], fused[4, 17], fused[5, 6]]
@test θ ≈ ProfileLikelihood.get_parameters(fused, (2, 0, -7, 17, 6))
ProfileLikelihood.get_parameters!(θ, fused, (0, 0, 0, 0, 0))
@test θ ≈ centre
@test θ ≈ ProfileLikelihood.get_parameters(fused, (0, 0, 0, 0, 0))
ProfileLikelihood.get_parameters!(θ, fused, (-11, -25, -21, -51, -99))
@test θ ≈ lb
@test θ ≈ ProfileLikelihood.get_parameters(fused, (-11, -25, -21, -51, -99))
ProfileLikelihood.get_parameters!(θ, fused, (2, 23, 19, 50, 17))
@test θ ≈ ub
@test θ ≈ ProfileLikelihood.get_parameters(fused, (2, 23, 19, 50, 17))
ProfileLikelihood.get_parameters!(θ, fused, [2, 5, -10, 12, 15])
@test θ ≈ [centre[1] + 2Δpos[1], centre[2] + 5Δpos[2], centre[3] - 10Δneg[3], centre[4] + 12Δpos[4], centre[5] + 15Δpos[5]]
@test θ ≈ ProfileLikelihood.get_parameters(fused, (2, 5, -10, 12, 15))
ProfileLikelihood.get_parameters!(θ, fused, [-2, -12, 12, 10, 9])
@test θ ≈ [centre[1] - 2Δneg[1], centre[2] - 12Δneg[2], centre[3] + 12Δpos[3], centre[4] + 10Δpos[4], centre[5] + 9Δpos[5]]
@test θ ≈ ProfileLikelihood.get_parameters(fused, (-2, -12, 12, 10, 9))
@test ProfileLikelihood.finite_bounds(fused)
@test fused.resolutions == ProfileLikelihood.get_resolution_tuples(res, 5)

## Storing the original resolutions 
lb = [2.0, 3.0, 1.0, 5.0]
ub = [15.0, 13.0, 27.0, 10.0]
centre = [7.3, 8.3, 2.3, 7.5]
fused = ProfileLikelihood.FusedRegularGrid(lb, ub, centre, 263; store_original_resolutions=true)
@test fused.resolutions == 263
@test fused.positive_grid.resolution == [263, 263, 263, 263]
@test fused.negative_grid.resolution == [263, 263, 263, 263]

lb = [2.0, 3.0, 1.0, 5.0]
ub = [15.0, 13.0, 27.0, 10.0]
centre = [7.3, 8.3, 2.3, 7.5]
fused = ProfileLikelihood.FusedRegularGrid(lb, ub, centre, [17, 23, 4, 50]; store_original_resolutions=true)
@test fused.resolutions == [17, 23, 4, 50]
@test fused.positive_grid.resolution == [17, 23, 4, 50]
@test fused.negative_grid.resolution == [17, 23, 4, 50]

lb = [2.0, 3.0, 1.0, 5.0]
ub = [15.0, 13.0, 27.0, 10.0]
centre = [7.3, 8.3, 2.3, 7.5]
fused = ProfileLikelihood.FusedRegularGrid(lb, ub, centre, [(17, 23), (48, 50), (11, 11), (17, 18)]; store_original_resolutions=true)
@test fused.resolutions == [(17, 23), (48, 50), (11, 11), (17, 18)]
@test fused.positive_grid.resolution == [17, 48, 11, 17]
@test fused.negative_grid.resolution == [23, 50, 11, 18]

## Sub-grid ranges
lb = [2.0, 3.0, 1.0, 5.0]
ub = [15.0, 13.0, 27.0, 10.0]
centre = [7.3, 8.3, 2.3, 7.5]
fused = ProfileLikelihood.FusedRegularGrid(lb, ub, centre, 263; store_original_resolutions=true)
@test get_range(fused, 1, -30, 73) == OffsetVector(get_range(fused, 1)[-30:73], -30:73)
@test get_range(fused, 1, 5, 73) == OffsetVector(get_range(fused, 1)[5:73], 5:73)
@test get_range(fused, 2, -30, 73) == OffsetVector(get_range(fused, 2)[-30:73], -30:73)
@test get_range(fused, 2, 5, 73) == OffsetVector(get_range(fused, 2)[5:73], 5:73)
@test get_range(fused, 3, -30, 73) == OffsetVector(get_range(fused, 3)[-30:73], -30:73)
@test get_range(fused, 3, 5, 73) == OffsetVector(get_range(fused, 3)[5:73], 5:73)
@test get_range(fused, 4, -30, 73) == OffsetVector(get_range(fused, 4)[-30:73], -30:73)
@test get_range(fused, 4, 5, 73) == OffsetVector(get_range(fused, 4)[5:73], 5:73)

lb = [2.0, 3.0, 1.0, 5.0]
ub = [15.0, 13.0, 27.0, 10.0]
centre = [7.3, 8.3, 2.3, 7.5]
fused = ProfileLikelihood.FusedRegularGrid(lb, ub, centre, [17, 23, 4, 50]; store_original_resolutions=true)
@test get_range(fused, 1, -10, 16) == OffsetVector(get_range(fused, 1)[-10:16], -10:16)
@test get_range(fused, 2, -20, 4) == OffsetVector(get_range(fused, 2)[-20:4], -20:4)
@test get_range(fused, 3, 0, 4) == OffsetVector(get_range(fused, 3)[0:4], 0:4)
@test get_range(fused, 4, -44, 49) == OffsetVector(get_range(fused, 4)[-44:49], -44:49)


