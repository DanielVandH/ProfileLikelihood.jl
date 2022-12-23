using ..ProfileLikelihood
######################################################
## ConfidenceInterval 
######################################################
local a, b
CI = ProfileLikelihood.ConfidenceInterval(0.1, 0.2, 0.95)
@test ProfileLikelihood.get_lower(CI) == CI.lower == 0.1
@test ProfileLikelihood.get_upper(CI) == CI.upper == 0.2
@test ProfileLikelihood.get_level(CI) == CI.level == 0.95
@test ProfileLikelihood.get_bounds(CI) == (CI.lower, CI.upper) == (0.1, 0.2)
@test CI[1] == CI.lower == 0.1
@test CI[2] == CI.upper == 0.2
@test CI[begin] == CI[1] == 0.1
@test CI[end] == CI[2] == 0.2
@test length(CI) == 0.1
a, b = CI
@test a == ProfileLikelihood.get_lower(CI)
@test b == ProfileLikelihood.get_upper(CI)
@test_throws BoundsError a, b, c = CI
@test 0.17 ∈ CI
@test 0.24 ∉ CI
@test 0.0 ∉ CI

######################################################
## ConfidenceRegion 
######################################################
x = rand(100)
y = rand(100)
CR = ProfileLikelihood.ConfidenceRegion(x,y,0.95)
@test ProfileLikelihood.get_x(CR) == x 
@test ProfileLikelihood.get_y(CR) == y 
@test ProfileLikelihood.get_level(CR) == 0.95 
@test length(CR) == 100 
@test eltype(CR) == NTuple{2,Float64}
cr_xy = collect(CR)
@test cr_xy == [(x, y) for (x, y) in zip(x, y)]