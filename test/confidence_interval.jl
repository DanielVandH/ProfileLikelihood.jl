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