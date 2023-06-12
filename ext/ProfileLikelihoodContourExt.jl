module ProfileLikelihoodContourExt 

using ProfileLikelihood 
@static if isdefined(Base, :get_extension) 
    using Contour 
else 
    using ..Contour 
end

function ProfileLikelihood._get_confidence_regions_contour!(confidence_regions, n, range_1, range_2, profile_values, threshold, conf_level)
    c = Contour.contour(range_1, range_2, profile_values, threshold)
    all_coords = reduce(vcat, [reduce(hcat, Contour.coordinates(xy)) for xy in Contour.lines(c)])
    region_x = all_coords[:, 1]
    region_y = all_coords[:, 2]
    ax = all_coords[1, 1]
    bx = all_coords[2, 1]
    cx = range_1[0]
    ay = all_coords[1, 2]
    by = all_coords[2, 2]
    cy = range_2[0]
    countour_is_clockwise = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx) < 0
    countour_is_clockwise && reverse!(region_x)
    countour_is_clockwise && reverse!(region_y)
    if region_x[end] == region_x[begin]
        pop!(region_x)
        pop!(region_y) # contour keeps the last value as being the same as the first
    end
    confidence_regions[n] = ProfileLikelihood.ConfidenceRegion(region_x, region_y, conf_level)
    return nothing
end