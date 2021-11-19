module Summary

import DivPopGa.ClusteredGa: Chromosome 
import Statistics 

function summary(population::Array{Chromosome, 1})::Dict{String, Any}
    costs = map(c -> c.cost, population)
    sorted_pop     = sort(population, by = ch -> ch.cost)
    mean_cost      = Statistics.mean(costs)
    median_cost    = Statistics.median(costs)
    min_cost       = minimum(costs)
    max_cost       = maximum(costs)
    q25_cost       = Statistics.quantile(costs, 0.25)
    q75_cost       = Statistics.quantile(costs, 0.75)
    best           = sorted_pop[1]

    result = Dict(
        "mean_cost"       => mean_cost,
        "min_cost"        => min_cost,
        "q25_cost"        => q25_cost,
        "median_cost"     => median_cost,
        "q75_cost"        => q75_cost, 
        "max_cost"        => max_cost,
        "best_solution"   => best          
    )

    return result
end

end # End of module 