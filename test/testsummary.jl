using Test 

import DivPopGa.ClusteredGa as CLGA
import DivPopGa.Summary as Summary 

@testset "Summary" begin 
    function costfn(vals)
        return abs(vals[1] - 3.141592) + abs(vals[2] - 2.71828)
    end 
    
    crossfn = CLGA.makelinearcrossover(costfn)
    mutatefn = CLGA.makenormalmutation(10.0,  #stddev
                                      0.10   #mutation prob
                )
    result = CLGA.ga(
        100, # popsize
        500, # generations 
        [-100.0, -100.0], #lower
        [100.0, 100.0],   #upper
         costfn, #costfunction 
         crossfn, #crossoverfunction 
         mutatefn, #mutation function 
         CLGA.GA_TYPE_CLASSIC # classical selection
    )

    s = Summary.summary(result.population)

    @test s isa Dict
    @test s["min_cost"] <= s["q25_cost"]
    @test s["q25_cost"] <= s["median_cost"]
    @test s["median_cost"] <= s["q75_cost"]
    @test s["q75_cost"] <= s["max_cost"]
    @test s["min_cost"] <= s["mean_cost"] <= s["max_cost"] 
    @test s["best_solution"] isa CLGA.Chromosome
end 