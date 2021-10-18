using Test 

import DivPopGa.ClusteredGa as CLGA

@testset "Create Chromosome" begin
    lower = [0.0, 1.0, 2.0, 3.0, 5.0]
    upper = [1.0, 2.0, 3.0, 4.0, 6.0]
    c = CLGA.Chromosome(lower, upper)
    for i in 1:length(upper)
        @test c.genes[i] <= upper[i]
        @test c.genes[i] >= lower[i]
    end
end

@testset "Linear Crossover" begin
    ch1 = CLGA.Chromosome(
        [1.0, 2.0],           # genes 
        Inf64,                # cost 
        -1                    # cluster id
    )

    ch2 = CLGA.Chromosome(
        [1.0, 2.0],
        Inf64,
        -1
    )

    function costfn(x::Array{Float64, 1})::Float64
        return abs(x[1] - 3.141592) + abs(x[2] - 2.71828)
    end

    xfn = CLGA.makelinearcrossover(costfn)
    
    child = xfn(ch1, ch2)
    g = child.genes 

    alternative1 = 0.5 * ch1.genes .+ 0.5 * ch2.genes
    alternative2 = 1.5 * ch1.genes .- 0.5 * ch2.genes
    alternative3 = 1.5 * ch2.genes .- 0.5 * ch1.genes

    @test g == alternative1 || g == alternative2 || g == alternative3      
end


@testset "Random normal mutation" begin
    mutfn = CLGA.makerandommutation(
        1.0,          # std deviation
        1.0           # mutation probability
    )

    ch = CLGA.Chromosome(
        [1.0, 1.0], # genes
        Inf64,      # cost of chromosome
        -1         # cluster id
    )

    mutated = mutfn(ch)

    @test mutated isa CLGA.Chromosome
    @test mutated.genes[1] < 1 + 5
    @test mutated.genes[1] > 1 - 5
    @test mutated.genes[2] < 1 + 5
    @test mutated.genes[2] > 1 - 5
end

@testset "Tournament selection" begin 
    pop = [
        CLGA.Chromosome([0.0, 0.0], 1, -1),
        CLGA.Chromosome([0.0, 0.0], 2, -1),
        CLGA.Chromosome([0.0, 0.0], 3, -1),
        CLGA.Chromosome([0.0, 0.0], 4, -1),
        CLGA.Chromosome([0.0, 0.0], 5, -1),
    ]
    chs = CLGA.tournamentselection(pop, 2)
    
    @test length(chs) == 2
    @test chs[1].cost < 5.0
    @test chs[2].cost < 5.0
end


@testset "KMeans Tournament selection" begin 
    pop = [
        CLGA.Chromosome([0.0, 0.0], 1, 1),
        CLGA.Chromosome([0.0, 0.0], 2, 1),
        CLGA.Chromosome([0.0, 0.0], 3, 1),
        CLGA.Chromosome([0.0, 0.0], 4, 1),
        CLGA.Chromosome([0.0, 0.0], 5, 2),
    ]
    chs = CLGA.kmeanstournamentselection(pop, 2)
    
    @test length(chs) == 2
    @test chs[1].clusterid != chs[2].clusterid
end

@testset "Pi and E with Kmeans clustering GA" begin
    function costfn(vals)
        return abs(vals[1] - 3.141592) + abs(vals[2] - 2.71828)
    end 
    crossfn = CLGA.makelinearcrossover(costfn)
    mutatefn = CLGA.makerandommutation(10.0,  #stddev
                                      0.10   #mutation prob
    )
    selectfn = CLGA.makekmeanstournamentselection(3 #= tournament count=#)
    result = CLGA.ga(
        100, # popsize
        1000, # generations 
        [-100.0, -100.0], #lower
        [100.0, 100.0],   #upper
         costfn, #costfunction 
         crossfn, #crossoverfunction 
         mutatefn, #mutation function 
         selectfn # selection function
    )
    best = result[1]
    @test best isa CLGA.Chromosome
    @test best.genes[1] > 3.0
    @test best.genes[1] < 4.0
    @test best.genes[2] > 2.0
    @test best.genes[2] < 3.0

    # Fine tuning
    finetuning_mutatefn = CLGA.makerandommutation(1.0,  #stddev
                                                 0.05   #mutation prob
    )
    pop = result 
    for i in 1:1000
        pop = CLGA.generation(pop, costfn, crossfn, finetuning_mutatefn, selectfn)
    end
    CLGA.calculatefitness(pop, costfn)
    sort!(pop, by = ch -> ch.cost)
    best = pop[1]
    @test best isa CLGA.Chromosome
    @test best.genes[1] > 3.1  # 3.141592
    @test best.genes[1] < 3.2
    @test best.genes[2] > 2.7  # 2.71828
    @test best.genes[2] < 2.8
end