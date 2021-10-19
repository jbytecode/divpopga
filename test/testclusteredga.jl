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

@testset "Distance of chromosomes" begin
    ch1 = CLGA.Chromosome([0.0, 0.0, 0.0], Inf64, -1)
    ch2 = CLGA.Chromosome([1.0, 1.0, 1.0], Inf64, -1)
    d = CLGA.distance(ch1, ch2)
    
    @test d isa Float64
    @test d == 3.0
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


@testset "Normal mutation" begin
    mutfn = CLGA.makenormalmutation(
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


@testset "Random mutation (within range)" begin
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [9.0, 9.0, 9.0, 9.0]
    ch = CLGA.Chromosome(lower, upper)
    
    mutatefn = CLGA.makerandommutation(lower, upper, 0.50)

    newch = mutatefn(ch)

    @test newch isa CLGA.Chromosome
    @test all(x -> x == 1,  newch.genes .< upper)
    @test all(x -> x == 1,  newch.genes .> lower) 
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

@testset "Simulated KMeans Tournament selection" begin 
    pop = [
        CLGA.Chromosome([0.0, 0.0], 1, -1),
        CLGA.Chromosome([0.0, 0.0], 2, -1),
        CLGA.Chromosome([1.0, 1.0], 3, -1),
    ]
    chs = CLGA.simulatedkmeanstournamentselection(pop, 3)
    
    @test length(chs) == 2
    @test CLGA.distance(chs[1], chs[2]) > 0.0
end


@testset "Pi and E with classical FPGA" begin
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
    
    best = result[1]
    @test best isa CLGA.Chromosome
    @test best.genes[1] > 2.0
    @test best.genes[1] < 5.0
    @test best.genes[2] > 1.0
    @test best.genes[2] < 4.0

    # Fine tuning
    finetuning_mutatefn = CLGA.makenormalmutation(1.0,  #stddev
                                                 0.05   #mutation prob
    )
    pop = result 
    for i in 1:500
        pop = CLGA.generation(pop, costfn, crossfn, finetuning_mutatefn, CLGA.GA_TYPE_CLASSIC)
    end
    CLGA.calculatefitness(pop, costfn)
    sort!(pop, by = ch -> ch.cost)
    best = pop[1]
    @test best isa CLGA.Chromosome
    @test best.genes[1] > 3.0  # 3.141592
    @test best.genes[1] < 3.3
    @test best.genes[2] > 2.6  # 2.71828
    @test best.genes[2] < 2.9
end




@testset "Pi and E with simulated kmeans" begin
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
         CLGA.GA_TYPE_CLUSTER_SIM # simulated kmeans selection
    )
    
    best = result[1]
    @test best isa CLGA.Chromosome
    @test best.genes[1] > 2.0
    @test best.genes[1] < 5.0
    @test best.genes[2] > 1.0
    @test best.genes[2] < 4.0

    # Fine tuning
    finetuning_mutatefn = CLGA.makenormalmutation(1.0,  #stddev
                                                 0.05   #mutation prob
    )
    pop = result 
    for i in 1:500
        pop = CLGA.generation(pop, costfn, crossfn, finetuning_mutatefn, CLGA.GA_TYPE_CLUSTER_SIM)
    end
    CLGA.calculatefitness(pop, costfn)
    sort!(pop, by = ch -> ch.cost)
    best = pop[1]
    @test best isa CLGA.Chromosome
    @test best.genes[1] > 3.0  # 3.141592
    @test best.genes[1] < 3.3
    @test best.genes[2] > 2.6  # 2.71828
    @test best.genes[2] < 2.9
end



@testset "Pi and E with Kmeans clustering GA" begin
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
         CLGA.GA_TYPE_CLUSTER # cluster based selection
    )
    
    best = result[1]
    @test best isa CLGA.Chromosome
    @test best.genes[1] > 2.0
    @test best.genes[1] < 5.0
    @test best.genes[2] > 1.0
    @test best.genes[2] < 4.0

    # Fine tuning
    finetuning_mutatefn = CLGA.makenormalmutation(1.0,  #stddev
                                                 0.05   #mutation prob
    )
    pop = result 
    for i in 1:500
        pop = CLGA.generation(pop, costfn, crossfn, finetuning_mutatefn, CLGA.GA_TYPE_CLUSTER)
    end
    CLGA.calculatefitness(pop, costfn)
    sort!(pop, by = ch -> ch.cost)
    best = pop[1]
    @test best isa CLGA.Chromosome
    @test best.genes[1] > 3.0  # 3.141592
    @test best.genes[1] < 3.3
    @test best.genes[2] > 2.6  # 2.71828
    @test best.genes[2] < 2.9
end





