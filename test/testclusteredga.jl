using Test 

import DivPopGa.ClusteredGa as CLGA

@testset "Chromosome constructor with genes" begin
    genes = zeros(Float64, 10)
    c = CLGA.Chromosome(genes)

    @test length(c.genes) == 10
    @test c.clusterid == -1
    @test isinf(c.cost)
end

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

@testset "Uniform crossover" begin
    ch1 = CLGA.Chromosome([1.0, 2.0, 3.0])
    ch2 = CLGA.Chromosome([10.0, 20.0, 30.0])

    crossfn = CLGA.makeuniformcrossover()

    child1 = crossfn(ch1, ch2)

    @test child1.genes[1] in [1.0, 10.0]
    @test child1.genes[2] in [2.0, 20.0]
    @test child1.genes[3] in [3.0, 30.0]
    @test isinf(child1.cost)
    @test child1.clusterid == -1
end

@testset "Weighted Crossover" begin
    ch1 = CLGA.Chromosome(
        [1.0, 2.0],           # genes 
        Inf64,                # cost 
        -1                    # cluster id
    )

    ch2 = CLGA.Chromosome(
        [10.0, 20.0],
        Inf64,
        -1
    )

    crossfn = CLGA.makeweightedcrossover()

    child1 = crossfn(ch1, ch2)
    child2 = crossfn(ch2, ch1)

    @test child1.genes[1] <= 10.0
    @test child1.genes[2] <= 20.0
    @test child1.genes[1] >= 1.0
    @test child1.genes[2] >= 2.0
    
    @test child2.genes[1] <= 10.0
    @test child2.genes[2] <= 20.0
    @test child2.genes[1] >= 1.0
    @test child2.genes[2] >= 2.0
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
    @test all(x -> x == 1,  newch.genes .<= upper)
    @test all(x -> x == 1,  newch.genes .>= lower) 
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



@testset "Pi and E with Kmeans clustering GA - with elitism" begin
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
         CLGA.GA_TYPE_CLUSTER, # cluster based selection
         elitism = 1
    )
    
    best = result[1]
    @info best
    
    @test best isa CLGA.Chromosome
    @test best.genes[1] > 2.0
    @test best.genes[1] < 5.0
    @test best.genes[2] > 1.0
    @test best.genes[2] < 4.0
end




@testset "(Hybrid) Pi and E with Kmeans clustering GA - with elitism - then classical ga with initial pop" begin
    function costfn(vals)
        return abs(vals[1] - 3.141592) + abs(vals[2] - 2.71828)
    end 
    crossfn = CLGA.makelinearcrossover(costfn)
    mutatefn = CLGA.makenormalmutation(10.0,  #stddev
                                      0.10   #mutation prob
    )
    result = CLGA.ga(
        100, # popsize
        100, # generations 
        [-100.0, -100.0], #lower
        [100.0, 100.0],   #upper
         costfn, #costfunction 
         crossfn, #crossoverfunction 
         mutatefn, #mutation function 
         CLGA.GA_TYPE_CLUSTER, # cluster based selection
         elitism = 1
    )
    
    result_classical = CLGA.ga(
        100, # popsize
        500, # generations 
        [-100.0, -100.0], #lower
        [100.0, 100.0],   #upper
         costfn, #costfunction 
         crossfn, #crossoverfunction 
         mutatefn, #mutation function 
         CLGA.GA_TYPE_CLASSIC, # classical FPGA
         elitism = 1,
         initialpopulation = result
    )
    best = result_classical[1]
    @info best
    
    @test best isa CLGA.Chromosome
    @test best.genes[1] > 2.0
    @test best.genes[1] < 5.0
    @test best.genes[2] > 1.0
    @test best.genes[2] < 4.0
end
