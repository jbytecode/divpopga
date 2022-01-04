module ClusteredGa

using StatsBase
using Clustering

import Base.Threads

const GA_TYPE_CLUSTER = 0
const GA_TYPE_CLUSTER_SIM = 1
const GA_TYPE_CLASSIC = 2

mutable struct Chromosome 
    genes::Array{Float64, 1}
    cost::Float64
    clusterid::Int
end

# Create chromosome with given genes, cost of Infinity,
# and cluster id of -1.
function Chromosome(genes::Array{Float64, 1})
    return Chromosome(
        genes, 
        Inf64,
        -1
    )
end

function Chromosome(lower::Array{Float64, 1}, upper::Array{Float64, 1})
    genes = lower .+ rand(length(lower)) .* (upper .- lower)
    return Chromosome(genes)
end

function uniformcrossover(ch1::Chromosome, ch2::Chromosome)::Chromosome
    L = length(ch1.genes)
    offspring = Chromosome(zeros(Float64, L))
    for i in 1:L
        if rand() < 0.5
            offspring.genes[i] = ch1.genes[i]
        else
            offspring.genes[i] = ch2.genes[i]
        end
    end
    return offspring
end

function makeuniformcrossover()::Function 
    function tempfn(ch1::Chromosome, ch2::Chromosome)
        return uniformcrossover(ch1, ch2)
    end
    return tempfn 
end

function weightedcrossover(ch1::Chromosome, ch2::Chromosome)::Chromosome
    alpha = rand()
    genes = alpha * ch1.genes .+ (1.0 - alpha) * ch2.genes 
    return Chromosome(genes)
end

function makeweightedcrossover()::Function 
    function tempfn(ch1::Chromosome, ch2::Chromosome)
        return weightedcrossover(ch1, ch2)
    end
    return tempfn 
end

function linearcrossover(costfn::Function, ch1::Chromosome, ch2::Chromosome)::Chromosome
    genes1 = 0.5 .* ch1.genes .+ 0.5 .* ch2.genes 
    genes2 = 1.5 .* ch1.genes .- 0.5 .* ch2.genes 
    genes3 = 1.5 .* ch2.genes .- 0.5 .* ch1.genes 
    allgenes = [genes1, genes2, genes3]
    costs = map(costfn, allgenes)
    bestindice = costs |> sortperm |> first
    return Chromosome(
        allgenes[bestindice],
        costs[bestindice],
        -1
    )   
end

function makelinearcrossover(costfn::Function)::Function 
    function tmpfn(ch1::Chromosome, ch2::Chromosome)::Chromosome
        return linearcrossover(costfn, ch1, ch2)
    end
    return tmpfn
end

function blxalphacrossover(ch1::Chromosome, ch2::Chromosome)::Chromosome
    L = length(ch1.genes)
    offspring = Chromosome(zeros(Float64, L))
    for i in 1:L
        d = abs(ch1.genes[i] - ch2.genes[i])
        alpha = rand()
        ad = d * alpha 
        umin = min(ch1.genes[i], ch2.genes[i]) - ad
        umax = max(ch1.genes[i], ch2.genes[i]) + ad 
        u = umin + rand() * (umax - umin)
        offspring.genes[i] = u
    end
    return offspring
end

function makeblxalphacrossover()::Function 
    function tmpfn(ch1::Chromosome, ch2::Chromosome)::Chromosome
        return blxalphacrossover(ch1, ch2)
    end
    return tmpfn
end


function normalmutation(stddev::Float64, mutationprob::Float64, ch::Chromosome)::Chromosome
    newgenes = copy(ch.genes)
    for i in 1:length(length(newgenes))
        if rand() < mutationprob
            newgenes[i] += randn() * stddev
        end 
    end 
    return Chromosome(newgenes)
end

function makenormalmutation(stddev::Float64, mutationprob::Float64)::Function 
    function tmpfn(ch::Chromosome)::Chromosome 
        return normalmutation(stddev, mutationprob, ch)
    end
    return tmpfn 
end


function randommutation(lower::Array{Float64, 1}, upper::Array{Float64, 1}, mutationprob::Float64, ch::Chromosome)::Chromosome
    newgenes = copy(ch.genes)
    for i in 1:length(newgenes)
        if rand() < mutationprob
            newgenes[i] = lower[i] + rand() * (upper[i] - lower[i])
        end 
    end 
    return Chromosome(newgenes)
end

function makerandommutation(lower::Array{Float64, 1}, upper::Array{Float64, 1}, mutationprob::Float64)::Function
    function tmpfn(ch::Chromosome)::Chromosome 
        return randommutation(lower, upper, mutationprob, ch)
    end
end

function tournamentselection(pop::Array{Chromosome, 1}, k::Int)::Array{Chromosome, 1}
    fathers = sort(sample(pop, k, replace = false), by = ch -> ch.cost)
    mothers = sort(sample(pop, k, replace = false), by = ch -> ch.cost)
    return [fathers[1], mothers[1]]
end

function maketournamentselection(k::Int)::Function 
    function tmpfn(pop::Array{Chromosome, 1})::Array{Chromosome, 1}
        return tournamentselection(pop, k)
    end
end

function distance(ch1::Chromosome, ch2::Chromosome)::Float64
    (ch1.genes .- ch2.genes) .|> (x -> x * x) |> sum  
end

function simulatedkmeanstournamentselection(pop::Array{Chromosome, 1}, k::Int)::Array{Chromosome, 1}
    fathers = sort(sample(pop, k, replace = false), by = ch -> ch.cost)
    mothers = sort(sample(pop, k, replace = false), by = ch -> ch.cost)
    
    thefather = fathers[1]
    distances = map(ch -> distance(thefather, ch), mothers)
    furthestindice = distances |> sortperm |> last 
    themother = mothers[furthestindice]

    return [thefather, themother]
end

function makesimulatedkmeanstournamentselection(k::Int)::Function
    function tmpfn(pop::Array{Chromosome, 1})::Array{Chromosome, 1}
        return simulatedkmeanstournamentselection(pop, k)
    end
end


function kmeanstournamentselection(pop::Array{Chromosome, 1}, k::Int)::Array{Chromosome, 1}
    father = sort(sample(pop, k, replace = false), by = ch -> ch.cost)[1]
    mothers = filter(ch -> ch.clusterid != father.clusterid, pop)
    if length(mothers) == 0
        mothers = pop
    end
    mother = sort(sample(mothers, k, replace = true), by = ch -> ch.cost)[1]
    return [father, mother]
end

function makekmeanstournamentselection(k::Int)::Function
    function tmpfn(pop::Array{Chromosome, 1})::Array{Chromosome, 1}
        return kmeanstournamentselection(pop, k)
    end
end

function randompopulation(popsize::Int, lower::Array{Float64, 1}, upper::Array{Float64, 1})::Array{Chromosome, 1}
    population = Array{Chromosome, 1}(undef, popsize)
    for i in 1:popsize
        population[i] = Chromosome(lower, upper)
    end
    return population
end

function calculatefitness(population::Array{Chromosome, 1}, costfn::Function)
    Threads.@threads for ch in population
        ch.cost = costfn(ch.genes)
    end
end

function population2matrix(population::Array{Chromosome, 1})::Array{Float64, 2}
    popsize = length(population)
    chsize = length(population[1].genes)
    mat = Array{Float64, 2}(undef, popsize, chsize)
    for i in 1:popsize
        mat[i,:] = population[i].genes
    end
    return mat
end

function assignclusterids(population::Array{Chromosome, 1})
    mat = population2matrix(population)
    n, _ = size(mat)
    k = Int(ceil(sqrt(n)))
    clust::ClusteringResult = kmeans(transpose(mat), k)
    clusterids = clust.assignments 
    for i in 1:n
        population[i].clusterid = clusterids[i]
    end
end

function generation(
    population::Array{Chromosome, 1}, costfn::Function, crossfn::Function, 
    mutatefn::Function, gatype::Int; elitism::Int = 0)::Array{Chromosome, 1}

    popsize = length(population)
    calculatefitness(population, costfn)
    if gatype == GA_TYPE_CLUSTER
        assignclusterids(population)
        selectfn = makekmeanstournamentselection(3)
    elseif  gatype == GA_TYPE_CLASSIC
        selectfn = maketournamentselection(3)
    elseif gatype == GA_TYPE_CLUSTER_SIM
        selectfn = makesimulatedkmeanstournamentselection(3)
    end
    newpop = Array{Chromosome, 1}(undef, 0)
    
    if elitism > 0
        sort!(population, by = ch -> ch.cost)
        for i in 1:elitism
            push!(newpop, population[i])
        end    
    end
    
    for _ in (elitism + 1):popsize 
        father, mother = selectfn(population)
        offspring = mutatefn(crossfn(father, mother))
        push!(newpop, offspring)
    end
    return newpop
end

function ga(
    popsize::Int, generations::Int, lower::Array{Float64, 1}, upper::Array{Float64, 1},
    costfn::Function, crossfn::Function, mutatefn::Function, gatype::Int; elitism::Int = 0, initialpopulation = nothing)

    if isnothing(initialpopulation)
        population = randompopulation(popsize, lower, upper)
    else
        population = initialpopulation
    end 
    
    for _ in 1:generations 
        population = generation(population, costfn, crossfn, mutatefn, gatype, elitism = elitism)
    end 

    calculatefitness(population, costfn)
    sort!(population, by = ch -> ch.cost)
    return population
end



function hybridga(popsize::Int, generations::Array{Int, 1}, lower::Array{Float64, 1}, upper::Array{Float64, 1},
    costfn::Function, crossfn::Function, mutatefn::Function; elitism::Int = 0)

    population = randompopulation(popsize, lower, upper)

    for _ in 1:generations[1] 
        population = generation(population, costfn, crossfn, mutatefn, GA_TYPE_CLUSTER, elitism = elitism)
    end 

    for _ in 1:generations[2] 
        population = generation(population, costfn, crossfn, mutatefn, GA_TYPE_CLASSIC, elitism = elitism)
    end 

    calculatefitness(population, costfn)
    sort!(population, by = ch -> ch.cost)
    return population
end



end # end of module ClusteredGa
