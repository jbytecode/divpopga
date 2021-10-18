module ClusteredGa

using StatsBase
using Clustering

mutable struct Chromosome 
    genes::Array{Float64, 1}
    cost::Float64
    clusterid::Int
end

function Chromosome(lower::Array{Float64, 1}, upper::Array{Float64, 1})
    genes = lower .+ rand(length(lower)) .* (upper .- lower)
    return Chromosome(
        genes, 
        Inf64,
        -1
    )
end

function linearcrossover(costfn::Function, ch1::Chromosome, ch2::Chromosome)::Chromosome
    genes1 = 0.5 * ch1.genes .+ 0.5 * ch2.genes 
    genes2 = 1.5 * ch1.genes .- 0.5 * ch2.genes 
    genes3 = 1.5 * ch2.genes .- 0.5 * ch1.genes 
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

function randommutation(stddev::Float64, mutationprob::Float64, ch::Chromosome)::Chromosome
    newgenes = copy(ch.genes)
    for i in 1:length(length(newgenes))
        if rand() < mutationprob
            newgenes[i] += randn() * stddev
        end 
    end 
    return Chromosome(
        newgenes, 
        Inf64,
        -1
    )
end

function makerandommutation(stddev::Float64, mutationprob::Float64)::Function 
    function tmpfn(ch::Chromosome)::Chromosome 
        return randommutation(stddev, mutationprob, ch)
    end
    return tmpfn 
end

#=
Takes an array of chromosomes
and returns two chromosomes
=#
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

function kmeanstournamentselection(pop::Array{Chromosome, 1}, k::Int)::Array{Chromosome, 1}
    father = sort(sample(pop, k, replace = false), by = ch -> ch.cost)[1]
    mothers = filter(ch -> ch.clusterid != father.clusterid, pop)
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
    for ch in population
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
    mutatefn::Function, selectfn::Function)::Array{Chromosome, 1}

    popsize = length(population)
    calculatefitness(population, costfn)
    assignclusterids(population)
    newpop = Array{Chromosome, 1}(undef, 0)
    for i in 1:popsize 
        father, mother = selectfn(population)
        offspring = mutatefn(crossfn(father, mother))
        push!(newpop, offspring)
    end
    return newpop
end

function ga(
    popsize::Int, generations::Int, lower::Array{Float64, 1}, upper::Array{Float64, 1},
    costfn::Function, crossfn::Function, mutatefn::Function, selectfn::Function)

    population = randompopulation(popsize, lower, upper)
    
    for iter in 1:generations 
        population = generation(population, costfn, crossfn, mutatefn, selectfn)
    end 

    calculatefitness(population, costfn)
    sort!(population, by = ch -> ch.cost)
    return population
end

end # end of module ClusteredGa