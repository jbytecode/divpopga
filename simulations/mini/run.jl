import DivPopGa.ClusteredGa as CLGA
using DataFrames
using CSV
include("lib.jl")

# SIMULATION SETTINGS
popsize = 50
generation = 100
mutationParameter1 = 3.0
mutationParameter2 = 0.01
num_elitists = 2
num_simulations = 1000
num_variables = [2, 10, 25, 100]
costfns = ["ackley", "griewank", "rastrigin", "schwefel"]
global resultSet = DataFrame()
# END

for myFun in costfns, numV in num_variables, numS in 1:num_simulations
    
    println("--> FUN:",myFun," VAR:",numV," SIM:",numS)

    # THE FUNCTIONS
    costfn = functionLoader(myFun)
    costfn = @eval $costfn
    crossfn = CLGA.makelinearcrossover(costfn)
    mutatefn  = CLGA.makenormalmutation(mutationParameter1, mutationParameter2)
    # END

    # CLASSIC GA
    res_ga = CLGA.ga(
        popsize,
        generations,
        lower,      
        upper,      
        costfn,     
        crossfn,    
        mutatefn,   
        CLGA.GA_TYPE_CLASSIC, 
        elitism = num_elitists)

    # KMEANS
    res_kmeans = CLGA.ga(
        popsize,    
        generations,
        lower,      
        upper,      
        costfn,     
        crossfn,    
        mutatefn,   
        CLGA.GA_TYPE_CLUSTER, 
        elitism = num_elitists)

    # KMEANSSIM
    res_kmeanssim = CLGA.ga(
        popsize,    
        generations,
        lower,      
        upper,      
        costfn,     
        crossfn,    
        mutatefn,   
        CLGA.GA_TYPE_CLUSTER_SIM, 
        elitism = num_elitists)

    global resultSet = append!(resultSet, DataFrame(
        fun = myFun,
        popsize = popsize,
        generation = generation,
        num_elitists = num_elitists,
        numV = numV,
        numS = numS,
        res_ga = res_ga,
        res_kmeans = res_kmeans,
        res_kmeanssim = res_kmeanssim
    ))

end

CSV.write("test.csv", resultSet)