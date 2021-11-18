include("lib.jl")

println("Starting... Number of threads: $(Threads.nthreads())")

# SIMULATION SETTINGS
const popsize = 50
const generations = 100
const mutationParameter1 = 3.0
const mutationParameter2 = 0.01
const num_elitists = 2
const num_simulations = 1000
const num_variables = [2, 10, 25, 100]
const costfns = [fun_ackley, fun_griewank, fun_rastrigin, fun_schwefel]
const lowers = [lower_ackley, lower_griewank, lower_rastrigin, lower_schwefel]
const uppers = [upper_ackley, upper_griewank, upper_rastrigin, upper_schwefel]
const bests = [best_ackley, best_griewank, best_rastrigin, best_schwefel]
global resultSet = DataFrame()
# END

for myFun in 1:length(costfns), numV in num_variables, numS in 1:num_simulations
    
    @info "new iteration" costfns[myFun] numV numS

    # THE FUNCTIONS
    costfn = costfns[myFun]
    crossfn = CLGA.makelinearcrossover(costfn)
    mutatefn  = CLGA.makenormalmutation(mutationParameter1, mutationParameter2)
    # END

    # CLASSIC GA
    res_ga = CLGA.ga(
        popsize,
        generations,
        repeat([lowers[myFun]], outer=numV),      
        repeat([uppers[myFun]], outer=numV),      
        costfn,     
        crossfn,    
        mutatefn,   
        CLGA.GA_TYPE_CLASSIC, 
        elitism = num_elitists)

    # KMEANS
    res_kmeans = CLGA.ga(
        popsize,    
        generations,
        repeat([lowers[myFun]], outer=numV),      
        repeat([uppers[myFun]], outer=numV),      
        costfn,     
        crossfn,    
        mutatefn,   
        CLGA.GA_TYPE_CLUSTER, 
        elitism = num_elitists)

    # KMEANSSIM
    res_kmeanssim = CLGA.ga(
        popsize,    
        generations,
        repeat([lowers[myFun]], outer=numV),      
        repeat([uppers[myFun]], outer=numV),      
        costfn,     
        crossfn,    
        mutatefn,   
        CLGA.GA_TYPE_CLUSTER_SIM, 
        elitism = num_elitists)

    global resultSet = append!(resultSet, DataFrame(
        fun = myFun,
        popsize = popsize,
        generations = generations,
        num_elitists = num_elitists,
        numV = numV,
        numS = numS,
        res_ga = res_ga[1].cost,
        res_ga_norm = abs(bests[myFun] - res_ga[1].cost),
        res_kmeans = res_kmeans[1].cost,
        res_kmeans_norm = abs(bests[myFun] - res_kmeans[1].cost),
        res_kmeanssim = res_kmeanssim[1].cost,
        res_kmeanssim_norm = abs(bests[myFun] - res_kmeanssim[1].cost)
    ))

end

@info "Simulation is completed."

# SAVING RESULTS
@info "Saving results in resultSet.csv..."
CSV.write("resultSet.csv", resultSet, delim=";")


# STATISTICAL ANALYSIS
@info "Starting statistical analysis..."
global tempResultSet = DataFrame()
global statistics = DataFrame()
for myFun in 1:length(costfns), numV in num_variables

    global tempResultSet = filter(row -> row.fun == myFun && row.numV == numV, resultSet)

    global statistics = append!(statistics, DataFrame(
        fun = myFun,
        numV = numV,
        mean_ga = mean(tempResultSet[!,"res_ga"]),
        median_ga = median(tempResultSet[!,"res_ga"]),
        mean_kmeans = mean(tempResultSet[!,"res_kmeans"]),
        median_kmeans = median(tempResultSet[!,"res_kmeans"]),
        mean_kmeanssim = mean(tempResultSet[!,"res_kmeanssim"]),
        median_kmeanssim = median(tempResultSet[!,"res_kmeanssim"]),
        p_ga_kmeans = round(pvalue(MannWhitneyUTest(tempResultSet[!,"res_ga"], tempResultSet[!,"res_kmeans"])), digits=4),
        p_ga_kmeanssim = round(pvalue(MannWhitneyUTest(tempResultSet[!,"res_ga"], tempResultSet[!,"res_kmeanssim"])), digits=4),
        p_kmeans_kmeanssim = round(pvalue(MannWhitneyUTest(tempResultSet[!,"res_kmeans"], tempResultSet[!,"res_kmeanssim"])), digits=4)
    ))

end

# SAVING RESULTS
@info "Saving statistical results in statistics.csv..."
CSV.write("statistics.csv", statistics, delim=";")


@info "Done."