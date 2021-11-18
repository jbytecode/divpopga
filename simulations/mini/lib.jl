import DivPopGa.ClusteredGa as CLGA
using DataFrames
using CSV
using Statistics
using HypothesisTests
include("fun.jl")

function functionLoader(funName)
    return Symbol("fun_",funName)
end