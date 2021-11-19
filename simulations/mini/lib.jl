import DivPopGa.ClusteredGa as CLGA
using DataFrames
using CSV
using Statistics
using HypothesisTests
using RCall

include("fun.jl")

function functionLoader(funName)
    return Symbol("fun_",funName)
end


# x: vector
# y: vector 
# alternative: String "less", "greater", or two.sided
function mannwhitney(x, y; alternative = "two.sided")
    @rput x
    @rput y
    if !(alternative in ["less", "greater", "two.sided"])
        @error "alternative should be one of the 'less', 'greater', or 'two.sided'"
        return -1 
    end
    result = R"wilcox.test(x, y, paired = FALSE, alternative = $(alternative))"
    pvalue = result[3]
    return pvalue
end

x = [1.0, 2, 3, 4, 5, 6, 7]
y = [5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0]

pval = mannwhitney(x, y, alternative = "less")
print(pval)
