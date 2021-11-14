include("fun.jl")

function functionLoader(funName)
    return Symbol("fun_",funName)
end