# SCHWEFEL
# https://www.sfu.ca/~ssurjano/schwef.html
# [-500, 500]
# 420.9687* => 0
const lower_schwefel = -500.0
const upper_schwefel = 500.0
const best_schwefel = 0.0
function fun_schwefel(xx::Vector{Float64})
    d = length(xx)
	
    sum = 0
    for i in 1:d
        sum = sum + xx[i]*sin(sqrt(abs(xx[i])))
    end
  
    y = 418.9829*d - sum
    return(y)
end

# RASTRIGIN
# https://www.sfu.ca/~ssurjano/rastr.html
# [-5.12, 5.12]
# 0* => 0
const lower_rastrigin = -5.12
const upper_rastrigin = 5.12
const best_rastrigin = 0.0
function fun_rastrigin(xx)
    d = length(xx)
	
    sum = 0
    for i in 1:d
        sum = sum + xx[i]*xx[i] - 10 * cos(2 * pi * xx[i])
    end
      
    y = 10*d + sum
    return(y)
end

# GRIEWANK
# https://www.sfu.ca/~ssurjano/griewank.html
# [-600, 600]
# 0* => 0
const lower_griewank = -600.0
const upper_griewank = 600.0
const best_griewank = 0.0
function fun_griewank(xx)
    d = length(xx)
    ii = 1:d
    
    sum = 0
    for i in 1:d
        sum = sum + (xx[i]*xx[i]) / 4000
    end

    prod = 1
    for i in 1:d
        prod = prod * cos(xx[i]/sqrt(i))
    end
          
    y = sum - prod + 1
    return(y)
end

# ACKLEY
# https://www.sfu.ca/~ssurjano/ackley.html
# [-32.768, 32.768]
# 0* => 0
const lower_ackley = -32.768
const upper_ackley = 32.768
const best_ackley = 0.0
function fun_ackley(xx)
    d = length(xx)
    a = 20
    b = 0.2
    c = 2 * pi
  
    sum1 = 0
    for i in 1:d
        sum1 = sum1 + xx[i]^2
    end

    sum2 = 0
    for i in 1:d
        sum2 = sum2 + cos(c * xx[i])
    end
  
    term1 = -a * exp(-b*sqrt(sum1/d))
    term2 = -exp(sum2/d)
  
    y = term1 + term2 + a + exp(1)

    return(y)
end