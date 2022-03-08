struct Connection 
    from
    to
    dist
end

function euclidean(v, u)
    v .- u .|> (x -> x * x) |> sum |> sqrt
end

function getnearest(datapoints, linked, unlinked)
    mindist = Inf 
    from = nothing 
    to = nothing 
    for i in linked
        for j in unlinked
            dist = euclidean(datapoints[i,:], datapoints[j,:])
            if dist < mindist 
                mindist = dist 
                from = i
                to = j
            end
        end
    end 
    return Connection(from, to, mindist)
end

function mst(datapoints)
    result = []
    n, p = size(datapoints)
    allindices = collect(1:n)
    linkedset = [1]

    while true
        unlinkedset = setdiff(allindices, linkedset)
        if length(unlinkedset) == 0
            break
        end
        conn = getnearest(datapoints, linkedset, unlinkedset)
        push!(result, conn)
        push!(linkedset, conn.to) 
    end 
    return result
end

function cutree(tree)
    n = length(tree)
    maxdist = maximum(x -> x.dist, tree)
    part1 = []
    part2 = []
    i = 1
    while i < n
        if tree[i].dist < maxdist
            push!(part1, tree[i])
        else 
            break
        end
        i += 1
    end 
    
    for j = (i+1):n
        push!(part2, tree[j])
    end
    return (part1, part2)
end