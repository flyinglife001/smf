module util

using Random
using Printf

function format_float(x)
    return @sprintf("%.2f",x)
end

function toint(s)
    return parse(Int64,s)
end

function tofloat(s)
    return parse(Float64,s)
end

function parse_command()
    dict = Dict{String,Any}()
    for i = 1:2:length(ARGS)
        dict[ARGS[i]] = ARGS[i + 1]
    end
    return dict
end


function read_yaml(fname)
    dict = Dict{String,Any}()

    array_reg = r"\[[^\]]+\]"
    for line in eachline(fname)
        idx = findfirst(isequal('#'),line)
        if idx != nothing
             line = line[1:idx - 1]
        end
        #println(line)
        if length(line) == 0
            continue
        end
        #println(line)
        items = split(line,':')
        key = strip(items[1])
        value = strip(items[2])
        if (v = tryparse(Int64,value)) != nothing
            #println("int ",v)
            dict[key] = v
        elseif (v = tryparse(Float64,value)) != nothing
            #println("float ",v)
            dict[key] = v
        elseif value[1] == '[' # parse the array
            array_values = split(value[2:end-1],',')
            dict[key] = map(x -> tryparse(Int64, x) != nothing ? parse(Int64,x) : parse(Float64,x), array_values)
            #println(dict[key])
        else
            dict[key] = String(value)
        end
    end
    #println(dict)
    return dict
end

#=config: dict =#
function random_search(config)
    cfg_list = Array{Any,1}()
    for i in 1:config["total_iter"]
        cfg = Dict{String,Any}()
        for kv in config
            # check if it is abstract array.
            if isa(kv[2],AbstractArray)
                #println(kv[1]," ",kv[2])
                cfg[kv[1]] = rand(kv[2]) # take from set randomly
            elseif isa(kv[2],String)
                items = split(kv[2],'$')
                if length(items) == 3
                    if items[1] == "float"
                        a = parse(Float64,items[2])
                        b = parse(Float64,items[3])
                        cfg[kv[1]] = a + (b - a)*rand() # generate [0,1) number.
                    elseif items[1] == "log"
                        a = parse(Float64,items[2])
                        b = parse(Float64,items[3])
                        c = log(a) + (log(b) - log(a))*rand()
                        cfg[kv[1]] = exp(c)
                    else
                        a = parse(Int64,items[2])
                        b = parse(Int64,items[3])
                        cfg[kv[1]] = rand(a:b)
                    end
                else
                    cfg[kv[1]] = items[1]
                end
            else
                cfg[kv[1]] = kv[2]
            end
        end
        #println(cfg)
        push!(cfg_list, cfg)
    end

    return cfg_list
end

end


#util.parse_command()
#Random.seed!(19)
#config = read_yaml("./matrixcompletion/cub-parts-pooling.yaml")
#random_search(config)
