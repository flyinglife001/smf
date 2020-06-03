using Distributed
#push!(LOAD_PATH, "./matrixcompletion/") # it is better to add current path to LOAD_PATH.
using LinearAlgebra
using Statistics
using CSV
using SparseArrays
using Random

# use the local moduleG
include("./util.jl")
using .util

using ChangePrecision
using StaticArrays
using Logging
using JSON
using Dates
using Base.Threads
using SharedArrays

#=
cross_fold n into 5
=#
function cross_fold(n, n_folds)
    # note that 1:n is UintRange{Int64}
    #println(typeof(1:10))
    A = Vector{Int64}(1:n)
    shuffle!(A)
    folds = Vector{Any}(undef, n_folds)
    # note that 1:n_folds represent a range.
    for k in 1:n_folds
        train = Vector{Int64}()
        test = Vector{Int64}()
        for i in 1:n
            if i%n_folds == k - 1
                push!(test,A[i])
            else
                push!(train,A[i])
            end
        end
        folds[k] = (train,test)
        #println(folds[k])
    end
    return folds
end


# passing by sharing, so need not to copy
function normalize_input(train, valid)
    mean_value = mean(train)
    std_value = 1e-6 + std(train,mean=mean_value)

    train = (train .- mean_value) ./ std_value
    valid = (valid .- mean_value) ./ std_value

    return train, valid, mean_value, std_value
end

function denormalize_input(test, mean_value, std_value)
    test = mean_value .+ test .* std_value
    return test
end


function read_csv(train_name, test_name, delim, n,m)
    println("load $train_name $test_name row:$n col:$m")
    # delim: union of char and string.
    data = CSV.read(train_name, delim = delim)
    # view function is to avoid to copy
    row = data[:,1]
    col = data[:,2]
    value = data[:,3]

    #println(typeof(value))

    #X = sparse(row,col,value)
    #@show(size(X))

    X = sparse(row,col,value,n,m)

    #println(typeof(X))

    # split the dataset into train and valid dataset.
    cnt = length(row)
    dataset_idx = cross_fold(cnt,5)

    train_row = map(x -> row[x], dataset_idx[1][1])
    train_col = map(x -> col[x], dataset_idx[1][1])
    train_value = map(x -> value[x], dataset_idx[1][1])
    valid_row = map(x -> row[x], dataset_idx[1][2])
    valid_col = map(x -> col[x], dataset_idx[1][2])
    valid_value = map(x -> value[x], dataset_idx[1][2])

    train = sparse(train_row, train_col, train_value, n, m)
    v = length(train_row)
    train_pat = sparse(train_row, train_col, ones(v), n, m)
    #println("train ",typeof(train))

    valid = sparse(valid_row, valid_col, valid_value, n, m)
    v = length(valid_row)
    valid_pat = sparse(valid_row, valid_col, ones(v), n, m)

    data = CSV.read(test_name, delim = delim)
    # view function is to avoid to copy
    row = data[:,1]
    col = data[:,2]
    value = data[:,3]

    test = sparse(row,col,value, n, m)
    v = length(row)
    test_pat = sparse(row,col,ones(v),n,m)

    #println("value ",typeof(value))
    #println(size(train),size(valid))
    return (train,train_pat),(valid,valid_pat),(test,test_pat)
end

#=
compute the rmse: √\|(P - O)_{Ω}\|_{F}^2.
=#
@everywhere function RMSE(target, pred)
    #n,m = size(target)
    #coeff = n*m
    I,J,V = findnz(target)
    return sqrt(norm(pred - target)^2/(length(I)))
end

#=
compute the entropy of the matrix
=#
function norm_dist(mat)
    m,n = size(mat)
    return map(i -> norm(mat[i,:]), 1:m)
end

#=
\min_{M} = \frac{1}{2}\|(H - M)_{Ω}\|_{F} + λ \|M\|_{*}
=#
function soft_impute(train, valid, config)
    println("begin to soft impute method ...")

    start = time()

    λ = config["lambda"]
    epoches = config["epoches"]
    Random.seed!(config["seed"])

    X = train[1]
    n,m = size(X)
    M = rand(n,m) # initinize the M with zeros.
    #println(typeof(M))
    epoch = 1
    pattern = train[2]
    orth_pattern = ones(n,m) - pattern
    valid_rmse = 0.0
    #ProgressMeter.@showprogress 1 "" for epoch = 1:epoches
    loss = train_rmse = valid_rmse = eclipse = cur_epsilon = 0

    for epoch = 1:epoches
        N = M
        X_bar = X + M.*orth_pattern # element-wise multiplication.
        #println("xbar ",typeof(X_bar))
        U,D,V = svd(X_bar)
        E = map(x -> x > λ ? x : zero(x),D) # (d_{ii} - λ)_{+}
        #println(typeof(E))
        M = U*Diagonal(E)*V' # diagonal is create the diagonal.

        #=norm compute the norm by a fashion of vector
        but opnorm computer the norm by the mean of matrix.=#
        loss = 1/2*norm(X - M.*pattern)^2 + λ*opnorm(M,2)
        train_rmse = RMSE(X, M.*pattern)
        valid_rmse = RMSE(valid[1],M.*valid[2])
        cur_epsilon = norm(M - N)/norm(N)

        eclipse = time() - start


        config["loss"] = loss
        config["train"] = train_rmse
        config["valid"] = valid_rmse
        config["time"] = eclipse
        config["error"] = cur_epsilon
        config["iter"] = epoch
        println(JSON.json(config))

    end
    return M
end

#
#\min_{M} = \frac{1}{2}\|(X - AB)_{Ω}\|_{F} + λ (\|A\|_{F} + \|B\|_{F})
#
# it really is softimpute-als algorithm.
function svt(train, valid, config)
    println("begin to svt method ...")
    start = time()
    λ = config["lambda"]
    k = util.toint(config["k"])
    epoches = config["epoches"]

    Random.seed!(config["seed"])

    X = train[1]
    pattern = train[2]
    m,n = size(train[1])
    #@show m,n
    #initalize U
    U = rand(m,k)
    # randomly initialize the orth column.
    U,Σ,V = svd(U)
    D = I # convert it to matrix
    V = rand(n,k)
    A = U*D; B = V*D
    #@show size(A),size(B)
    M = Matrix{Float64}(undef,m,n)
    valid_rmse = 0.0

    loss = train_rmse = valid_rmse = eclipse = cur_epsilon = 0
    for epoch = 1:epoches
        #update B.
        AtB = A*B'
        Xbar = (X - AtB).*pattern + AtB
        #@show size(Xbar)
        D2I = D*D + λ*I
        DUX = D*U'*Xbar
        Bt = D2I\DUX
        V,d,R = svd(Bt'*D)
        #@show typeof(D)
        D = Diagonal(sqrt.(d))
        #@show size(D),size(V)
        B = V*D

        #update A.o
        BAt = B*A'
        Xbart = (X' - BAt).*(pattern') + BAt
        #@show size(Xbart)
        D2I = D.^2 + λ*I
        DUX = D*V'*Xbart
        At = D2I\DUX
        U,d,R = svd(At'*D)
        D = Diagonal(sqrt.(d))
        A = U*D
        #@show size(A),size(B)

        #@show size(Xbart),size(V)
        N = Xbart'*V
        U,d,R = svd(N)
        d = map(x -> x > λ ? x - λ : 0.0, d)
        M = Matrix(U*Diagonal(d)*((V*R)'))

        loss = norm((X - M).*pattern)^2 + λ*(norm(A)^2 + norm(B)^2)
        #@show size(X),size(M),size(pattern)
        train_rmse = RMSE(X, M.*pattern)
        valid_rmse = RMSE(valid[1],M.*valid[2])
        cur_epsilon = norm(A*B' - AtB)/norm(AtB)

        eclipse = time() - start

        config["normA"] = norm_dist(A')
        config["normB"] = norm_dist(B')
        config["loss"] = loss
        config["train"] = train_rmse
        config["valid"] = valid_rmse
        config["time"] = eclipse
        config["error"] = cur_epsilon
        config["iter"] = epoch
        println(JSON.json(config))
    end
    return M
end

#######################################################################################
#  Nie at al.  Efficient and Robust Feature Selection via
#  Joint l2,1-Norms Minimization.
#   \min_W \|X^* - A W^T\|_{2,1} + λ \|W\|_{2,1}
#

@everywhere function iterative_l21(X,A,config)
    #println("use the new iteration")
    #U = hcat(A', λ*I)
    c,n = size(X)
    r,m = size(A)

    #@show c,n

    R1 = I # D^{-1}
    R2 = I
    epoch = 1
    V1 = zeros(r,m)
    V2 = zeros(m,m)
    λ = config["lambda"]
    epsilon = config["epsilon"]

    pre_obj = 1e12
    while true

        #Y = pinv(A'*R1*A + λ * λ * R2)*X
        # (U R U^t)
        # (U R U^t)^{-1} X^*
        # V = R U^t (U R U^t)^{-1} X
        Y = (A'*R1*A + λ * λ * R2)\X

        V1 = R1*A*Y
        V2 = λ*R2*Y

        # R_ii = 2\|v^i\|_2

        R1 = Diagonal(map(i -> 2*norm(V1[i,:]),1:r))
        R2 = Diagonal(map(i -> 2*norm(V2[i,:]),1:m))

        # compute the l21-norm
        cur_obj = sum(tr(R1)) + sum(tr(R2))

        #println("$epoch epoch loss: $cur_obj")

        if abs(cur_obj - pre_obj) < epsilon
            break
        end

        epoch += 1

        pre_obj = cur_obj
    end

    return V1
end


#
#\min_{M} = \frac{1}{2}\|(X - A^T B)_{Ω}\|_{F} + λ (\|A\|_{2,1} + \|B\|_{2,1})
#
@everywhere function fnorm_iteration(AX, AA,config)
    λ = config["lambda"]
    epsilon = config["epsilon"]
    m,n = size(AX)
    D = I

    B = zeros(m,n)

    epoches = 1
    last_loss = 0
    while true
        B = (AA + λ*D)\AX

        #  D_{ii} = 1/(2*|B|_2)
        D = Diagonal(map(i -> 1.0/(2*norm(B[i,:])),1:m))
        #  B = (AA^T + λ D )^{-1}AX

        # loss = X^t X - 2 tr(B^T AX) + tr(B B^T AA) + lambda|B\|_{2,1}
        loss = -2*tr(B' * AX) + tr(B'*AA*B) + λ*sum(map(i-> norm(B[i,:]),1:m))

        #println("epoch $epoches losses: $loss")

        if norm(last_loss - loss) < 1e-4
            break
        end

        epoches += 1
        last_loss = loss

    end

    return B
end


@everywhere function norm_mc(train, valid, config, is_nmc)
    if is_nmc
        println("begin to norm mc method ...")
    else
        println("begin to factor mc method ...")
    end

    #rng = MersenneTwister(config["seed"])
    Random.seed!(config["seed"])

    start = time()
    λ = config["lambda"]
    epoches = config["epoches"]
    ϵ = config["epsilon"] = util.tofloat(config["epsilon"])
    k = config["k"] = util.toint(config["k"])

    X = train[1]
    pattern = train[2]
    m,n = size(train[1])
    #println("begint robust mc $m $n")

    M = zeros(m,n) # initinize the M with rand.
    Mbar = zeros(m,n) # initinize the M with rand.
    valid_rmse = 0.0

    loss = train_rmse = valid_rmse = eclipse = cur_epsilon = sq_loss = 0
    A = rand(k,m)
    B = rand(k,n)
    AtB = zeros(m,n) # initinize the M with zeros.
    for epoch = 1:epoches
        M = A'*B
        Xbar = (X - M).*pattern + M

        if is_nmc
            AX = A*Xbar
            AA = A*A'
            B = fnorm_iteration(AX,AA,config)
        else
            #
            #\min_{M} = \frac{1}{2}\|(X - AB)_{Ω}\|_{2,1} + λ (\|A\|_{2,1} + \|B\|_{2,1})
            #
            B = iterative_l21(Xbar,A,config)
        end

        # update the Xbar
        AtB = A'*B
        Xbar = (X - AtB).*pattern + AtB

        if is_nmc
            AX = B*Xbar'
            AA = B*B'
            A = fnorm_iteration(AX,AA,config)
        else
            A = iterative_l21(Xbar',B,config)
        end

        Mbar = A'*B

        if is_nmc
             sq_loss = 0.0
             loss = norm((X - Mbar).*pattern)^2 + λ*(sum(map(i-> norm(A[i,:]),1:k)) + sum(map(i-> norm(B[i,:]),1:k)))
        else
             sq_loss = norm((X - Mbar).*pattern) + λ*(sum(map(i-> norm(A[i,:]),1:k)) + sum(map(i-> norm(B[i,:]),1:k)))
             loss = norm((X - Mbar).*pattern)^2 + λ*(sum(map(i-> norm(A[i,:]),1:k)) + sum(map(i-> norm(B[i,:]),1:k)))
         end


        #@show size(X),size(M),size(pattern)
        train_rmse = RMSE(X, Mbar.*pattern)
        valid_rmse = RMSE(valid[1],Mbar.*valid[2])
        cur_epsilon = (norm(M - Mbar) + 1e-6)/(norm(M) + 1e-6)
        #cur_epsilon = norm(M - Mbar)

        eclipse = time() - start

        config["iter"] = epoch
        config["normA"] = norm_dist(A)
        config["normB"] = norm_dist(B)
        config["loss"] = loss
        config["train"] = train_rmse
        config["valid"] = valid_rmse
        config["time"] = eclipse
        config["error"] = cur_epsilon
        config["sq_loss"] = sq_loss

        println(JSON.json(config))
    end

#     config["loss"] = loss
#     config["train"] = train_rmse
#     config["valid"] = valid_rmse
#     config["time"] = eclipse
#     config["error"] = cur_epsilon
#     config["sq_loss"] = sq_loss

    return M
end


#########################################
# Fixed point and Bregman iteratiive methods for matrix rank
# minimization.  2009
###############################################################

# A is index of elements.
function AMVMx(A,x)
    return x[A]
end

function AtMVMx(A, b, m, n)
    y = zeros(m*n, 1)
    y[A] = b
    return reshape(y,m,n)
end

function gradient(x, m, n, A, Atb)
    #@show size(A)
    #@show size(x),m,n
    Ax = AMVMx(A,reshape(x, m*n, 1))
    return AtMVMx(A, Ax, m, n) - Atb
end

function linearSVD(A, c, k, p)
    m, n = size(A)
    # note if type is no specified, it will return float.
    c = round(Int,c); k = round(Int,k);
    C = zeros(m,c)
    bp = zeros(n,1)
    bp[1] = p[1]
    for j = 2:n
        bp[j] = bp[j - 1] + p[j]
    end

    for t = 1:c
        rr = rand()
        ind = findall(x -> x >= rr, bp)
        it = ind[1]
        C[:,t] = A[:,it]/sqrt(c*p[it])
    end
    U, S, V = svd(C'*C)
    sigma = sqrt.(S)
    sigma = sigma[1:k]
    H = zeros(m, k)
    for t = 1:k
        h = C*U[:, t]
        nh = norm(h)
        if nh == 0
            H[:, t] = zeros(size(h))
        else
            H[:, t] = h/nh
        end
    end

    return H, sigma
end

function fpca_computation(train, valid, config)
    println("begin to fpca method ...")

    start = time()
    λ = config["lambda"]
    k = util.toint(config["k"])
    epoches = config["epoches"]

    Random.seed!(config["seed"])

    #ϵ = config["epsilon"]
    valid_rmse = 0.0

    X = train[1]
    pattern = train[2]
    m,n = size(train[1])

    sr = 0.5; p = round(Int, m*n*sr);
    fr = k*(m + n - k)/p; maxr = round(Int, (m + n - sqrt((m + n)^2 - 4*p))/2);
    #@show fr,maxr
    if sr <= 0.5*fr || fr >= 0.38
        hard = 1
        println("This is hard problem!")
    else
        hard = 0;
        println("This is an easy problem!");
    end


    if hard == 1 && max(m,n) < 1000
        mu = 1e-8; xtol = 1e-6; maxinneriter = 500; tau = 1.0;
    else
        mu = 1e-4; xtol = 1e-6; maxinneriter = 10; tau = 2.0;
    end

    fastsvd_ratio_leading = 1e-2
    mxitr = epoches;  eta = 1.0/4.0;  print = 0.0;

    #convert the matrix into a vector.
    Xvec = reshape(X, m*n, 1);
    A = reshape(pattern, m*n, 1)
    #find the non-zero elements.
    Aindex = findall(x -> x > 0.0, A)
    b = Xvec[Aindex]
    Atb = AtMVMx(Aindex, b, m, n)
    x0 = zeros(m, n)

    if max(m, n) > 1000
        U, sigma = linearSVD(Atb, min(m/2, n/2, 1000),
                                  min(m/2, n/2, 3),
                                  ones(n,1)/n)
        nrm2Atb = sigma[1]
    else
        nrm2Atb = norm(Atb)
    end

    #@show nrm2Atb

    muf = mu; x = x0; mu = nrm2Atb*eta
    if mu < muf
        mu = muf
    end

    innercount = 0; nrmxxp = Inf; count_nrmxxp_increase = 0;
    extra_rank = 0; pp = ones(n, 1)/n
    sn = min(m/2, n/2, round(Int, 2*maxr - 2))
    g = gradient(x, m, n, Aindex, Atb)

    s = nothing
    xp_good = false

    loss = train_rmse = valid_rmse = eclipse = cur_epsilon = 0

    for i = 1:mxitr
        xp = x; nrmxxpp = nrmxxp;
        y = x - tau*g
        if i == 1
            xp_good = false
            if max(m, n) < 1000
                U,S,V = svd(y)
                sigma = S
            else
                U,sigma = linearSVD(y, sn, maxr, pp)
                d = map(x -> abs(x) > 0.0 ? 1.0/x : 0.0, sigma)
                V = Matrix((Diagonal(d)*U'*y)')
            end
        else
            sp = s[s .> 0.0]
            #@show typeof(sp)
            mx = maximum(sp)
            kk = length(findall(x -> x > mx*fastsvd_ratio_leading, sp))
            kk = max(1, min(kk + extra_rank, sn))
            U, sigma = linearSVD(y, sn, kk, pp)
            d = map(x -> abs(x) > 0 ? 1.0/x : 0.0, sigma)
            V = Matrix((Diagonal(d)*U'*y)')
        end

        s = sigma
        #@show s

        ind = findall(x -> x > 0, s);
        Ue = U[:, ind]; Ve = V[:,ind]; s = s[ind];
        nu = tau*mu;
        s = map(x -> x > nu ? x - nu : 0.0, s)
        x = Ue*Diagonal(s)*Ve'
        g = gradient(x,m,n,Aindex,Atb)

        nrmxxp = norm(x - xp)

        if xp_good
            if nrmxxp > nrmxxpp
                count_nrmxxp_increase += 1
            end
            if count_nrmxxp_increase >= 10
                extra_rank = 1
                count_nrmxxp_increase = 0
            else
                extra_rank = 0
            end
        end

        xp_good = true
        critx = nrmxxp/max(norm(xp),1.0); innercount += 1
        if mu == muf
            maxinneriter = 500
        end

        if (critx < xtol) ||  (innercount >= maxinneriter)
            innercount = 0; xp_good = false
            if mu == muf
                outx = x; outiter = i
                break
            end
            mu = eta*mu
            mu = max(mu, muf)
        end

        train_rmse = RMSE(X, x.*pattern)
        valid_rmse = RMSE(valid[1],x.*valid[2])
        eclipse = time() - start

        cur_epsilon = norm(x - xp)/norm(xp)


        config["loss"] = loss
        config["train"] = train_rmse
        config["valid"] = valid_rmse
        config["time"] = eclipse
        config["error"] = cur_epsilon
        config["iter"] = i
        println(JSON.json(config))
    end

    outx = x; outiter = mxitr
    return x
end


function main()
    config = util.parse_command()
    #@show config

    config["r"] = util.toint(config["r"])
    config["c"] = util.toint(config["c"])
    config["seed"] = util.toint(config["seed"])
    config["lambda"] = util.tofloat(config["lambda"])
    config["epoches"] = util.toint(config["epoches"])
    config["normalize"] = util.toint(config["normalize"])

    train,valid,test = read_csv(config["train"], config["test"],
                    config["delimiter"], config["r"], config["c"])

    Random.seed!(config["seed"])
    lambda = config["lambda"]

    if config["normalize"] == 1
        train_input,valid_input,min_value, interval = normalize_input(train[1],valid[1])
        train = (train_input,train[2])
        valid = (valid_input,valid[2])
    end

    config["lambda"] = norm(train[1])/lambda
    println("current cfg $config")

    if config["method"] == "svt"
        M = svt(train, valid, config)
    elseif config["method"] == "softimpute"
        M = soft_impute(train, valid, config)
    elseif config["method"] == "nmc"
        M = norm_mc(train, valid, config, true)
    elseif config["method"] == "fmc"
        M = norm_mc(train, valid, config, false)
        #M = factor_mc(train, valid, config)
    elseif config["method"] == "fpca"
        M = fpca_computation(train, valid, config)
    else
        println("there is no such method!")
        return
    end

    if config["normalize"] == 1
        M = denormalize_input(M, min_value, interval)
    end

    pred_Mat = M.*test[2]
    test_rmse = RMSE(test[1], pred_Mat)
    config["test"] = test_rmse
    config["inv-lambda"] = lambda # recover the value of lambda.

    println(JSON.json(config))
    end


let
      main()
end
