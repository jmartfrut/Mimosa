D_x, D_R_nl = TrainingSet_Read()

U_x_u, U_R_nl_u, U_x_𝜑, U_R_nl_𝜑, U_R_nl = SVD_X_R_Plot(D_x,D_R_nl)

## Parameters and evaluation
# Stored in root example folder

k,l,m,n = 25,25,25,25
k,l,m,n = 50,50,50,50
# Parameters selected based on the POD singular values plot and the Greedy error plot
k,l,m,n = 35,35,10,5
POD_DEIM_Matrices(k,l,m,n)
POD_DEIM_Matrices2(k,l,m,n)
POD_DEIM_Matrices3(k,l,m,n)

x_list, time_step, total_time = runs();
Red_Error(x_list,k,l,m,n)
Error_tip_(x_list,k,l,m,n)
## POD DEIM Matrix Generation (Multiple POD Multiple DEIM)
# Stored in Matrices Folder inside of example folder

#Set 1
k_list = [5,15,20,25,30,35,45,55] # ϕ sol modes
l_list = [15,25,35,45,55] # ϕ res modes
m_list = [10,20,30] # 𝜑 sol modes
n_list = [5,10,15] # 𝜑 res modes

#Set 1.1 inside of Set 1
k_list = [15,35,55] # ϕ sol modes
l_list = [15,35,55] # ϕ res modes
m_list = [10,20] # 𝜑 sol modes
n_list = [5,10] # 𝜑 res modes

# Set 1.2 inside of Set 1
k_list = [15,35,55] # ϕ sol modes
l_list = [25,45] # ϕ res modes
m_list = [10,20] # 𝜑 sol modes
n_list = [5,10] # 𝜑 res modes

# Set 1.2 inside of Set 1
k_list = [15,35,55,100] # ϕ sol modes
l_list = [16,18,20,22] # ϕ res modes
m_list = [10,20] # 𝜑 sol modes
n_list = [5,10] # 𝜑 res modes


# Multi matrix generation 

for l in l_list
    println("l = $l")
    @threads for n in n_list 
        POD_DEIM_Multiple_Matrices(k_list,l,m_list,n)
    end
end

# single reduced evaluation at a fixed set of reduction paramters stored en root example folder
x_list, time_step, total_time = runs();

# Single reduced run at a set of reduction parmeters with previously defined matrices in file
x_list, time_step, total_time = runs(35,55,20,10)

# Single reduced run at a set of reduction parmeters with previously defined matrices in file
x_list, time_step, total_time = runs_compare(35,35,10,5)
Red_Error(x_list,35,35,10,5)
# Analysis of the dictionary of POD-DEIM matrices of a single case
k,l,m,n = 55,55,20,10
k,l,m,n = 15,16,10,5
k,l,m,n = 35,35,10,5
DEIM_Matrices = load("scripts/MB Ex 2/PB_DEIM_matrices/Mat_$([k,l,m,n]).jld2")
𝛟_u = DEIM_Matrices["𝛟_u"]
𝛟_𝜑 = DEIM_Matrices["𝛟_𝜑"]

𝛟_u_ = 𝛟_u[[u_dofs+1:u_dofs+𝜑_dofs...],:]
maximum(𝛟_u_)

𝛟_𝜑_ = 𝛟_𝜑[[1:u_dofs...],:]
maximum(𝛟_𝜑_)

U_x_𝜑_ = U_x_𝜑[[1:u_dofs...],:]
maximum(U_x_𝜑_)
argmax(U_x_𝜑_)
plot(U_x_𝜑[:,1])
max_U_x_ϕ_ = [maximum(i) for i in eachcol(abs.(U_x_𝜑_))]

plot(max_U_x_ϕ_, label = "max_U_x_ϕ_")
maximum(D_x_𝜑)
argmax(D_x_𝜑)

maximum(abs.(D_R_nl_𝜑))
argmax(D_R_nl_𝜑)

U_x_u_ = U_x_u[[u_dofs+1:u_dofs+𝜑_dofs...],:]
maximum(U_x_u_)

U_R_nl_𝜑_ = U_R_nl_𝜑[[1:u_dofs...],:]

max_U_R_nl_ϕ_ = [maximum(i) for i in eachcol(abs.(U_R_nl_𝜑_))]
maxarg_U_R_nl_ϕ_ = [argmax(i) for i in eachcol(abs.(U_R_nl_𝜑_))]
plot(max_U_R_nl_ϕ_, label = "max_U_R_nl_ϕ_")
plot(maxarg_U_R_nl_ϕ_, label = "max_U_R_nl_ϕ_")

max_U_x_u = [maximum(i) for i in eachcol(abs.(U_x_u))]

plot(max_U_x_u, label = "max_U_x_u")

plot(U_x_u[:,13])

U_x_u[:,26]

max_U_x_ϕ = [maximum(i) for i in eachcol(abs.(U_x_𝜑))]
plot(max_U_x_ϕ, label = "max_U_x_ϕ")

argmax_U_x_ϕ = [argmax(i) for i in eachcol(abs.(U_x_𝜑))]
plot(argmax_U_x_ϕ, label = "argmax_U_x_ϕ")
plot(U_x_𝜑[:,20])




U_R_nl, σ_i_R_nl, V_R_nl = svd(D_R_nl)

𝛀 = U_R_nl

q = Vector{Int}(undef,903)
W_𝛀 = copy(𝛀)
e = copy(I(u_dofs+𝜑_dofs))
q[1] = argmax(abs.(W_𝛀[:,1]))
Z = copy(e[:,q[1]])
𝛀 = W_𝛀[:,1]

count = 0

for s in 2:100
    
    if count >= 3
        break
    end

    Zᵀ𝛀_inv = inv(Z'*𝛀)
    Error = maximum(abs.(D_R_nl-𝛀*Zᵀ𝛀_inv*Z'*D_R_nl))
    R_red = Zᵀ𝛀_inv*Z'*W_𝛀[:,s]
    # println(size(R_red))
    r = W_𝛀[:,s] - 𝛀*R_red
    
    q[s] = argmax(abs.(r))
    if q[s] > u_dofs
        count += 1
    end
    println("s = $s count = $count Max residual from greedy = $(maximum(abs.(r))) location = $(q[s]) Error_s-1 = $Error")
    Z = hcat(Z,e[:,q[s]])
    𝛀 = hcat(𝛀,W_𝛀[:,s])
end

for i in 1:u_dofs
    for j in 1:m
        𝛟_𝜑[i,j]=0.0
    end
end
# 𝛀_u = DEIM_Matrices["𝛀_u"]
# 𝛀_𝜑 = DEIM_Matrices["𝛀_𝜑"]
𝛀_u = nothing
𝛀_𝜑 = nothing
𝛀 = DEIM_Matrices["𝛀"]
# Z_u = DEIM_Matrices["Z_u"]
# Z_𝜑 = DEIM_Matrices["Z_𝜑"]
Z_u = nothing
Z_𝜑 = nothing
Z = DEIM_Matrices["Z"]
# Zᵀ𝛀_inv_u = DEIM_Matrices["Zᵀ𝛀_inv_u"]
# Zᵀ𝛀_inv_𝜑 = DEIM_Matrices["Zᵀ𝛀_inv_𝜑"]
Zᵀ𝛀_inv_u = nothing
Zᵀ𝛀_inv_𝜑 = nothing
Zᵀ𝛀_inv = DEIM_Matrices["Zᵀ𝛀_inv"]
q_u = DEIM_Matrices["q_u"]
q_𝜑 = DEIM_Matrices["q_𝜑"]

q_u_ = DEIM_Matrices["q_u"]
q_𝜑_ = DEIM_Matrices["q_𝜑"]


# Collect data at a set of material paramters factors defined in fuction
collect_data()

Dict0 = load("scripts/MB Ex 2/Red_Soltions/SolutionSet1.jld2")
x_list_ = Dict0["x_list_"]
time_list = Dict0["time_step_"]
total_time_ = Dict0["total_time"]
time_step_ = Dict0["time_step_"]
total_time_ = Dict()
for k in k_list
    for l in l_list
        for m in m_list
            for n in n_list
                id = [k,l,m,n]
                println("Reduction parameters = $id")
                try
                    # println(time_list["$id"])
                    total_time_["$id"] = time_list["$id"][end]
                catch
                    println("Error")
                end
            end
            
        end
        
    end
end

jldsave("scripts/MB Ex 2/Red_Soltions/SolutionSet2.jld2", k_list=k_list, l_list=l_list, m_list=m_list, n_list=n_list,
   x_list_=x_list_, time_step_=time_step_, total_time_=total_time_)

bar(collect(keys(total_time_)), collect(values(total_time_)), xrotation=90, size=(800,800),legend=false, xticks = :all)

Error_dict = Dict()
for k in k_list
    for l in l_list
        for m in m_list
            for n in n_list
                id = [k,l,m,n]
                println("Reduction parameters = $id")
                try
                    Error_0 = Error_tip(x_list_["$id"],k,l,m,n)
                    Error_dict["$id"] = Error_0
                catch
                    println("Error")
                end
            end
            
        end
        
    end
end

jldsave("scripts/MB Ex 2/Red_Soltions/SolutionSet2.jld2", k_list=k_list, l_list=l_list, m_list=m_list, n_list=n_list,
   x_list_=x_list_, time_step_=time_step_, total_time_=total_time_, Error_dict=Error_dict)

pp = plot(size = (800,800), yscale = :log10)
for l in l_list
    for m in m_list
        for n in n_list
            y = []
            x = []
            for k in k_list
                try
                    id = [k,l,m,n]
                    push!(y,Error_dict["$id"])
                    push!(x,k)
                catch
                end
            end
            pp = plot!(x,y,label = "$([l,m,n])")
        end
        
    end
    
end
display(pp)

x_list_ = Dict()
time_step_ = Dict()
total_time_ = Dict()
for k in k_list
    for l in l_list
        for m in m_list
            for n in n_list
                id = [k,l,m,n]
                println("Reduction parameters = $id")
                x_list, time_step, total_time = runs(k,l,m,n)
                x_list_["$id"] = x_list
                time_step_["$id"] = time_step
                total_time_["$id"] = total_time
            end
            
        end
        
    end
end

x_list, time_step, total_time = runs_compare(15,18,10,5)

x_list, time_step, total_time = runs_compare(15,18,10,5)
Red_Error(x_list,15,18,10,5)
Error_tip(x_list,15,18,10,5)

x_list, time_step, total_time = runs_compare(35,35,10,5)
Red_Error(x_list,35,35,10,5)
Error_tip(x_list,35,35,10,5)