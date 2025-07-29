# Libraries
using Mimosa
using Mimosa.Drivers
using Gridap
using GridapGmsh
using LineSearches: BackTracking
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.Arrays
using CSV
using DataFrames
using SparseMatricesCSR
using LinearAlgebra
using WriteVTK
using JLD2
using Plots
using Base.Threads

# Data Import and apriori data
𝜑_dofs = 7245
u_dofs = 36147


function TrainingSet_Read()
    D_x = []
    D_R_nl= []
    f_list = [0.9,1.0,1.1]
    for i in 1:3
        f = f_list[i]
        file_name = "data/csv/EM_PB_test_V2/MaterialModel$f/x_.csv"
        _X = CSV.File(file_name) |> Tables.matrix
        push!(D_x,_X)
        file_name = "data/csv/EM_PB_test_V2/MaterialModel$f/R_nl_.csv"
        _R_nl = CSV.File(file_name) |> Tables.matrix
        push!(D_R_nl,_R_nl)
    end
    D_x = reduce(hcat,D_x)
    D_R_nl = reduce(hcat,D_R_nl)
    return D_x, D_R_nl
end

# SVD
function SVD_X_R_Plot(D_x,D_R_nl)

    U_R_nl, σ_i_R_nl, V_R_nl = svd(D_R_nl)
    U_x, σ_i_x, V_x = svd(D_x)

    total_dofs, n_s = size(D_x)

    D_x_u = D_x[[1:u_dofs...],:]
    D_x_𝜑 = D_x[[u_dofs+1:u_dofs+𝜑_dofs...],:]

    D_x_u = vcat(D_x_u,zeros(𝜑_dofs,n_s))
    D_x_𝜑 = vcat(zeros(u_dofs,n_s),D_x_𝜑)


    D_R_nl_u = D_R_nl[[1:u_dofs...],:]
    D_R_nl_𝜑 = D_R_nl[[u_dofs+1:u_dofs+𝜑_dofs...],:]

    D_R_nl_u = vcat(D_R_nl_u,zeros(𝜑_dofs,n_s))
    D_R_nl_𝜑 = vcat(zeros(u_dofs,n_s),D_R_nl_𝜑)

    U_x_u, σ_i_x_u, V_x_u = svd(D_x_u)
    U_R_nl_u, σ_i_R_nl_u, V_R_nl_u = svd(D_R_nl_u)

    U_x_𝜑, σ_i_x_𝜑, V_x_𝜑 = svd(D_x_𝜑)
    U_R_nl_𝜑, σ_i_R_nl_𝜑, V_R_nl_𝜑 = svd(D_R_nl_𝜑)

    σ_i_x_rel = σ_i_x./σ_i_x[1]
    σ_i_R_nl_rel = σ_i_R_nl./σ_i_R_nl[1]

    σ_i_x_rel_u = σ_i_x_u./σ_i_x_u[1]
    σ_i_R_nl_rel_u = σ_i_R_nl_u./σ_i_R_nl_u[1]

    σ_i_x_rel_𝜑 = σ_i_x_𝜑./σ_i_x_𝜑[1]
    σ_i_R_nl_rel_𝜑 = σ_i_R_nl_𝜑./σ_i_R_nl_𝜑[1]
    σ_i_x_tot_u = sum(σ_i_x_u)
    σ_i_R_nl_tot_u = sum(σ_i_R_nl_u)

    σ_i_x_tot_𝜑 = sum(σ_i_x_𝜑)
    σ_i_R_nl_tot_𝜑 = sum(σ_i_R_nl_𝜑)

    σ_i_x_acc_u = [sum(σ_i_x_u[[1:i...]])/σ_i_x_tot_u for i in 1:n_s]

    σ_i_x_rel_u = σ_i_x_u./σ_i_x_tot_u
    σ_i_R_nl_rel_u = σ_i_R_nl_u./σ_i_R_nl_tot_u

    σ_i_x_rel_𝜑 = σ_i_x_𝜑./σ_i_x_tot_𝜑
    σ_i_R_nl_rel_𝜑 = σ_i_R_nl_𝜑./σ_i_R_nl_tot_𝜑
    ## Plot of singular values
    pp = plot(
        [σ_i_x_rel[[1:100...]],σ_i_R_nl_rel[[1:100...]],σ_i_x_rel_u[[1:100...]],σ_i_R_nl_rel_u[[1:100...]],σ_i_x_rel_𝜑[[1:100...]],σ_i_R_nl_rel_𝜑[[1:100...]]],
        yscale=:log10,
        label = ["Dₓ" "Dᵣ" "Dₓ_u" "Dᵣ_u" "Dₓ_phi" "Dᵣ_phi"],
        ylims = (10^-float(20), 1),
        yticks=[10^-float(i*4) for i in 0:5]
    )
    display(pp)
    return U_x_u, U_R_nl_u, U_x_𝜑, U_R_nl_𝜑, U_R_nl
end
# Apriori Greedy Error and plot
## Function
function POD_DEIM_Matrices_test(l1,l2,n1,n2)
    Error = Matrix{Float64}(undef,l2-l1+1,n2-n1+1)
    for l in l1:l2
        println("----------------------------")
        println("Row = $l of $l2")
        println("----------------------------")
        println("----------------------------")
        @threads for n in n1:n2
            
            𝛀_u = U_R_nl_u[:,[1:l...]]
            𝛀_𝜑 = U_R_nl_𝜑[:,[1:n...]]
            
            q_u = Vector{Int}(undef,l)
            W_𝛀 = copy(𝛀_u)
            e = copy(I(u_dofs+𝜑_dofs))
            q_u[1] = argmax(abs.(W_𝛀[:,1]))
            Z_u = copy(e[:,q_u[1]])
            𝛀 = W_𝛀[:,1]
            
            for s in 2:l
                R_red = inv(Z_u'*𝛀)*Z_u'*W_𝛀[:,s]
                # println(size(R_red))
                r = W_𝛀[:,s] - 𝛀*R_red
                # println("Max residual from greedy = $(maximum(abs.(r)))")
                q_u[s] = argmax(abs.(r))
                Z_u = hcat(Z_u,e[:,q_u[s]])
                𝛀 = hcat(𝛀,W_𝛀[:,s])
            end
            
            Z = Z_u
            
            # Zᵀ𝛀_inv_u = inv(Z_u'*𝛀_u)
            
            q_𝜑 = Vector{Int}(undef,n)
            W_𝛀 = copy(𝛀_𝜑)
            e = copy(I(u_dofs+𝜑_dofs))
            q_𝜑[1] = argmax(abs.(W_𝛀[:,1]))
            Z_𝜑 = copy(e[:,q_𝜑[1]])
            𝛀 = W_𝛀[:,1]
            
            for s in 2:n
                R_red = inv(Z_𝜑'*𝛀)*Z_𝜑'*W_𝛀[:,s]
                # println(size(R_red))
                r = W_𝛀[:,s] - 𝛀*R_red
                # println("Max residual from greedy = $(maximum(abs.(r)))")
                q_𝜑[s] = argmax(abs.(r))
                Z_𝜑 = hcat(Z_𝜑,e[:,q_𝜑[s]])
                𝛀 = hcat(𝛀,W_𝛀[:,s])
            end
            
            # Zᵀ𝛀_inv_𝜑 = inv(Z_𝜑'*𝛀_𝜑)
            
            Z = hcat(Z,Z_𝜑)
            𝛀 = hcat(𝛀_u,𝛀_𝜑)
            Zᵀ𝛀_inv = inv(Z'*𝛀)
            Error[l-l1+1,n-n1+1] = maximum(abs.(D_R_nl-𝛀*Zᵀ𝛀_inv*Z'*D_R_nl))
            println("Column = $n of $n2")
        end
    end
    return Error
end
# POD and DEIM matrices
## Function (Save to file)
function POD_DEIM_Matrices(k,l,m,n)
    𝛟_u = U_x_u[:,[1:k...]]
    𝛀_u = U_R_nl_u[:,[1:l...]]
    
    𝛟_𝜑 = U_x_𝜑[:,[1:m...]]
    𝛀_𝜑 = U_R_nl_𝜑[:,[1:n...]]

    𝛀 = hcat(𝛀_u,𝛀_𝜑)
    
    q = Vector{Int}(undef,l+n)
    W_𝛀 = copy(𝛀)
    e = copy(I(u_dofs+𝜑_dofs))
    q[1] = argmax(abs.(W_𝛀[:,1]))
    Z = copy(e[:,q[1]])
    𝛀 = W_𝛀[:,1]
    
    for s in 2:l+n
        R_red = inv(Z'*𝛀)*Z'*W_𝛀[:,s]
        println(size(R_red))
        r = W_𝛀[:,s] - 𝛀*R_red
        println("Max residual from greedy = $(maximum(abs.(r)))")
        q[s] = argmax(abs.(r))
        Z = hcat(Z,e[:,q[s]])
        𝛀 = hcat(𝛀,W_𝛀[:,s])
    end
    
    Zᵀ𝛀_inv = inv(Z'*𝛀)
    
    jldsave("scripts/MB Ex 2/PB_DEIM_matrices_u_phi_sep.jld2", 𝛟_u=𝛟_u, 𝛟_𝜑=𝛟_𝜑,
        𝛀=𝛀, Z=Z, Zᵀ𝛀_inv=Zᵀ𝛀_inv, q_u=q[[1:l...]], q_𝜑=q[[l+1:n+l...]]
        )  
end

function POD_DEIM_Matrices3(k,l,m,n)
    𝛟_u = U_x_u[:,[1:k...]]
    𝛀_u = U_R_nl_u[:,[1:l...]]
    
    𝛟_𝜑 = U_x_𝜑[:,[1:m...]]
    𝛀_𝜑 = U_R_nl_𝜑[:,[1:n...]]

    𝛀 = U_R_nl[:,[1:l+n...]]
    
    q = Vector{Int}(undef,l+n)
    W_𝛀 = copy(𝛀)
    e = copy(I(u_dofs+𝜑_dofs))
    q[1] = argmax(abs.(W_𝛀[:,1]))
    Z = copy(e[:,q[1]])
    𝛀 = W_𝛀[:,1]
    
    for s in 2:l+n
        R_red = inv(Z'*𝛀)*Z'*W_𝛀[:,s]
        println(size(R_red))
        r = W_𝛀[:,s] - 𝛀*R_red
        q[s] = argmax(abs.(r))
        Z = hcat(Z,e[:,q[s]])
        𝛀 = hcat(𝛀,W_𝛀[:,s])
        println("Max residual from greedy = $(maximum(abs.(r))) location = $(q[s])")
    end
    
    Zᵀ𝛀_inv = inv(Z'*𝛀)
    
    jldsave("scripts/MB Ex 2/PB_DEIM_matrices_u_phi_sep.jld2", 𝛟_u=𝛟_u, 𝛟_𝜑=𝛟_𝜑,
        𝛀=𝛀, Z=Z, Zᵀ𝛀_inv=Zᵀ𝛀_inv, q_u=q[[1:l...]], q_𝜑=q[[l+1:n+l...]]
        )  
end


function POD_DEIM_Matrices2(k,l,m,n)
    𝛟_u = U_x_u[:,[1:k...]]
    𝛀_u = U_R_nl_u[:,[1:l...]]
    
    𝛟_𝜑 = U_x_𝜑[:,[1:m...]]
    𝛀_𝜑 = U_R_nl_𝜑[:,[1:n...]]

    
    q_u = Vector{Int}(undef,l)
    W_𝛀 = copy(𝛀_u)
    e = copy(I(u_dofs+𝜑_dofs))
    q_u[1] = argmax(abs.(W_𝛀[:,1]))
    # q_u[1] = argmax((W_𝛀[:,1]))
    Z_u = copy(e[:,q_u[1]])
    𝛀 = W_𝛀[:,1]
    
    for s in 2:l
        R_red = inv(Z_u'*𝛀)*Z_u'*W_𝛀[:,s]
        # println(size(R_red))
        r = W_𝛀[:,s] - 𝛀*R_red
        # println("Max residual from greedy = $(maximum(abs.(r)))")
        q_u[s] = argmax(abs.(r))
        # q_u[s] = argmax((r))
        Z_u = hcat(Z_u,e[:,q_u[s]])
        𝛀 = hcat(𝛀,W_𝛀[:,s])
    end
    
    Z = Z_u
    
    # Zᵀ𝛀_inv_u = inv(Z_u'*𝛀_u)
    
    q_𝜑 = Vector{Int}(undef,n)
    W_𝛀 = copy(𝛀_𝜑)
    e = copy(I(u_dofs+𝜑_dofs))
    q_𝜑[1] = argmax(abs.(W_𝛀[:,1]))
    # q_𝜑[1] = argmax((W_𝛀[:,1]))
    Z_𝜑 = copy(e[:,q_𝜑[1]])
    𝛀 = W_𝛀[:,1]
    
    for s in 2:n
        R_red = inv(Z_𝜑'*𝛀)*Z_𝜑'*W_𝛀[:,s]
        # println(size(R_red))
        r = W_𝛀[:,s] - 𝛀*R_red
        # println("Max residual from greedy = $(maximum(abs.(r)))")
        q_𝜑[s] = argmax(abs.(r))
        # q_𝜑[s] = argmax((r))
        Z_𝜑 = hcat(Z_𝜑,e[:,q_𝜑[s]])
        𝛀 = hcat(𝛀,W_𝛀[:,s])
    end
    
    # Zᵀ𝛀_inv_𝜑 = inv(Z_𝜑'*𝛀_𝜑)
    
    Z = hcat(Z,Z_𝜑)
    𝛀 = hcat(𝛀_u,𝛀_𝜑)
    Zᵀ𝛀_inv = inv(Z'*𝛀)
    
    jldsave("scripts/MB Ex 2/PB_DEIM_matrices_u_phi_sep.jld2", 𝛟_u=𝛟_u, 𝛟_𝜑=𝛟_𝜑,
        𝛀=𝛀, Z=Z, Zᵀ𝛀_inv=Zᵀ𝛀_inv, q_u=q_u, q_𝜑=q_𝜑
        )  
end

function POD_DEIM_Matrices4(k,l,m)
    𝛟_u = U_x_u[:,[1:k...]]
    𝛀_u = U_R_nl_u[:,[1:l...]]
    
    𝛟_𝜑 = U_x_𝜑[:,[1:m...]]
    𝛀_𝜑 = U_R_nl_𝜑[:,[1:n...]]

    𝛀 = U_R_nl[:,[1:l+n...]]
    
    q = Vector{Int}(undef,l+n)
    W_𝛀 = copy(𝛀)
    e = copy(I(u_dofs+𝜑_dofs))
    q[1] = argmax(abs.(W_𝛀[:,1]))
    Z = copy(e[:,q[1]])
    𝛀 = W_𝛀[:,1]
    
    for s in 2:l+n
        R_red = inv(Z'*𝛀)*Z'*W_𝛀[:,s]
        println(size(R_red))
        r = W_𝛀[:,s] - 𝛀*R_red
        println("Max residual from greedy = $(maximum(abs.(r)))")
        q[s] = argmax(abs.(r))
        Z = hcat(Z,e[:,q[s]])
        𝛀 = hcat(𝛀,W_𝛀[:,s])
    end
    
    Zᵀ𝛀_inv = inv(Z'*𝛀)
    
    jldsave("scripts/MB Ex 2/PB_DEIM_matrices_u_phi_sep.jld2", 𝛟_u=𝛟_u, 𝛟_𝜑=𝛟_𝜑,
        𝛀=𝛀, Z=Z, Zᵀ𝛀_inv=Zᵀ𝛀_inv, q_u=q[[1:l...]], q_𝜑=q[[l+1:n+l...]]
        )  
end

function POD_DEIM_Multiple_Matrices(k_list,l,m_list,n)
    
    𝛀_u = U_R_nl_u[:,[1:l...]]
    𝛀_𝜑 = U_R_nl_𝜑[:,[1:n...]]
    𝛀 = hcat(𝛀_u,𝛀_𝜑)
    q = Vector{Int}(undef,l+n)
    W_𝛀 = copy(𝛀)
    e = copy(I(u_dofs+𝜑_dofs))
    q[1] = argmax(abs.(W_𝛀[:,1]))
    Z = copy(e[:,q[1]])
    𝛀 = W_𝛀[:,1]
    
    for s in 2:l+n
        R_red = inv(Z'*𝛀)*Z'*W_𝛀[:,s]
        # println(size(R_red))
        r = W_𝛀[:,s] - 𝛀*R_red
        # println("Max residual from greedy = $(maximum(abs.(r)))")
        q[s] = argmax(abs.(r))
        Z = hcat(Z,e[:,q[s]])
        𝛀 = hcat(𝛀,W_𝛀[:,s])
    end
    
    Zᵀ𝛀_inv = inv(Z'*𝛀)
    
    for k in k_list
        for m in m_list
            𝛟_u = U_x_u[:,[1:k...]]
            𝛟_𝜑 = U_x_𝜑[:,[1:m...]]
            jldsave("scripts/MB Ex 2/PB_DEIM_matrices/Mat_$([k,l,m,n]).jld2", 𝛟_u=𝛟_u, 𝛟_𝜑=𝛟_𝜑,
            𝛀=𝛀, Z=Z, Zᵀ𝛀_inv=Zᵀ𝛀_inv, q_u=q[[1:l...]], q_𝜑=q[[l+1:n+l...]]
            )  
        end 
        
    end
end

# Full order simuation
## Preparation MIMOSA functions
function get_trian_and_measure()
    model = GmshDiscreteModel("data/models/PlateBeame10S_BC.msh")
    degree = 4
    Ω = Triangulation(model)
    dΩ = Measure(Ω,degree)
    return model, Ω, dΩ
end

function get_DirichletBC(𝜑ᵇ)
    n_sec = 10
    conf = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
    sw = Array{Int}(undef, (n_sec))
    for i in 1:n_sec
        for j in 0:3
            if conf[[(((i-1)*2)+1):(((i-1)*2)+2)...]] == digits(j,base=2,pad=2)
                sw[i] = j
                break
            end
        end
    end
    evolu(Λ) = 1.0
    dir_u_tags = ["point_zy","point_z","fixedup_1"]
    dir_u_values = [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]
    masks = [(true,true,true),(true,false,true),(true,false,false)]
    dir_u_timesteps = [evolu,evolu,evolu]
    Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps,masks)
    evolφ(Λ) = Λ
    earth_loc = Vector{String}()
    power_loc = Vector{String}()
    dir_φ_timesteps = Vector{typeof(evolφ)}()
    earth_val = []
    power_val = []
    for i in 1:n_sec
    append!(earth_val,0.0)
    append!(earth_loc, ["midsurf_$i"])
    push!(dir_φ_timesteps,evolφ)
    if sw[i]==1 #iseven(i)
        append!(power_loc,["bottomsurf_$i"])
        append!(power_val,𝜑ᵇ)
        push!(dir_φ_timesteps,evolφ)
    elseif sw[i]==2
        append!(power_loc,["topsurf_$i"])
        append!(power_val,𝜑ᵇ)
        push!(dir_φ_timesteps,evolφ)
    elseif sw[i]==3 #iseven(i)
        append!(power_loc,["bottomsurf_$i"])
        append!(power_val,𝜑ᵇ)
        push!(dir_φ_timesteps,evolφ)
        append!(power_loc,["topsurf_$i"])
        append!(power_val,𝜑ᵇ)
        push!(dir_φ_timesteps,evolφ)
    end
    end
    dir_φ_tags = Vector{String}()
    append!(dir_φ_tags,earth_loc)
    append!(dir_φ_tags,power_loc)
    dir_φ_values = []
    append!(dir_φ_values,earth_val)
    append!(dir_φ_values,power_val)
    Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)
    dirichletbc = MultiFieldBoundaryCondition([Du, Dφ])
    return dirichletbc
end

function get_symbolic_res_and_jac(dΩ,f)
    diffstrat = "autodiff"
    soltype = "monolithic"
    modmec = Yeoh(C₁ = f*0.0693e6, C₂ = -8.88e2*f, C₃ = f*16.7, κ = 0.0693e8)
    modelec = IdealDielectric(ε=8.8542e-12*4.0)
    consmodel = ElectroMech(modmec, modelec)
    Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = consmodel(DerivativeStrategy{Symbol(diffstrat)}())
    ctype = CouplingStrategy{Symbol(soltype)}()
    res((u, φ), (v, vφ)) = residual_EM(ctype, (u, φ), (v, vφ), (∂Ψu, ∂Ψφ), dΩ)
    jac((u, φ), (du, dφ), (v, vφ)) = jacobian_EM(ctype, (u, φ), (du, dφ), (v, vφ), (∂Ψuu, ∂Ψφu, ∂Ψφφ), dΩ)
    return res, jac
end

function get_fe_spaces(model,dirichletbc)
    order = 2
    regtype = "statics"
    soltype = "monolithic"
    problem = ElectroMechProblem{Symbol(soltype), Symbol(regtype)}()
    fe_spaces = Drivers.get_FE_spaces(problem, model, order, dirichletbc)
    return fe_spaces
end
function get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
    RES = res(ph,get_fe_basis(fe_spaces.V))
    σₖ = get_cell_dof_ids(fe_spaces.U)
    assem = SparseMatrixAssembler(fe_spaces.U,fe_spaces.V)
    rs = ([RES[Ω]],[σₖ])
    b = allocate_vector(assem,rs)
    assemble_vector!(b,assem,rs)
    JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    rs = ([JAC[Ω]],[σₖ],[σₖ])
    K_T = MultiField.allocate_matrix(assem,rs)
    assemble_matrix!(K_T,assem,rs)
    return b, K_T
end

## Increment Solver Function
function run(x0,step,nsteps,𝜑ᵇ,f,model, Ω, dΩ,cache,x_list,b_list,R_nl_list,K_T_init,bisect)
    res, jac = get_symbolic_res_and_jac(dΩ,f)
    dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps))
    fe_spaces = get_fe_spaces(model,dirichletbc)
    norm_res = 1
    count = 0
    x0_copy = copy(x0)
    println("==============================================")
    println("step = $step of $nsteps")
    while norm_res>1e-12
        if count>15 || norm_res>1e2
            bisect += 1
            println("bisect = $bisect")
            x0, cache, _, _, _ = run(x0_copy,step-(1/(2^bisect)),nsteps,𝜑ᵇ,f,model, Ω, dΩ,cache,x_list,b_list,R_nl_list,K_T_init,bisect)
            count = 1
        end
        # ph = get_initial_guess(fe_spaces,x0)
        # t0 = time()
        # time_ = []
        ph = FEFunction(fe_spaces.U, x0)
        b, K_T = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
        # push!(time_,time()-t0)
        # println("assembly time = $(time_[end])")
        # if count==0
        #     fjac2 = K_T'*K_T
        #     lambda = 1e6*sqrt(length(b)*eps())*norm(fjac2, 1)
        #     K_T = (fjac2 + lambda * I)
        # end
        R_nl = b-K_T_init*x0
        if count==0
            push!(x_list,copy(x0))
            push!(b_list,copy(b))
            push!(R_nl_list,copy(R_nl))
        end


        Δx = K_T\(-b)
        # push!(time_,time()-(t0+sum(time_)))
        # println("time to solve = $(time_[end])")

        # println("Total time = $(sum(time_))")
        # if isnothing(x0)
        #     copyto!(x0,Δx)
        # else
        #     copyto!(x0,x0+Δx)           
        # end
        copyto!(x0,x0+Δx)
        norm_res = maximum(abs.(b))
        count += 1
        println("iter = $count norm_res = $norm_res norm_Δx = $(norm(Δx))  size_K_T = $(size(K_T)) norm_x0 = $(norm(x0))")
    end
    # nlsolver = get_FE_solver()
    # ph = get_initial_guess(fe_spaces,x0)
    # op = FEOperator(res, jac, fe_spaces.U, fe_spaces.V)
    # RES = op.res(ph,get_fe_basis(fe_spaces.V))
    # ph, cache = solve!(ph, nlsolver, op, cache)
    # return get_free_dof_values(ph), cache
    return x0, cache, x_list, b_list, R_nl_list
end
## Incremental Solver Function
function runs(f)
    model, Ω, dΩ =  get_trian_and_measure()
    nsteps = 500
    # step = 1
    # x0 = nothing
    𝜑ᵇ = 5000.0
    dirichletbc = get_DirichletBC(𝜑ᵇ/nsteps)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
    xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
    x0 = vcat(xu, xφ)
    println("number of dofs = $(length(x0))")
    ph = FEFunction(fe_spaces.U, x0)
    res, jac = get_symbolic_res_and_jac(dΩ,f)
    _, K_T_init = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
    x_list = []
    b_list = []
    R_nl_list = []
    cache = nothing
    bisect = 0
    for step in 1:nsteps
        @time x0, cache, x_list, b_list, R_nl_list = run(x0,step,nsteps,𝜑ᵇ,f,model, Ω, dΩ,cache,x_list,b_list,R_nl_list,K_T_init,bisect)
    end
    ph = FEFunction(fe_spaces.U, x0)
    b, K_T = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
    R_nl = b-K_T_init*x0
    push!(x_list,copy(x0))
    push!(b_list,copy(b))
    push!(R_nl_list,copy(R_nl))
    return x_list, b_list, R_nl_list
end
# Training data Collection (or single evaluation)
function collect_data()
    f_list = [0.9,1.0,1.1,1.05] # [1.05] #[0.9,1.0,1.1]
    for i in 1:lastindex(f_list) #1:3
        f = f_list[i]
        x_list, b_list, R_nl_list = runs(f)
        df_x = DataFrame(x_list, :auto)
        df_b = DataFrame(b_list, :auto)
        df_R_nl = DataFrame(R_nl_list, :auto)
        mkpath("data/csv/EM_PB_test_V3/MaterialModel$f")
        CSV.write("data/csv/EM_PB_test_V3/MaterialModel$f/x_.csv",df_x)
        CSV.write("data/csv/EM_PB_test_V3/MaterialModel$f/b_.csv",df_b)
        CSV.write("data/csv/EM_PB_test_V3/MaterialModel$f/R_nl_.csv",df_R_nl)
    end
end
# Reduced order simulation
## Gridap aditional functions for reduced assembly

function Gridap.FESpaces.assemble_matrix!(mat,a::SparseMatrixAssembler,matdata,q)
    LinearAlgebra.fillstored!(mat,zero(eltype(mat)))
    assemble_matrix_add!(mat,a,matdata,q)
end

function Gridap.FESpaces.assemble_matrix_add!(mat,a::SparseMatrixAssembler,matdata,q)
    numeric_loop_matrix!(mat,a,matdata,q)
    Gridap.FESpaces.create_from_nz(mat)
end

function Gridap.FESpaces.numeric_loop_matrix!(A,a::SparseMatrixAssembler,matdata,q)
strategy = Gridap.FESpaces.get_assembly_strategy(a)
# println("Check!")
h=0
for (cellmat,_cellidsrows,_cellidscols) in zip(matdata...)
    cellidsrows = Gridap.FESpaces.map_cell_rows(strategy,_cellidsrows)
    cellidscols = Gridap.FESpaces.map_cell_cols(strategy,_cellidscols)
    @assert length(cellidscols) == length(cellidsrows)
    @assert length(cellmat) == length(cellidsrows)
    if length(cellmat) > 0
        rows_cache = array_cache(cellidsrows)
        cols_cache = array_cache(cellidscols)
        vals_cache = array_cache(cellmat)
        mat1 = getindex!(vals_cache,cellmat,1)
        rows1 = getindex!(rows_cache,cellidsrows,1)
        cols1 = getindex!(cols_cache,cellidscols,1)
        # if h==0
        #     println(rows1)
        # end
        add! = AddEntriesMap(+)
        add_cache = return_cache(add!,A,mat1,rows1,cols1)
        caches = add_cache, vals_cache, rows_cache, cols_cache
        Gridap.FESpaces._numeric_loop_matrix!(A,caches,cellmat,cellidsrows,cellidscols,q)
        h+=1
    end
    # println("Count 1 = $h")
end
A
end

@noinline function Gridap.FESpaces._numeric_loop_matrix!(mat,caches,cell_vals,cell_rows,cell_cols,q)
    # println("Check!")
    add_cache, vals_cache, rows_cache, cols_cache = caches
    add! = AddEntriesMap(+)
    h=0
    for cell in 1:length(cell_cols)
        rows = getindex!(rows_cache,cell_rows,cell)
        rows_ = vcat(rows[1],rows[2])
        In_red = maximum([i in rows_ for i in q])
        if In_red
            cols = getindex!(cols_cache,cell_cols,cell)
            vals = getindex!(vals_cache,cell_vals,cell)
            evaluate!(add_cache,add!,mat,vals,rows,cols)
            h+=1
            # println("Count 2 = $h")
        end
        # if h==0
        #     println(rows)
        # end


        # cols = getindex!(cols_cache,cell_cols,cell)
        # vals = getindex!(vals_cache,cell_vals,cell)
        # rows_ = vcat(rows[1],rows[2])
        # In_red = maximum([i in rows_ for i in q])
        # if In_red
        #     evaluate!(add_cache,add!,mat,vals,rows,cols)
        #     h+=1
        #     # println("Count 2 = $h")
        # end
    end
end

function Gridap.FESpaces.allocate_matrix(a::SparseMatrixAssembler,matdata,q)
    m1 = Gridap.FESpaces.nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
    symbolic_loop_matrix!(m1,a,matdata,q)
    m2 = Gridap.FESpaces.nz_allocation(m1)
    symbolic_loop_matrix!(m2,a,matdata,q)
    m3 = Gridap.FESpaces.create_from_nz(m2)
    m3
end

function Gridap.FESpaces.symbolic_loop_matrix!(A,a::SparseMatrixAssembler,matdata,q)
    get_mat(a::Tuple) = a[1]
    get_mat(a) = a
    if Gridap.FESpaces.LoopStyle(A) == Gridap.FESpaces.DoNotLoop()
      return A
    end
    strategy = Gridap.FESpaces.get_assembly_strategy(a)
    for (cellmat,_cellidsrows,_cellidscols) in zip(matdata...)
      cellidsrows = Gridap.FESpaces.map_cell_rows(strategy,_cellidsrows)
      cellidscols = Gridap.FESpaces.map_cell_cols(strategy,_cellidscols)
      @assert length(cellidscols) == length(cellidsrows)
      if length(cellidscols) > 0
        rows_cache = array_cache(cellidsrows)
        cols_cache = array_cache(cellidscols)
        mat1 = get_mat(first(cellmat))
        rows1 = getindex!(rows_cache,cellidsrows,1)
        cols1 = getindex!(cols_cache,cellidscols,1)
        touch! = TouchEntriesMap()
        touch_cache = return_cache(touch!,A,mat1,rows1,cols1)
        caches = touch_cache, rows_cache, cols_cache
        Gridap.FESpaces._symbolic_loop_matrix!(A,caches,cellidsrows,cellidscols,mat1,q)
      end
    end
    A
end

@noinline function Gridap.FESpaces._symbolic_loop_matrix!(A,caches,cell_rows,cell_cols,mat1,q)
    touch_cache, rows_cache, cols_cache = caches
    touch! = TouchEntriesMap()
    for cell in 1:length(cell_cols)
        rows = getindex!(rows_cache,cell_rows,cell)
        rows_ = vcat(rows[1],rows[2])
        In_red = maximum([i in rows_ for i in q])
        h=0
        if In_red
            cols = getindex!(cols_cache,cell_cols,cell)
            evaluate!(touch_cache,touch!,A,mat1,rows,cols)
            h+=1
            # println("Count 2 = $h")
        end
    #   cols = getindex!(cols_cache,cell_cols,cell)
    #   evaluate!(touch_cache,touch!,A,mat1,rows,cols)
    end
end

function Gridap.FESpaces.assemble_vector!(b,a::SparseMatrixAssembler,vecdata,q)
    fill!(b,zero(eltype(b)))
    assemble_vector_add!(b,a,vecdata,q)
end

function Gridap.FESpaces.assemble_vector_add!(b,a::SparseMatrixAssembler,vecdata,q)
    numeric_loop_vector!(b,a,vecdata,q)
    Gridap.FESpaces.create_from_nz(b)
end

function Gridap.FESpaces.numeric_loop_vector!(b,a::SparseMatrixAssembler,vecdata,q)
    strategy = Gridap.FESpaces.get_assembly_strategy(a)
    for (cellvec, _cellids) in zip(vecdata...)
      cellids = Gridap.FESpaces.map_cell_rows(strategy,_cellids)
      if length(cellvec) > 0
        rows_cache = array_cache(cellids)
        vals_cache = array_cache(cellvec)
        vals1 = getindex!(vals_cache,cellvec,1)
        rows1 = getindex!(rows_cache,cellids,1)
        add! = AddEntriesMap(+)
        add_cache = return_cache(add!,b,vals1,rows1)
        caches = add_cache, vals_cache, rows_cache
        Gridap.FESpaces._numeric_loop_vector!(b,caches,cellvec,cellids,q)
      end
    end
    b
end

@noinline function Gridap.FESpaces._numeric_loop_vector!(vec,caches,cell_vals,cell_rows,q)
    add_cache, vals_cache, rows_cache = caches
    @assert length(cell_vals) == length(cell_rows)
    add! = AddEntriesMap(+)
    for cell in 1:length(cell_rows)
      rows = getindex!(rows_cache,cell_rows,cell)
      rows_ = vcat(rows[1],rows[2])
      In_red = maximum([i in rows_ for i in q])
      h=0
      if In_red
        vals = getindex!(vals_cache,cell_vals,cell)
        evaluate!(add_cache,add!,vec,vals,rows)
        h+=1
        # println("Count 2 = $h")
      end
    #   vals = getindex!(vals_cache,cell_vals,cell)
    #   evaluate!(add_cache,add!,vec,vals,rows)
    end
end

function Gridap.FESpaces.allocate_vector(a::SparseMatrixAssembler,vecdata,q)
    v1 = Gridap.FESpaces.nz_counter(get_vector_builder(a),(get_rows(a),))
    symbolic_loop_vector!(v1,a,vecdata,q)
    v2 = Gridap.FESpaces.nz_allocation(v1)
    symbolic_loop_vector!(v2,a,vecdata,q) 
    v3 = Gridap.FESpaces.create_from_nz(v2)
    v3
end

function Gridap.FESpaces.symbolic_loop_vector!(b,a::SparseMatrixAssembler,vecdata,q)
    get_vec(a::Tuple) = a[1]
    get_vec(a) = a
    if Gridap.FESpaces.LoopStyle(b) == Gridap.FESpaces.DoNotLoop()
      return b
    end
    strategy = Gridap.FESpaces.get_assembly_strategy(a)
    for (cellvec,_cellids) in zip(vecdata...)
      cellids = Gridap.FESpaces.map_cell_rows(strategy,_cellids)
      if length(cellids) > 0
        rows_cache = array_cache(cellids)
        vec1 = get_vec(first(cellvec))
        rows1 = getindex!(rows_cache,cellids,1)
        touch! = TouchEntriesMap()
        touch_cache = return_cache(touch!,b,vec1,rows1)
        caches = touch_cache, rows_cache
        _symbolic_loop_vector!(b,caches,cellids,vec1,q) 
      end
    end
    b
end

@noinline function Gridap.FESpaces._symbolic_loop_vector!(A,caches,cellids,vec1,q)
    touch_cache, rows_cache = caches
    touch! = TouchEntriesMap()
    for cell in 1:length(cellids)
      rows = getindex!(rows_cache,cellids,cell)
      rows_ = vcat(rows[1],rows[2])
      In_red = maximum([i in rows_ for i in q])
      h=0
      if In_red
        evaluate!(touch_cache,touch!,A,vec1,rows)
        h+=1
         # println("Count 2 = $h")
      end
    #   evaluate!(touch_cache,touch!,A,vec1,rows)
    end
end
## MIMOSA Reduced Assembly execution functions

function get_numeric_res_and_jac_red(ph,fe_spaces,Ω,res,jac,q)
    RES = res(ph,get_fe_basis(fe_spaces.V))
    σₖ = get_cell_dof_ids(fe_spaces.U)
    assem = SparseMatrixAssembler(fe_spaces.U,fe_spaces.V)
    rs = ([RES[Ω]],[σₖ])
    b = allocate_vector(assem,rs,q)
    assemble_vector!(b,assem,rs,q)
    JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    rs = ([JAC[Ω]],[σₖ],[σₖ])
    K_T = allocate_matrix(assem,rs,q)
    assemble_matrix!(K_T,assem,rs,q)
    return b, K_T
end
## Increment Solver
### No time function

function run_no_time(
    x0,step,nsteps,𝜑ᵇ, f,
    model, Ω, dΩ,cache,
    x_list,b_list,R_nl_list,
    K_T_init,K_0_red,
    K_0_red_uu, K_0_red_u𝜑,
    K_0_red_𝜑u, K_0_red_𝜑𝜑,
    k, l, m, n_,
    𝛟_u, 𝛟_𝜑, 𝛀_u, 
    𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
    Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
    bisect
    )

    res, jac = get_symbolic_res_and_jac(dΩ,f)
    dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps))
    fe_spaces = get_fe_spaces(model,dirichletbc)
    norm_res = 1
    count = 0
    n, m_k = size(𝛟_u)
    prev_step = step-(1/2^bisect)
    # K_T_DEIM_ = Matrix{Float64}(undef, m+k, n)
    K_T_DEIM = Matrix{Float64}(undef, m+k, n)
    # K_T_DEIM_red_ = Matrix{Float64}(undef, n, m+k)
    K_T_DEIM_red = Matrix{Float64}(undef, m+k, m+k)
    # for i in 1:u_dofs
    #     for j in 1:m
    #         𝛟_𝜑[i,j]=0.0
    #     end
    # end
    𝛟 = hcat(𝛟_u, 𝛟_𝜑)
    M_DEIM = 𝛟'*𝛀*Zᵀ𝛀_inv
    q = vcat(q_u,q_𝜑)
    K_T_init_rows =  @view K_T_init[q,:]
    x0_copy = copy(x0)
    println("==============================================")
    println("step = $step of $nsteps current potential = $(𝜑ᵇ*(step/nsteps))")
    while norm_res>1e-10
        # t0 = time()
        time_ = []
        if count>100 || norm_res>1e2
            bisect += 1
            println("bisect = $bisect")
            x0 = x0_copy
            for i in 1:2^bisect
                x0, cache, _, _, _ = run_no_time(x0,prev_step+(i/2^bisect),nsteps,𝜑ᵇ, f,
                    model, Ω, dΩ,cache,
                    x_list,b_list,R_nl_list,
                    K_T_init,K_0_red,
                    K_0_red_uu, K_0_red_u𝜑,
                    K_0_red_𝜑u, K_0_red_𝜑𝜑,
                    k, l, m, n_,
                    𝛟_u, 𝛟_𝜑, 𝛀_u, 
                    𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
                    Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
                    bisect
                )
                if prev_step+(i/2^bisect)>step
                    break                    
                end
            end
            count = 0
        end
        # append!(time_,time()-t0) #1
        x_u = 𝛟_u*x0[[1:k...]]
        x_𝜑 = 𝛟_𝜑*x0[[k+1:m+k...]]
        x_ = x_u+x_𝜑
        # append!(time_,time()-(t0+sum(time_))) #2
        ph = FEFunction(fe_spaces.U, x_)
        b, K_T = get_numeric_res_and_jac_red(ph,fe_spaces,Ω,res,jac,q)
        # b, K_T = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)

        # append!(time_,time()-(t0+sum(time_)))  #3
        K_T = @view K_T[q,:]
        # append!(time_,time()-(t0+sum(time_)))  #8
        R_nl = b-K_T_init*x_

        # R_nl_red_u = 𝛟_u'*𝛀_u*Zᵀ𝛀_inv_u*Z_u'*R_nl
        # R_nl_red_𝜑 = 𝛟_𝜑'*𝛀_𝜑*Zᵀ𝛀_inv_𝜑*Z_𝜑'*R_nl
        # R_nl_red = vcat(R_nl_red_u,R_nl_red_𝜑)

        # R_nl_red = 𝛟'*𝛀*Zᵀ𝛀_inv*Z'*R_nl
        # append!(time_,time()-(t0+sum(time_)))  #4
        R_nl_red = 𝛟'*𝛀*Zᵀ𝛀_inv*R_nl[q]
        # append!(time_,time()-(t0+sum(time_)))  #5

        R = K_0_red*x0 + R_nl_red

        # R = b
        # append!(time_,time()-(t0+sum(time_)))  #6
        # @show typeof(K_T_init_rows)
        # @show typeof(K_T)
        K_T = K_T - K_T_init_rows
        # append!(time_,time()-(t0+sum(time_)))  #7
        
        # K_T_DEIM = 𝛀*Zᵀ𝛀_inv*Z'*K_T

        # println("size of K_T = $(size(K_T))")


        # mul!(K_T_DEIM_,Zᵀ𝛀_inv,K_T)
        # mul!(K_T_DEIM,𝛀,K_T_DEIM_)


        # K_T_DEIM = 𝛀*Zᵀ𝛀_inv*K_T

        mul!(K_T_DEIM,M_DEIM,K_T)

        # append!(time_,time()-(t0+sum(time_)))  #9
        # K_T_red_uu = K_0_red_uu + 𝛟_u'*K_T_DEIM*𝛟_u
        # K_T_red_u𝜑 = K_0_red_u𝜑 + 𝛟_u'*K_T_DEIM*𝛟_𝜑
        # K_T_red_𝜑u = K_0_red_𝜑u + 𝛟_𝜑'*K_T_DEIM*𝛟_u
        # K_T_red_𝜑𝜑 = K_0_red_𝜑𝜑 + 𝛟_𝜑'*K_T_DEIM*𝛟_𝜑
        # K_T_red = hcat(vcat(K_T_red_uu,K_T_red_𝜑u),vcat(K_T_red_u𝜑,K_T_red_𝜑𝜑))

        # mul!(K_T_DEIM_red_,K_T_DEIM,𝛟)
        # mul!(K_T_DEIM_red,𝛟',K_T_DEIM_red_)
        # K_T_red = K_0_red + K_T_DEIM_red


        # K_T_red = K_0_red + 𝛟'*K_T_DEIM*𝛟


        mul!(K_T_DEIM_red,K_T_DEIM,𝛟)
        K_T_red = K_0_red + K_T_DEIM_red

        # K_T_red = K_T
        # append!(time_,time()-(t0+sum(time_)))  #10
        # println("Time DEIM: $(time_[end])")
        
        Δx = K_T_red\(-R)
        # append!(time_,time()-(t0+sum(time_)))  #11
        copyto!(x0,x0+Δx)
        # copyto!(x0,x0+𝛟'Δx)
        # norm_res = maximum(abs.(R))
        norm_res = maximum(abs.(R))

        count += 1
        println("iter = $count norm_res = $norm_res norm_Δx = $(norm(Δx)) argmax_Δx = $(argmax(Δx)) size_K_T_red = $(size(K_T_red))")
        # append!(time_,time()-(t0+sum(time_)))  #12
        # println("Time for iteration = $(sum(time_)) - time per section = $(round.(time_,digits=3))")
end
# nlsolver = get_FE_solver()
# ph = get_initial_guess(fe_spaces,x0)
# op = FEOperator(res, jac, fe_spaces.U, fe_spaces.V)
# RES = op.res(ph,get_fe_basis(fe_spaces.V))
# ph, cache = solve!(ph, nlsolver, op, cache)
# return get_free_dof_values(ph), cache
return x0, cache, x_list, b_list, R_nl_list, bisect
end
### Timed function 
#Requires update!
function run(
        x0,step,nsteps,𝜑ᵇ, f,
        model, Ω, dΩ,cache,
        x_list,b_list,R_nl_list,
        K_T_init,K_0_red,
        𝛟, M_DEIM, Z, 𝛀,
        bisect
        )
    res, jac = get_symbolic_res_and_jac(dΩ,f)
    dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps))
    fe_spaces = get_fe_spaces(model,dirichletbc)
    norm_res = 1
    count = 0
    prev_step = step-(1/2^bisect)
    x0_copy = copy(x0)
    println("==============================================")
    println("step = $step of $nsteps")
    while norm_res>1e-6
        if count>15 || norm_res>1e2
            bisect += 1
            println("bisect = $bisect")
            x0 = x0_copy
            for i in 1:2^bisect
                x0, cache, _, _, _ = run(x0,prev_step+(i/2^bisect),nsteps,𝜑ᵇ, f,
                model, Ω, dΩ,cache,
                x_list,b_list,R_nl_list,
                K_T_init,K_0_red,
                𝛟, M_DEIM, Z, 𝛀,
                bisect
                )
                if prev_step+(i/2^bisect)>step
                    break                  
                end
            end
            count = 0
        end
        x_ = 𝛟*x0
        # ph = get_initial_guess(fe_spaces,x0)
        ph = FEFunction(fe_spaces.U, x_)
        b, K_T = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
        # if count==0
        #     fjac2 = K_T'*K_T
        #     lambda = 1e6*sqrt(length(b)*eps())*norm(fjac2, 1)
        #     K_T = (fjac2 + lambda * I)
        # end
        R_nl = b-K_T_init*x_
        R = K_0_red*x0 + M_DEIM*Z'*R_nl
        K_T = K_T - K_T_init
        # K_T_red = K_0_red + 𝛟'*K_T*𝛟
        K_T_red = K_0_red + M_DEIM*Z'*K_T*𝛟
        # if count==0
        #     push!(x_list,copy(x0))
        #     push!(b_list,copy(b))
        #     push!(R_nl_list,copy(R_nl))
        # end


        Δx = K_T_red\(-R)
        # if isnothing(x0)
        #     copyto!(x0,Δx)
        # else
        #     copyto!(x0,x0+Δx)           
        # end
        copyto!(x0,x0+Δx)
        norm_res = maximum(abs.(R))
        count += 1
        println("iter = $count norm_res = $norm_res norm_Δx = $(norm(Δx))  det_K_T = $(det(K_T)) norm_x0 = $(norm(x0))")
    end
    # nlsolver = get_FE_solver()
    # ph = get_initial_guess(fe_spaces,x0)
    # op = FEOperator(res, jac, fe_spaces.U, fe_spaces.V)
    # RES = op.res(ph,get_fe_basis(fe_spaces.V))
    # ph, cache = solve!(ph, nlsolver, op, cache)
    # return get_free_dof_values(ph), cache
    return x0, cache, x_list, b_list, R_nl_list
end

## Incremental Solver (whole problem)
### Function

function runs()
    f = 1.05
    DEIM_Matrices = load("scripts/MB Ex 2/PB_DEIM_matrices_u_phi_sep.jld2")
    𝛟_u = DEIM_Matrices["𝛟_u"]
    𝛟_𝜑 = DEIM_Matrices["𝛟_𝜑"]
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
    n, k = size(𝛟_u)
    l = length(q_u)
    _, m = size(𝛟_𝜑)
    n_ = length(q_𝜑)
    model, Ω, dΩ =  get_trian_and_measure()
    nsteps = 300
    # step = 1
    # x0 = nothing
    dirichletbc = get_DirichletBC(00.0)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
    xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
    x0_ = vcat(xu, xφ)
    println("number of dofs = $(length(x0_))")
    ph = FEFunction(fe_spaces.U, x0_)
    res, jac = get_symbolic_res_and_jac(dΩ,f)
    _, K_T_init = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
    K_0_red_uu= 𝛟_u'*K_T_init*𝛟_u
    K_0_red_u𝜑= 𝛟_u'*K_T_init*𝛟_𝜑
    K_0_red_𝜑u= 𝛟_𝜑'*K_T_init*𝛟_u
    K_0_red_𝜑𝜑= 𝛟_𝜑'*K_T_init*𝛟_𝜑
    K_0_red = hcat(vcat(K_0_red_uu,K_0_red_𝜑u),vcat(K_0_red_u𝜑,K_0_red_𝜑𝜑))
    x0 = zeros(Float64,m+k)
    # K_T_DEIM_ = Matrix{Float64}(undef, 2*m_k, n)
    # K_T_DEIM = Matrix{Float64}(undef, n, n)
    # K_T_DEIM_red_ = Matrix{Float64}(undef, n, 2*m_k)
    # K_T_DEIM_red = Matrix{Float64}(undef, 2*m_k, 2*m_k)
    x_list = [copy(x0)]
    x_ = zeros(Float64,k+m)
    t0 = time()
    time_step = []
    x_list = []
    b_list = []
    R_nl_list = []
    cache = nothing
    bisect = 0
    𝜑ᵇ = 5000.0
    for step in 1:nsteps
        @time x0, cache, x_list, b_list, R_nl_list, bisect = run_no_time(
            x0,step,nsteps,𝜑ᵇ, f,
            model, Ω, dΩ,cache,
            x_list,b_list,R_nl_list,
            K_T_init,K_0_red,
            K_0_red_uu, K_0_red_u𝜑,
            K_0_red_𝜑u, K_0_red_𝜑𝜑,
            k, l, m, n_,
            𝛟_u, 𝛟_𝜑, 𝛀_u, 
            𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
            Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
            bisect
            )
        copyto!(x_,x0)
        push!(x_list,copy(x0))
        push!(time_step,time()-t0)
    end
    # ph = FEFunction(fe_spaces.U, x0)
    # b, K_T = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
    # R_nl = b-K_T_init*x0
    # push!(x_list,copy(x0))
    # push!(b_list,copy(b))
    # push!(R_nl_list,copy(R_nl))
    # return x_list, b_list, R_nl_list
    total_time = time()-t0
    return x_list, time_step, total_time
end

## Reduced Simulations Evaluation
### Function

function run_Compare(
    x0,step,nsteps,𝜑ᵇ, f,
    model, Ω, dΩ,cache,
    x_list,b_list,R_nl_list,
    K_T_init,K_0_red,
    K_0_red_uu, K_0_red_u𝜑,
    K_0_red_𝜑u, K_0_red_𝜑𝜑,
    k, l, m, n_,
    𝛟_u, 𝛟_𝜑, 𝛀_u, 
    𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
    Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
    bisect
    )

res, jac = get_symbolic_res_and_jac(dΩ,f)
dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps))
fe_spaces = get_fe_spaces(model,dirichletbc)
norm_res = 1
count = 0
n, m_k = size(𝛟_u)
prev_step = step-(1/2^bisect)
# K_T_DEIM_ = Matrix{Float64}(undef, m+k, n)
K_T_DEIM = Matrix{Float64}(undef, m+k, n)
# K_T_DEIM_red_ = Matrix{Float64}(undef, n, m+k)
K_T_DEIM_red = Matrix{Float64}(undef, m+k, m+k)
𝛟 = hcat(𝛟_u, 𝛟_𝜑)
M_DEIM = 𝛟'*𝛀*Zᵀ𝛀_inv
q = vcat(q_u,q_𝜑)
K_T_init_rows =  @view K_T_init[q,:]
x0_copy = copy(x0)
println("==============================================")
println("step = $step of $nsteps current potential = $(𝜑ᵇ*(step/nsteps))")
while norm_res>1e-10
    # t0 = time()
    time_ = []
    if count>100 || norm_res>1e2
        bisect += 1
        println("bisect = $bisect")
        x0 = x0_copy
        for i in 1:2^bisect
            x0, cache, _, _, _ = run_Compare(x0,prev_step+(i/2^bisect),nsteps,𝜑ᵇ, f,
                model, Ω, dΩ,cache,
                x_list,b_list,R_nl_list,
                K_T_init,K_0_red,
                K_0_red_uu, K_0_red_u𝜑,
                K_0_red_𝜑u, K_0_red_𝜑𝜑,
                k, l, m, n_,
                𝛟_u, 𝛟_𝜑, 𝛀_u, 
                𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
                Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
                bisect
            )
            if prev_step+(i/2^bisect)>step
                break                    
            end
        end
        count = 0
    end
    # append!(time_,time()-t0) #1
    x_u = 𝛟_u*x0[[1:k...]]
    x_𝜑 = 𝛟_𝜑*x0[[k+1:m+k...]]
    x_ = x_u+x_𝜑
    # append!(time_,time()-(t0+sum(time_))) #2
    ph = FEFunction(fe_spaces.U, x_)
    b, K_T = get_numeric_res_and_jac_red(ph,fe_spaces,Ω,res,jac,q)
    b_fo, K_T_fo = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
    # b, K_T = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)

    # append!(time_,time()-(t0+sum(time_)))  #3
    K_T = @view K_T[q,:]
    # append!(time_,time()-(t0+sum(time_)))  #8
    R_nl = b-K_T_init*x_
    R_nl_fo = b_fo-K_T_init*x_


    # R_nl_red_u = 𝛟_u'*𝛀_u*Zᵀ𝛀_inv_u*Z_u'*R_nl
    # R_nl_red_𝜑 = 𝛟_𝜑'*𝛀_𝜑*Zᵀ𝛀_inv_𝜑*Z_𝜑'*R_nl
    # R_nl_red = vcat(R_nl_red_u,R_nl_red_𝜑)

    # R_nl_red = 𝛟'*𝛀*Zᵀ𝛀_inv*Z'*R_nl
    # append!(time_,time()-(t0+sum(time_)))  #4
    R_nl_red = 𝛟'*𝛀*Zᵀ𝛀_inv*R_nl[q]
    metric1_ = R_nl_fo - 𝛀*Zᵀ𝛀_inv*R_nl[q]
    metric1_at2 = metric1_[2]
    metric1 = maximum(abs.(metric1_))
    metric1_loc = argmax(abs.(metric1_))
    # append!(time_,time()-(t0+sum(time_)))  #5

    R = K_0_red*x0 + R_nl_red

    # R = b
    # append!(time_,time()-(t0+sum(time_)))  #6
    # @show typeof(K_T_init_rows)
    # @show typeof(K_T)
    K_T = K_T - K_T_init_rows
    # append!(time_,time()-(t0+sum(time_)))  #7
    
    # K_T_DEIM = 𝛀*Zᵀ𝛀_inv*Z'*K_T

    # println("size of K_T = $(size(K_T))")


    # mul!(K_T_DEIM_,Zᵀ𝛀_inv,K_T)
    # mul!(K_T_DEIM,𝛀,K_T_DEIM_)


    K_T_DEIM_ = 𝛀*Zᵀ𝛀_inv*K_T
    metric2_  = K_T_fo -K_T_init - K_T_DEIM_
    metric2_at2 = metric2_[2,2]
    metric2 = maximum(abs.(metric2_))
    metric2_loc = argmax(abs.(metric2_))



    mul!(K_T_DEIM,M_DEIM,K_T)

    # append!(time_,time()-(t0+sum(time_)))  #9
    # K_T_red_uu = K_0_red_uu + 𝛟_u'*K_T_DEIM*𝛟_u
    # K_T_red_u𝜑 = K_0_red_u𝜑 + 𝛟_u'*K_T_DEIM*𝛟_𝜑
    # K_T_red_𝜑u = K_0_red_𝜑u + 𝛟_𝜑'*K_T_DEIM*𝛟_u
    # K_T_red_𝜑𝜑 = K_0_red_𝜑𝜑 + 𝛟_𝜑'*K_T_DEIM*𝛟_𝜑
    # K_T_red = hcat(vcat(K_T_red_uu,K_T_red_𝜑u),vcat(K_T_red_u𝜑,K_T_red_𝜑𝜑))

    # mul!(K_T_DEIM_red_,K_T_DEIM,𝛟)
    # mul!(K_T_DEIM_red,𝛟',K_T_DEIM_red_)
    # K_T_red = K_0_red + K_T_DEIM_red


    # K_T_red = K_0_red + 𝛟'*K_T_DEIM*𝛟


    mul!(K_T_DEIM_red,K_T_DEIM,𝛟)
    K_T_red = K_0_red + K_T_DEIM_red

    # K_T_red = K_T
    # append!(time_,time()-(t0+sum(time_)))  #10
    # println("Time DEIM: $(time_[end])")
    
    Δx = K_T_red\(-R)
    # append!(time_,time()-(t0+sum(time_)))  #11
    copyto!(x0,x0+Δx)
    # copyto!(x0,x0+𝛟'Δx)
    # norm_res = maximum(abs.(R))
    norm_res = maximum(abs.(R))

    count += 1
    println("iter = $count norm_res = $norm_res norm_Δx = $(norm(Δx)) argmax_Δx = $(argmax(Δx)) size_K_T_red = $(size(K_T_red))")
    println("iter = $count Error_R = $metric1 Error_R_loc = $metric1_loc Error_R_at2 = $metric1_at2 Error_K = $metric2 Error_K_loc = $metric2_loc Error_K_at2 = $metric2_at2")
    # append!(time_,time()-(t0+sum(time_)))  #12
    # println("Time for iteration = $(sum(time_)) - time per section = $(round.(time_,digits=3))")
end
# nlsolver = get_FE_solver()
# ph = get_initial_guess(fe_spaces,x0)
# op = FEOperator(res, jac, fe_spaces.U, fe_spaces.V)
# RES = op.res(ph,get_fe_basis(fe_spaces.V))
# ph, cache = solve!(ph, nlsolver, op, cache)
# return get_free_dof_values(ph), cache
return x0, cache, x_list, b_list, R_nl_list, bisect
end


function runs_compare(k,l,m,n)
    f = 1.05
    DEIM_Matrices = load("scripts/MB Ex 2/PB_DEIM_matrices/Mat_$([k,l,m,n]).jld2")
    𝛟_u = DEIM_Matrices["𝛟_u"]
    𝛟_𝜑 = DEIM_Matrices["𝛟_𝜑"]
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
    n, k = size(𝛟_u)
    l = length(q_u)
    _, m = size(𝛟_𝜑)
    n_ = length(q_𝜑)
    model, Ω, dΩ =  get_trian_and_measure()
    nsteps = 300
    # step = 1
    # x0 = nothing
    dirichletbc = get_DirichletBC(00.0)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
    xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
    x0_ = vcat(xu, xφ)
    println("number of dofs = $(length(x0_))")
    ph = FEFunction(fe_spaces.U, x0_)
    res, jac = get_symbolic_res_and_jac(dΩ,f)
    _, K_T_init = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
    K_0_red_uu= 𝛟_u'*K_T_init*𝛟_u
    K_0_red_u𝜑= 𝛟_u'*K_T_init*𝛟_𝜑
    K_0_red_𝜑u= 𝛟_𝜑'*K_T_init*𝛟_u
    K_0_red_𝜑𝜑= 𝛟_𝜑'*K_T_init*𝛟_𝜑
    K_0_red = hcat(vcat(K_0_red_uu,K_0_red_𝜑u),vcat(K_0_red_u𝜑,K_0_red_𝜑𝜑))
    x0 = zeros(Float64,m+k)
    # K_T_DEIM_ = Matrix{Float64}(undef, 2*m_k, n)
    # K_T_DEIM = Matrix{Float64}(undef, n, n)
    # K_T_DEIM_red_ = Matrix{Float64}(undef, n, 2*m_k)
    # K_T_DEIM_red = Matrix{Float64}(undef, 2*m_k, 2*m_k)
    x_list = [copy(x0)]
    x_ = zeros(Float64,k+m)
    t0 = time()
    time_step = []
    x_list = []
    b_list = []
    R_nl_list = []
    cache = nothing
    bisect = 0
    𝜑ᵇ = 5000.0
    for step in 1:nsteps
        if step < 299 && step ≠ 25 && step ≠ 50 && step ≠ 100 && step ≠ 150
            @time x0, cache, x_list, b_list, R_nl_list, bisect_0 = run_no_time(
            x0,step,nsteps,𝜑ᵇ, f,
            model, Ω, dΩ,cache,
            x_list,b_list,R_nl_list,
            K_T_init,K_0_red,
            K_0_red_uu, K_0_red_u𝜑,
            K_0_red_𝜑u, K_0_red_𝜑𝜑,
            k, l, m, n_,
            𝛟_u, 𝛟_𝜑, 𝛀_u, 
            𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
            Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
            bisect
            )
        elseif step == 25
            @time x0, cache, x_list, b_list, R_nl_list, bisect_0 = run_Compare(
            x0,step,nsteps,𝜑ᵇ, f,
            model, Ω, dΩ,cache,
            x_list,b_list,R_nl_list,
            K_T_init,K_0_red,
            K_0_red_uu, K_0_red_u𝜑,
            K_0_red_𝜑u, K_0_red_𝜑𝜑,
            k, l, m, n_,
            𝛟_u, 𝛟_𝜑, 𝛀_u, 
            𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
            Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
            bisect
            )
        elseif step == 50
            @time x0, cache, x_list, b_list, R_nl_list, bisect_0 = run_Compare(
            x0,step,nsteps,𝜑ᵇ, f,
            model, Ω, dΩ,cache,
            x_list,b_list,R_nl_list,
            K_T_init,K_0_red,
            K_0_red_uu, K_0_red_u𝜑,
            K_0_red_𝜑u, K_0_red_𝜑𝜑,
            k, l, m, n_,
            𝛟_u, 𝛟_𝜑, 𝛀_u, 
            𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
            Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
            bisect
            )
        elseif step == 100
            @time x0, cache, x_list, b_list, R_nl_list, bisect_0 = run_Compare(
            x0,step,nsteps,𝜑ᵇ, f,
            model, Ω, dΩ,cache,
            x_list,b_list,R_nl_list,
            K_T_init,K_0_red,
            K_0_red_uu, K_0_red_u𝜑,
            K_0_red_𝜑u, K_0_red_𝜑𝜑,
            k, l, m, n_,
            𝛟_u, 𝛟_𝜑, 𝛀_u, 
            𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
            Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
            bisect
            )
        elseif step == 150
            @time x0, cache, x_list, b_list, R_nl_list, bisect_0 = run_Compare(
            x0,step,nsteps,𝜑ᵇ, f,
            model, Ω, dΩ,cache,
            x_list,b_list,R_nl_list,
            K_T_init,K_0_red,
            K_0_red_uu, K_0_red_u𝜑,
            K_0_red_𝜑u, K_0_red_𝜑𝜑,
            k, l, m, n_,
            𝛟_u, 𝛟_𝜑, 𝛀_u, 
            𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
            Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
            bisect
            )
        else
            @time x0, cache, x_list, b_list, R_nl_list, bisect_0 = run_Compare(
            x0,step,nsteps,𝜑ᵇ, f,
            model, Ω, dΩ,cache,
            x_list,b_list,R_nl_list,
            K_T_init,K_0_red,
            K_0_red_uu, K_0_red_u𝜑,
            K_0_red_𝜑u, K_0_red_𝜑𝜑,
            k, l, m, n_,
            𝛟_u, 𝛟_𝜑, 𝛀_u, 
            𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
            Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
            bisect
            )
        end
        
        copyto!(x_,x0)
        push!(x_list,copy(x0))
        push!(time_step,time()-t0)
        if bisect > 15
            break
        end
    end
    # ph = FEFunction(fe_spaces.U, x0)
    # b, K_T = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
    # R_nl = b-K_T_init*x0
    # push!(x_list,copy(x0))
    # push!(b_list,copy(b))
    # push!(R_nl_list,copy(R_nl))
    # return x_list, b_list, R_nl_list
    total_time = time()-t0
    return x_list, time_step, total_time
end

function runs(k,l,m,n)
    f = 1.05
    DEIM_Matrices = load("scripts/MB Ex 2/PB_DEIM_matrices/Mat_$([k,l,m,n]).jld2")
    𝛟_u = DEIM_Matrices["𝛟_u"]
    𝛟_𝜑 = DEIM_Matrices["𝛟_𝜑"]
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
    n, k = size(𝛟_u)
    l = length(q_u)
    _, m = size(𝛟_𝜑)
    n_ = length(q_𝜑)
    model, Ω, dΩ =  get_trian_and_measure()
    nsteps = 300
    # step = 1
    # x0 = nothing
    dirichletbc = get_DirichletBC(00.0)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
    xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
    x0_ = vcat(xu, xφ)
    println("number of dofs = $(length(x0_))")
    ph = FEFunction(fe_spaces.U, x0_)
    res, jac = get_symbolic_res_and_jac(dΩ,f)
    _, K_T_init = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
    K_0_red_uu= 𝛟_u'*K_T_init*𝛟_u
    K_0_red_u𝜑= 𝛟_u'*K_T_init*𝛟_𝜑
    K_0_red_𝜑u= 𝛟_𝜑'*K_T_init*𝛟_u
    K_0_red_𝜑𝜑= 𝛟_𝜑'*K_T_init*𝛟_𝜑
    K_0_red = hcat(vcat(K_0_red_uu,K_0_red_𝜑u),vcat(K_0_red_u𝜑,K_0_red_𝜑𝜑))
    x0 = zeros(Float64,m+k)
    # K_T_DEIM_ = Matrix{Float64}(undef, 2*m_k, n)
    # K_T_DEIM = Matrix{Float64}(undef, n, n)
    # K_T_DEIM_red_ = Matrix{Float64}(undef, n, 2*m_k)
    # K_T_DEIM_red = Matrix{Float64}(undef, 2*m_k, 2*m_k)
    x_list = [copy(x0)]
    x_ = zeros(Float64,k+m)
    t0 = time()
    time_step = []
    x_list = []
    b_list = []
    R_nl_list = []
    cache = nothing
    bisect = 0
    𝜑ᵇ = 5000.0
    for step in 1:nsteps
        @time x0, cache, x_list, b_list, R_nl_list, bisect_0 = run_no_time(
        x0,step,nsteps,𝜑ᵇ, f,
        model, Ω, dΩ,cache,
        x_list,b_list,R_nl_list,
        K_T_init,K_0_red,
        K_0_red_uu, K_0_red_u𝜑,
        K_0_red_𝜑u, K_0_red_𝜑𝜑,
        k, l, m, n_,
        𝛟_u, 𝛟_𝜑, 𝛀_u, 
        𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
        Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
        bisect
        )
        
        copyto!(x_,x0)
        push!(x_list,copy(x0))
        push!(time_step,time()-t0)
        if bisect > 15
            break
        end
    end
    # ph = FEFunction(fe_spaces.U, x0)
    # b, K_T = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
    # R_nl = b-K_T_init*x0
    # push!(x_list,copy(x0))
    # push!(b_list,copy(b))
    # push!(R_nl_list,copy(R_nl))
    # return x_list, b_list, R_nl_list
    total_time = time()-t0
    return x_list, time_step, total_time
end

function Error_tip_(x_list,k,l,m,n)
    DEIM_Matrices = load("scripts/MB Ex 2/PB_DEIM_matrices_u_phi_sep.jld2")
    𝛟_u = DEIM_Matrices["𝛟_u"]
    𝛟_𝜑 = DEIM_Matrices["𝛟_𝜑"]
    𝜑_dofs = 7245
    u_dofs = 36147
    f = 1.05
    file_name = "data/csv/EM_PB_test_V2/MaterialModel$f/x_.csv"
    _X = CSV.File(file_name) |> Tables.matrix
    Error_list = []
    for i in 1:(lastindex(eachcol(_X))-1)
        x = 𝛟_u*x_list[i][[1:k...]]+𝛟_𝜑*x_list[i][[1+k:k+m...]]
        _x = _X[:,i+1]
        Error_u = abs.(_x[[1:u_dofs...]] - x[[1:u_dofs...]])./maximum(abs.(_x[[1:u_dofs...]]))
        push!(Error_list,maximum(Error_u))
    end
    return sum(Error_list)/length(Error_list)
end

function Error_tip(x_list,k,l,m,n)
    DEIM_Matrices = load("scripts/MB Ex 2/PB_DEIM_matrices/Mat_$([k,l,m,n]).jld2")
    𝛟_u = DEIM_Matrices["𝛟_u"]
    𝛟_𝜑 = DEIM_Matrices["𝛟_𝜑"]
    𝜑_dofs = 7245
    u_dofs = 36147
    f = 1.05
    file_name = "data/csv/EM_PB_test_V2/MaterialModel$f/x_.csv"
    _X = CSV.File(file_name) |> Tables.matrix
    Error_list = []
    for i in 1:(lastindex(eachcol(_X))-1)
        x = 𝛟_u*x_list[i][[1:k...]]+𝛟_𝜑*x_list[i][[1+k:k+m...]]
        _x = _X[:,i+1]
        Error_u = abs.(_x[[1:u_dofs...]] - x[[1:u_dofs...]])./maximum(abs.(_x[[1:u_dofs...]]))
        push!(Error_list,maximum(Error_u))
    end
    return sum(Error_list)/length(Error_list)
end

function Red_Error(x_list,k,l,m,n)
    DEIM_Matrices = load("scripts/MB Ex 2/PB_DEIM_matrices/Mat_$([k,l,m,n]).jld2")
    𝛟_u = DEIM_Matrices["𝛟_u"]
    𝛟_𝜑 = DEIM_Matrices["𝛟_𝜑"]
    𝜑_dofs = 7245
    u_dofs = 36147
    model, Ω, dΩ =  get_trian_and_measure()
    f = 1.05
    file_name_red = "data/sims/EM_PB_test_2/RedDEIM_$k-$l-$m-$n])"
    file_name = "data/csv/EM_PB_test_V2/MaterialModel$f/x_.csv"
    _X = CSV.File(file_name) |> Tables.matrix
    # println("Norm of 𝛟*x_red-x = $(norm((𝛟_u*x_list[end][[1:m_k...]]+𝛟_𝜑*x_list[end][[1+m_k:2*m_k...]])-_X[:,end])./maximum(abs.(_X[:,end]))))")
    # println("Max Abs of 𝛟*x_red-x = $(maximum(abs.((𝛟_u*x_list[end][[1:m_k...]]+𝛟_𝜑*x_list[end][[1+m_k:2*m_k...]]-_X[:,end])./maximum(abs.(_X[:,end])))))")
    mkpath(file_name_red*"/Result_withError_POD-DEIM_u_phi_sep_$f")
    pvd = paraview_collection(file_name_red*"/Result_withError_POD-DEIM_u_phi_sep_$f" * "/Results", append=false)
    writevtk(model, file_name_red*"/Result_withError_POD-DEIM_u_phi_sep_$f" * "/DiscreteModel")
    dirichletbc = get_DirichletBC(0.0) #lastindex(x_list)))
    fe_spaces = get_fe_spaces(model,dirichletbc)
    U_ = fe_spaces.U
    # _x = _X[:,i]
    # maximum(abs.(x[[1:u_dofs...]]))
    for i in 1:(lastindex(eachcol(_X))-1)
        dirichletbc = get_DirichletBC(5000.0*(i/lastindex(x_list))) #lastindex(x_list)))
        fe_spaces = get_fe_spaces(model,dirichletbc)
        U = fe_spaces.U
        x = 𝛟_u*x_list[i][[1:k...]]+𝛟_𝜑*x_list[i][[1+k:k+m...]]
        uh_red = FEFunction(U,x)
        _x = _X[:,i+1]
        uh = FEFunction(U,_x)
        Error_u = abs.(_x[[1:u_dofs...]] - x[[1:u_dofs...]])./maximum(abs.(x[[1:u_dofs...]]))
        Error_𝜑 = abs.(_x[[u_dofs+1:u_dofs+𝜑_dofs...]] - x[[u_dofs+1:u_dofs+𝜑_dofs...]])./maximum(abs.(x[[u_dofs+1:u_dofs+𝜑_dofs...]]))
        Error_rel = vcat(Error_u,Error_𝜑)
        error_rel = FEFunction(U_,Error_rel)
        Error = abs.(_x - x)
        error = FEFunction(U_,Error)
        Λ = i/(lastindex(x_list)-1)
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        Λ_ = i
        pvd[Λ_] = createvtk(
            Ω,
            file_name_red*"/Result_withError_POD-DEIM_u_phi_sep_$f"* "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
            cellfields=["u"=>uh[1], "phi"=>uh[2], "u_red"=>uh_red[1], "phi_red"=>uh_red[2],"error_u"=>error[1], "error_phi"=>error[2], "error_u_rel"=>error_rel[1], "error_phi_rel"=>error_rel[2]]
        )
        print("\r  $Λ_  ")
    end
    vtk_save(pvd)
end
