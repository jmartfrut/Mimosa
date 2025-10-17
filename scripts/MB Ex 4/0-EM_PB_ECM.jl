##

#region Libraries

using Mimosa
using Mimosa.Drivers
using Gridap
using GridapGmsh
using LineSearches: BackTracking
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.Arrays
using Gridap.Geometry
using CSV
using DataFrames
using SparseMatricesCSR
using LinearAlgebra
using JacobiSVD
using TSVD
using WriteVTK
using JLD2
using Plots
using Base.Threads
 ENV["PYTHON"] = "C:/Users/mjbarillas/AppData/Local/Programs/Python/Python312/python.exe"
using PyCall
using SparseArrays

#endregion

##
##

#region Full Order Solver

function get_trian_and_measure()
    model = GmshDiscreteModel("data/models/PlateBeame10S_BC.msh")
    degree = 4
    Ω = Triangulation(model)
    dΩ = Measure(Ω,degree)
    return model, Ω, dΩ
end

function get_DirichletBC(𝜑ᵇ,conf)
    n_sec = 10
    # conf = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
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
        append!(power_loc,["topsurf_$i"])
        append!(power_val,0.0)
        push!(dir_φ_timesteps,evolφ)
    elseif sw[i]==2
        append!(power_loc,["topsurf_$i"])
        append!(power_val,𝜑ᵇ)
        push!(dir_φ_timesteps,evolφ)
        append!(power_loc,["bottomsurf_$i"])
        append!(power_val,0.0)
        push!(dir_φ_timesteps,evolφ)
    elseif sw[i]==3 #iseven(i)
        append!(power_loc,["bottomsurf_$i"])
        append!(power_val,𝜑ᵇ)
        push!(dir_φ_timesteps,evolφ)
        append!(power_loc,["topsurf_$i"])
        append!(power_val,𝜑ᵇ)
        push!(dir_φ_timesteps,evolφ)
    elseif sw[i]==0 #iseven(i)
        append!(power_loc,["bottomsurf_$i"])
        append!(power_val,0.0)
        push!(dir_φ_timesteps,evolφ)
        append!(power_loc,["topsurf_$i"])
        append!(power_val,0.0)
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

function Increment_Solver(x0,conf,step,nsteps,𝜑ᵇ,f,model, Ω, dΩ,cache,x_list,b_list,bisect,trace=true)
    res, jac = get_symbolic_res_and_jac(dΩ,f)
    dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps),conf)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    norm_res = 1
    count = 0
    x0_copy = copy(x0)
    if trace
        println("\n==============================================")
        println("Material parameter = $f :: step = $step of $nsteps ")
    end
    while norm_res>1e-12
        if count>15 || norm_res>1e2
            bisect += 1
            println("bisect = $bisect")
            x0, cache, _, _, _ = run(x0_copy,conf,step-(1/(2^bisect)),nsteps,𝜑ᵇ,f,model, Ω, dΩ,cache,x_list,b_list,bisect)
            count = 1
        end
        ph = FEFunction(fe_spaces.U, x0)
        b, K_T = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
        Δx = K_T\(-b)
        copyto!(x0,x0+Δx)
        norm_res = maximum(abs.(b))
        count += 1
        a1 = log10(norm_res)
        a2 = log10(norm(Δx))
        if trace
            print("\riter = $count norm_res = $(round(10.0^(a1-trunc(a1)),digits = 3))e$(Int64(trunc(a1))) norm_Δx = $(round(10.0^(a2-trunc(a2)),digits = 3))e$(Int64(trunc(a2)))")
        end
        if norm_res<1e-12
            push!(x_list,copy(x0))
            push!(b_list,copy(b))
        end
    end
    return x0, cache, x_list, b_list
end

function Incremental_Solver(f,conf,trace=true)
    model, Ω, dΩ = nothing, nothing, nothing
    lock(gmsh_lock) do
        model, Ω, dΩ =  get_trian_and_measure()
    end
    nsteps = 300
    𝜑ᵇ = 5000.0
    dirichletbc = get_DirichletBC(0.0,conf)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
    xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
    x0 = vcat(xu, xφ)
    x_list = []
    b_list = []
    cache = nothing
    bisect = 0
    for step in 1:nsteps
        x0, cache, x_list, b_list = Increment_Solver(x0,conf,step,nsteps,𝜑ᵇ,f,model, Ω, dΩ,cache,x_list,b_list,bisect,trace)
    end
    return x_list, b_list
end

# Training data Collection (or single evaluation)
const gmsh_lock = ReentrantLock()
function collect_data()
    # conf_list = [[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]]

    # conf_ = zeros(Int32,20)
    # conf_list = [conf_]
    # for i in 1:20
    #     c = copy(conf_)
    #     c[i] = 1
    #     push!(conf_list,c)
    # end
        
    conf = zeros(Int32,20)
    conf[1] = 1
    conf[3] = 1
    conf[19] = 1

    conf_list = [conf]
    # gmsh_lock = ReentrantLock()
    @threads for conf in conf_list
        f = 1.0
        println("conf = $conf")
        x_list, b_list = Incremental_Solver(f,conf,false)
        df_x = DataFrame(x_list, :auto)
        df_b = DataFrame(b_list, :auto)
        folder = "scripts/MB Ex 4/Full Order Solutions/V1/"
        mkpath(folder * "MaterialModel$(f)_$conf")
        CSV.write(folder*"MaterialModel$(f)_$conf/x_.csv",df_x)
        CSV.write(folder*"MaterialModel$(f)_$conf/b_.csv",df_b)
    end
end

#endregion

##

##

#region POD Matrices and analysis

function Jacobi_SVDs_POD(D_x)

    D_x_u = D_x[[1:u_dofs...],:]
    D_x_𝜑 = D_x[[u_dofs+1:u_dofs+𝜑_dofs...],:]
    U_x_u, σ_i_x_u, V_x_u = jsvd!(D_x_u)
    U_x_𝜑, σ_i_x_𝜑, V_x_𝜑 = jsvd!(D_x_𝜑)

    return U_x_u, σ_i_x_u, V_x_u, U_x_𝜑, σ_i_x_𝜑, V_x_𝜑
end

function SingulaVals_Rel_to_max(σ_i_x_u, σ_i_x_𝜑)
    σ_i_x_rel_u = σ_i_x_u./σ_i_x_u[1]
    σ_i_x_rel_𝜑 = σ_i_x_𝜑./σ_i_x_𝜑[1]
    return σ_i_x_rel_u, σ_i_x_rel_𝜑
end

function MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)
    Zeros_u = zeros(Float64,𝜑_dofs,k)
    Zeros_𝜑 = zeros(Float64,u_dofs,l)
    𝛷 = hcat(vcat(U_x_u[:,[1:k...]],Zeros_u),vcat(Zeros_𝜑,U_x_𝜑[:,[1:l...]]))
    return 𝛷
end

#endregion

##

##

#region POD Solver

function POD_Increment_Solver(
    x0,step,nsteps,𝜑ᵇ,f,conf,
    model, Ω, dΩ,
    cache,x_list,b_list,
    bisect,
    𝛷,
    trace=true
    )
    prev_step = step-(1/2^bisect)
    x0_copy = copy(x0)
    n, k_l = size(𝛷)
    b_POD = Vector{Float64}(undef,k_l)
    K_T_POD_pos_mult = Matrix{Float64}(undef,n,k_l)
    K_T_POD = Matrix{Float64}(undef,k_l,k_l)
    res, jac = get_symbolic_res_and_jac(dΩ,f)
    dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps),conf)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    norm_res = 1
    count = 0
    if trace
        println("\n==============================================")
        println("Material parameter = $f :: step = $step of $nsteps k+l = $k_l")
    end
    if !trace
        print("\rMaterial parameter = $f :: step = $step of $nsteps k+l = $k_l    ")
    end
    while norm_res>1e-12
        if count>20 || norm_res>1e5
            bisect += 1
            if bisect > 5
                throw("bisect = $bisect is too high")            
            end
            if trace
                println("bisect = $bisect")
            end
            x0 = x0_copy
            for i in 1:2^bisect
                x0, cache, _, _ = POD_Increment_Solver(
                    x0,prev_step+(i/2^bisect),nsteps,𝜑ᵇ,f,conf,
                    model, Ω, dΩ,
                    cache,x_list,b_list,
                    bisect,
                    𝛷,
                    trace
                    )
                if prev_step+(i/2^bisect)>step
                    break                    
                end
            end
            count = 0
        end
        ph = FEFunction(fe_spaces.U, 𝛷*x0)
        b, K_T = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
        mul!(b_POD,𝛷',b)
        mul!(K_T_POD_pos_mult,K_T,𝛷)
        mul!(K_T_POD,𝛷',K_T_POD_pos_mult)
        Δx = K_T_POD\(-b_POD)
        copyto!(x0,x0+Δx)
        norm_res = maximum(abs.(b_POD))
        count += 1
        a1 = log10(norm_res)
        a2 = log10(norm(Δx))
        if trace
            print("\riter = $count norm_res = $(round(10.0^(a1-trunc(a1)),digits = 3))e$(Int64(trunc(a1))) norm_Δx = $(round(10.0^(a2-trunc(a2)),digits = 3))e$(Int64(trunc(a2)))")
        end
        if norm_res<1e-12
            push!(x_list,copy(x0))
            push!(b_list,copy(b_POD))
        end
    end
    return x0, cache, x_list, b_list
end

function POD_Incremental_Solver(𝛷,f,conf,trace=true)
    model, Ω, dΩ = nothing, nothing, nothing
    lock(gmsh_lock) do
        model, Ω, dΩ =  get_trian_and_measure()
    end
    nsteps = 300
    𝜑ᵇ = 5000.0
    n, k_l = size(𝛷)
    x0 = zeros(Float64,k_l)
    x_list = []
    b_list = []
    cache = nothing
    bisect = 0
    for step in 1:nsteps
        x0, cache, x_list, b_list = POD_Increment_Solver(
            x0,step,nsteps,𝜑ᵇ,f,conf,
            model, Ω, dΩ,
            cache,x_list,b_list,
            bisect,
            𝛷,
            trace
        )
    end
    return x_list, b_list
end

function compute_residual_contributions(model,dΩ,x0,𝛷,f,conf,step,nsteps,𝜑ᵇ)
    dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps),conf)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    assem = SparseMatrixAssembler(fe_spaces.U,fe_spaces.V)
    ph = FEFunction(fe_spaces.U, 𝛷*x0)
    res, jac = get_symbolic_res_and_jac(dΩ,f)
    RES = res(ph,get_fe_basis(fe_spaces.V))
    res_contributions = collect_cell_vector(fe_spaces.V,RES)
    glo_res_element_contributions = []
    for i in eachindex(res_contributions[2][1])
        b = allocate_vector(assem,res_contributions)
        b[filter(x -> x >= 0, res_contributions[2][1][i][1])] = res_contributions[1][1][i][1][findall(x -> x > 0, res_contributions[2][1][i][1])]
        b[filter(x -> x >= 0, res_contributions[2][1][i][2])] = res_contributions[1][1][i][1][findall(x -> x > 0, res_contributions[2][1][i][2])]
        push!(glo_res_element_contributions,𝛷'*b)
    end
    return glo_res_element_contributions
end

function POD_Incremental_Solver_store_contributions(𝛷,f,conf,trace=true)
    model, Ω, dΩ = nothing, nothing, nothing
    lock(gmsh_lock) do
        model, Ω, dΩ =  get_trian_and_measure()
    end
    nsteps = 300
    𝜑ᵇ = 5000.0
    n, k_l = size(𝛷)
    x0 = zeros(Float64,k_l)
    x_list = []
    b_list = []
    r_contri_list  = []
    cache = nothing
    bisect = 0
    for step in 1:nsteps
        x0, cache, x_list, b_list = POD_Increment_Solver(
            x0,step,nsteps,𝜑ᵇ,f,
            model, Ω, dΩ,
            cache,x_list,b_list,
            bisect,
            𝛷,
            trace
        )
        r_contribution = compute_residual_contributions(model,dΩ,x0,𝛷,f,conf,step,nsteps,𝜑ᵇ)
        push!(r_contri_list,r_contribution)
    end
    return x_list, b_list, r_contri_list
end

# const gmsh_lock = ReentrantLock()
function POD_collect_data()
    f = 1.05
    k_list = [1,3,5,10,15,17,20,25,30]
    l_list = [1,3,5,10,12,15,20]
    k_l_list = []
    for k in k_list, l in l_list
        push!(k_l_list,(k,l))
    end
    total = length(k_l_list)
    # gmsh_lock = ReentrantLock()
    i = 0
    @threads for k_l in k_l_list
        k, l = k_l
        𝛷 = MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)
        println("Current k l = $k_l")
        x_list, b_list = POD_Incremental_Solver(𝛷,f,conf,false)
        df_x = DataFrame(x_list, :auto)
        df_b = DataFrame(b_list, :auto)
        folder = "scripts/MB Ex 3/POD_red_Solutions/V1/"
        mkpath(folder * "MaterialParameter$f")
        CSV.write(folder*"MaterialParameter$f/RedParam_k_$(k)_l_$(l)x_.csv",df_x)
        CSV.write(folder*"MaterialParameter$f/RedParam_k_$(k)_l_$(l)b_.csv",df_b)
        i += 1
        E = Max_Error_rel(x_list,k,l,f)
        println("Current executed k l = $k_l :: Executed = $i out of $total :: Error = $E")
    end
end

function POD_read_or_collect_data()
    f = 1.05
    k_list = [1,3,5,10,15,17,20,25,30]
    l_list = [1,3,5,10,12,15,20]
    k_l_list = []
    for k in k_list, l in l_list
        push!(k_l_list,(k,l))
    end
    total = length(k_l_list)
    # gmsh_lock = ReentrantLock()
    i = 0
    folder = "scripts/MB Ex 3/POD_red_Solutions/V1/"
    for k_l in k_l_list
        k, l = k_l
        try
            file_name = folder*"MaterialParameter$f/RedParam_k_$(k)_l_$(l)x_.csv"
            x_list = CSV.File(file_name) |> Tables.matrix
            println("Current read k l = $k_l :: Executed = $i out of $total")
            E = Max_Error_rel(eachcol(x_list),k,l,f)
            println("Current read k l = $k_l :: Executed = $i out of $total :: Error = $E")
        catch
            𝛷 = MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)
            println("Current computation k l = $k_l")
            try
                x_list, b_list = POD_Incremental_Solver(𝛷,f,false)
                df_x = DataFrame(x_list, :auto)
                df_b = DataFrame(b_list, :auto)
                mkpath(folder * "MaterialParameter$f")
                CSV.write(folder*"MaterialParameter$f/RedParam_k_$(k)_l_$(l)x_.csv",df_x)
                CSV.write(folder*"MaterialParameter$f/RedParam_k_$(k)_l_$(l)b_.csv",df_b)
                i += 1
                E = Max_Error_rel(x_list,k,l,f)
                println("Current executed k l = $k_l :: Executed = $i out of $total :: Error = $E")
            catch
                println("Failed simulation")
            end
        end
    end
end

function POD_read_and_error()
    f = 1.05
    k_list = [1,3,5,10,15,17,20,25,30]
    l_list = [1,3,5,10,12,15,20]
    k_l_list = []
    for k in k_list, l in l_list
        push!(k_l_list,(k,l))
    end
    total = length(k_l_list)
    # gmsh_lock = ReentrantLock()
    i = 0
    folder = "scripts/MB Ex 3/POD_red_Solutions/V1/"
    Error_list = Dict()
    for k_l in k_l_list
        k, l = k_l
        try
            file_name = folder*"MaterialParameter$f/RedParam_k_$(k)_l_$(l)x_.csv"
            x_list = CSV.File(file_name) |> Tables.matrix
            println("Current read k l = $k_l :: Executed = $i out of $total")
            E = Max_Error_rel(eachcol(x_list),k,l,f)
            Error_list["$k_l"] = E
            println("Current read k l = $k_l :: Executed = $i out of $total :: Error = $E")
        catch
            E = 10
            Error_list["$k_l"] = E
            println("Current read k l = $k_l :: Executed = $i out of $total :: Error = $E")
            
        end
    end
    jldsave("scripts/MB Ex 3/POD_red_Solutions/V1/MaterialParameter$f/Error_list.jld2",Error_list = Error_list)
end

function POD_collect_data_res_contributions()
    f_list = [0.9,1.0,1.1]
    k, l = 25,18
    i = 0
    for f in f_list
        𝛷 = MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)
        println("Current f = $f")
        x_list, b_list, r_contri_list = POD_Incremental_Solver_store_contributions(𝛷,f,false)
        folder = "scripts/MB Ex 3/POD_red_Solutions/V4_r_contributions/"
        mkpath(folder * "MaterialParameter$f")
        jldsave(folder*"MaterialParameter$f/RedParam_k_$(k)_l_$(l)_f_$f.jld2",x_list = x_list, b_list = b_list, r_contri_list = r_contri_list )
        i += 1
        println("Current executed f = $f")
    end
end

#endregion
##

##

##

#region Read Full Order training data

𝜑_dofs = 4830
u_dofs = 36147

function Training_Set_Read(v)
    folder = "scripts/MB Ex 4/Full Order Solutions/V$v/"

    D_x = []
    D_T= []

    conf_ = zeros(Int32,20)
    conf_list = [conf_]
    for i in 1:20
        c = copy(conf_)
        c[i] = 1
        push!(conf_list,c)
    end

    for conf in conf_list
        f = 1.0
        file_name = folder*"MaterialModel$(f)_$conf" * "/x_.csv"
        _X = CSV.File(file_name) |> Tables.matrix
        push!(D_x,_X)
        file_name = folder*"MaterialModel$(f)_$conf" * "/b_.csv"
        _T = CSV.File(file_name) |> Tables.matrix
        push!(D_T,_T)
    end

    D_x = reduce(hcat,D_x)
    D_T = reduce(hcat,D_T)
    return D_x, D_T
end

function Training_Set_Read(v,conf_list)
    folder = "scripts/MB Ex 4/Full Order Solutions/V$v/"

    D_x = []
    D_T= []

    for conf in conf_list
        f = 1.0
        file_name = folder*"MaterialModel$(f)_$conf" * "/x_.csv"
        _X = CSV.File(file_name) |> Tables.matrix
        push!(D_x,_X)
        file_name = folder*"MaterialModel$(f)_$conf" * "/b_.csv"
        _T = CSV.File(file_name) |> Tables.matrix
        push!(D_T,_T)
    end

    D_x = reduce(hcat,D_x)
    D_T = reduce(hcat,D_T)
    return D_x, D_T
end

#endregion

##

##

#region Error evaluation

function Max_Error_rel(x_list,k,l,f,conf)
    folder = "scripts/MB Ex 4/Full Order Solutions/V1/"
    file_name = folder*"MaterialModel$(f)_$conf/x_.csv"
    _X = CSV.File(file_name) |> Tables.matrix
    𝛷 = MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)
    i = 1
    Error_list = []
    for x_red in x_list
        x = 𝛷*x_red
        _x = _X[:,i]
        Error_u = abs.(_x[[1:u_dofs...]] - x[[1:u_dofs...]])./maximum(abs.(_x[[1:u_dofs...]]))
        push!(Error_list,maximum(Error_u))
        i += 1
    end
    return sum(Error_list)/length(Error_list)
end

function Max_Error_rel_list(x_list,k,l,f,conf)
    folder = "scripts/MB Ex 4/Full Order Solutions/V1/"
    file_name = folder*"MaterialModel$(f)_$conf/x_.csv"
    _X = CSV.File(file_name) |> Tables.matrix
    𝛷 = MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)
    i = 1
    Error_list = []
    for x_red in x_list
        x = 𝛷*x_red
        _x = _X[:,i]
        Error_u = abs.(_x[[1:u_dofs...]] - x[[1:u_dofs...]])./maximum(abs.(_x[[1:u_dofs...]]))
        push!(Error_list,maximum(Error_u))
        i += 1
    end
    return Error_list
end

function Max_Error_rel_VTK(x_list,k,l,f,conf)
    model, Ω, dΩ =  get_trian_and_measure()
    folder = "scripts/MB Ex 4/Full Order Solutions/V1/"
    file_name = folder*"MaterialModel$(f)_$conf/x_.csv"
    _X = CSV.File(file_name) |> Tables.matrix
    𝛷 = MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)
    dirichletbc = get_DirichletBC(0.0,conf) #lastindex(x_list)))
    fe_spaces = get_fe_spaces(model,dirichletbc)
    U_ = fe_spaces.U
    i = 1
    Error_list = []
    mkpath(folder*"/Result_withError_POD-k_$(k)_l_$(l)")
    pvd = paraview_collection(folder*"/Result_withError_POD-k_$(k)_l_$(l)" * "/Results", append=false)
    writevtk(model, folder*"/Result_withError_POD-k_$(k)_l_$(l)" * "/DiscreteModel")
    for x_red in x_list
        x = 𝛷*x_red
        _x = _X[:,i]
        Error_u = abs.(_x[[1:u_dofs...]] - x[[1:u_dofs...]])./maximum(abs.(_x[[1:u_dofs...]]))
        Error_𝜑 = abs.(_x[[u_dofs+1:u_dofs+𝜑_dofs...]] - x[[u_dofs+1:u_dofs+𝜑_dofs...]])./maximum(abs.(x[[u_dofs+1:u_dofs+𝜑_dofs...]]))
        dirichletbc = get_DirichletBC(5000.0*(i/lastindex(x_list)),conf) #lastindex(x_list)))
        fe_spaces = get_fe_spaces(model,dirichletbc)
        U = fe_spaces.U
        uh_red = FEFunction(U,x)
        uh = FEFunction(U,_x)
        Error_rel = vcat(Error_u,Error_𝜑)
        error_rel = FEFunction(U_,Error_rel)
         Λ = i/(lastindex(x_list))
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        Λ_ = i
        pvd[Λ_] = createvtk(
            Ω,
            folder*"/Result_withError_POD-k_$(k)_l_$(l)"* "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
            cellfields=["u"=>uh[1], "phi"=>uh[2], "u_red"=>uh_red[1], "phi_red"=>uh_red[2], "error_u_rel"=>error_rel[1], "error_phi_rel"=>error_rel[2]]
        )
        print("\r  $Λ_  ")
        push!(Error_list,maximum(Error_u))
        i += 1
    end
    vtk_save(pvd)
end

#endregion

##

##

#region Initial tests
folder = "scripts/MB Ex 4/Full Order Solutions/V1/"


model, Ω, dΩ = get_trian_and_measure()
dirichletbc = get_DirichletBC(0.0,[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
fe_spaces = get_fe_spaces(model,dirichletbc)
xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
x0 = vcat(xu, xφ)

v = 1
D_x, D_T = Training_Set_Read(v)
U_x_u, σ_i_x_u, V_x_u, U_x_𝜑, σ_i_x_𝜑, V_x_𝜑 = Jacobi_SVDs_POD(D_x)
σ_i_x_rel_u, σ_i_x_rel_𝜑 = SingulaVals_Rel_to_max(σ_i_x_u, σ_i_x_𝜑)
plotlyjs()
w = 500
plot(
    [σ_i_x_rel_u[[1:w...]],σ_i_x_rel_𝜑[[1:w...]]],
    yscale=:log10,
    xlabel = "Modes", ylabel = "σ_i/σ_max",
    label = ["Dₓ_u" "Dₓ_phi" ],
    ylims = (10^-float(20), 1),
    yticks=[10^-float(i*4) for i in 0:5]
)

jldsave(folder * "POD_SVD_matrix.jld2",U_x_u=U_x_u, σ_i_x_u=σ_i_x_u, V_x_u=V_x_u, U_x_𝜑=U_x_𝜑, σ_i_x_𝜑=σ_i_x_𝜑, V_x_𝜑=V_x_𝜑)

σ_i_x_selection_u = findall(x -> x>1e-12,σ_i_x_rel_u)
σ_i_x_selection_𝜑 = findall(x -> x>1e-12,σ_i_x_rel_𝜑)

k = length(σ_i_x_selection_u)
l = length(σ_i_x_selection_𝜑)

𝛷 = MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)

f = 1.0
conf = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]


x_list, b_list = POD_Incremental_Solver(𝛷,f,conf,true)
Max_Error_rel(x_list,k,l,f,conf)
Error_list = Max_Error_rel_list(x_list,k,l,f,conf)
plot(Error_list)
Max_Error_rel_VTK(x_list,k,l,f,conf)


σ_i_x_selection_u = findall(x -> x>1e-16,σ_i_x_rel_u)
σ_i_x_selection_𝜑 = findall(x -> x>1e-16,σ_i_x_rel_𝜑)

k = length(σ_i_x_selection_u)
l = length(σ_i_x_selection_𝜑)

𝛷 = MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)

f = 1.0
conf = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]


x_list, b_list = POD_Incremental_Solver(𝛷,f,conf,true)
Max_Error_rel(x_list,k,l,f,conf)
Error_list = Max_Error_rel_list(x_list,k,l,f,conf)
plot(Error_list)

conf = zeros(Int16,20)
conf[1] = 1
conf[3] = 1
x_list, b_list = POD_Incremental_Solver(𝛷,f,conf,true)
Max_Error_rel(x_list,k,l,f,conf)
Error_list = Max_Error_rel_list(x_list,k,l,f,conf)
plot(Error_list)

conf_list = []
conf = zeros(Int32,20)
conf[1] = 1
push!(conf_list,conf)
conf = zeros(Int32,20)
conf[3] = 1
push!(conf_list,conf)

D_x, D_T = Training_Set_Read(v,conf_list)

U_x_u, σ_i_x_u, V_x_u, U_x_𝜑, σ_i_x_𝜑, V_x_𝜑 = Jacobi_SVDs_POD(D_x)
σ_i_x_rel_u, σ_i_x_rel_𝜑 = SingulaVals_Rel_to_max(σ_i_x_u, σ_i_x_𝜑)
plotlyjs()
w = 200
plot(
    [σ_i_x_rel_u[[1:w...]],σ_i_x_rel_𝜑[[1:w...]]],
    yscale=:log10,
    xlabel = "Modes", ylabel = "σ_i/σ_max",
    label = ["Dₓ_u" "Dₓ_phi" ],
    ylims = (10^-float(20), 1),
    yticks=[10^-float(i*4) for i in 0:5]
)

σ_i_x_selection_u = findall(x -> x>1e-16,σ_i_x_rel_u)
σ_i_x_selection_𝜑 = findall(x -> x>1e-16,σ_i_x_rel_𝜑)

k = length(σ_i_x_selection_u)
l = length(σ_i_x_selection_𝜑)

𝛷 = MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)

f = 1.0

conf = zeros(Int16,20)
conf[1] = 1
conf[3] = 1

x_list, b_list = POD_Incremental_Solver(𝛷,f,conf,true)
Max_Error_rel(x_list,k,l,f,conf)
Error_list = Max_Error_rel_list(x_list,k,l,f,conf)
plot(Error_list)
Max_Error_rel_VTK(x_list,k,l,f,conf)


conf_list = []
conf = zeros(Int32,20)
conf[1] = 1
push!(conf_list,conf)
conf = zeros(Int32,20)
conf[3] = 1
push!(conf_list,conf)
conf = zeros(Int32,20)
conf[1] = 1
conf[5] = 1
push!(conf_list,conf)

D_x, D_T = Training_Set_Read(v,conf_list)

U_x_u, σ_i_x_u, V_x_u, U_x_𝜑, σ_i_x_𝜑, V_x_𝜑 = Jacobi_SVDs_POD(D_x)
σ_i_x_rel_u, σ_i_x_rel_𝜑 = SingulaVals_Rel_to_max(σ_i_x_u, σ_i_x_𝜑)
plotlyjs()
w = 200
plot(
    [σ_i_x_rel_u[[1:w...]],σ_i_x_rel_𝜑[[1:w...]]],
    yscale=:log10,
    xlabel = "Modes", ylabel = "σ_i/σ_max",
    label = ["Dₓ_u" "Dₓ_phi" ],
    ylims = (10^-float(20), 1),
    yticks=[10^-float(i*4) for i in 0:5]
)

σ_i_x_selection_u = findall(x -> x>1e-16,σ_i_x_rel_u)
σ_i_x_selection_𝜑 = findall(x -> x>1e-16,σ_i_x_rel_𝜑)

k = length(σ_i_x_selection_u)
l = length(σ_i_x_selection_𝜑)

𝛷 = MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)

f = 1.0

conf = zeros(Int32,20)
conf[1] = 1
conf[3] = 1

x_list, b_list = POD_Incremental_Solver(𝛷,f,conf,true)
Max_Error_rel(x_list,k,l,f,conf)
Error_list = Max_Error_rel_list(x_list,k,l,f,conf)
plot(Error_list)
Max_Error_rel_VTK(x_list,k,l,f,conf)

conf_list = []
conf = zeros(Int32,20)
conf[1] = 1
push!(conf_list,conf)
conf = zeros(Int32,20)
conf[3] = 1
push!(conf_list,conf)
conf = zeros(Int32,20)
conf[1] = 1
conf[5] = 1
push!(conf_list,conf)
conf = zeros(Int32,20)
conf[1] = 1
conf[3] = 1
conf[19] = 1
push!(conf_list,conf)

v = 1
D_x, D_T = Training_Set_Read(v,conf_list)

g,h = size(D_x)

skip = 9
D_ = D_x[:,1]
count = 1
for i in 1:h
    if i==count+skip
        D_ = hcat(D_,D_x[:,i])
        count = count+skip
    end
end


U_x_u, σ_i_x_u, V_x_u, U_x_𝜑, σ_i_x_𝜑, V_x_𝜑 = Jacobi_SVDs_POD(D_)
σ_i_x_rel_u, σ_i_x_rel_𝜑 = SingulaVals_Rel_to_max(σ_i_x_u, σ_i_x_𝜑)
plotlyjs()
w = 100
plot(
    [σ_i_x_rel_u[[1:w...]],σ_i_x_rel_𝜑[[1:w...]]],
    yscale=:log10,
    xlabel = "Modes", ylabel = "σ_i/σ_max",
    label = ["Dₓ_u" "Dₓ_phi" ],
    ylims = (10^-float(20), 1),
    yticks=[10^-float(i*4) for i in 0:5]
)

σ_i_x_selection_u = findall(x -> x>1e-16,σ_i_x_rel_u)
σ_i_x_selection_𝜑 = findall(x -> x>1e-16,σ_i_x_rel_𝜑)

k = length(σ_i_x_selection_u)
l = length(σ_i_x_selection_𝜑)

𝛷 = MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)

f = 1.0

conf = zeros(Int32,20)
conf[1] = 1
conf[3] = 1

x_list, b_list = POD_Incremental_Solver(𝛷,f,conf,true)
Max_Error_rel(x_list,k,l,f,conf)
Error_list = Max_Error_rel_list(x_list,k,l,f,conf)
plot(Error_list)
Max_Error_rel_VTK(x_list,k,l,f,conf)

folder = "scripts/MB Ex 4/Full Order Solutions/V1/"
file_name = folder*"MaterialModel$(f)_$conf/x_.csv"
_X = CSV.File(file_name) |> Tables.matrix
𝛷 = MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)
i = 1
x = 𝛷*x_list[end]
_x = _X[:,end]
Error_u = abs.(_x[[1:u_dofs...]] - x[[1:u_dofs...]])./maximum(abs.(_x[[1:u_dofs...]]))
maximum(Error_u)
i += 1
#endregion

##