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

function Increment_Solver(x0,step,nsteps,𝜑ᵇ,f,model, Ω, dΩ,cache,x_list,b_list,bisect,trace=true)
    res, jac = get_symbolic_res_and_jac(dΩ,f)
    dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps))
    fe_spaces = get_fe_spaces(model,dirichletbc)
    norm_res = 1
    count = 0
    x0_copy = copy(x0)
    if trace
        println("\n==============================================")
        print("Material parameter = $f :: step = $step of $nsteps ")
    else
        print("\rMaterial parameter = $f :: step = $step of $nsteps ")
    end
    while norm_res>1e-12
        if count>15 || norm_res>1e2
            bisect += 1
            println("bisect = $bisect")
            x0, cache, _, _, _ = run(x0_copy,step-(1/(2^bisect)),nsteps,𝜑ᵇ,f,model, Ω, dΩ,cache,x_list,b_list,bisect)
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

function Incremental_Solver(f,trace=true)
    model, Ω, dΩ = nothing, nothing, nothing
    lock(gmsh_lock) do
        model, Ω, dΩ =  get_trian_and_measure()
    end
    nsteps = 300
    𝜑ᵇ = 5000.0
    dirichletbc = get_DirichletBC(0.0)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
    xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
    x0 = vcat(xu, xφ)
    x_list = []
    b_list = []
    cache = nothing
    bisect = 0
    for step in 1:nsteps
        x0, cache, x_list, b_list = Increment_Solver(x0,step,nsteps,𝜑ᵇ,f,model, Ω, dΩ,cache,x_list,b_list,bisect,trace)
    end
    return x_list, b_list
end

function Incremental_Solver_n_steps(f,nsteps,trace=true)
    model, Ω, dΩ = nothing, nothing, nothing
    lock(gmsh_lock) do
        model, Ω, dΩ =  get_trian_and_measure()
    end
    # nsteps = 300
    𝜑ᵇ = 5000.0
    dirichletbc = get_DirichletBC(0.0)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
    xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
    x0 = vcat(xu, xφ)
    x_list = []
    b_list = []
    cache = nothing
    bisect = 0
    for step in 1:nsteps
        x0, cache, x_list, b_list = Increment_Solver(x0,step,nsteps,𝜑ᵇ,f,model, Ω, dΩ,cache,x_list,b_list,bisect,trace)
    end
    return x_list, b_list
end

# Training data Collection (or single evaluation)
const gmsh_lock = ReentrantLock()
function collect_data()
    f_list = [0.9,1.0,1.05,1.1]
    # gmsh_lock = ReentrantLock()
    @threads for i in 1:lastindex(f_list)
        f = f_list[i]
        println("f = $f")
        x_list, b_list = Incremental_Solver(f,false)
        df_x = DataFrame(x_list, :auto)
        df_b = DataFrame(b_list, :auto)
        folder = "scripts/MB Ex 3/Full Order Solutions/V1/"
        mkpath(folder * "MaterialModel$f")
        CSV.write(folder*"MaterialModel$f/x_.csv",df_x)
        CSV.write(folder*"MaterialModel$f/b_.csv",df_b)
    end
end

#endregion

##

##

#region Read Full Order training data

𝜑_dofs = 7245
u_dofs = 36147

function Training_Set_Read(v)
    folder = "scripts/MB Ex 3/Full Order Solutions/V$v/"
    D_x = []
    D_T= []
    f_list = [0.9,1.0,1.1]
    for i in 1:3
        f = f_list[i]
        file_name = folder*"MaterialModel$f/x_.csv"
        _X = CSV.File(file_name) |> Tables.matrix
        push!(D_x,_X)
        file_name = folder*"MaterialModel$f/b_.csv"
        _T = CSV.File(file_name) |> Tables.matrix
        push!(D_T,_T)
    end

    D_x = reduce(hcat,D_x)
    D_T = reduce(hcat,D_T)
    return D_x, D_T
end

function Assemble_contributions(v,k,l)
    folder = "scripts/MB Ex 3/POD_red_Solutions/V$(v)_r_contributions/"
    f_list = [0.9,1.0,1.1]
    C_list = []
    for f in f_list
        sol = jldopen(folder*"MaterialParameter$f/RedParam_k_$(k)_l_$(l)_f_$f.jld2")
        C_ = reduce(vcat,reduce.(hcat,sol["r_contri_list"]))
        push!(C_list,C_)
    end
    C = reduce(vcat,C_list)
    b = vec(sum(C,dims=2))
    return C, b
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

#region ECM Matrices

function RondomizedSVD(C,tol=1e-8)
    pushfirst!(pyimport("sys")."path", "C:/Users/mjbarillas/Documents/GitHub/Mimosa/scripts/Scratch")
    RandomizedSVD_module = pyimport("randomized_singular_value_decomposition")
    u, _, _, _ = RandomizedSVD_module.RandomizedSingularValueDecomposition().Calculate(C', truncation_tolerance=tol)
    return u
end

function ECM_Selection(u,tol=1e-6)
    pushfirst!(pyimport("sys")."path", "C:/Users/mjbarillas/Documents/GitHub/Mimosa/scripts/Scratch")
    ECM_module = pyimport("empirical_cubature_method")
    ecm = ECM_module.EmpiricalCubatureMethod(ECM_tolerance=tol)
    ecm.SetUp(
        ResidualsBasis=u,
        InitialCandidatesSet=nothing,
        constrain_sum_of_weights=true,
        constrain_conditions=false
    )
    ecm.Run()
    return   ecm.z.+1, ecm.w
end

#endregion

##


##

#region POD Solver

function POD_Increment_Solver(
    x0,step,nsteps,𝜑ᵇ,f,
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
    dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps))
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
                    x0,prev_step+(i/2^bisect),nsteps,𝜑ᵇ,f,
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

function POD_Incremental_Solver(𝛷,f,trace=true)
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
            x0,step,nsteps,𝜑ᵇ,f,
            model, Ω, dΩ,
            cache,x_list,b_list,
            bisect,
            𝛷,
            trace
        )
    end
    return x_list, b_list
end

function compute_residual_contributions(model,dΩ,x0,𝛷,f,step,nsteps,𝜑ᵇ)
    dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps))
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

function POD_Incremental_Solver_store_contributions(𝛷,f,trace=true)
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
        r_contribution = compute_residual_contributions(model,dΩ,x0,𝛷,f,step,nsteps,𝜑ᵇ)
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
        x_list, b_list = POD_Incremental_Solver(𝛷,f,false)
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

#region ECM Solver

function el_proyection(𝛷,z,model,dΩ,f)
    dirichletbc = get_DirichletBC(0.0)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
    xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
    x0 = vcat(xu, xφ)
    ph = FEFunction(fe_spaces.U, x0)
    _, jac = get_symbolic_res_and_jac(dΩ,f)
    JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    n, k_l = size(𝛷)
    jac_contributions = collect_cell_matrix(fe_spaces.U,fe_spaces.V,JAC)
    𝛷_el_z = []
    for el_id in z
        el_dofs = vcat(jac_contributions[2][1][el_id][1],jac_contributions[2][1][el_id][2])
        Inci_el = sparse(filter(x -> x >= 0,el_dofs),findall(x -> x >= 0,el_dofs), fill(1,count(x -> x >= 0, el_dofs)),n,108)
        𝛷_el = Inci_el'*𝛷
        push!(𝛷_el_z,𝛷_el)
    end
    return 𝛷_el_z
end

function ECM_POD_get_numeric_res_and_jac(ph,fe_spaces,res,jac,z,w,𝛷_el_z,k_l)
    RES = res(ph,get_fe_basis(fe_spaces.V))
    JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    res_contributions = collect_cell_vector(fe_spaces.V,RES)
    jac_contributions = collect_cell_matrix(fe_spaces.U,fe_spaces.V,JAC)
    b_red = zeros(Float64,k_l)
    k_red = zeros(Float64,k_l,k_l)
    for (z_,w_,𝛷_el) in zip(z,w,𝛷_el_z)
        b_el = vcat(res_contributions[1][1][z_][1],res_contributions[1][1][z_][2])
        b_red += w_*𝛷_el'*b_el
        k_el = vcat(hcat(jac_contributions[1][1][z_][1],jac_contributions[1][1][z_][3]),hcat(jac_contributions[1][1][z_][2],jac_contributions[1][1][z_][4]))
        k_red += w_*𝛷_el'*k_el*𝛷_el
    end
    return b_red, k_red
end

function ECM_POD_get_numeric_res_and_jac_threads(ph,fe_spaces,res,jac,z,w,𝛷_el_z,k_l)
    RES = res(ph,get_fe_basis(fe_spaces.V))
    JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    res_contributions = collect_cell_vector(fe_spaces.V,RES)
    jac_contributions = collect_cell_matrix(fe_spaces.U,fe_spaces.V,JAC)
    b_red = zeros(Float64,k_l)
    k_red = zeros(Float64,k_l,k_l)
    a = lastindex(z)
    b_red_ = [Vector{Float64}(undef,k_l) for _ in 1:a]
    k_red_ = [Matrix{Float64}(undef,k_l,k_l) for _ in 1:a]
    @threads for i in 1:a
        z_,w_,𝛷_el = z[i],w[i],𝛷_el_z[i]
        b = res_contributions[1][1][z_].array;
        b_el = vcat(b[1],b[2])
        b_red_[i] = w_*𝛷_el'*b_el
        # Threads.atomic_add!(b_red,w_*𝛷_el'*b_el)
        a = jac_contributions[1][1][z_].array;
        k_el = hvcat((2,2),a[1,1],a[1,2],a[2,1],a[2,2]);
        k_red_[i] = w_*𝛷_el'*k_el*𝛷_el
        # Threads.atomic_add!(k_red,w_*𝛷_el'*k_el*𝛷_el)
    end
    b_red = sum(b_red_)
    k_red = sum(k_red_)
    return b_red, k_red
end

function POD_ECM_Increment_Solver(
    x0,step,nsteps,𝜑ᵇ,f,
    model, Ω, dΩ,
    cache,x_list,b_list,
    bisect,
    𝛷, z, w, 𝛷_el_z,
    trace=true
    )
    prev_step = step-(1/2^bisect)
    x0_copy = copy(x0)
    n, k_l = size(𝛷)
    res, jac = get_symbolic_res_and_jac(dΩ,f)
    dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps))
    fe_spaces = get_fe_spaces(model,dirichletbc)
    norm_res = 1
    count = 0
    if trace
        println("\n==============================================")
        print("Material parameter = $f :: step = $step of $nsteps k+l = $k_l elements = $(length(z)) ")
    else
        print("\rMaterial parameter = $f :: step = $step of $nsteps k+l = $k_l elements = $(length(z)) ")
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
                x0, cache, _, _ = POD_ECM_Increment_Solver(
                    x0,prev_step+(i/2^bisect),nsteps,𝜑ᵇ,f,
                    model, Ω, dΩ,
                    cache,x_list,b_list,
                    bisect,
                    𝛷, z, w, 𝛷_el_z,
                    )
                if prev_step+(i/2^bisect)>step
                    break                    
                end
            end
            count = 0
        end
        ph = FEFunction(fe_spaces.U, 𝛷*x0)
        b_POD, K_T_POD = ECM_POD_get_numeric_res_and_jac_threads(ph,fe_spaces,res,jac,z,w,𝛷_el_z,k_l)
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

function POD_ECM_Incremental_Solver(𝛷,z,w,f,trace=true)
    model, Ω, dΩ = nothing, nothing, nothing
    lock(gmsh_lock) do
        model, Ω, dΩ =  get_trian_and_measure()
    end
    nsteps = 300
    𝜑ᵇ = 5000.0
    n, k_l = size(𝛷)
    x0 = zeros(Float64,k_l)
    𝛷_el_z = el_proyection(𝛷,z,model,dΩ,f)
    x_list = []
    b_list = []
    cache = nothing
    bisect = 0
    for step in 1:nsteps
        x0, cache, x_list, b_list = POD_ECM_Increment_Solver(
            x0,step,nsteps,𝜑ᵇ,f,
            model, Ω, dΩ,
            cache,x_list,b_list,
            bisect,
            𝛷, z, w, 𝛷_el_z,
            trace
            )
    end
    return x_list, b_list
end

function POD_ECM_Incremental_Solver_n_steps(𝛷,z,w,f,nsteps,trace=true)
    model, Ω, dΩ = nothing, nothing, nothing
    lock(gmsh_lock) do
        model, Ω, dΩ =  get_trian_and_measure()
    end
    # nsteps = 300
    𝜑ᵇ = 5000.0
    n, k_l = size(𝛷)
    x0 = zeros(Float64,k_l)
    𝛷_el_z = el_proyection(𝛷,z,model,dΩ,f)
    x_list = []
    b_list = []
    cache = nothing
    bisect = 0
    for step in 1:nsteps
        x0, cache, x_list, b_list = POD_ECM_Increment_Solver(
            x0,step,nsteps,𝜑ᵇ,f,
            model, Ω, dΩ,
            cache,x_list,b_list,
            bisect,
            𝛷, z, w, 𝛷_el_z,
            trace
            )
    end
    return x_list, b_list
end

#endregion

##


##

#region Error evaluation

function Max_Error_rel(x_list,k,l,f)
    folder = "scripts/MB Ex 3/Full Order Solutions/V1/"
    file_name = folder*"MaterialModel$f/x_.csv"
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

#endregion

##


##

#region POD analysis

D_x, D_T = Training_Set_Read(1)
U_x_u, σ_i_x_u, V_x_u, U_x_𝜑, σ_i_x_𝜑, V_x_𝜑 = Jacobi_SVDs_POD(D_x)

D_x_u = U_x_u[:,[1:20...]]*diagm(σ_i_x_u)[[1:20...],[1:20...]]*V_x_u[[1:20...],:]
D_x_𝜑 = U_x_𝜑[:,[1:20...]]*diagm(σ_i_x_𝜑)[[1:20...],[1:20...]]*V_x_𝜑[[1:20...],:]
D_x_ = vcat(D_x_u,D_x_𝜑)
norm(D_x-D_x_)/norm(D_x)
heatmap(D_x-D_x_)


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
savefig("scripts/MB Ex 3/POD_red_Solutions/V1/MaterialParameter$f/POD_SingularValuesDecay.svg")
savefig("scripts/MB Ex 3/POD_red_Solutions/V1/MaterialParameter$f/POD_SingularValuesDecay.png")
savefig("C:/Users/mjbarillas/Documents/LaTeX/POD_ECM_Notes/POD_SingularValuesDecay.svg")
σ_i_x_selection_u = findall(x -> x>1e-12,σ_i_x_rel_u)
σ_i_x_selection_𝜑 = findall(x -> x>1e-12,σ_i_x_rel_𝜑)

σ_i_x_selection_u = findall(x -> x>1e-8,σ_i_x_rel_u)
σ_i_x_selection_𝜑 = findall(x -> x>1e-8,σ_i_x_rel_𝜑)

σ_i_x_selection_u = findall(x -> x>1e-15,σ_i_x_rel_u)
σ_i_x_selection_𝜑 = findall(x -> x>1e-15,σ_i_x_rel_𝜑)

k = length(σ_i_x_selection_u)
l = length(σ_i_x_selection_𝜑)
Zeros_u = zeros(Float64,𝜑_dofs,k)
Zeros_𝜑 = zeros(Float64,u_dofs,l)
𝛷 = hcat(vcat(U_x_u[:,σ_i_x_selection_u],Zeros_u),vcat(Zeros_𝜑,U_x_𝜑[:,σ_i_x_selection_𝜑]))
𝛷 = hcat(vcat(U_x_u[:,[1:k...]],Zeros_u),vcat(Zeros_𝜑,U_x_𝜑[:,[1:l...]]))

f = 1.05
x_list, b_list = POD_Incremental_Solver(𝛷,f,true)
E = Max_Error_rel(x_list,k,l,f)

POD_collect_data()

POD_read_or_collect_data()

POD_collect_data_res_contributions()

POD_read_and_error()

f = 1.05
Error_list = jldopen("scripts/MB Ex 3/POD_red_Solutions/V1/MaterialParameter$f/Error_list.jld2")
Error_list = Error_list["Error_list"]
k_list = [1,3,5,10,15,17,20,25,30]
l_list = [1,3,5,10,12,15,20]
pl = plot(
    yscale = :log10, yticks = [10.0^i for i in -9:2], xlabel = "k", 
    ylabel = "tip displacement error",
    title = "Error vs POD u modes (k) at POD phi modes (l)"
    )
for l in l_list
    x = []
    y = []
    for k in k_list
        try
            push!(y,Error_list["$((k,l))"])
            push!(x,k)
        catch
        end
    end
    pl = plot!(x,y,label = "l = $l", marker = 4)
end
display(pl)
savefig("scripts/MB Ex 3/POD_red_Solutions/V1/MaterialParameter$f/POD_ErrorSensibility.svg")
savefig("scripts/MB Ex 3/POD_red_Solutions/V1/MaterialParameter$f/POD_ErrorSensibility.png")


#endregion

##


##

#region Scratch to test the extraction of elemental contributions

f = 1.0
model, Ω, dΩ =  get_trian_and_measure()
dirichletbc = get_DirichletBC(1000.0)
fe_spaces = get_fe_spaces(model,dirichletbc)
assem = SparseMatrixAssembler(fe_spaces.U,fe_spaces.V)
xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
x0 = vcat(xu, xφ)
println("number of dofs = $(length(x0))")
ph = FEFunction(fe_spaces.U, x0)
res, jac = get_symbolic_res_and_jac(dΩ,f)
RES = res(ph,get_fe_basis(fe_spaces.V))

JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))

res_contributions = collect_cell_vector(fe_spaces.V,RES)
glo_res_element_contributions = []
for i in eachindex(res_contributions[2][1])
    b = allocate_vector(assem,res_contributions)
    b[filter(x -> x >= 0, res_contributions[2][1][i][1])] = res_contributions[1][1][i][1][findall(x -> x > 0, res_contributions[2][1][i][1])]
    b[filter(x -> x >= 0, res_contributions[2][1][i][2])] = res_contributions[1][1][i][1][findall(x -> x > 0, res_contributions[2][1][i][2])]
    push!(glo_res_element_contributions,b)
end
glo_res_element_contributions[1]

plot(glo_res_element_contributions[1])

@time jac_contributions = collect_cell_matrix(fe_spaces.U,fe_spaces.V,JAC);

z_ = 451
@time jac_contributions[1][1][z_][1],jac_contributions[1][1][z_][3],jac_contributions[1][1][z_][2],jac_contributions[1][1][z_][4];
@time a = jac_contributions[1][1][z_].array;
@time hvcat((2,2),a[1,1],a[1,2],a[2,1],a[2,2]);
@time k_el = hvcat((2,2),jac_contributions[1][1][z_][1],jac_contributions[1][1][z_][3],jac_contributions[1][1][z_][2],jac_contributions[1][1][z_][4]);
@time get_array(jac_contributions[1][1][z_])

@time b = res_contributions[1][1][z_].array;
vcat(b[1],b[2])


jac_contributions[1][1][1]
jac_contributions[1][1][1][1]
jac_contributions[2][1][1][1] == jac_contributions[3][1][1][1]
jac_contributions[2][1][1][2]
n = length(x0)
k_el = vcat(hcat(jac_contributions[1][1][1][1],jac_contributions[1][1][1][3]),hcat(jac_contributions[1][1][1][2],jac_contributions[1][1][1][4]))
el_dofs = vcat(jac_contributions[2][1][1][1],jac_contributions[2][1][1][2])
Inci_el = sparse(filter(x -> x >= 0,el_dofs),findall(x -> x >= 0,el_dofs), fill(1,count(x -> x >= 0, el_dofs)),n,108)
sparse(𝛷)
𝛷_el = Inci_el'*𝛷
k_el_red = 𝛷_el'*k_el*𝛷_el
using BlockArrays
k_el  = hvcat((2,2),jac_contributions[1][1][1].array...)
k_el_red = 𝛷_el'*k_el*𝛷_el

A = BlockArray(rand(4, 4), [2, 2], [2, 2])
v = rand(4)

# Matrix-vector multiplication
b = A * v

# Matrix-matrix multiplication
C = A * A

res_contributions[1][1][1][2]

@time b_el = vcat(res_contributions[1][1][1][1],res_contributions[1][1][1][2]);

b_el = zeros(Float64,108)
@time b_el[[1:81...]] = res_contributions[1][1][1][1]; b_el[[82:end...]] = res_contributions[1][1][1][2]


K = allocate_matrix(assem,jac_contributions)

𝛷_el_z = el_proyection(𝛷,z,model,dΩ,f)

using Base.Threads

V = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

# Use a generator expression with @threads for a clean reduction
total_sum = sum(sum(v) for v in V)

function el_proyection(𝛷,z,model,dΩ,f)
    dirichletbc = get_DirichletBC(0.0)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
    xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
    x0 = vcat(xu, xφ)
    ph = FEFunction(fe_spaces.U, x0)
    _, jac = get_symbolic_res_and_jac(dΩ,f)
    JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    n, k_l = size(𝛷)
    jac_contributions = collect_cell_matrix(fe_spaces.U,fe_spaces.V,JAC)
    𝛷_el_z = []
    for el_id in z
        el_dofs = vcat(jac_contributions[2][1][el_id][1],jac_contributions[2][1][el_id][2])
        Inci_el = sparse(filter(x -> x >= 0,el_dofs),findall(x -> x >= 0,el_dofs), fill(1,count(x -> x >= 0, el_dofs)),n,108)
        𝛷_el = Inci_el'*𝛷
        push!(𝛷_el_z,𝛷_el)
    end
    return 𝛷_el_z
end

function get_numeric_res_and_jac_timed(ph,fe_spaces,Ω,res,jac)
    @time RES = res(ph,get_fe_basis(fe_spaces.V))
    @time σₖ = get_cell_dof_ids(fe_spaces.U)
    @time assem = SparseMatrixAssembler(fe_spaces.U,fe_spaces.V)
    @time rs = ([RES[Ω]],[σₖ])
    @time b = allocate_vector(assem,rs)
    @time assemble_vector!(b,assem,rs)
    @time JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    @time rs = ([JAC[Ω]],[σₖ],[σₖ])
    @time K_T = MultiField.allocate_matrix(assem,rs)
    @time assemble_matrix!(K_T,assem,rs)
    return b, K_T
end

function ECM_POD_get_numeric_res_and_jac_timed(ph,fe_spaces,Ω,res,jac,z,w,𝛷)
    @time RES = res(ph,get_fe_basis(fe_spaces.V))
    @time σₖ = get_cell_dof_ids(fe_spaces.U)
    @time assem = SparseMatrixAssembler(fe_spaces.U,fe_spaces.V)
    @time rs = ([RES[Ω]],[σₖ])
    @time b = allocate_vector(assem,rs)
    @time assemble_vector!(b,assem,rs)
    @time res_contributions = collect_cell_vector(fe_spaces.V,RES)
    n, k_l = size(𝛷)
    t = zeros(Float64,k_l)
    @time for (z_,w_) in zip(z,w)
        b = allocate_vector(assem,res_contributions)
        b[filter(x -> x >= 0, res_contributions[2][1][z_][1])] = res_contributions[1][1][z_][1][findall(x -> x > 0, res_contributions[2][1][z_][1])]
        b[filter(x -> x >= 0, res_contributions[2][1][z_][2])] = res_contributions[1][1][z_][1][findall(x -> x > 0, res_contributions[2][1][z_][2])]
        t += w_*𝛷'*b
    end
    @time JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    @time rs = ([JAC[Ω]],[σₖ],[σₖ])
    @time K_T = MultiField.allocate_matrix(assem,rs)
    @time assemble_matrix!(K_T,assem,rs)
    @time jac_contributions = collect_cell_matrix(fe_spaces.U,fe_spaces.V,JAC)
    K = zeros(Float64,k_l,k_l)
    count = 0 
    k = allocate_matrix(assem,jac_contributions)
    Temp1 = zeros(Float64,n,k_l)
    Temp2 = zeros(Float64,k_l,k_l)
    @time for (z_, w_) in zip(z,w)
        @time k[filter(x -> x >= 0, jac_contributions[3][1][z_][1]),filter(x -> x >= 0, jac_contributions[2][1][z_][1])] = jac_contributions[1][1][z_][1][findall(x -> x > 0, jac_contributions[3][1][z_][1]),findall(x -> x > 0, jac_contributions[2][1][z_][1])]
        @time k[filter(x -> x >= 0, jac_contributions[3][1][z_][2]),filter(x -> x >= 0, jac_contributions[2][1][z_][2])] = jac_contributions[1][1][z_][2][findall(x -> x > 0, jac_contributions[3][1][z_][2]),findall(x -> x > 0, jac_contributions[2][1][z_][2])]
        @time mul!(Temp1,k,𝛷)
        @time mul!(Temp2,𝛷',Temp1,w_,1.0)
        @time K += Temp2
        count += 1
        @time k[filter(x -> x >= 0, jac_contributions[3][1][z_][1]),filter(x -> x >= 0, jac_contributions[2][1][z_][1])] .= 0.0
        @time k[filter(x -> x >= 0, jac_contributions[3][1][z_][2]),filter(x -> x >= 0, jac_contributions[2][1][z_][2])] .= 0.0
        print("\r$count")
    end
    return b, K_T
end

function ECM_POD_get_numeric_res_and_jac_timed(ph,fe_spaces,res,jac,z,w,𝛷_el_z,k_l)
    @time RES = res(ph,get_fe_basis(fe_spaces.V))
    @time JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    @time res_contributions = collect_cell_vector(fe_spaces.V,RES)
    @time jac_contributions = collect_cell_matrix(fe_spaces.U,fe_spaces.V,JAC)
    b_red = zeros(Float64,k_l)
    k_red = zeros(Float64,k_l,k_l)
    @time for (z_,w_,𝛷_el) in zip(z,w,𝛷_el_z)
        println(z_)
        # @time b_el = vcat(res_contributions[1][1][z_][1],res_contributions[1][1][z_][2])
        @time b = res_contributions[1][1][z_].array;
        @time b_el = vcat(b[1],b[2])
        @time b_red += w_*𝛷_el'*b_el
        # @time k_el = vcat(hcat(jac_contributions[1][1][z_][1],jac_contributions[1][1][z_][3]),hcat(jac_contributions[1][1][z_][2],jac_contributions[1][1][z_][4]))
        # @time k_el = hvcat((2,2),jac_contributions[1][1][z_][1],jac_contributions[1][1][z_][3],jac_contributions[1][1][z_][2],jac_contributions[1][1][z_][4])
        @time a = jac_contributions[1][1][z_].array;
        @time k_el = hvcat((2,2),a[1,1],a[1,2],a[2,1],a[2,2]);
        @time k_red += w_*𝛷_el'*k_el*𝛷_el
    end
    return b_red, k_red
end

function ECM_POD_get_numeric_res_and_jac(ph,fe_spaces,res,jac,z,w,𝛷_el_z,k_l)
    RES = res(ph,get_fe_basis(fe_spaces.V))
    JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    res_contributions = collect_cell_vector(fe_spaces.V,RES)
    jac_contributions = collect_cell_matrix(fe_spaces.U,fe_spaces.V,JAC)
    b_red = zeros(Float64,k_l)
    k_red = zeros(Float64,k_l,k_l)
    for (z_,w_,𝛷_el) in zip(z,w,𝛷_el_z)
        b = res_contributions[1][1][z_].array;
        b_el = vcat(b[1],b[2])
        b_red += w_*𝛷_el'*b_el
        a = jac_contributions[1][1][z_].array;
        k_el = hvcat((2,2),a[1,1],a[1,2],a[2,1],a[2,2]);
        k_red += w_*𝛷_el'*k_el*𝛷_el
    end
    return b_red, k_red
end
my_lock_1 = ReentrantLock();
my_lock_2 = ReentrantLock();

function ECM_POD_get_numeric_res_and_jac_threads(ph,fe_spaces,res,jac,z,w,𝛷_el_z,k_l)
    RES = res(ph,get_fe_basis(fe_spaces.V))
    JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    res_contributions = collect_cell_vector(fe_spaces.V,RES)
    jac_contributions = collect_cell_matrix(fe_spaces.U,fe_spaces.V,JAC)
    b_red = zeros(Float64,k_l)
    k_red = zeros(Float64,k_l,k_l)
    a = lastindex(z)
    b_red_ = [Vector{Float64}(undef,k_l) for _ in 1:a]
    k_red_ = [Matrix{Float64}(undef,k_l,k_l) for _ in 1:a]
    @threads for i in 1:a
        z_,w_,𝛷_el = z[i],w[i],𝛷_el_z[i]
        b = res_contributions[1][1][z_].array;
        b_el = vcat(b[1],b[2])
        b_red_[i] = w_*𝛷_el'*b_el
        # Threads.atomic_add!(b_red,w_*𝛷_el'*b_el)
        a = jac_contributions[1][1][z_].array;
        k_el = hvcat((2,2),a[1,1],a[1,2],a[2,1],a[2,2]);
        k_red_[i] = w_*𝛷_el'*k_el*𝛷_el
        # Threads.atomic_add!(k_red,w_*𝛷_el'*k_el*𝛷_el)
    end
    b_red = sum(b_red_)
    k_red = sum(k_red_)
    return b_red, k_red
end

ECM_POD_get_numeric_res_and_jac_timed(ph,fe_spaces,Ω,res,jac,z,w,𝛷);
get_numeric_res_and_jac_timed(ph,fe_spaces,Ω,res,jac);
k_l = k+l
ECM_POD_get_numeric_res_and_jac_timed(ph,fe_spaces,res,jac,z,w,𝛷_el_z,k_l);
@time A_1 = ECM_POD_get_numeric_res_and_jac(ph,fe_spaces,res,jac,z,w,𝛷_el_z,k_l);

@time A_2 = ECM_POD_get_numeric_res_and_jac_threads(ph,fe_spaces,res,jac,z,w,𝛷_el_z,k_l);

A_1 == A_2

#endregion


##


##

#region Assembly of the contributions matrix for ECM

k,l = 17, 12
f = 0.9

folder = "scripts/MB Ex 3/POD_red_Solutions/V1_r_contributions/"

sol = jldopen(folder*"MaterialParameter$f/RedParam_k_$(k)_l_$(l)_f_$f.jld2")

sol["r_contri_list"][1][1]

C = reduce(vcat,reduce.(hcat,sol["r_contri_list"]))

function Assemble_contributions()
    folder = "scripts/MB Ex 3/POD_red_Solutions/V1_r_contributions/"
    k,l = 17, 12
    f_list = [0.9,1.0,1.1]
    C_list = []
    for f in f_list
        sol = jldopen(folder*"MaterialParameter$f/RedParam_k_$(k)_l_$(l)_f_$f.jld2")
        C_ = reduce(vcat,reduce.(hcat,sol["r_contri_list"]))
        push!(C_list,C_)
    end
    C = reduce(vcat,C_list)
    b = vec(sum(C,dims=2))
    return C
end

C, b = Assemble_contributions()

u = RondomizedSVD(C,1e-8)

z, w =ECM_Selection(u, 1e-6)

x_list, b_list = POD_ECM_Incremental_Solver(𝛷,z,w,f,true)

Max_Error_rel(x_list,k,l,f)

#endregion


##

##

#region ECM test

v,k,l = 1,17,12
v,k,l = 2,17,3 # 17, 12
v,k,l = 3,8,5
v,k,l = 4,25,18
f = 1.05
D_x, D_T = Training_Set_Read(1)
U_x_u, σ_i_x_u, V_x_u, U_x_𝜑, σ_i_x_𝜑, V_x_𝜑 = Jacobi_SVDs_POD(D_x)
Zeros_u = zeros(Float64,𝜑_dofs,k)
Zeros_𝜑 = zeros(Float64,u_dofs,l)
𝛷 = hcat(vcat(U_x_u[:,[1:k...]],Zeros_u),vcat(Zeros_𝜑,U_x_𝜑[:,[1:l...]]))
C, b = Assemble_contributions(v,k,l)
u = RondomizedSVD(C,1e-8) #1e.8
z, w =ECM_Selection(u, 1e-6)
length(z)
x_list, b_list = POD_ECM_Incremental_Solver(𝛷,z,w,f,true);
Max_Error_rel(x_list,k,l,f)  # 3.7461891480533494e-8 k,l,f = 17, 12, 1.05 & ECM tol = 1e-6




ECM_tol_list = [10.0^i for i in -6:-2]
append!(ECM_tol_list,5e-2)
sol = Dict()
for tol in ECM_tol_list
    z, w =ECM_Selection(u, tol)
    sol["tol=$tol : z"] = z
    sol["tol=$tol : w"] = w
    sol["tol=$tol : el_num"] = length(z)
    x_list, b_list = POD_ECM_Incremental_Solver(𝛷,z,w,f,false)
    sol["tol=$tol : x_list"] = x_list
    E = Max_Error_rel(x_list,k,l,f)
    sol["tol=$tol : E"] = E
end

folder = "scripts/MB Ex 3/POD_ECM_red_Solutions/V4/"
mkpath(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/")
jldsave(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/" * "SolutionsVsTolerances2_tol_ECM.jld2",sol = sol)

E_list = [sol["tol=$tol : E"] for tol in ECM_tol_list]
el_num_list = [sol["tol=$tol : el_num"] for tol in ECM_tol_list]

p = plot(ECM_tol_list,el_num_list, xscale = :log10,
            xticks = ECM_tol_list, marker = 4, label = "Selected elements",
            xlabel = "Tolerance in selection algorithm"
            )
plot!(p, ECM_tol_list, fill(NaN, length(ECM_tol_list)), label="Tip point error", lc=:red,
     xscale = :log10, xticks = ECM_tol_list, marker = 4
)
plot!(twinx(),ECM_tol_list, E_list,  yscale = :log10, xscale = :log10,
            xticks = ECM_tol_list, yticks = [10.0^i for i in -8:-3],
            ylims = (1e-8,1e-3),marker = 4, lc=:red, mc = :red, label = ""
            )

plot!(p, title="Tolerance Sensibility analysis",
      ylabel="Selected elements",
      y2label="Tip point error (Log)",
      legend=:topleft)

savefig(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/" * "POD_ECM_ErrorSensibility2.svg")
savefig(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/" * "POD_ECM_ErrorSensibility2.png")



SVD_tol_list = [10.0^i for i in -9:-2]
sol = Dict()
for tol in SVD_tol_list
    u = RondomizedSVD(C,tol)
    sol["tol=$tol : SVD_tol"] = tol
    z, w =ECM_Selection(u, 1e-6)
    sol["tol=$tol : z"] = z
    sol["tol=$tol : w"] = w
    sol["tol=$tol : el_num"] = length(z)
    x_list, b_list = POD_ECM_Incremental_Solver(𝛷,z,w,f,true)
    sol["tol=$tol : x_list"] = x_list
    E = Max_Error_rel(x_list,k,l,f)
    sol["tol=$tol : E"] = E
end

folder = "scripts/MB Ex 3/POD_ECM_red_Solutions/V3/"
mkpath(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/")
jldsave(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/" * "SolutionsVsTolerances3_tol_SVD.jld2",sol = sol)

E_list = [sol["tol=$tol : E"] for tol in SVD_tol_list]
el_num_list = [sol["tol=$tol : el_num"] for tol in SVD_tol_list]

p = plot(SVD_tol_list,el_num_list, xscale = :log10,
            xticks = SVD_tol_list, marker = 4, label = "Selected elements",
            xlabel = "Tolerance in RSVD algorithm"
            )
plot!(p, SVD_tol_list, fill(NaN, length(SVD_tol_list)), label="Tip point error", lc=:red,
     xscale = :log10, xticks = SVD_tol_list, marker = 4
)
plot!(twinx(),SVD_tol_list, E_list,  yscale = :log10, xscale = :log10,
            xticks = SVD_tol_list, yticks = [10.0^i for i in -8:-2],
            ylims = (1e-8,1e-2),marker = 4, lc=:red, mc = :red, label = ""
            )

plot!(p, title="Tolerance Sensibility analysis",
      y2label="Selected elements",
      ylabel=["Selected elements" "Tip point error (Log)"],
      legend=:topleft)

savefig(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/" * "POD_ECM_ErrorSensibility3_SVDtol.svg")
savefig(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/" * "POD_ECM_ErrorSensibility2_SVDtol.png")
#endregion

##


##

#region Plot composition

list = []
v,k,l = 1,17,12
push!(list,[v,k,l])
v,k,l = 2,17,3 # 17, 12
push!(list,[v,k,l])
v,k,l = 3,8,5
push!(list,[v,k,l])
v,k,l = 4,25,18
push!(list,[v,k,l])
f = 1.05
p_list = []
for li in list 
    v,k,l = li
    folder = "scripts/MB Ex 3/POD_ECM_red_Solutions/V$v/"
    sol = load(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/" * "SolutionsVsTolerances3_tol_SVD.jld2")
    sol = sol["sol"]
    E_list = [sol["tol=$tol : E"] for tol in SVD_tol_list]
    el_num_list = [sol["tol=$tol : el_num"] for tol in SVD_tol_list]

    p = plot(SVD_tol_list,el_num_list, xscale = :log10,
                xticks = SVD_tol_list, marker = 4, label = "Selected elements",
                xlabel = "Tolerance in RSVD algorithm"
                )
    plot!(p, SVD_tol_list, fill(NaN, length(SVD_tol_list)), label="Tip point error", lc=:red,
        xscale = :log10, xticks = SVD_tol_list, marker = 4
    )
    plot!(twinx(),SVD_tol_list, E_list,  yscale = :log10, xscale = :log10,
                xticks = SVD_tol_list, yticks = [10.0^i for i in -8:-2],
                ylims = (1e-8,1e-2),marker = 4, lc=:red, mc = :red, label = ""
                )

    plot!(p, title="Tolerance Sensibility Analysis at POD k,l = $k,$l",
        ylabel=["Selected elements" "Tip point error (Log)"],
        legend=:topleft)
    push!(p_list,p)
end
plot(p_list...,layout=(2,2),size=(1100,1100))
savefig("C:/Users/mjbarillas/Documents/LaTeX/POD_ECM_Notes/RSVDtolerance_sensibilityAnalysis.svg")

p_list = []
SVD_tol_list = [10.0^i for i in -6:-2]
for li in list 
    v,k,l = li
    folder = "scripts/MB Ex 3/POD_ECM_red_Solutions/V$v/"
    sol = load(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/" * "SolutionsVsTolerances2_tol_ECM.jld2")
    sol = sol["sol"]
    E_list = [sol["tol=$tol : E"] for tol in SVD_tol_list]
    el_num_list = [sol["tol=$tol : el_num"] for tol in SVD_tol_list]

    p = plot(SVD_tol_list,el_num_list, xscale = :log10,
                xticks = SVD_tol_list, marker = 4, label = "Selected elements",
                xlabel = "Tolerance in ECM greedy element selection algoith"
                )
    plot!(p, SVD_tol_list, fill(NaN, length(SVD_tol_list)), label="Tip point error", lc=:red,
        xscale = :log10, xticks = SVD_tol_list, marker = 4
    )
    plot!(twinx(),SVD_tol_list, E_list,  yscale = :log10, xscale = :log10,
                xticks = SVD_tol_list, yticks = [10.0^i for i in -8:-2],
                ylims = (1e-8,1e-2),marker = 4, lc=:red, mc = :red, label = ""
                )

    plot!(p, title="Tolerance Sensibility Analysis at POD k,l = $k,$l",
        ylabel=["Selected elements" "Tip point error (Log)"],
        legend=:topleft)
    push!(p_list,p)
end
plot(p_list...,layout=(2,2),size=(1100,1100))
savefig("C:/Users/mjbarillas/Documents/LaTeX/POD_ECM_Notes/GreedyelementSelectionTolerance_sensibilityAnalysis.svg")
#endregion

##

##

#region Time evaluation

v = 3
D_x, D_T = Training_Set_Read(1)
U_x_u, σ_i_x_u, V_x_u, U_x_𝜑, σ_i_x_𝜑, V_x_𝜑 = Jacobi_SVDs_POD(D_x)

σ_i_x_rel_u, σ_i_x_rel_𝜑 = SingulaVals_Rel_to_max(σ_i_x_u, σ_i_x_𝜑)

σ_i_x_selection_u = findall(x -> x>1e-12,σ_i_x_rel_u)
σ_i_x_selection_𝜑 = findall(x -> x>1e-12,σ_i_x_rel_𝜑)

σ_i_x_selection_u = findall(x -> x>1e-8,σ_i_x_rel_u)
σ_i_x_selection_𝜑 = findall(x -> x>1e-8,σ_i_x_rel_𝜑)

σ_i_x_selection_u = findall(x -> x>1e-15,σ_i_x_rel_u)
σ_i_x_selection_𝜑 = findall(x -> x>1e-15,σ_i_x_rel_𝜑)

k = length(σ_i_x_selection_u)
l = length(σ_i_x_selection_𝜑)
Zeros_u = zeros(Float64,𝜑_dofs,k)
Zeros_𝜑 = zeros(Float64,u_dofs,l)
𝛷 = hcat(vcat(U_x_u[:,σ_i_x_selection_u],Zeros_u),vcat(Zeros_𝜑,U_x_𝜑[:,σ_i_x_selection_𝜑]))
𝛷 = hcat(vcat(U_x_u[:,[1:k...]],Zeros_u),vcat(Zeros_𝜑,U_x_𝜑[:,[1:l...]]))

f = 1.05
C, b = Assemble_contributions(v,k,l)
u = RondomizedSVD(C,1e-8) #1e.8
z, w =ECM_Selection(u, 1e-6)

nsteps = 20
@time x_list, b_list = POD_ECM_Incremental_Solver_n_steps(𝛷,z,w,f,nsteps,true) # 21.438298 seconds (170.71 M allocations: 115.575 GiB, 32.12% gc time)
@time x_list, b_list = Incremental_Solver_n_steps(f,nsteps,true) #595.284146 seconds (553.53 M allocations: 215.913 GiB, 2.46% gc time, 0.96% compilation time)

function time_comparison(nsteps,compute_ref_time=true)
    if compute_ref_time
        t_0 = time()
        x_list, b_list = Incremental_Solver_n_steps(f,nsteps,true)
        t_1 = time() - t_0
        x_list, b_list = POD_ECM_Incremental_Solver_n_steps(𝛷,z,w,f,nsteps,true)
        t_2 = time() - t_1
    else
        t_1 = 595.284146
        t_0 = time()
        x_list, b_list = POD_ECM_Incremental_Solver_n_steps(𝛷,z,w,f,nsteps,true)
        t_2 = time() - t_0
    end

    return t_2/t_1, t_2, t_1
end

time_comparison(nsteps,false) # 0.03529

function time_comparison_set(nsteps,compute_ref_time=true,trace=true)
    t_list_SVD = []
    t_list_ECM_sel = []
    t_list_ECM_solve = []
    v,k,l = 1,17,12
    f = 1.05
    if compute_ref_time
        t_0 = time()
        x_list, b_list = Incremental_Solver_n_steps(f,nsteps,trace)
        t_ref = time() - t_0
        println("reference time computed = $t_ref !")
    else
        t_ref = 615.914999961853
        println("reference time defined!")
    end
    println("Reading training Set")
    D, _ = Training_Set_Read(1)
    println("Computing POD basis")
    U_x_u, σ_i_x_u, V_x_u, U_x_𝜑, σ_i_x_𝜑, V_x_𝜑 = Jacobi_SVDs_POD(D)

    Zeros_u = zeros(Float64,𝜑_dofs,k)
    Zeros_𝜑 = zeros(Float64,u_dofs,l)
    𝛷 = hcat(vcat(U_x_u[:,[1:k...]],Zeros_u),vcat(Zeros_𝜑,U_x_𝜑[:,[1:l...]]))
    
    println("Reading training set for ECM")
    C, b = Assemble_contributions(v,k,l)
    println("time calculations")
    SVD_tol_list = [10.0^i for i in -9:-2]
    for tol in SVD_tol_list
        if !trace
            println("Computing tol = $tol")
        end
        t_0 = time()
        u = RondomizedSVD(C,tol)
        push!(t_list_SVD,time()-t_0)
        t_0 = time()
        z, w =ECM_Selection(u, 1e-6)
        push!(t_list_ECM_sel,time()-t_0)
        t_0 = time()
        x_list, b_list = POD_ECM_Incremental_Solver_n_steps(𝛷,z,w,f,nsteps,trace)
        push!(t_list_ECM_solve,time()-t_0)
    end
    return t_ref, t_list_SVD, t_list_ECM_sel, t_list_ECM_solve, SVD_tol_list
end

t_ref, t_list_SVD, t_list_ECM_sel, t_list_ECM_solve, SVD_tol_list = time_comparison_set(20,false,false)

folder = "scripts/MB Ex 3/POD_ECM_red_Solutions/V4/"
mkpath(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/")
jldsave(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/" * "timeReductionVsTolerances2_tol_ECM.jld2", t_ref = t_ref,
    t_list_SVD = t_list_SVD, t_list_ECM_sel = t_list_ECM_sel, t_list_ECM_solve = t_list_ECM_solve, SVD_tol_list = SVD_tol_list)

plot(SVD_tol_list,t_list_ECM_solve./t_ref, xscale = :log10,
                xticks = SVD_tol_list, marker = 4, label = "Time reduction",
                xlabel = "Tolerance in RSVD algorithm", ylabel = "Time reduction"
                )
savefig("C:/Users/mjbarillas/Documents/LaTeX/POD_ECM_Notes/GreedyelementSelectionTolerance_timereduction.svg")

plot(SVD_tol_list,t_list_ECM_sel./t_ref, xscale = :log10,
                xticks = SVD_tol_list, marker = 4, label = "Time reduction",
                xlabel = "Tolerance in RSVD algorithm", ylabel = "Time reduction"
                )

#endregion

##