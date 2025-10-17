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
    model = GmshDiscreteModel("data/models/"*"CircularMembrane5.msh")
    degree = 4
    Ω = Triangulation(model)
    dΩ = Measure(Ω,degree)
    return model, Ω, dΩ
end


function get_DirichletBC(𝜑ᵇ)
    sw = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
    evolu(Λ) = 1.0
    dir_u_tags = ["point_xyz","point_xy","point_x"]
    dir_u_values = [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]
    masks = [(true,true,true),(true,true,false),(true,false,false)]
    dir_u_timesteps = [evolu,evolu,evolu]
    Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps,masks)
    # println(Du)
  
    evolφ(Λ) = Λ
    bottom_surf_list = [
      "bottom_surf_1", "bottom_surf_2", "bottom_surf_3", "bottom_surf_4",
      "bottom_surf_5", "bottom_surf_6", "bottom_surf_7", "bottom_surf_8",
      "bottom_surf_9", "bottom_surf_10", "bottom_surf_11", "bottom_surf_12", 
      "bottom_surf_13", "bottom_surf_14", "bottom_surf_15", "bottom_surf_16"
    ]
    earth_loc = [
      "mid_surf_1", "mid_surf_2", "mid_surf_3", "mid_surf_4",
      "mid_surf_5", "mid_surf_6", "mid_surf_7", "mid_surf_8",
      "mid_surf_9", "mid_surf_10", "mid_surf_11", "mid_surf_12",
      "mid_surf_13", "mid_surf_14", "mid_surf_15", "mid_surf_16"
    ]
    top_surf_list = [
      "top_surf_1", "top_surf_2", "top_surf_3", "top_surf_4",
      "top_surf_5", "top_surf_6", "top_surf_7", "top_surf_8",
      "top_surf_9", "top_surf_10", "top_surf_11", "top_surf_12",
      "top_surf_13", "top_surf_14", "top_surf_15", "top_surf_16"
    ]
    power_loc = []
    i = 1
    for io in sw
        if io==1 && i <= 16
          push!(power_loc,top_surf_list[i])
        elseif io==1 && i >= 16
          push!(power_loc,bottom_surf_list[i-16])
        end
        i += 1
    end
    earth_val = [0.0 for _ in earth_loc]
    power_val = [𝜑ᵇ for _ in power_loc]
    dir_φ_tags = Vector{String}()
    append!(dir_φ_tags,earth_loc)
    append!(dir_φ_tags,power_loc)
    dir_φ_timesteps = [evolφ for i in dir_φ_tags]
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
    end
    println("Material parameter = $f :: step = $step of $nsteps ")
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
        folder = "scripts/MB Ex 3/Full Order Solutions/CM_V1/"
        mkpath(folder * "MaterialModel$f")
        CSV.write(folder*"MaterialModel$f/x_.csv",df_x)
        CSV.write(folder*"MaterialModel$f/b_.csv",df_b)
    end
end

#endregion

##

##

#region Read Full Order training data

𝜑_dofs = 30226
u_dofs = 152409

function Training_Set_Read(v)
    folder = "scripts/MB Ex 3/Full Order Solutions/CM_V$v/"
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
    folder = "scripts/MB Ex 3/POD_red_Solutions/CM_V$(v)_r_contributions/"
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

#region Error evaluation

function Max_Error_rel(x_list,k,l,f)
    folder = "scripts/MB Ex 3/Full Order Solutions/CM_V1/"
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
        println("Material parameter = $f :: step = $step of $nsteps k+l = $k_l elements = $(length(z))")
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
        if norm_res<1e-12 && bisect<1
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
        if !trace
        print("\r# of elements = $(length(w)) k + l = $k_l")
        end
    end
    println("")
    return x_list, b_list
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
        if norm_res<1e-12 & bisect<1
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
    k_list = [5,15,25,55,85,100]
    l_list = [5,15,35,60,80]
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
        folder = "scripts/MB Ex 3/POD_red_Solutions/CM_V1/"
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
    k_list = [5,15,25,55,85,100]
    l_list = [5,15,35,60,80]
    k_l_list = []
    for k in k_list, l in l_list
        push!(k_l_list,(k,l))
    end
    total = length(k_l_list)
    # gmsh_lock = ReentrantLock()
    i = 0
    folder = "scripts/MB Ex 3/POD_red_Solutions/CM_V1/"
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
    k_list = [5,15,25,55,85,100]
    l_list = [5,15,35,60,80]
    k_l_list = []
    for k in k_list, l in l_list
        push!(k_l_list,(k,l))
    end
    total = length(k_l_list)
    # gmsh_lock = ReentrantLock()
    i = 0
    folder = "scripts/MB Ex 3/POD_red_Solutions/CM_V1/"
    Error_list = Dict()
    @threads for k_l in k_l_list
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
        i*=1
    end
    jldsave("scripts/MB Ex 3/POD_red_Solutions/CM_V1/MaterialParameter$f/Error_list.jld2",Error_list = Error_list)
end

function POD_collect_data_res_contributions()
    f_list = [0.9,1.0,1.1]
    # k, l = 55, 36
    k, l = 26, 14

    i = 0
    @threads for f in f_list
        𝛷 = MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)
        println("Current f = $f")
        x_list, b_list, r_contri_list = POD_Incremental_Solver_store_contributions(𝛷,f,false)
        folder = "scripts/MB Ex 3/POD_red_Solutions/CM_V1_r_contributions/"
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

#region POD analysis

D_x, D_T = Training_Set_Read(1)
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

folder = "scripts/MB Ex 3/POD_red_Solutions/CM_V1_r_contributions/"
savefig(folder * "SingularValuesDecay.svg")
savefig(folder * "singularValuesDecay.png")

σ_i_x_selection_u = findall(x -> x>1e-12,σ_i_x_rel_u)
σ_i_x_selection_𝜑 = findall(x -> x>1e-12,σ_i_x_rel_𝜑)

σ_i_x_selection_u = findall(x -> x>1e-8,σ_i_x_rel_u)
σ_i_x_selection_𝜑 = findall(x -> x>1e-8,σ_i_x_rel_𝜑)

σ_i_x_selection_u = findall(x -> x>1e-15,σ_i_x_rel_u)
σ_i_x_selection_𝜑 = findall(x -> x>1e-15,σ_i_x_rel_𝜑)

POD_collect_data_res_contributions()

k = length(σ_i_x_selection_u)
l = length(σ_i_x_selection_𝜑)
Zeros_u = zeros(Float64,𝜑_dofs,k)
Zeros_𝜑 = zeros(Float64,u_dofs,l)
𝛷 = hcat(vcat(U_x_u[:,σ_i_x_selection_u],Zeros_u),vcat(Zeros_𝜑,U_x_𝜑[:,σ_i_x_selection_𝜑]))
𝛷 = hcat(vcat(U_x_u[:,[1:k...]],Zeros_u),vcat(Zeros_𝜑,U_x_𝜑[:,[1:l...]]))

f = 1.05
x_list, b_list = POD_Incremental_Solver(𝛷,f,true)
E = Max_Error_rel(x_list,k,l,f)

v=1
C, b = Assemble_contributions(v,k,l)


u = RondomizedSVD(C,1e-8) #1e.8
z, w =ECM_Selection(u, 1e-6)
length(z)
x_list, b_list = POD_ECM_Incremental_Solver(𝛷,z,w,f,true);
E = Max_Error_rel(x_list,k,l,f)

folder = "scripts/MB Ex 3/POD_ECM_red_Solutions/CM_V1/"
mkpath(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/")
jldsave(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/" * "SingleSol_ECM_El_$(length(z)).jld2",x_list = x_list)

SVD_tol_list = [10.0^i for i in -9:-2]
sol = Dict()
for tol in SVD_tol_list
    println(" current SVD tol = $tol")
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
sol["tol=$(SVD_tol_list[6]) : x_list"]
sol["tol=$(SVD_tol_list[5]) : E"]
k, l = 26, 14

folder = "scripts/MB Ex 3/POD_ECM_red_Solutions/CM_V1/"
mkpath(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/")
jldsave(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/" * "SolutionsVsTolerances3_tol_SVD.jld2",sol = sol)

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
            xticks = SVD_tol_list, yticks = [10.0^i for i in -5:-0],
            ylims = (1e-5,1e-0),marker = 4, lc=:red, mc = :red, label = ""
            )

plot!(p, title="Tolerance Sensibility analysis for k = $k and l = $l",
      y2label="Selected elements",
      ylabel=["Selected elements" "Tip point error (Log)"],
      legend=:topleft)

savefig(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/" * "POD_ECM_ErrorSensibility3_SVDtol.svg")
savefig(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/" * "POD_ECM_ErrorSensibility2_SVDtol.png")

list = []
v,k,l = 1,26,14
push!(list,[v,k,l])
v,k,l = 1,55,36 # 17, 12
push!(list,[v,k,l])
v,k,l = 1,85,60
push!(list,[v,k,l])
p_list = []
for li in list 
    v,k,l = li
    folder = "scripts/MB Ex 3/POD_ECM_red_Solutions/CM_V$v/"
    sol = load(folder * "MaterialParameter$(f)_k_$(k)_l_$(l)/" * "SolutionsVsTolerances3_tol_SVD.jld2")
    sol = sol["sol"]
    E_list = [sol["tol=$tol : E"] for tol in SVD_tol_list]
    el_num_list = [sol["tol=$tol : el_num"] for tol in SVD_tol_list]

    p = plot(SVD_tol_list,el_num_list, xscale = :log10,
                xticks = SVD_tol_list, marker = 4, label = "Selected elements",
                xlabel = "Tolerance in ECM greedy element selection algorithm"
                )
    plot!(p, SVD_tol_list, fill(NaN, length(SVD_tol_list)), label="Tip point error", lc=:red,
        xscale = :log10, xticks = SVD_tol_list, marker = 4
    )
    plot!(twinx(),SVD_tol_list, E_list,  yscale = :log10, xscale = :log10,
                xticks = SVD_tol_list, yticks = [10.0^i for i in -5:-0],
                ylims = (1e-5,1e-0),marker = 4, lc=:red, mc = :red, label = ""
                )

    plot!(p, title="Tolerance Sensibility Analysis at POD k,l = $k,$l",
        ylabel=["Selected elements" "Tip point error (Log)"],
        legend=:topleft)
    push!(p_list,p)
end
plot(p_list...,layout=(2,2),size=(1100,1100))
savefig("C:/Users/mjbarillas/Documents/LaTeX/POD_ECM_Notes/CM_RSVDtolerance_sensibilityAnalysis.svg")


POD_collect_data()

POD_read_and_error()

f = 1.05
Error_list = jldopen("scripts/MB Ex 3/POD_red_Solutions/CM_V1/MaterialParameter$f/Error_list.jld2")
Error_list = Error_list["Error_list"]
k_list = [5,15,25,55,85,100]
l_list = [5,15,35,60,80]
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
savefig("scripts/MB Ex 3/POD_red_Solutions/CM_V1/MaterialParameter$f/POD_ErrorSensibility.svg")
savefig("scripts/MB Ex 3/POD_red_Solutions/CM_V1/MaterialParameter$f/POD_ErrorSensibility.png")

#endregion

##

