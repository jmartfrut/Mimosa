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
    return   ecm.z, ecm.w
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
    end
    println("Material parameter = $f :: step = $step of $nsteps k+l = $k_l")
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

function compute_residual_contributions(x0,𝛷,f,step,nsteps,𝜑ᵇ)
    model, Ω, dΩ = nothing, nothing, nothing
    lock(gmsh_lock) do
        model, Ω, dΩ =  get_trian_and_measure()
    end
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
        r_contribution = compute_residual_contributions(x0,f,step,nsteps,𝜑ᵇ)
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
            E = Max_Error_rel(eachcol(x_list),k,l,f)
            println("Current read k l = $k_l :: Executed = $i out of $total :: Error = $E")
        catch
            𝛷 = MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)
            println("Current computation k l = $k_l")
            x_list, b_list = POD_Incremental_Solver(𝛷,f,false)
            df_x = DataFrame(x_list, :auto)
            df_b = DataFrame(b_list, :auto)
            mkpath(folder * "MaterialParameter$f")
            CSV.write(folder*"MaterialParameter$f/RedParam_k_$(k)_l_$(l)x_.csv",df_x)
            CSV.write(folder*"MaterialParameter$f/RedParam_k_$(k)_l_$(l)b_.csv",df_b)
            i += 1
            E = Max_Error_rel(x_list,k,l,f)
            println("Current executed k l = $k_l :: Executed = $i out of $total :: Error = $E")
        end
    end
end

function POD_collect_data_res_contributions()
    f_list = [0.9,1.0,1.1]
    k, l = 17,12
    i = 0
    for f in f_list
        𝛷 = MultiField_Tuncated_Basis(U_x_u, U_x_𝜑, k, l)
        println("Current f = $f")
        x_list, b_list, r_contri_list = POD_Incremental_Solver_store_contributions(𝛷,f,false)
        folder = "scripts/MB Ex 3/POD_red_Solutions/V1_r_contributions/"
        mkpath(folder * "MaterialParameter$f")
        jldsave(folder*"MaterialParameter$f/RedParam_k_$(k)_l_$(l)_f_$f.jld2",x_list = x_list, b_list = b_list, r_contri_list = r_contri_list )
        i += 1
        println("Current executed f = $f")
    end
end

#endregion
##

##

#region ECM Gridap Functions

function Gridap.FESpaces.assemble_matrix!(mat,a::SparseMatrixAssembler,matdata,z,w)
    LinearAlgebra.fillstored!(mat,zero(eltype(mat)))
    assemble_matrix_add!(mat,a,matdata,z,w)
end

function Gridap.FESpaces.assemble_matrix_add!(mat,a::SparseMatrixAssembler,matdata,z,w)
    numeric_loop_matrix!(mat,a,matdata,z,w)
    Gridap.FESpaces.create_from_nz(mat)
end

function Gridap.FESpaces.numeric_loop_matrix!(A,a::SparseMatrixAssembler,matdata,z,w)
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
            Gridap.FESpaces._numeric_loop_matrix!(A,caches,cellmat,cellidsrows,cellidscols,z,w)
            h+=1
        end
        # println("Count 1 = $h")
    end
    A
end

@noinline function Gridap.FESpaces._numeric_loop_matrix!(mat,caches,cell_vals,cell_rows,cell_cols,z,w)
    cells = zip(z,w)
    add_cache, vals_cache, rows_cache, cols_cache = caches
    add! = AddEntriesMap(+)
    for cell in cells
    rows = getindex!(rows_cache,cell_rows,cell[1])
    cols = getindex!(cols_cache,cell_cols,cell[1])
    vals = getindex!(vals_cache,cell_vals,cell[1])
    evaluate!(add_cache,add!,mat,cell[2]*vals,rows,cols)
    end
end

function Gridap.FESpaces.allocate_matrix(a::SparseMatrixAssembler,matdata,z,w)
    m1 = Gridap.FESpaces.nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
    symbolic_loop_matrix!(m1,a,matdata,z,w)
    m2 = Gridap.FESpaces.nz_allocation(m1)
    symbolic_loop_matrix!(m2,a,matdata,z,w)
    m3 = Gridap.FESpaces.create_from_nz(m2)
    m3
end

function Gridap.FESpaces.symbolic_loop_matrix!(A,a::SparseMatrixAssembler,matdata,z,w)
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
        Gridap.FESpaces._symbolic_loop_matrix!(A,caches,cellidsrows,cellidscols,mat1,z,w)
      end
    end
    A
end

@noinline function Gridap.FESpaces._symbolic_loop_matrix!(
  A, caches, cell_rows, cell_cols, mat1, z, w
)
    cells = zip(z,w)
    touch_cache, rows_cache, cols_cache = caches
    touch! = TouchEntriesMap()
    for cell in cells
    rows = getindex!(rows_cache,cell_rows,cell[1])
    cols = getindex!(cols_cache,cell_cols,cell[1])
    evaluate!(touch_cache,touch!,A,mat1,rows,cols)
    end
end

function Gridap.FESpaces.assemble_vector!(b,a::SparseMatrixAssembler,vecdata,z,w)
    fill!(b,zero(eltype(b)))
    assemble_vector_add!(b,a,vecdata,z,w)
end

function Gridap.FESpaces.assemble_vector_add!(b,a::SparseMatrixAssembler,vecdata,z,w)
    numeric_loop_vector!(b,a,vecdata,z,w)
    Gridap.FESpaces.create_from_nz(b)
end

function Gridap.FESpaces.numeric_loop_vector!(b,a::SparseMatrixAssembler,vecdata,z,w)
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
        Gridap.FESpaces._numeric_loop_vector!(b,caches,cellvec,cellids,z,w)
      end
    end
    b
end

@noinline function Gridap.FESpaces._numeric_loop_vector!(
  vec, caches, cell_vals, cell_rows, z, w
)
    cells = zip(z,w)
    add_cache, vals_cache, rows_cache = caches
    @assert length(cell_vals) == length(cell_rows)
    add! = AddEntriesMap(+)
    for cell in cells
    rows = getindex!(rows_cache,cell_rows,cell[1])
    vals = getindex!(vals_cache,cell_vals,cell[1])
    evaluate!(add_cache,add!,vec,cell[2]*vals,rows)
    end
end


function Gridap.FESpaces.allocate_vector(a::SparseMatrixAssembler,vecdata,z,w)
    v1 = Gridap.FESpaces.nz_counter(get_vector_builder(a),(get_rows(a),))
    symbolic_loop_vector!(v1,a,vecdata,z,w)
    v2 = Gridap.FESpaces.nz_allocation(v1)
    symbolic_loop_vector!(v2,a,vecdata,z,w) 
    v3 = Gridap.FESpaces.create_from_nz(v2)
    v3
end

function Gridap.FESpaces.symbolic_loop_vector!(b,a::SparseMatrixAssembler,vecdata,z,w)
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
        Gridap.FESpaces._symbolic_loop_vector!(b,caches,cellids,vec1,z,w)
      end
    end
    b
end


@noinline function Gridap.FESpaces._symbolic_loop_vector!(
  A, caches, cell_rows, vec1, z, w
  )
  cells = zip(z,w)
  touch_cache, rows_cache = caches
  touch! = TouchEntriesMap()
  for cell in cells
    rows = getindex!(rows_cache,cell_rows,cell[1])
    evaluate!(touch_cache,touch!,A,vec1,rows)
  end
end

#endregion

##



##

#region ECM Solver

function POD_ECM_Increment_Solver(
    x0,step,nsteps,𝜑ᵇ,f,
    model, Ω, dΩ,
    cache,x_list,b_list,
    bisect,
    𝛷, z, w,
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
    end
    println("Material parameter = $f :: step = $step of $nsteps k+l = $k_l")
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
        b, K_T = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac,z,w)
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
σ_i_x_rel_u, σ_i_x_rel_𝜑 = SingulaVals_Rel_to_max(σ_i_x_u, σ_i_x_𝜑)
plotlyjs()
w = 200
plot(
    [σ_i_x_rel_u[[1:w...]],σ_i_x_rel_𝜑[[1:w...]]],
    yscale=:log10,
    label = ["Dₓ_u" "Dₓ_phi" ],
    ylims = (10^-float(20), 1),
    yticks=[10^-float(i*4) for i in 0:5]
)
σ_i_x_selection_u = findall(x -> x>1e-12,σ_i_x_rel_u)
σ_i_x_selection_𝜑 = findall(x -> x>1e-12,σ_i_x_rel_𝜑)
k = length(σ_i_x_selection_u)
l = length(σ_i_x_selection_𝜑)
Zeros_u = zeros(Float64,𝜑_dofs,k)
Zeros_𝜑 = zeros(Float64,u_dofs,l)
𝛷 = hcat(vcat(U_x_u[:,σ_i_x_selection_u],Zeros_u),vcat(Zeros_𝜑,U_x_𝜑[:,σ_i_x_selection_𝜑]))
f = 1.05
x_list, b_list = POD_Incremental_Solver(𝛷,f,true)
E = Max_Error_rel(x_list,k,l,f)

POD_collect_data()

POD_read_or_collect_data()

POD_collect_data_res_contributions()


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
res_contributions = collect_cell_vector(fe_spaces.V,RES)
glo_res_element_contributions = []
for i in eachindex(res_contributions[2][1])
    b = allocate_vector(assem,res_contributions)
    b[filter(x -> x >= 0, res_contributions[2][1][i][1])] = res_contributions[1][1][i][1][findall(x -> x > 0, res_contributions[2][1][i][1])]
    push!(glo_res_element_contributions,b)
end
glo_res_element_contributions[1]

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

#endregion


##

