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
    x0,step,nsteps,𝜑ᵇ,f,
    model, Ω, dΩ,
    cache,x_list,b_list,
    bisect,
    𝛷,
    trace=true
    )
    n, k_l = size(𝛷)
    b_POD = Vector{Float64}(undef,k_l)
    K_T_POD_pos_mult = Matrix{Float64}(undef,n,k_l)
    K_T_POD = Matrix{Float64}(undef,k_l,k_l)
    res, jac = get_symbolic_res_and_jac(dΩ,f)
    dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps))
    fe_spaces = get_fe_spaces(model,dirichletbc)
    norm_res = 1
    count = 0
    x0_copy = copy(x0)
    if trace
        println("\n==============================================")
    end
    println("Material parameter = $f :: step = $step of $nsteps k+l = $k_l")
    while norm_res>1e-12
        if count>15 || norm_res>1e2
            bisect += 1
            println("bisect = $bisect")
            x0, cache, _, _, _ = run(x0_copy,step-(1/(2^bisect)),nsteps,𝜑ᵇ,f,model, Ω, dΩ,cache,x_list,b_list,bisect)
            count = 1
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

#endregion

##