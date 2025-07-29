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

function get_symbolic_res_and_jac(dΩ)
    diffstrat = "autodiff"
    soltype = "monolithic"
    # Notes: percentage applied: 0.9 = Model0 ; 1.0 = model1 ; 1.1 = model2
    modmec = Yeoh(C₁ = 1.1*0.0693e6, C₂ = -8.88e2*1.1, C₃ = 1.1*16.7, κ = 0.0693e8)
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

function run(x0,step,nsteps,𝜑ᵇ,model, Ω, dΩ,cache,x_list,b_list,R_nl_list,K_T_init,bisect)
    res, jac = get_symbolic_res_and_jac(dΩ)
    dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps))
    fe_spaces = get_fe_spaces(model,dirichletbc)
    norm_res = 1
    count = 0
    x0_copy = copy(x0)
    println("==============================================")
    println("step = $step of $nsteps current potential =$(𝜑ᵇ*(step/nsteps))")
    while norm_res>1e-12
        if count>15 || norm_res>1e2
            bisect += 1
            println("bisect = $bisect")
            x0, cache, _, _, _ = run(x0_copy,step-(1/(2^bisect)),nsteps,𝜑ᵇ,model, Ω, dΩ,cache,x_list,b_list,R_nl_list,K_T_init,bisect)
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
        println("iter = $count norm_res = $norm_res norm_Δx = $(norm(Δx))  size_K_T = $(size(K_T))")
    end
    # nlsolver = get_FE_solver()
    # ph = get_initial_guess(fe_spaces,x0)
    # op = FEOperator(res, jac, fe_spaces.U, fe_spaces.V)
    # RES = op.res(ph,get_fe_basis(fe_spaces.V))
    # ph, cache = solve!(ph, nlsolver, op, cache)
    # return get_free_dof_values(ph), cache
    return x0, cache, x_list, b_list, R_nl_list
end

function runs()
    model, Ω, dΩ =  get_trian_and_measure()
    nsteps = 300
    # step = 1
    # x0 = nothing
    dirichletbc = get_DirichletBC(40.0)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
    xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
    x0 = vcat(xu, xφ)
    println("number of dofs = $(length(x0))")
    ph = FEFunction(fe_spaces.U, x0)
    res, jac = get_symbolic_res_and_jac(dΩ)
    _, K_T_init = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
    x_list = []
    b_list = []
    R_nl_list = []
    cache = nothing
    bisect = 0
    𝜑ᵇ = 5000.0
    for step in 1:nsteps
        @time x0, cache, x_list, b_list, R_nl_list = run(x0,step,nsteps,𝜑ᵇ,model, Ω, dΩ,cache,x_list,b_list,R_nl_list,K_T_init,bisect)
    end
    ph = FEFunction(fe_spaces.U, x0)
    b, K_T = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
    R_nl = b-K_T_init*x0
    push!(x_list,copy(x0))
    push!(b_list,copy(b))
    push!(R_nl_list,copy(R_nl))
    return x_list, b_list, R_nl_list
end

function collect_data()
    @time x_list, b_list, R_nl_list = runs()
    df_x = DataFrame(x_list, :auto)
    df_b = DataFrame(b_list, :auto)
    df_R_nl = DataFrame(R_nl_list, :auto)
    path_name = "data/csv/EM_CM_test_3/MaterialModel2"
    mkpath(path_name)
    CSV.write(path_name*"/x_.csv",df_x)
    CSV.write(path_name*"/b_.csv",df_b)
    CSV.write(path_name*"/R_nl_.csv",df_R_nl)
end

function compute_VTK()
    model, Ω, dΩ =  get_trian_and_measure()
    dirichletbc = get_DirichletBC(50.0)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    file_name = "data/csv/EM_CM_test/MaterialModel0/x_.csv"
    _X = CSV.File(file_name) |> Tables.matrix
    mkpath("data/sims/EM_CM_test/MaterialModel0")
    pvd = paraview_collection("data/sims/EM_CM_test/MaterialModel0"* "/Results", append=false)
    writevtk(model, "data/sims/EM_CM_test/MaterialModel0"* "/DiscreteModel")
    i = 1
    for _x in eachcol(_X)
        𝜑ᵇ = 4000.0
        # Λ = (i-1)/(lastindex(eachcol(_X))-1)
        Λ = i/(100) 
        dirichletbc = get_DirichletBC(Λ*𝜑ᵇ)
        fe_spaces = get_fe_spaces(model,dirichletbc)
        ph = FEFunction(fe_spaces.U,_x)
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        Λ_ = i
        pvd[Λ_] = createvtk(
            Ω,
            "data/sims/EM_CM_test/MaterialModel0" * "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
            cellfields=["u"=>ph[1], "phi"=>ph[2]]
        )
        print("\r$Λ_  ")
        i += 1
    end
    vtk_save(pvd)
    
    end