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
    model = GmshDiscreteModel("data/models/EMCube_Coarse.msh")
    degree = 4
    Ω = Triangulation(model)
    dΩ = Measure(Ω,degree)
    return model, Ω, dΩ
end

function get_symbolic_res_and_jac(dΩ)
    diffstrat = "autodiff"
    soltype = "monolithic"
    modmec = Yeoh(C₁ = 0.0693e6, C₂ = -8.88e2, C₃ = 16.7, κ = 0.0693e8)
    modelec = IdealDielectric(ε=8.8542e-12*4.0)
    consmodel = ElectroMech(modmec, modelec)
    Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = consmodel(DerivativeStrategy{Symbol(diffstrat)}())
    ctype = CouplingStrategy{Symbol(soltype)}()
    res((u, φ), (v, vφ)) = residual_EM(ctype, (u, φ), (v, vφ), (∂Ψu, ∂Ψφ), dΩ)
    jac((u, φ), (du, dφ), (v, vφ)) = jacobian_EM(ctype, (u, φ), (du, dφ), (v, vφ), (∂Ψuu, ∂Ψφu, ∂Ψφφ), dΩ)
    # lφ(vφ) = -1.0 * residual_EM(CouplingStrategy{:staggered_E}(), (uh, φh), vφ, ∂Ψφ, dΩ)
    # aφ(dφ, vφ) = jacobian_EM(CouplingStrategy{:staggered_E}(), (uh, φh), dφ, vφ, ∂Ψφφ, dΩ)
    return res, jac
end


function get_DirichletBC(𝜑ᵇ)
    evolu(Λ) = 1.0
    x3_1 = ["surf_1","surf_7","surf_13","surf_19"]
    x3_0 = ["surf_2","surf_8","surf_14","surf_20"]
    x1_0 = ["surf_6","surf_24"]
    x2_0 = ["surf_3","surf_9"]
    dir_u_tags =  Vector{String}([])
    append!(dir_u_tags,x3_1,x3_0,x1_0,x2_0)
    x3_1_m = [(true,true,false),(true,true,false),(true,true,false),(true,true,false)]
    x3_0_m = [(false,false,true),(false,false,true),(false,false,true),(false,false,true)]
    x1_0_m = [(true,false,false),(true,false,false)]
    x2_0_m = [(false,true,false),(false,true,false)]
    dir_tags_masks = []
    append!(dir_tags_masks,x3_1_m,x3_0_m,x1_0_m,x2_0_m)
    dir_u_values = [[0.0, 0.0, 0.0] for _ in 1:length(dir_u_tags)]
    dir_u_timesteps = [evolu for _ in 1:length(dir_u_tags)]
    Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps,dir_tags_masks)
    # println(Du)
    evolφ(Λ) = Λ
    # 𝜑ᵇ = 100.0
    earth_loc = ["surf_2","surf_8","surf_14","surf_20"]
    power_loc = ["surf_1"]
    earth_val = [0.0 for _ in 1:length(earth_loc)]
    power_val = [𝜑ᵇ]
    # display(dir_φ_timesteps)
    dir_φ_tags = Vector{String}()
    append!(dir_φ_tags,earth_loc)
    append!(dir_φ_tags,power_loc)
    # display(dir_φ_tags)
    dir_φ_values = []
    append!(dir_φ_values,earth_val)
    append!(dir_φ_values,power_val)
    # display(dir_φ_values)
    dir_φ_timesteps = [evolφ for _ in 1:length(dir_φ_tags)]
    Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

    dirichletbc = MultiFieldBoundaryCondition([Du, Dφ])
    return dirichletbc
end

function get_fe_spaces(model,dirichletbc)
    order = 2
    regtype = "statics"
    soltype = "monolithic"
    problem = ElectroMechProblem{Symbol(soltype), Symbol(regtype)}()
    fe_spaces = Drivers.get_FE_spaces(problem, model, order, dirichletbc)
    return fe_spaces
end


function get_initial_guess(fe_spaces,x0::Nothing)
    xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
    xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
    x0 = vcat(xu, xφ)
    ph = FEFunction(fe_spaces.U, x0)
    return ph
end

function get_initial_guess(fe_spaces,x0)
    ph = FEFunction(fe_spaces.U, x0)
    return ph
end

function get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
    RES = res(ph,get_fe_basis(fe_spaces.V))
    # σₖ = MultiField.get_cell_dof_ids(fe_spaces.U,Ω,MultiFieldStyle(fe_spaces.V))
    σₖ = get_cell_dof_ids(fe_spaces.U)
    assem = SparseMatrixAssembler(fe_spaces.U,fe_spaces.V)
    # assem = MultiField.SparseMatrixAssembler(SparseMatrixCSR{0,Float64,Int},fe_spaces.U,fe_spaces.V)

    rs = ([RES[Ω]],[σₖ])
    # RES[Ω][1]
    b = allocate_vector(assem,rs)
    assemble_vector!(b,assem,rs)

    JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    # JAC[Ω][1]
    rs = ([JAC[Ω]],[σₖ],[σₖ])
    # pair_arrays(RES[Ω],JAC[Ω])
    # rs = (([pair_arrays(RES[Ω],JAC[Ω])],[σₖ],[σₖ]),([JAC[Ω]],[σₖ],[σₖ]),([RES[Ω]],[σₖ]))
    # K_T_init = MultiField.allocate_matrix_and_vector(assem,rs)
    K_T = MultiField.allocate_matrix(assem,rs)
    assemble_matrix!(K_T,assem,rs)
    
    # op = FEOperator(res, jac, fe_spaces.U, fe_spaces.V)
    return b, K_T
end

function get_FE_solver()
    nls_ = NLSolver(show_trace=true,
        method=:newton,
        # linesearch=MoreThuente(),
        iterations=15,
        ftol=1e-10)
    FESolver(nls_)
end

function run(x0,step,nsteps,𝜑ᵇ,model, Ω, dΩ,cache,x_list,b_list,R_nl_list,K_T_init,bisect)
    res, jac = get_symbolic_res_and_jac(dΩ)
    dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps))
    fe_spaces = get_fe_spaces(model,dirichletbc)
    norm_res = 1
    count = 0
    x0_copy = copy(x0)
    println("==============================================")
    println("step = $step of $nsteps")
    while norm_res>1e-15
        if count>15 || norm_res>1e2
            bisect += 1
            println("bisect = $bisect")
            x0, cache, _, _, _ = run(x0_copy,step-(1/(2^bisect)),nsteps,𝜑ᵇ,model, Ω, dΩ,cache,x_list,b_list,R_nl_list,K_T_init,bisect)
            count = 1
        end
        # ph = get_initial_guess(fe_spaces,x0)
        ph = FEFunction(fe_spaces.U, x0)
        b, K_T = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
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
        # if isnothing(x0)
        #     copyto!(x0,Δx)
        # else
        #     copyto!(x0,x0+Δx)           
        # end
        copyto!(x0,x0+Δx)
        norm_res = maximum(abs.(b))
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

function runs()
    model, Ω, dΩ =  get_trian_and_measure()
    nsteps = 10
    # x0 = nothing
    dirichletbc = get_DirichletBC(50.0)
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
    𝜑ᵇ = 4000.0
    for step in 1:nsteps
        x0, cache, x_list, b_list, R_nl_list = run(x0,step,nsteps,𝜑ᵇ,model, Ω, dΩ,cache,x_list,b_list,R_nl_list,K_T_init,bisect)
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
    x_list, b_list, R_nl_list = runs()
    df_x = DataFrame(x_list, :auto)
    df_b = DataFrame(b_list, :auto)
    df_R_nl = DataFrame(R_nl_list, :auto)
    CSV.write("data/csv/EMCube_test/x_"*"MaterialModel0"*".csv",df_x)
    CSV.write("data/csv/EMCube_test/b_"*"MaterialModel0"*".csv",df_b)
    CSV.write("data/csv/EMCube_test/R_nl_"*"MaterialModel0"*".csv",df_R_nl)
end

function compute_VTK()
model, Ω, dΩ =  get_trian_and_measure()
dirichletbc = get_DirichletBC(50.0)
fe_spaces = get_fe_spaces(model,dirichletbc)
file_name = "data/csv/EMCube_test/x_"*"MaterialModel0"*".csv"
_X = CSV.File(file_name) |> Tables.matrix
mkpath("data/sims/EMCube_test/MaterialModel0")
pvd = paraview_collection("data/sims/EMCube_test/MaterialModel0"* "/Results", append=false)
writevtk(model, "data/sims/EMCube_test/MaterialModel0"* "/DiscreteModel")
i = 1
for _x in eachcol(_X)
    𝜑ᵇ = 4000.0
    Λ = (i-1)/(lastindex(eachcol(_X))-1)
    dirichletbc = get_DirichletBC(Λ*𝜑ᵇ)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    ph = FEFunction(fe_spaces.U,_x)
    Λstring = replace(string(round(Λ, digits=2)), "." => "_")
    Λ_ = i
    pvd[Λ_] = createvtk(
        Ω,
        "data/sims/EMCube_test/MaterialModel0" * "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
        cellfields=["u"=>ph[1], "phi"=>ph[2]]
    )
    print("\r$Λ_  ")
    i += 1
end
vtk_save(pvd)

end