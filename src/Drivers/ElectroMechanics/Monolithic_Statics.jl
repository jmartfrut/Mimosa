
function execute(problem::ElectroMechProblem{:monolithic,:statics}; kwargs...)

    # Problem setting
    pname = _get_kwarg(:problemName, kwargs)
    ptype = "ElectroMechanics"
    soltype = "monolithic"
    regtype = "statics"
    diffstrat = _get_kwarg(:diffstrat, kwargs, "analytic")
    ctype = CouplingStrategy{Symbol(soltype)}()
    printinfo = @dict ptype soltype regtype pname
    print_heading(printinfo)

    is_vtk = _get_kwarg(:is_vtk, kwargs, false)
    is_P_F = _get_kwarg(:is_P_F, kwargs, false)
    csv_bool = _get_kwarg(:csv_bool, kwargs, false)
    csv_funct_ = _get_kwarg(:csv_funct_, kwargs, nothing)
    f₀ = _get_kwarg(:f₀, kwargs, nothing)
    simdir_ = datadir("sims", pname)
    csvdir_ = datadir("csv", pname)
    setupfolder(simdir_)
    setupfolder(csvdir_)
 
    # Constitutive models
    consmodel = _get_kwarg(:consmodel, kwargs)
    @assert consmodel isa ElectroMech

    # Derivatives
    # Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = consmodel(DerivativeStrategy{:analytic}())
    Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = consmodel(DerivativeStrategy{Symbol(diffstrat)}())
    DΨ = @ntuple Ψ ∂Ψu ∂Ψφ ∂Ψuu ∂Ψφu ∂Ψφφ

    # grid model
    mesh_file = _get_kwarg(:meshfile, kwargs)
    model = GmshDiscreteModel(datadir("models", mesh_file))
    if is_vtk 
        writevtk(model, simdir_ * "/DiscreteModel")
    end
    # Setup integration
    order = _get_kwarg(:order, kwargs, 1)
    degree = 2 * order
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    # Dirichlet boundary conditions
    dirichletbc = _get_kwarg(:dirichletbc, kwargs)

    # FE spaces
    fe_spaces = get_FE_spaces(problem, model, order, dirichletbc)

    # WeakForms
    res((u, φ), (v, vφ)) = residual_EM(ctype, (u, φ), (v, vφ), (∂Ψu, ∂Ψφ), f₀, dΩ) # Add Neumann BC 
    jac((u, φ), (du, dφ), (v, vφ)) = jacobian_EM(ctype, (u, φ), (du, dφ), (v, vφ), (∂Ψuu, ∂Ψφu, ∂Ψφφ), dΩ)

    @timeit pname begin
        println("Defining Nonlinear solver")
        # NewtonRaphson solver
        solveropt = _get_kwarg(:solveropt, kwargs)
        nlsolver = get_FE_solver(solveropt)
        init_sol_bool = _get_kwarg(:init_sol_bool, kwargs)

        # Initialization
        if init_sol_bool
            U_ap = _get_kwarg(:U_ap, kwargs)
            X_ap = _get_kwarg(:X_ap, kwargs)
            ph = interpolate([U_ap,X_ap],fe_spaces.U)
            writevtk(Ω, simdir_ * "/results_MB",cellfields=["uh"=>ph[1], "phi"=>ph[2]])
        else
            xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
            xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
            x0 = vcat(xu, xφ)
            ph = FEFunction(fe_spaces.U, x0)
            # writevtk(Ω, simdir_ * "/results_MB_NOinit",cellfields=["uh"=>ph[1], "phi"=>ph[2]])
        end
        @show size(get_free_dof_values(ph))

        post_params = @dict Ω is_vtk simdir_ csvdir_ is_P_F csv_bool csv_funct_
        solver_params = @dict fe_spaces dirichletbc Ω dΩ DΨ res jac solveropt nlsolver post_params

        ph,cache = IncrementalSolver(problem, ctype, ph, solver_params)

    end
end

function ΔSolver!(problem::ElectroMechProblem{:monolithic,:statics}, ctype::CouplingStrategy{:monolithic}, ph, Λ, Λ_inc, params, cache)

    fe_spaces = _get_kwarg(:fe_spaces, params)
    dirichletbc = _get_kwarg(:dirichletbc, params)
    res = _get_kwarg(:res, params)
    jac = _get_kwarg(:jac, params)
    nlsolver = _get_kwarg(:nlsolver, params)
    DΨ = _get_kwarg(:DΨ, params)
    dΩ = _get_kwarg(:dΩ, params)
    is_P_F = _get_kwarg(:is_P_F, params[:post_params])
    # simdir_ = _get_kwarg(:simdir_, params[:post_params])
    # Ω = _get_kwarg(:Ω, params[:post_params])

    # update x0 with dirichlet incrementos   
    uh = ph[1] # not hard copy
    φh = ph[2] # not hard copy
    # Test and trial spaces for Λ_inc
    fe_spaces = get_FE_spaces!(problem, fe_spaces, dirichletbc, Λ_inc)
    # Update Dirichlet for electro problem
    lφ(vφ) = -1.0 * residual_EM(CouplingStrategy{:staggered_E}(), (uh, φh), vφ, DΨ.∂Ψφ, dΩ)
    aφ(dφ, vφ) = jacobian_EM(CouplingStrategy{:staggered_E}(), (uh, φh), dφ, vφ, DΨ.∂Ψφφ, dΩ)
    opφ = AffineFEOperator(aφ, lφ, fe_spaces.Uφ, fe_spaces.Vφ)
    dφh = solve(opφ)

    pφh = get_free_dof_values(φh)
    pdφh = get_free_dof_values(dφh)
    pφh .+= pdφh

    fe_spaces = get_FE_spaces!(problem, fe_spaces, dirichletbc, Λ)

    # writevtk(Ω, simdir_ * "/results_MB_2$Λ",cellfields=["uh"=>ph[1], "phi"=>ph[2]])
    
    op = FEOperator(res, jac, fe_spaces.U, fe_spaces.V)
    # println((ph, nlsolver, op, cache))
    ph, cache = solve!(ph, nlsolver, op, cache)
    # println(cache.result)


    if is_P_F
        uh = ph[1]
        ϕh = ph[2]
        P = DΨ.∂Ψu∘(∇(uh)',∇(ϕh))
    
        return ph, cache, P
    else
        return ph, cache
    end
end


function computeOutputs!(::ElectroMechProblem{:monolithic,:statics}, pvd, ph, P, Λ, Λ_, post_params)

    println("STEP: $Λ_, LAMBDA: $Λ")
    println("============================")

    Ω = _get_kwarg(:Ω, post_params)
    is_vtk = _get_kwarg(:is_vtk, post_params)
    filePath = _get_kwarg(:simdir_, post_params)
    filecsvPath = _get_kwarg(:csvdir_, post_params)
    csv_bool = _get_kwarg(:csv_bool, post_params)
    csv_funct_ = _get_kwarg(:csv_funct_, post_params)
    if csv_bool
        Λstring_ = replace(string(round(Λ, digits=2)), "." => "_")
        xline_df = csv_funct_(ph)
        CSV.write(filecsvPath * "/_Λ_" * Λstring_ * ".csv", xline_df)
        jldsave(filecsvPath * "/_Λ_" * Λstring_ * ".jld2";ph)
    end

    uh = ph[1]
    φh = ph[2]

    F, H, J, b, n, e, εₐ = _getKinematic()
    F_ = F∘(∇(uh)')
    J_ = J∘(F_)
    σ = (1/J_) * P * F_'
    e_ = e∘(b∘(F_))
    εₐ_ = εₐ∘(n∘(F_),e_)



    if is_vtk
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        pvd[Λ_] = createvtk(
            Ω,
            filePath * "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
            cellfields=["u" => uh, "φ" => φh, "F" => F_, "P" => P, "σ" => σ, "e" => e_, "εₐ" => εₐ_, "J"  => J_]
        )
    end

    return pvd
end

function computeOutputs!(::ElectroMechProblem{:monolithic,:statics}, pvd, ph, Λ, Λ_, post_params)

    println("STEP: $Λ_, LAMBDA: $Λ")
    println("============================")

    Ω = _get_kwarg(:Ω, post_params)
    is_vtk = _get_kwarg(:is_vtk, post_params)
    filePath = _get_kwarg(:simdir_, post_params)
    filecsvPath = _get_kwarg(:csvdir_, post_params)
    csv_bool = _get_kwarg(:csv_bool, post_params)
    csv_funct_ = _get_kwarg(:csv_funct_, post_params)

    if csv_bool
        Λstring_ = replace(string(round(Λ, digits=2)), "." => "_")
        xline_df = csv_funct_(ph)
        CSV.write(filecsvPath * "/_Λ_" * Λstring_ * ".csv", xline_df)
        jldsave(filecsvPath * "/_Λ_" * Λstring_ * ".jld2";ph)
    end

    uh = ph[1]
    φh = ph[2]

    if is_vtk
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        pvd[Λ_] = createvtk(
            Ω,
            filePath * "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
            cellfields=["u" => uh, "φ" => φh]
        )
    end

    return pvd
end