using DrWatson
using Mimosa
using Gridap.Visualization
using Gridap.TensorValues
using CSV
using DataFrames



function get_parameters(pot, sw, St, Sl, csv_funct)

    problemName = "TubeBeam_V2_grav1" 
    problemName = problemName*"_ϕ$pot"
    for s in sw
        problemName = problemName*"_$s"
    end
    ptype = "ElectroMechanics"
    soltype = "monolithic"
    regtype = "statics"
    diffstrat = "autodiff"
    meshfile = "TubeBeam_V2_SecT_4-SecL_4.msh"

    # problemName = "TB"*"$(St)ST"*"$(Sl)SL"*"-O2-/$diffstrat/Elastomer-X_Yeoh/"*problemName
    problemName = "TB"*"$(St)ST"*"$(Sl)SL"*"-O2-Nograv/$diffstrat/VHB4910_Yeoh/"*problemName


    # modmec = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0)
    # modmec = NeoHookean3D(λ=0, μ=0.03911e6)
    # modmec = Carroll(a = 2.868e05, b = 1.4183e-01, c =  7.846e05, κ = 2.868e07)
    modmec = Yeoh(C₁ = 0.0693e6, C₂ = -8.88e2, C₃ = 16.7, κ = 0.0693e8)
    # modmec = Yeoh(C₁ = 0.170747e6, C₂ = 0.00451313e6, C₃ = -28.5168, κ = 0.0693e8)
    modelec = IdealDielectric(ε=8.8542e-12*4.0)
    consmodel = ElectroMech(modmec, modelec)

    # Boundary conditions 

    evolu(Λ) = 1.0
    dir_u_tags = ["fixed_surf_1", "fixed_surf_2", "fixed_surf_3", "fixed_surf_4"]
    dir_u_values = [[0.0,0.0,0.0] for i in dir_u_tags]
    dir_u_timesteps = [evolu for i in dir_u_tags]
    masks = [(true,true,true) for i in dir_u_tags]
    Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps, masks)

    evolφ(Λ) = Λ
    ext_surf_list = ["ext_surf_1", "ext_surf_2", "ext_surf_3", "ext_surf_4", "ext_surf_5",
     "ext_surf_6", "ext_surf_7", "ext_surf_8", "ext_surf_9", "ext_surf_10", "ext_surf_11",
      "ext_surf_12", "ext_surf_13", "ext_surf_14", "ext_surf_15", "ext_surf_16"]
    earth_loc = ["int_surf_1", "int_surf_2", "int_surf_3", "int_surf_4", "int_surf_5",
     "int_surf_6", "int_surf_7", "int_surf_8", "int_surf_9", "int_surf_10", "int_surf_11",
      "int_surf_12", "int_surf_13", "int_surf_14", "int_surf_15", "int_surf_16"]
    power_loc = []
    i = 1
    for io in sw
        if io == 1
            push!(power_loc,ext_surf_list[i])
        end
        i += 1
    end
    earth_val = [0.0 for i in earth_loc]
    power_val = [pot for i in power_loc]
    dir_φ_tags = Vector{String}()
    append!(dir_φ_tags,earth_loc)
    append!(dir_φ_tags,power_loc)
    dir_φ_timesteps = [evolφ for i in dir_φ_tags]
    dir_φ_values = []
    append!(dir_φ_values,earth_val)
    append!(dir_φ_values,power_val)
    
    Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

    dirichletbc = MultiFieldBoundaryCondition([Du, Dφ])

    # Body force
    # f₀=VectorValue(9.81*960.0,0.0,0.0)
    f₀=VectorValue(0.0,0.0,0.0)

    # FE parameters
    order = 2

    # NewtonRaphson parameters
    nr_show_trace = true
    nr_iter = 15
    nr_ftol = 1e-12

    # Incremental solver
    nsteps = 25
    nbisec = 10

    solveropt = @dict nr_show_trace nr_iter nr_ftol nsteps nbisec

    # Postprocessing
    is_vtk = true
    is_P_F = true

    init_sol_bool, U_ap, X_ap = false, nothing, nothing

    csv_bool = false
    csv_funct_ = csv_funct


    return @dict problemName ptype soltype diffstrat regtype meshfile consmodel dirichletbc order solveropt is_vtk is_P_F init_sol_bool U_ap X_ap csv_bool csv_funct_ f₀
end

function CenterLine(ph)

    u = ph[1]
    vd = visualization_data(u.cell_field.trian,"",cellfields=["u"=>u])
    vd = vd[1]
    X_grid = vd.grid.sub_grid.node_coordinates
    u_grid = vd.nodaldata["u"]
    x_grid = X_grid .+ u_grid
    sort_inidices = sortperm(X_grid, by = x -> x[1])
    x0 = X_grid[sort_inidices[1]][1]
    x_ = VectorValue(0.0,0.0,0.0)
    n = 0
    x_line = []
    tol = 0.00001
    for i in sort_inidices
        if (1-tol)*x0<=X_grid[i][1]<=(1+tol)*x0
            x_ += x_grid[i]
            n += 1
        else
            push!(x_line, x_./n)
            x0 = X_grid[i][1]
            x_ = x_grid[i]
            n = 1
        end
    end
    push!(x_line, x_./n)
    # plot(getproperty.(x_line,:data))
    return x_line
end

function CenterLine_(ph)

    u = ph[1]
    vd = visualization_data(u.cell_field.trian,"",cellfields=["u"=>u])
    vd = vd[1]
    X_grid = vd.grid.sub_grid.node_coordinates
    u_grid = vd.nodaldata["u"]
    x_grid = X_grid .+ u_grid
    sort_inidices = sortperm(X_grid, by = x -> x[1])
    x0 = X_grid[sort_inidices[1]][1]
    x_ = VectorValue(0.0,0.0,0.0)
    n = 0
    x_line = []
    tol = 0.00001
    for i in sort_inidices
        if (1-tol)*x0<=X_grid[i][1]<=(1+tol)*x0
            x_ += x_grid[i]
            n += 1
        else
            push!(x_line, x_./n)
            x0 = X_grid[i][1]
            x_ = x_grid[i]
            n = 1
        end
    end
    push!(x_line, x_./n)
    x_line_list = [x_line]
    x_line_list = [getproperty.(x_line,:data) for x_line in x_line_list]
    x_line_list = [x_line_list[i][j][k] for i in 1:lastindex(x_line_list), j in 1:lastindex(x_line_list[1]), k in 1:lastindex(x_line_list[1][1])]
    df = DataFrame(vcat(x_line_list[:,:,1],x_line_list[:,:,2],x_line_list[:,:,3]), :auto)
    return df
end

function run_(start, finish)
    div = 10
    pots = [5000.0]
    conf_list = CSV.File("data/csv/EM_TB_ST4_SL4_ConfRand.csv") |> Tables.matrix
    St = 4
    Sl = 4
    ph_list = []
    x_line_list = []
    n = lastindex(eachcol(conf_list))
    # start = (div-1)*(n/10) + 1
    # finish =  div*(n/10)
    t0 = time()
    for i in Int(start):Int(finish)
        t0_ = time()
        conf = [1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1] # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0] #[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1] # conf_list[:,i]
        println(" ")
        println("!!!!!!!!    Configuration number $i / $(Int(finish))  (total = $n)   !!!!!!!")
        println("!!!!!!!!        Configuration $conf        !!!!!!!")
        println(" ")
        ph, chache = main(; get_parameters(pots[1], conf, St, Sl,CenterLine_)...)
        x_line = CenterLine(ph)
        push!(ph_list,ph)
        push!(x_line_list,x_line)
        t_ = time() - t0_
        T_ = time() - t0
        println("Evaluation time = $t_ s = $(t_/60.0) min")
        println("Total elapsed time = $T_ s = $(T_/60.0) min")
    end
end

function run_candidates(start, finish)
    div = 10
    pots = [5000.0]
    conf_list = CSV.File("data/csv/EM_TB_ST4_SL4_CandidateConf.csv") |> Tables.matrix
    St = 4
    Sl = 4
    ph_list = []
    x_line_list = []
    n = lastindex(eachcol(conf_list))
    # start = (div-1)*(n/10) + 1
    # finish =  div*(n/10)
    time = []
    for i in Int(start):Int(finish)
        t0 = time()
        conf = conf_list[:,i]
        println(" ")
        println("!!!!!!!!    Configuration number $i / $(Int(finish))  (total = $n)   !!!!!!!")
        println("!!!!!!!!        Configuration $conf        !!!!!!!")
        println(" ")
        main(; get_parameters(pots[1], conf, St, Sl,CenterLine_)...)
        t_ = time() - t0
        push!(time,t)
    end
    println("total time = $(sum(time_))")
    df = DataFrame(time__ = time)
    CSV.write("data/csv/Time_TB_4ST_4SL_Conf_from_$(start)_to_$(finish).csv",df)
end

# main(; get_parameters(5000.0, [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0], 4, 4,CenterLine_)...)