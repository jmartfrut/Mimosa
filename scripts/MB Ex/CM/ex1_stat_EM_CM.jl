using DrWatson
using Mimosa
using Gridap
using Plots
using CSV
using DataFrames
using Gridap.Visualization
using Gridap.TensorValues
using LineSearches: BackTracking


function get_parameters(sw, pot)

  problemName = "5-CM-O2-PL" #PB = PlateBeam; S = # of sections; O2 = Order of elements; PL = Potential Location; ϕ = potential magnitude
  for s in sw
    problemName = problemName*"_$s"
  end
  pot_10 = 10*pot
  problemName = problemName*"_ϕ$pot_10"
  ptype = "ElectroMechanics"
  soltype = "monolithic"
  regtype = "statics"
  # diffstrat = "analytic"
  diffstrat = "autodiff"
  # meshfile = "PlateBeam4SecSI.msh"
  meshfile = "CircularMembrane5.msh"


  # modmec = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0)
  # modmec = NeoHookean3D(λ=0, μ=0.03911e6)
  modmec = Yeoh(C₁ = 0.0693e6, C₂ = -8.88e2, C₃ = 16.7, κ = 0.0693e8)
  # modmec = NeoHookean3DNearlyIncomp(λ=0.03911e6*1e3, μ=0.03911e6)
  # modmec = NeoHookean3D(λ=1e5, μ=0.03911e6)
  modelec = IdealDielectric(ε=8.8542e-12*4.0)
  consmodel = ElectroMech(modmec, modelec)

  # Boundary conditions 

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
      if io == 1
        push!(power_loc,top_surf_list[i])
      elseif io == 2
        push!(power_loc,bottom_surf_list[i])
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

  # FE parameters
  order = 2

  # NewtonRaphson parameters
  nr_show_trace = true
  nr_iter = 10
  nr_ftol = 1e-12

  # Incremental solver
  nsteps = 10
  nbisec = 5

  solveropt = @dict nr_show_trace nr_iter nr_ftol nsteps nbisec

  # Postprocessing
  is_vtk = true
  is_P_F = true

  init_sol_bool, U_ap, X_ap = false, nothing, nothing

  csv_bool = true
  csv_funct_ = Surf_

  return @dict problemName ptype soltype regtype diffstrat meshfile consmodel dirichletbc order solveropt is_vtk is_P_F init_sol_bool U_ap X_ap csv_bool csv_funct_
end

# function Surf_(ph)
#   u = ph[1]
#   n = 16
#   r_list = [0.0e-3:1e-3:49.0e-3...]
#   θ_list = [(((i-1)/n)*2*pi)*(1.0) for i in 1:n]
#   X_list = []
#   for θ in θ_list
#     # println("θ = $θ")
#     for r in r_list
#       # println("r = $r")
#       push!(X_list,VectorValue(r*cos(θ),r*sin(θ),0.0e-3))
#     end
#   end
#   for X in X_list
#     println(X)
#     u(X)
#   end
#   u_list = u.(X_list)
#   x_list = X_list .+ u_list
#   x_list = [getproperty.(x_list,:data)]
#   x_list = [x_list[i][j][k] for i in 1:lastindex(x_list), j in 1:lastindex(x_list[1]), k in 1:lastindex(x_list[1][1])]
#   df = DataFrame(vcat(x_list[:,:,1],x_list[:,:,2],x_list[:,:,3]), :auto)
#   return df
# end

function Surf_(ph)
  u = ph[1]
  vd = visualization_data(u.cell_field.trian,"",cellfields=["u"=>u])
  vd = vd[1]
  X_grid = vd.grid.sub_grid.node_coordinates
  u_grid = vd.nodaldata["u"]
  x_grid = X_grid .+ u_grid
  X_grid = getproperty.(X_grid,:data)
  u_grid = getproperty.(u_grid,:data)
  x_grid = getproperty.(x_grid,:data)
  X_grid = [[X[1],X[2],X[3]] for X in X_grid]
  u_grid = [[u[1],u[2],u[3]] for u in u_grid]
  x_grid = [[x[1],x[2],x[3]] for x in x_grid]
  X_grid = reduce(hcat,X_grid)'
  u_grid = reduce(hcat,u_grid)'
  x_grid = reduce(hcat,x_grid)'
  data = hcat(X_grid,u_grid,x_grid)
  df = DataFrame(data, :auto)
  return df
end

function run()
  pot = 4000.0
  # sw = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
  # sw = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  conf_list = []
  for i in 0:2, j in 0:2, k in 0:2, l in 0:2
    push!(conf_list,[i,j,k,l,i,j,k,l,i,j,k,l,i,j,k,l])
  end
  count = 1
  t0 = time()
  err_conf = []
  for i in 75:75 #lastindex(conf_list)
    sw = conf_list[i]
    println("------------------ $count / $(length(conf_list)) ---------------")
    println("------------------ $i / $(length(conf_list)) ---------------")
    #try
      main(; get_parameters(sw, pot)...)
    #catch
     # push!(err_conf,i)
    #end
    
    println("")
    println("  -----   Total elapsed time $(time()-t0)")
    count += 1
  end
end
