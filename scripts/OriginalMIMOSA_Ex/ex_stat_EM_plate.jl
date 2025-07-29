using DrWatson
using Mimosa


function get_parameters()

  problemName = "EMPLATE"
  ptype = "ElectroMechanics"
  soltype = "monolithic"
  regtype = "statics"
  meshfile = "ex2_mesh.msh"
  diffstrat = "autodiff"



  # modmec = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0)
  modmec = NeoHookean3D(λ=10.0, μ=1.0)
  modelec = IdealDielectric(ε=1.0)
  consmodel = ElectroMech(modmec, modelec)

  # Boundary conditions 

  evolu(Λ) = 1.0
  dir_u_tags = ["fixedup"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [evolu]
  masks = [(true,true,true)]
  Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps,masks)

  evolφ(Λ) = Λ
  dir_φ_tags = ["midsuf", "topsuf"]
  dir_φ_values = [0.0, 0.2]
  dir_φ_timesteps = [evolφ, evolφ]
  Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

  dirichletbc = MultiFieldBoundaryCondition([Du, Dφ])

  # FE parameters
  order = 1

  # NewtonRaphson parameters
  nr_show_trace = true
  nr_iter = 20
  nr_ftol = 1e-12

  # Incremental solver
  nsteps = 5
  nbisec = 10

  solveropt = @dict nr_show_trace nr_iter nr_ftol nsteps nbisec

  # Postprocessing
  is_vtk = true
  is_P_F = false

  init_sol_bool, U_ap, X_ap = false, nothing, nothing

  csv_bool = false
  csv_funct_ = nothing

  return @dict problemName ptype soltype diffstrat regtype meshfile consmodel dirichletbc order solveropt is_vtk is_P_F init_sol_bool U_ap X_ap csv_bool csv_funct_

end

main(; get_parameters()...)

#  using PProf

#  PProf.Allocs.pprof(from_c=false)