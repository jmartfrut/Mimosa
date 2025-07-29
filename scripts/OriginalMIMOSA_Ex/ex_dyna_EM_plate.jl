using DrWatson
using Mimosa
using Gridap
using Plots

function get_parameters()

  problemName = "dynamicEMPLATE"
  ptype = "ElectroMechanics"
  soltype = "monolithic"
  regtype = "dynamics"
  meshfile = "ex2_mesh.msh"

  modmec = NeoHookean3D(λ=10.0, μ=1.0, ρ=0.001)
  # modmec = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0, ρ=0.01)
  modelec = IdealDielectric(ε=1.0)
  consmodel = ElectroMech(modmec, modelec)

  # Boundary conditions 

  evolu(Λ) = 1.0
  dir_u_tags = ["fixedup"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [evolu]
  masks = [(true,true,true)]
  Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps,masks)

  evolφ(t) = min(t / 5.0, 1.0)
  dir_φ_tags = ["midsuf", "topsuf"]
  dir_φ_values = [0.0, 0.2]
  dir_φ_timesteps = [evolφ, evolφ]
  Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

  dirichletbc = MultiFieldBoundaryCondition([Du, Dφ])

  # initial conditions
  velocity(x) = VectorValue(0.0, 0.0, 0.0)

  # FE parameters
  order = 1

  # NewtonRaphson parameters
  nr_show_trace = true
  nr_iter = 20
  nr_ftol = 1e-12

  # Midpoint solver
  Δt = 0.01
  nsteps = 5000
  solveropt = @dict nr_show_trace nr_iter nr_ftol Δt nsteps

  # Postprocessing
  is_vtk = true

  return @dict problemName ptype soltype regtype meshfile consmodel dirichletbc order solveropt is_vtk velocity
end

ph, KE, EE = main(; get_parameters()...)

plot(KE, label="Kinetic Energy")
plot!(EE, label="Elastic Energy")
plot!(KE + EE, label="Total Energy")

#  using PProf

#  PProf.Allocs.pprof(from_c=false)