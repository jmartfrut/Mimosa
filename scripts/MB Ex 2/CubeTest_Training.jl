using Gridap
using GridapGmsh
using LineSearches: BackTracking
using Gridap.FESpaces
using CSV
using DataFrames

model = GmshDiscreteModel("data/models/Cube.msh")

degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

neumanntags = ["surf_1"]
Γ = BoundaryTriangulation(model,tags=neumanntags)
dΓ = Measure(Γ,degree)

pressure = -320

x3_1 = ["surf_1","surf_7","surf_13","surf_19"]
x3_0 = ["surf_2","surf_8","surf_14","surf_20"]
x1_0 = ["surf_6","surf_24"]
x2_0 = ["surf_3","surf_9"]

dir_tags =  Vector{String}([])
append!(dir_tags,x3_1,x3_0,x1_0,x2_0)

x3_1_m = [(true,true,false),(true,true,false),(true,true,false),(true,true,false)]
x3_0_m = [(false,false,true),(false,false,true),(false,false,true),(false,false,true)]
x1_0_m = [(true,false,false),(true,false,false)]
x2_0_m = [(false,true,false),(false,true,false)]

dir_tags_m = []
append!(dir_tags_m,x3_1_m,x3_0_m,x1_0_m,x2_0_m)


reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},1)
V = TestFESpace(model,reffe,
  conformity=:H1,dirichlet_tags = dir_tags,
  dirichlet_masks = dir_tags_m)
v = get_fe_basis(V)

g(x) = VectorValue(0.0,0.0,0.0)
g_ = [g for _ in 1:length(dir_tags)]
U = TrialFESpace(V,g_)
u = get_trial_fe_basis(U)

# λ_list = 333288.89, 416611.11, 499933.33, 583255.55, 666577.77 μ_list = 66.67, 83.34, 100.01, 116.67, 133.34


const λ = 666577.77
const μ = 133.34
# Deformation Gradient

F(∇u) = one(∇u) + ∇u'

J(F) = sqrt(det(C(F)))

#Green strain

#E(F) = 0.5*( F'*F - one(F) )

dE(∇du,∇u) = 0.5*( ∇du⋅F(∇u) + (∇du⋅F(∇u))' )

# Right Cauchy-green deformation tensor

C(F) = (F')⋅F

# Constitutive law (Neo hookean)

function S(∇u)
  Cinv = inv(C(F(∇u)))
  μ*(one(∇u)-Cinv) + λ*log(J(F(∇u)))*Cinv
end

function dS(∇du,∇u)
  Cinv = inv(C(F(∇u)))
  _dE = dE(∇du,∇u)
  λ*(Cinv⊙_dE)*Cinv + 2*(μ-λ*log(J(F(∇u))))*Cinv⋅_dE⋅(Cinv')
end

# Cauchy stress tensor

σ(∇u) = (1.0/J(F(∇u)))*F(∇u)⋅S(∇u)⋅(F(∇u))'




# res(u,v) = ∫( (dE∘(∇(v),∇(u))) ⊙ (S∘∇(u)) )*dΩ - ∫(t⋅v)*dΓ

jac_mat(u,du,v) =  ∫( (dE∘(∇(v),∇(u))) ⊙ (dS∘(∇(du),∇(u))) )*dΩ

jac_geo(u,du,v) = ∫( ∇(v) ⊙ ( (S∘∇(u))⋅∇(du) ) )*dΩ

jac(u,du,v) = jac_mat(u,du,v) + jac_geo(u,du,v)

# Initial tangent matrix
x0 = zeros(Float64,num_free_dofs(V))
uh = FEFunction(U,x0)
JAC = jac(uh,u,v)[Ω]
assem = SparseMatrixAssembler(U,V)
σₖ = get_cell_dof_ids(U)
rs = ([JAC],[σₖ],[σₖ])
K_T_init = allocate_matrix(assem,rs)
assemble_matrix!(K_T_init,assem,rs)

# Setup non-linear solver
nls = NLSolver(
  show_trace=true,
  extended_trace=false,
  store_trace=false,
  method=:newton,
  # linesearch=BackTracking()
  )

solver = FESolver(nls)

function run(x0,step,nsteps,cache,x_list,b_list,R_nl_list)
    Δinc = step/nsteps

    t(x) = VectorValue(0.0,0.0,Δinc*pressure)
    res(u,v) = ∫( (dE∘(∇(v),∇(u))) ⊙ (S∘∇(u)) )*dΩ - ∫(t⋅v)*dΓ

    #FE problem
    op = FEOperator(res,jac,U,V)
  
    println("\n+++ Solving for step $step of $nsteps +++\n")
  
    uh = FEFunction(U,x0)

    RES = res(uh,v)
    rs = ([RES[Ω]],[σₖ])
    b = allocate_vector(op.assem,rs)
    assemble_vector!(b,op.assem,rs)
    R_nl = b-K_T_init*x0

    push!(x_list,copy(x0))
    push!(b_list,copy(b))
    push!(R_nl_list,copy(R_nl))
  
    uh, cache = solve!(uh,solver,op,cache)
  
    # writevtk(Ω,"res_$(lpad(step,3,'0'))",cellfields=["uh"=>uh,"sigma"=>σ∘∇(uh),"S"=>S∘∇(uh)])
  
    return get_free_dof_values(uh), cache, x_list, b_list, R_nl_list, res
  
end

function runs()
  x0 = zeros(Float64,num_free_dofs(V))
  cache = nothing
  nsteps = 100
  x_list = []
  b_list = []
  R_nl_list = []
  res = nothing
  for step in 1:nsteps 
    x0, cache, x_list, b_list, R_nl_list, res = run(x0,step,nsteps,cache,x_list,b_list,R_nl_list)
  end
  uh = FEFunction(U,x0)

  RES = res(uh,v)
  rs = ([RES[Ω]],[σₖ])
  b = allocate_vector(assem,rs)
  assemble_vector!(b,assem,rs)
  R_nl = b-K_T_init*x0

  push!(x_list,copy(x0))
  push!(b_list,copy(b))
  push!(R_nl_list,copy(R_nl))

  x_list = reduce(hcat,x_list)
  b_list = reduce(hcat,b_list)
  R_nl_list = reduce(hcat,R_nl_list)

  return x_list, b_list, R_nl_list
end

function collect_data()
  x_list, b_list, R_nl_list = runs()
  df_x = DataFrame(x_list, :auto)
  df_b = DataFrame(b_list, :auto)
  df_R_nl = DataFrame(R_nl_list, :auto)
  CSV.write("data/csv/Cube_test/x_"*"Lambda_$λ"*"Mu_$μ"*".csv",df_x)
  CSV.write("data/csv/Cube_test/b_"*"Lambda_$λ"*"Mu_$μ"*".csv",df_b)
  CSV.write("data/csv/Cube_test/R_nl_"*"Lambda_$λ"*"Mu_$μ"*".csv",df_R_nl)
end