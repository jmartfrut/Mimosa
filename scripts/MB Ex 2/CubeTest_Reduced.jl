using Gridap
using GridapGmsh
using LineSearches: BackTracking
using Gridap.FESpaces
using CSV
using DataFrames
using JLD2
using NLsolve
using WriteVTK
using LightXML

DEIM_Matrices = load("scripts/MB Ex 2/DEIM_matrices.jld2")
𝛟 = DEIM_Matrices["𝛟"]
M_DEIM = DEIM_Matrices["M_DEIM"]
Z = DEIM_Matrices["Z"]
𝛀 = DEIM_Matrices["𝛀"]
_, m_k = size(𝛟)

λ_list = [333288.89, 416611.11, 499933.33, 583255.55, 666577.77]
μ_list = [66.67, 83.34, 100.01, 116.67, 133.34]

D_x = []
D_R_nl= []
for i in 1:5
    λ = λ_list[i]
    μ = μ_list[i]
    file_name = "data/csv/Cube_test/x_"*"Lambda_$λ"*"Mu_$μ"*".csv"
    _X = CSV.File(file_name) |> Tables.matrix
    push!(D_x,_X)
    file_name = "data/csv/Cube_test/R_nl_"*"Lambda_$λ"*"Mu_$μ"*".csv"
    _R_nl = CSV.File(file_name) |> Tables.matrix
    push!(D_R_nl,_R_nl)
end

D_x = reduce(hcat,D_x)
D_R_nl = reduce(hcat,D_R_nl)

k=3
β = 5e-7
Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Λ, U, U_, Ḡ, G = kPOD(Κ,D_R_nl,k)
Z_ = real.(U_'*Ḡ)


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


const λ = 400889.87
const μ = 80.194
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
σₕ = get_cell_dof_ids(U,Γ)
rs = ([JAC],[σₖ],[σₖ])
K_T_init = allocate_matrix(assem,rs)
assemble_matrix!(K_T_init,assem,rs)
K_0_red = 𝛟'*K_T_init*𝛟
norm(K_T_init)


function run(x0,step,nsteps)
    
    Δinc = step/nsteps
    t(x) = VectorValue(0.0,0.0,Δinc*pressure)
    res(u,v) = ∫( (dE∘(∇(v),∇(u))) ⊙ (S∘∇(u)) )*dΩ - ∫(t⋅v)*dΓ
    function f!(F_,x_0)
        x_ = 𝛟*x_0
        uh = FEFunction(U,x_)
        RES = res(uh,v)
        assem = SparseMatrixAssembler(U,V)
        rs = ([RES[Ω]],[σₖ])
        T = allocate_vector(assem,rs)
        assemble_vector!(T,assem,rs)
        rs = ([RES[Γ]],[σₕ])
        T_nl = T-K_T_init*x_
        F = allocate_vector(assem,rs)
        assemble_vector!(F,assem,rs)
        # R = K_0_red*x0 + 𝛟'*T_nl + 𝛟'*F
        R = K_0_red*x0 + M_DEIM*Z'*T_nl + 𝛟'*F
        copyto!(F_,R)
    end

    function j!(J,x0)
        x_ = 𝛟*x0
        uh = FEFunction(U,x_)
        JAC = jac(uh,u,v)[Ω]
        assem = SparseMatrixAssembler(U,V)
        σₖ = get_cell_dof_ids(U)
        rs = ([JAC],[σₖ],[σₖ])
        K_T = allocate_matrix(assem,rs)
        assemble_matrix!(K_T,assem,rs)
        K_T = K_T - K_T_init
        # K_T_red = K_0_red + 𝛟'*K_T*𝛟
        K_T_red = K_0_red + M_DEIM*Z'*K_T*𝛟
        copyto!(J,K_T_red)
    end

    F_ = zeros(Float64,m_k)
    J_ = zeros(Float64,m_k,m_k)

    # f!(F_,x0)
    # # println(F_)
    # df = OnceDifferentiable(f!, j!, x0, F_)

    # r = NLsolve.nlsolve(df, x0,
    # show_trace=true,
    # extended_trace=false,
    # store_trace=false,
    # method=:newton,
    # )
    norm_F = 1
    count = 1
    while norm_F>1e-10
        f!(F_,x0)
        j!(J_,x0)
        Δx = inv(J_)*(-F_)
        println("count = $count f norm = $(norm(F_)) j norm = $(norm(J_)) Δx = $(norm(𝛟*Δx))")
        copyto!(x0,x0 + Δx)
        norm_F = norm(F_)
        count += 1
    end

    return x0

end

function runs()
    nsteps = 100
    x0 = zeros(Float64,m_k)
    x_list = [copy(x0)]
    x_ = zeros(Float64,m_k)
    t0 = time()
    time_step = []
    for step in 1:nsteps
        println("Increment step = $step")
        x0 = run(x0,step,nsteps)
        println("norm of x0 = $(norm(x0))")
        copyto!(x_,x0)
        push!(x_list,copy(x0))
        push!(time_step,time()-t0)
    end
    total_time = time()-t0
    return x_list, time_step, total_time
end

# x_list, time_step, total_time = runs()

function Red_VTK(x0)
    x = 𝛟*x0
    uh = FEFunction(U,x)
    writevtk(Ω,"data/sims/Cube_test/Result_POD-DEIM_"*"Lambda_$λ"*"Mu_$μ",cellfields=["uh"=>uh,"sigma"=>σ∘∇(uh),"S"=>S∘∇(uh)])
end

function Red_Error_VTK(x0)
    x = 𝛟*x0
    uh_red = FEFunction(U,x)
    file_name = "data/csv/Cube_test/x_"*"Lambda_$λ"*"Mu_$μ"*".csv"
    _X = CSV.File(file_name) |> Tables.matrix
    uh = FEFunction(U,_X[:,end])
    Error = abs.(_X[:,end] - x)./maximum(abs.(_X[:,end]))
    error = FEFunction(U,Error)
    writevtk(Ω,"data/sims/Cube_test/Result_withError_POD-DEIM_"*"Lambda_$λ"*"Mu_$μ",cellfields=["uh"=>uh, "uh_red"=>uh_red,"sigma"=>σ∘∇(uh),"S"=>S∘∇(uh), "sigma_red"=>σ∘∇(uh_red),"S_red"=>S∘∇(uh_red), "error"=>error])
end

function Red_Error_last(x_list)
    file_name = "data/csv/Cube_test/x_"*"Lambda_$λ"*"Mu_$μ"*".csv"
    _X = CSV.File(file_name) |> Tables.matrix
    println("Norm of 𝛟*x_red-x = $(norm((𝛟*x_list[end]-_X[:,end])./maximum(abs.(_X[:,end]))))")
    println("Max Abs of 𝛟*x_red-x = $(maximum(abs.((𝛟*x_list[end]-_X[:,end])./maximum(abs.(_X[:,end])))))")
    Red_VTK(x_list[end])
    Red_Error_VTK(x_list[end])
end

function Red_Error(x_list)
    file_name = "data/csv/Cube_test/x_"*"Lambda_$λ"*"Mu_$μ"*".csv"
    _X = CSV.File(file_name) |> Tables.matrix
    println("Norm of 𝛟*x_red-x = $(norm((𝛟*x_list[end]-_X[:,end])./maximum(abs.(_X[:,end]))))")
    println("Max Abs of 𝛟*x_red-x = $(maximum(abs.((𝛟*x_list[end]-_X[:,end])./maximum(abs.(_X[:,end])))))")
    mkpath("data/sims/Cube_test/Result_withError_POD-DEIM_"*"Lambda_$λ"*"Mu_$μ")
    pvd = paraview_collection("data/sims/Cube_test/Result_withError_POD-DEIM_"*"Lambda_$λ"*"Mu_$μ" * "/Results", append=false)
    writevtk(model, "data/sims/Cube_test/Result_withError_POD-DEIM_"*"Lambda_$λ"*"Mu_$μ" * "/DiscreteModel")
    for i in 1:lastindex(x_list)
        x0 = x_list[i]
        x = 𝛟*x0
        uh_red = FEFunction(U,x)
        _x = _X[:,i]
        uh = FEFunction(U,_x)
        Error = abs.(_x - x)./maximum(abs.(_x))
        error = FEFunction(U,Error)
        Error_ = abs.(_x - x)
        error_ = FEFunction(U,Error_)
        Λ = i/(lastindex(x_list)-1)
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        Λ_ = i
        pvd[Λ_] = createvtk(
            Ω,
            "data/sims/Cube_test/Result_withError_POD-DEIM_"*"Lambda_$λ"*"Mu_$μ" * "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
            cellfields=["u"=>uh, "u_red"=>uh_red,"sigma"=>σ∘∇(uh),"S"=>S∘∇(uh), "sigma_red"=>σ∘∇(uh_red),"S_red"=>S∘∇(uh_red), "error_"=>error, "error_abs"=>error_]
        )
        print(Λ_)
    end
    vtk_save(pvd)
end

function run_kPCA(x0,z,step,nsteps)
    
    Δinc = step/nsteps
    t(x) = VectorValue(0.0,0.0,Δinc*pressure)
    res(u,v) = ∫( (dE∘(∇(v),∇(u))) ⊙ (S∘∇(u)) )*dΩ - ∫(t⋅v)*dΓ
    function f!(F_,x_0,Δx,z,T_nl_n_1)
        Δx_ = 𝛟*Δx
        x_ = 𝛟*x_0
        uh = FEFunction(U,x_)
        RES = res(uh,v)
        JAC = jac(uh,u,v)[Ω]
        assem = SparseMatrixAssembler(U,V)
        σₖ = get_cell_dof_ids(U)
        rs = ([JAC],[σₖ],[σₖ])
        K_T = allocate_matrix(assem,rs)
        assemble_matrix!(K_T,assem,rs)
        K_T = K_T - K_T_init
        ΔT_nl = K_T*Δx_
        g = [-β*Κ(T_nl_n_1,i)*dot(ΔT_nl,T_nl_n_1-i)*2 for i in eachcol(D_R_nl)]
        ns = length(g)
        g = g - (1/ns)*ones(ns,ns)*g
        Z_new = z + real.(U_'*g)
        println("Δz norm = $(norm(real.(U_'*g)))")
        T_nl, _, _ = ReverseMap(D_R_nl,Z_,Z_new,5)
        rs = ([RES[Γ]],[σₕ])
        F = allocate_vector(assem,rs)
        assemble_vector!(F,assem,rs)
        R = K_0_red*x0 + 𝛟'*T_nl + 𝛟'*F
        # R = K_0_red*x0 + M_DEIM*Z'*T_nl + 𝛟'*F
        copyto!(F_,R)
        return Z_new, T_nl
    end

    function j!(J,x0)
        x_ = 𝛟*x0
        uh = FEFunction(U,x_)
        JAC = jac(uh,u,v)[Ω]
        assem = SparseMatrixAssembler(U,V)
        σₖ = get_cell_dof_ids(U)
        rs = ([JAC],[σₖ],[σₖ])
        K_T = allocate_matrix(assem,rs)
        assemble_matrix!(K_T,assem,rs)
        K_T = K_T - K_T_init
        K_T_red = K_0_red + 𝛟'*K_T*𝛟
        # K_T_red = K_0_red + M_DEIM*Z'*K_T*𝛟
        copyto!(J,K_T_red)
    end

    F_ = zeros(Float64,m_k)
    J_ = zeros(Float64,m_k,m_k)
    Δx = zeros(Float64,m_k)
    
    T_nl_n_1, _, _ = ReverseMap(D_R_nl,Z_,z,4)
    norm_F = 1
    count = 1
    while norm_F>1e-10
        z,T_nl_n_1 = f!(F_,x0,Δx,z,T_nl_n_1)
        j!(J_,x0)
        Δx = inv(J_)*(-F_)
        println("count = $count f norm = $(norm(F_)) j norm = $(norm(J_)) Δx = $(norm(𝛟*Δx))")
        copyto!(x0,x0 + Δx)
        # println(x0)
        norm_F = norm(F_)
        count += 1
    end
    return x0, z
end

function runs_kPCA()
    nsteps = 100
    x0 = zeros(Float64,m_k)
    x_list = [copy(x0)]
    x_ = zeros(Float64,m_k)
    z = _Interpolation([200,250,300,350,400],Z_[:,[1,102,203,304,405]],275)
    t0 = time()
    time_step = []
    for step in 1:nsteps
        println("===============================================")
        println("Increment step = $step")
        x0, z = run_kPCA(x0,z,step,nsteps)
        z = _Interpolation([200,250,300,350,400],Z_[:,[step+1,step+102,step+203,step+304,step+405]],275)
        println("norm of x0 = $(norm(x0))")
        # println(x0)
        copyto!(x_,x0)
        push!(x_list,copy(x0))
        push!(time_step,time()-t0)
    end
    total_time = time()-t0
    return x_list, time_step, total_time
end

# x_list, time_step, total_time = runs_kPCA()
# Red_VTK(X[end])
# file_name = "data/csv/Cube_test/x_"*"Lambda_$λ"*"Mu_$μ"*".csv"
# _X = CSV.File(file_name) |> Tables.matrix

# norm(𝛟*X[end]-_X[:,end])
# maximum(abs.(𝛟*X[end]-_X[:,end]))