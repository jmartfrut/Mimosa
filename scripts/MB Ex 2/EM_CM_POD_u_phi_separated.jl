using Mimosa
using Mimosa.Drivers
using Gridap
using GridapGmsh
using LineSearches: BackTracking
using Gridap.FESpaces
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.Arrays
using CSV
using DataFrames
using SparseMatricesCSR
using LinearAlgebra
using WriteVTK
using JLD2

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
    modmec = Yeoh(C₁ = 0.9*0.0693e6, C₂ = -8.88e2*0.9, C₃ = 0.9*16.7, κ = 0.0693e8)
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
    time_ = []
    t0 = time()
    RES = res(ph,get_fe_basis(fe_spaces.V))
    append!(time_,time()-t0)
    σₖ = get_cell_dof_ids(fe_spaces.U)
    append!(time_,time()-(t0+sum(time_)))
    assem = SparseMatrixAssembler(fe_spaces.U,fe_spaces.V)
    append!(time_,time()-(t0+sum(time_)))
    rs = ([RES[Ω]],[σₖ])
    append!(time_,time()-(t0+sum(time_)))
    b = allocate_vector(assem,rs)
    append!(time_,time()-(t0+sum(time_)))
    assemble_vector!(b,assem,rs)
    append!(time_,time()-(t0+sum(time_)))
    JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    append!(time_,time()-(t0+sum(time_)))
    rs = ([JAC[Ω]],[σₖ],[σₖ])
    append!(time_,time()-(t0+sum(time_)))
    K_T = MultiField.allocate_matrix(assem,rs)
    append!(time_,time()-(t0+sum(time_)))
    assemble_matrix!(K_T,assem,rs)
    append!(time_,time()-(t0+sum(time_)))
    println("res and jac time = $(round.(time_,digits=3))")

    
    # RES = res(ph,get_fe_basis(fe_spaces.V))
    # σₖ = get_cell_dof_ids(fe_spaces.U)
    # assem = SparseMatrixAssembler(fe_spaces.U,fe_spaces.V)
    # rs = ([RES[Ω]],[σₖ])
    # b = allocate_vector(assem,rs)
    # assemble_vector!(b,assem,rs)
    # JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    # rs = ([JAC[Ω]],[σₖ],[σₖ])
    # K_T = MultiField.allocate_matrix(assem,rs)
    # assemble_matrix!(K_T,assem,rs)
    return b, K_T
end

function Gridap.FESpaces.assemble_matrix!(mat,a::SparseMatrixAssembler,matdata,q)
    LinearAlgebra.fillstored!(mat,zero(eltype(mat)))
    assemble_matrix_add!(mat,a,matdata,q)
end

function Gridap.FESpaces.assemble_matrix_add!(mat,a::SparseMatrixAssembler,matdata,q)
    numeric_loop_matrix!(mat,a,matdata,q)
    Gridap.FESpaces.create_from_nz(mat)
end

function Gridap.FESpaces.numeric_loop_matrix!(A,a::SparseMatrixAssembler,matdata,q)
strategy = Gridap.FESpaces.get_assembly_strategy(a)
# println("Check!")
h=0
for (cellmat,_cellidsrows,_cellidscols) in zip(matdata...)
    cellidsrows = Gridap.FESpaces.map_cell_rows(strategy,_cellidsrows)
    cellidscols = Gridap.FESpaces.map_cell_cols(strategy,_cellidscols)
    @assert length(cellidscols) == length(cellidsrows)
    @assert length(cellmat) == length(cellidsrows)
    if length(cellmat) > 0
        rows_cache = array_cache(cellidsrows)
        cols_cache = array_cache(cellidscols)
        vals_cache = array_cache(cellmat)
        mat1 = getindex!(vals_cache,cellmat,1)
        rows1 = getindex!(rows_cache,cellidsrows,1)
        cols1 = getindex!(cols_cache,cellidscols,1)
        # if h==0
        #     println(rows1)
        # end
        add! = AddEntriesMap(+)
        add_cache = return_cache(add!,A,mat1,rows1,cols1)
        caches = add_cache, vals_cache, rows_cache, cols_cache
        Gridap.FESpaces._numeric_loop_matrix!(A,caches,cellmat,cellidsrows,cellidscols,q)
        h+=1
    end
    # println("Count 1 = $h")
end
A
end

@noinline function Gridap.FESpaces._numeric_loop_matrix!(mat,caches,cell_vals,cell_rows,cell_cols,q)
    # println("Check!")
    add_cache, vals_cache, rows_cache, cols_cache = caches
    add! = AddEntriesMap(+)
    h=0
    for cell in 1:length(cell_cols)
        rows = getindex!(rows_cache,cell_rows,cell)
        rows_ = vcat(rows[1],rows[2])
        In_red = maximum([i in rows_ for i in q])
        if In_red
            cols = getindex!(cols_cache,cell_cols,cell)
            vals = getindex!(vals_cache,cell_vals,cell)
            evaluate!(add_cache,add!,mat,vals,rows,cols)
            h+=1
            # println("Count 2 = $h")
        end
        # if h==0
        #     println(rows)
        # end


        # cols = getindex!(cols_cache,cell_cols,cell)
        # vals = getindex!(vals_cache,cell_vals,cell)
        # rows_ = vcat(rows[1],rows[2])
        # In_red = maximum([i in rows_ for i in q])
        # if In_red
        #     evaluate!(add_cache,add!,mat,vals,rows,cols)
        #     h+=1
        #     # println("Count 2 = $h")
        # end
    end
end

function Gridap.FESpaces.allocate_matrix(a::SparseMatrixAssembler,matdata,q)
    m1 = Gridap.FESpaces.nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
    symbolic_loop_matrix!(m1,a,matdata,q)
    m2 = Gridap.FESpaces.nz_allocation(m1)
    symbolic_loop_matrix!(m2,a,matdata,q)
    m3 = Gridap.FESpaces.create_from_nz(m2)
    m3
end

function Gridap.FESpaces.symbolic_loop_matrix!(A,a::SparseMatrixAssembler,matdata,q)
    get_mat(a::Tuple) = a[1]
    get_mat(a) = a
    if Gridap.FESpaces.LoopStyle(A) == Gridap.FESpaces.DoNotLoop()
      return A
    end
    strategy = Gridap.FESpaces.get_assembly_strategy(a)
    for (cellmat,_cellidsrows,_cellidscols) in zip(matdata...)
      cellidsrows = Gridap.FESpaces.map_cell_rows(strategy,_cellidsrows)
      cellidscols = Gridap.FESpaces.map_cell_cols(strategy,_cellidscols)
      @assert length(cellidscols) == length(cellidsrows)
      if length(cellidscols) > 0
        rows_cache = array_cache(cellidsrows)
        cols_cache = array_cache(cellidscols)
        mat1 = get_mat(first(cellmat))
        rows1 = getindex!(rows_cache,cellidsrows,1)
        cols1 = getindex!(cols_cache,cellidscols,1)
        touch! = TouchEntriesMap()
        touch_cache = return_cache(touch!,A,mat1,rows1,cols1)
        caches = touch_cache, rows_cache, cols_cache
        Gridap.FESpaces._symbolic_loop_matrix!(A,caches,cellidsrows,cellidscols,mat1,q)
      end
    end
    A
end

@noinline function Gridap.FESpaces._symbolic_loop_matrix!(A,caches,cell_rows,cell_cols,mat1,q)
    touch_cache, rows_cache, cols_cache = caches
    touch! = TouchEntriesMap()
    for cell in 1:length(cell_cols)
        rows = getindex!(rows_cache,cell_rows,cell)
        rows_ = vcat(rows[1],rows[2])
        In_red = maximum([i in rows_ for i in q])
        h=0
        if In_red
            cols = getindex!(cols_cache,cell_cols,cell)
            evaluate!(touch_cache,touch!,A,mat1,rows,cols)
            h+=1
            # println("Count 2 = $h")
        end
    #   cols = getindex!(cols_cache,cell_cols,cell)
    #   evaluate!(touch_cache,touch!,A,mat1,rows,cols)
    end
end

function Gridap.FESpaces.assemble_vector!(b,a::SparseMatrixAssembler,vecdata,q)
    fill!(b,zero(eltype(b)))
    assemble_vector_add!(b,a,vecdata,q)
end

function Gridap.FESpaces.assemble_vector_add!(b,a::SparseMatrixAssembler,vecdata,q)
    numeric_loop_vector!(b,a,vecdata,q)
    Gridap.FESpaces.create_from_nz(b)
end

function Gridap.FESpaces.numeric_loop_vector!(b,a::SparseMatrixAssembler,vecdata,q)
    strategy = Gridap.FESpaces.get_assembly_strategy(a)
    for (cellvec, _cellids) in zip(vecdata...)
      cellids = Gridap.FESpaces.map_cell_rows(strategy,_cellids)
      if length(cellvec) > 0
        rows_cache = array_cache(cellids)
        vals_cache = array_cache(cellvec)
        vals1 = getindex!(vals_cache,cellvec,1)
        rows1 = getindex!(rows_cache,cellids,1)
        add! = AddEntriesMap(+)
        add_cache = return_cache(add!,b,vals1,rows1)
        caches = add_cache, vals_cache, rows_cache
        Gridap.FESpaces._numeric_loop_vector!(b,caches,cellvec,cellids,q)
      end
    end
    b
end

@noinline function Gridap.FESpaces._numeric_loop_vector!(vec,caches,cell_vals,cell_rows,q)
    add_cache, vals_cache, rows_cache = caches
    @assert length(cell_vals) == length(cell_rows)
    add! = AddEntriesMap(+)
    for cell in 1:length(cell_rows)
      rows = getindex!(rows_cache,cell_rows,cell)
      rows_ = vcat(rows[1],rows[2])
      In_red = maximum([i in rows_ for i in q])
      h=0
      if In_red
        vals = getindex!(vals_cache,cell_vals,cell)
        evaluate!(add_cache,add!,vec,vals,rows)
        h+=1
        # println("Count 2 = $h")
      end
    #   vals = getindex!(vals_cache,cell_vals,cell)
    #   evaluate!(add_cache,add!,vec,vals,rows)
    end
end

function Gridap.FESpaces.allocate_vector(a::SparseMatrixAssembler,vecdata,q)
    v1 = Gridap.FESpaces.nz_counter(get_vector_builder(a),(get_rows(a),))
    symbolic_loop_vector!(v1,a,vecdata,q)
    v2 = Gridap.FESpaces.nz_allocation(v1)
    symbolic_loop_vector!(v2,a,vecdata,q) 
    v3 = Gridap.FESpaces.create_from_nz(v2)
    v3
end

function Gridap.FESpaces.symbolic_loop_vector!(b,a::SparseMatrixAssembler,vecdata,q)
    get_vec(a::Tuple) = a[1]
    get_vec(a) = a
    if Gridap.FESpaces.LoopStyle(b) == Gridap.FESpaces.DoNotLoop()
      return b
    end
    strategy = Gridap.FESpaces.get_assembly_strategy(a)
    for (cellvec,_cellids) in zip(vecdata...)
      cellids = Gridap.FESpaces.map_cell_rows(strategy,_cellids)
      if length(cellids) > 0
        rows_cache = array_cache(cellids)
        vec1 = get_vec(first(cellvec))
        rows1 = getindex!(rows_cache,cellids,1)
        touch! = TouchEntriesMap()
        touch_cache = return_cache(touch!,b,vec1,rows1)
        caches = touch_cache, rows_cache
        _symbolic_loop_vector!(b,caches,cellids,vec1,q) 
      end
    end
    b
end

@noinline function Gridap.FESpaces._symbolic_loop_vector!(A,caches,cellids,vec1,q)
    touch_cache, rows_cache = caches
    touch! = TouchEntriesMap()
    for cell in 1:length(cellids)
      rows = getindex!(rows_cache,cellids,cell)
      rows_ = vcat(rows[1],rows[2])
      In_red = maximum([i in rows_ for i in q])
      h=0
      if In_red
        evaluate!(touch_cache,touch!,A,vec1,rows)
        h+=1
         # println("Count 2 = $h")
      end
    #   evaluate!(touch_cache,touch!,A,vec1,rows)
    end
end

function get_numeric_res_and_jac_red(ph,fe_spaces,Ω,res,jac,q)
    time_ = []
    t0 = time()
    RES = res(ph,get_fe_basis(fe_spaces.V))
    append!(time_,time()-t0)
    σₖ = get_cell_dof_ids(fe_spaces.U)
    append!(time_,time()-(t0+sum(time_)))
    assem = SparseMatrixAssembler(fe_spaces.U,fe_spaces.V)
    append!(time_,time()-(t0+sum(time_)))
    rs = ([RES[Ω]],[σₖ])
    append!(time_,time()-(t0+sum(time_)))
    b = allocate_vector(assem,rs,q)
    append!(time_,time()-(t0+sum(time_)))
    assemble_vector!(b,assem,rs,q)
    append!(time_,time()-(t0+sum(time_)))
    JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    append!(time_,time()-(t0+sum(time_)))
    rs = ([JAC[Ω]],[σₖ],[σₖ])
    append!(time_,time()-(t0+sum(time_)))
    K_T = allocate_matrix(assem,rs,q)
    append!(time_,time()-(t0+sum(time_)))
    assemble_matrix!(K_T,assem,rs,q)
    append!(time_,time()-(t0+sum(time_)))
    println("res and jac time = $(round.(time_,digits=3))")
    
    # RES = res(ph,get_fe_basis(fe_spaces.V))
    # σₖ = get_cell_dof_ids(fe_spaces.U)
    # assem = SparseMatrixAssembler(fe_spaces.U,fe_spaces.V)
    # rs = ([RES[Ω]],[σₖ])
    # b = allocate_vector(assem,rs)
    # assemble_vector!(b,assem,rs)
    # JAC = jac(ph,get_trial_fe_basis(fe_spaces.U),get_fe_basis(fe_spaces.V))
    # rs = ([JAC[Ω]],[σₖ],[σₖ])
    # K_T = MultiField.allocate_matrix(assem,rs)
    # # q=nothing
    # assemble_matrix!(K_T,assem,rs,q)
    return b, K_T
end

function run(
        x0,step,nsteps,𝜑ᵇ,
        model, Ω, dΩ,cache,
        x_list,b_list,R_nl_list,
        K_T_init,K_0_red,
        K_0_red_uu, K_0_red_u𝜑,
        K_0_red_𝜑u, K_0_red_𝜑𝜑,
        m_k,
        𝛟_u, 𝛟_𝜑, 𝛀_u, 
        𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
        Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
        bisect
        )
    
    res, jac = get_symbolic_res_and_jac(dΩ)
    dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps))
    fe_spaces = get_fe_spaces(model,dirichletbc)
    norm_res = 1
    count = 0
    n, m_k = size(𝛟_u)
    prev_step = step-(1/2^bisect)
    K_T_DEIM_ = Matrix{Float64}(undef, 2*m_k, n)
    K_T_DEIM = Matrix{Float64}(undef, 2*m_k, n)
    K_T_DEIM_red_ = Matrix{Float64}(undef, n, 2*m_k)
    K_T_DEIM_red = Matrix{Float64}(undef, 2*m_k, 2*m_k)
    𝛟 = hcat(𝛟_u, 𝛟_𝜑)
    M_DEIM = 𝛟'*𝛀*Zᵀ𝛀_inv
    q = vcat(q_u,q_𝜑)
    K_T_init_rows =  @view K_T_init[q,:]
    x0_copy = copy(x0)
    println("==============================================")
    println("step = $step of $nsteps current potential = $(𝜑ᵇ*(step/nsteps))")
    while norm_res>1e-10
        t0 = time()
        time_ = []
        if count>15 || norm_res>1e2
            bisect += 1
            println("bisect = $bisect")
            x0 = x0_copy
            for i in 1:2^bisect
                x0, cache, _, _, _ = run(x0,prev_step+(i/2^bisect),nsteps,𝜑ᵇ,
                    model, Ω, dΩ,cache,
                    x_list,b_list,R_nl_list,
                    K_T_init,K_0_red,
                    K_0_red_uu, K_0_red_u𝜑,
                    K_0_red_𝜑u, K_0_red_𝜑𝜑,
                    m_k,
                    𝛟_u, 𝛟_𝜑, 𝛀_u, 
                    𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
                    Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
                    bisect
                )
                if prev_step+(i/2^bisect)>step
                    break                    
                end
            end
            count = 0
        end
        append!(time_,time()-t0) #1
        x_u = 𝛟_u*x0[[1:m_k...]]
        x_𝜑 = 𝛟_𝜑*x0[[m_k+1:2*m_k...]]
        x_ = x_u+x_𝜑
        append!(time_,time()-(t0+sum(time_))) #2
        ph = FEFunction(fe_spaces.U, x_)
        b, K_T = get_numeric_res_and_jac_red(ph,fe_spaces,Ω,res,jac,q)
        append!(time_,time()-(t0+sum(time_)))  #3
        K_T = @view K_T[q,:]
        append!(time_,time()-(t0+sum(time_)))  #8
        R_nl = b-K_T_init*x_

        # R_nl_red_u = 𝛟_u'*𝛀_u*Zᵀ𝛀_inv_u*Z_u'*R_nl
        # R_nl_red_𝜑 = 𝛟_𝜑'*𝛀_𝜑*Zᵀ𝛀_inv_𝜑*Z_𝜑'*R_nl
        # R_nl_red = vcat(R_nl_red_u,R_nl_red_𝜑)

        # R_nl_red = 𝛟'*𝛀*Zᵀ𝛀_inv*Z'*R_nl
        append!(time_,time()-(t0+sum(time_)))  #4
        R_nl_red = 𝛟'*𝛀*Zᵀ𝛀_inv*R_nl[q]
        append!(time_,time()-(t0+sum(time_)))  #5

        R = K_0_red*x0 + R_nl_red

        # R = b
        append!(time_,time()-(t0+sum(time_)))  #6
        @show typeof(K_T_init_rows)
        @show typeof(K_T)
        K_T = K_T - K_T_init_rows
        append!(time_,time()-(t0+sum(time_)))  #7
        
        # K_T_DEIM = 𝛀*Zᵀ𝛀_inv*Z'*K_T

        # println("size of K_T = $(size(K_T))")


        # mul!(K_T_DEIM_,Zᵀ𝛀_inv,K_T)
        # mul!(K_T_DEIM,𝛀,K_T_DEIM_)


        # K_T_DEIM = 𝛀*Zᵀ𝛀_inv*K_T

        mul!(K_T_DEIM,M_DEIM,K_T)

        append!(time_,time()-(t0+sum(time_)))  #9
        # K_T_red_uu = K_0_red_uu + 𝛟_u'*K_T_DEIM*𝛟_u
        # K_T_red_u𝜑 = K_0_red_u𝜑 + 𝛟_u'*K_T_DEIM*𝛟_𝜑
        # K_T_red_𝜑u = K_0_red_𝜑u + 𝛟_𝜑'*K_T_DEIM*𝛟_u
        # K_T_red_𝜑𝜑 = K_0_red_𝜑𝜑 + 𝛟_𝜑'*K_T_DEIM*𝛟_𝜑
        # K_T_red = hcat(vcat(K_T_red_uu,K_T_red_𝜑u),vcat(K_T_red_u𝜑,K_T_red_𝜑𝜑))

        # mul!(K_T_DEIM_red_,K_T_DEIM,𝛟)
        # mul!(K_T_DEIM_red,𝛟',K_T_DEIM_red_)
        # K_T_red = K_0_red + K_T_DEIM_red


        # K_T_red = K_0_red + 𝛟'*K_T_DEIM*𝛟


        mul!(K_T_DEIM_red,K_T_DEIM,𝛟)
        K_T_red = K_0_red + K_T_DEIM_red

        # K_T_red = K_T
        append!(time_,time()-(t0+sum(time_)))  #10
        # println("Time DEIM: $(time_[end])")
        
        Δx = K_T_red\(-R)
        append!(time_,time()-(t0+sum(time_)))  #11
        copyto!(x0,x0+Δx)
        # copyto!(x0,x0+𝛟'Δx)
        norm_res = maximum(abs.(R))
        count += 1
        println("iter = $count norm_res = $norm_res norm_Δx = $(norm(Δx)) norm_x0 = $(norm(x0)) size_K_T_red = $(size(K_T_red))")
        append!(time_,time()-(t0+sum(time_)))  #12
        println("Time for iteration = $(sum(time_)) - time per section = $(round.(time_,digits=3))")
    end
    # nlsolver = get_FE_solver()
    # ph = get_initial_guess(fe_spaces,x0)
    # op = FEOperator(res, jac, fe_spaces.U, fe_spaces.V)
    # RES = op.res(ph,get_fe_basis(fe_spaces.V))
    # ph, cache = solve!(ph, nlsolver, op, cache)
    # return get_free_dof_values(ph), cache
    return x0, cache, x_list, b_list, R_nl_list
end

function run_no_time(
    x0,step,nsteps,𝜑ᵇ,
    model, Ω, dΩ,cache,
    x_list,b_list,R_nl_list,
    K_T_init,K_0_red,
    K_0_red_uu, K_0_red_u𝜑,
    K_0_red_𝜑u, K_0_red_𝜑𝜑,
    k, l, m, n_,
    𝛟_u, 𝛟_𝜑, 𝛀_u, 
    𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
    Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
    bisect
    )

res, jac = get_symbolic_res_and_jac(dΩ)
dirichletbc = get_DirichletBC(𝜑ᵇ*(step/nsteps))
fe_spaces = get_fe_spaces(model,dirichletbc)
norm_res = 1
count = 0
n, m_k = size(𝛟_u)
prev_step = step-(1/2^bisect)
# K_T_DEIM_ = Matrix{Float64}(undef, m+k, n)
K_T_DEIM = Matrix{Float64}(undef, m+k, n)
# K_T_DEIM_red_ = Matrix{Float64}(undef, n, m+k)
K_T_DEIM_red = Matrix{Float64}(undef, m+k, m+k)
𝛟 = hcat(𝛟_u, 𝛟_𝜑)
M_DEIM = 𝛟'*𝛀*Zᵀ𝛀_inv
q = vcat(q_u,q_𝜑)
K_T_init_rows =  @view K_T_init[q,:]
x0_copy = copy(x0)
println("==============================================")
println("step = $step of $nsteps current potential = $(𝜑ᵇ*(step/nsteps))")
while norm_res>1e-10
    # t0 = time()
    time_ = []
    if count>15 || norm_res>1e2
        bisect += 1
        println("bisect = $bisect")
        x0 = x0_copy
        for i in 1:2^bisect
            x0, cache, _, _, _ = run_no_time(x0,prev_step+(i/2^bisect),nsteps,𝜑ᵇ,
                model, Ω, dΩ,cache,
                x_list,b_list,R_nl_list,
                K_T_init,K_0_red,
                K_0_red_uu, K_0_red_u𝜑,
                K_0_red_𝜑u, K_0_red_𝜑𝜑,
                k, l, m, n_,
                𝛟_u, 𝛟_𝜑, 𝛀_u, 
                𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
                Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
                bisect
            )
            if prev_step+(i/2^bisect)>step
                break                    
            end
        end
        count = 0
    end
    # append!(time_,time()-t0) #1
    x_u = 𝛟_u*x0[[1:k...]]
    x_𝜑 = 𝛟_𝜑*x0[[k+1:m+k...]]
    x_ = x_u+x_𝜑
    # append!(time_,time()-(t0+sum(time_))) #2
    ph = FEFunction(fe_spaces.U, x_)
    b, K_T = get_numeric_res_and_jac_red(ph,fe_spaces,Ω,res,jac,q)
    # b, K_T = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)

    # append!(time_,time()-(t0+sum(time_)))  #3
    K_T = @view K_T[q,:]
    # append!(time_,time()-(t0+sum(time_)))  #8
    R_nl = b-K_T_init*x_

    # R_nl_red_u = 𝛟_u'*𝛀_u*Zᵀ𝛀_inv_u*Z_u'*R_nl
    # R_nl_red_𝜑 = 𝛟_𝜑'*𝛀_𝜑*Zᵀ𝛀_inv_𝜑*Z_𝜑'*R_nl
    # R_nl_red = vcat(R_nl_red_u,R_nl_red_𝜑)

    # R_nl_red = 𝛟'*𝛀*Zᵀ𝛀_inv*Z'*R_nl
    # append!(time_,time()-(t0+sum(time_)))  #4
    R_nl_red = 𝛟'*𝛀*Zᵀ𝛀_inv*R_nl[q]
    # append!(time_,time()-(t0+sum(time_)))  #5

    R = K_0_red*x0 + R_nl_red

    # R = b
    # append!(time_,time()-(t0+sum(time_)))  #6
    # @show typeof(K_T_init_rows)
    # @show typeof(K_T)
    K_T = K_T - K_T_init_rows
    # append!(time_,time()-(t0+sum(time_)))  #7
    
    # K_T_DEIM = 𝛀*Zᵀ𝛀_inv*Z'*K_T

    # println("size of K_T = $(size(K_T))")


    # mul!(K_T_DEIM_,Zᵀ𝛀_inv,K_T)
    # mul!(K_T_DEIM,𝛀,K_T_DEIM_)


    # K_T_DEIM = 𝛀*Zᵀ𝛀_inv*K_T

    mul!(K_T_DEIM,M_DEIM,K_T)

    # append!(time_,time()-(t0+sum(time_)))  #9
    # K_T_red_uu = K_0_red_uu + 𝛟_u'*K_T_DEIM*𝛟_u
    # K_T_red_u𝜑 = K_0_red_u𝜑 + 𝛟_u'*K_T_DEIM*𝛟_𝜑
    # K_T_red_𝜑u = K_0_red_𝜑u + 𝛟_𝜑'*K_T_DEIM*𝛟_u
    # K_T_red_𝜑𝜑 = K_0_red_𝜑𝜑 + 𝛟_𝜑'*K_T_DEIM*𝛟_𝜑
    # K_T_red = hcat(vcat(K_T_red_uu,K_T_red_𝜑u),vcat(K_T_red_u𝜑,K_T_red_𝜑𝜑))

    # mul!(K_T_DEIM_red_,K_T_DEIM,𝛟)
    # mul!(K_T_DEIM_red,𝛟',K_T_DEIM_red_)
    # K_T_red = K_0_red + K_T_DEIM_red


    # K_T_red = K_0_red + 𝛟'*K_T_DEIM*𝛟


    mul!(K_T_DEIM_red,K_T_DEIM,𝛟)
    K_T_red = K_0_red + K_T_DEIM_red

    # K_T_red = K_T
    # append!(time_,time()-(t0+sum(time_)))  #10
    # println("Time DEIM: $(time_[end])")
    
    Δx = K_T_red\(-R)
    # append!(time_,time()-(t0+sum(time_)))  #11
    copyto!(x0,x0+Δx)
    # copyto!(x0,x0+𝛟'Δx)
    # norm_res = maximum(abs.(R))
    norm_res = maximum(abs.(𝛟*R))

    count += 1
    println("iter = $count norm_res = $norm_res norm_Δx = $(norm(Δx)) argmax_Δx = $(argmax(Δx)) size_K_T_red = $(size(K_T_red))")
    # append!(time_,time()-(t0+sum(time_)))  #12
    # println("Time for iteration = $(sum(time_)) - time per section = $(round.(time_,digits=3))")
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
    DEIM_Matrices = load("scripts/MB Ex 2/CM_DEIM_matrices_u_phi_sep.jld2")
    𝛟_u = DEIM_Matrices["𝛟_u"]
    𝛟_𝜑 = DEIM_Matrices["𝛟_𝜑"]
    # 𝛀_u = DEIM_Matrices["𝛀_u"]
    # 𝛀_𝜑 = DEIM_Matrices["𝛀_𝜑"]
    𝛀_u = nothing
    𝛀_𝜑 = nothing
    𝛀 = DEIM_Matrices["𝛀"]
    # Z_u = DEIM_Matrices["Z_u"]
    # Z_𝜑 = DEIM_Matrices["Z_𝜑"]
    Z_u = nothing
    Z_𝜑 = nothing
    Z = DEIM_Matrices["Z"]
    # Zᵀ𝛀_inv_u = DEIM_Matrices["Zᵀ𝛀_inv_u"]
    # Zᵀ𝛀_inv_𝜑 = DEIM_Matrices["Zᵀ𝛀_inv_𝜑"]
    Zᵀ𝛀_inv_u = nothing
    Zᵀ𝛀_inv_𝜑 = nothing
    Zᵀ𝛀_inv = DEIM_Matrices["Zᵀ𝛀_inv"]
    q_u = DEIM_Matrices["q_u"]
    q_𝜑 = DEIM_Matrices["q_𝜑"]
    n, k = size(𝛟_u)
    l = length(q_u)
    _, m = size(𝛟_𝜑)
    n_ = length(q_𝜑)
    model, Ω, dΩ =  get_trian_and_measure()
    nsteps = 200
    # step = 1
    # x0 = nothing
    dirichletbc = get_DirichletBC(00.0)
    fe_spaces = get_fe_spaces(model,dirichletbc)
    xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
    xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
    x0_ = vcat(xu, xφ)
    println("number of dofs = $(length(x0_))")
    ph = FEFunction(fe_spaces.U, x0_)
    res, jac = get_symbolic_res_and_jac(dΩ)
    _, K_T_init = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
    K_0_red_uu= 𝛟_u'*K_T_init*𝛟_u
    K_0_red_u𝜑= 𝛟_u'*K_T_init*𝛟_𝜑
    K_0_red_𝜑u= 𝛟_𝜑'*K_T_init*𝛟_u
    K_0_red_𝜑𝜑= 𝛟_𝜑'*K_T_init*𝛟_𝜑
    K_0_red = hcat(vcat(K_0_red_uu,K_0_red_𝜑u),vcat(K_0_red_u𝜑,K_0_red_𝜑𝜑))
    x0 = zeros(Float64,m+k)
    # K_T_DEIM_ = Matrix{Float64}(undef, 2*m_k, n)
    # K_T_DEIM = Matrix{Float64}(undef, n, n)
    # K_T_DEIM_red_ = Matrix{Float64}(undef, n, 2*m_k)
    # K_T_DEIM_red = Matrix{Float64}(undef, 2*m_k, 2*m_k)
    x_list = [copy(x0)]
    x_ = zeros(Float64,k+m)
    t0 = time()
    time_step = []
    x_list = []
    b_list = []
    R_nl_list = []
    cache = nothing
    bisect = 0
    𝜑ᵇ = 5000.0
    for step in 1:199#nsteps
        @time x0, cache, x_list, b_list, R_nl_list = run_no_time(
            x0,step,nsteps,𝜑ᵇ,
            model, Ω, dΩ,cache,
            x_list,b_list,R_nl_list,
            K_T_init,K_0_red,
            K_0_red_uu, K_0_red_u𝜑,
            K_0_red_𝜑u, K_0_red_𝜑𝜑,
            k, l, m, n_,
            𝛟_u, 𝛟_𝜑, 𝛀_u, 
            𝛀_𝜑, 𝛀, Z_u, Z_𝜑, Z, Zᵀ𝛀_inv_u,
            Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv, q_u, q_𝜑,
            bisect
            )
        copyto!(x_,x0)
        push!(x_list,copy(x0))
        push!(time_step,time()-t0)
    end
    # ph = FEFunction(fe_spaces.U, x0)
    # b, K_T = get_numeric_res_and_jac(ph,fe_spaces,Ω,res,jac)
    # R_nl = b-K_T_init*x0
    # push!(x_list,copy(x0))
    # push!(b_list,copy(b))
    # push!(R_nl_list,copy(R_nl))
    # return x_list, b_list, R_nl_list
    total_time = time()-t0
    return x_list, time_step, total_time
end

# @time x_list, time_step, total_time = runs();

function Red_Error(x_list)
    DEIM_Matrices = load("scripts/MB Ex 2/CM_DEIM_matrices_u_phi_sep.jld2")
    𝛟_u = DEIM_Matrices["𝛟_u"]
    𝛟_𝜑 = DEIM_Matrices["𝛟_𝜑"]
    𝜑_dofs = 30226
    u_dofs = 152409
    model, Ω, dΩ =  get_trian_and_measure()
    _, m_k = size(𝛟_u)
    file_name = "data/csv/EM_CM_test_2/MaterialModel2/x_.csv"
    _X = CSV.File(file_name) |> Tables.matrix
    println("Norm of 𝛟*x_red-x = $(norm((𝛟_u*x_list[end][[1:m_k...]]+𝛟_𝜑*x_list[end][[1+m_k:2*m_k...]])-_X[:,end])./maximum(abs.(_X[:,end]))))")
    println("Max Abs of 𝛟*x_red-x = $(maximum(abs.((𝛟_u*x_list[end][[1:m_k...]]+𝛟_𝜑*x_list[end][[1+m_k:2*m_k...]]-_X[:,end])./maximum(abs.(_X[:,end])))))")
    mkpath("data/sims/EM_CM_test_2/Red/Result_withError_POD-DEIM_u_phi_sep")
    pvd = paraview_collection("data/sims/EM_CM_test_2/Red/Result_withError_POD-DEIM_u_phi_sep" * "/Results", append=false)
    writevtk(model, "data/sims/EM_CM_test_2/Red/Result_withError_POD-DEIM_u_phi_sep" * "/DiscreteModel")
    dirichletbc = get_DirichletBC(0.0) #lastindex(x_list)))
    fe_spaces = get_fe_spaces(model,dirichletbc)
    U_ = fe_spaces.U
    # _x = _X[:,i]
    # maximum(abs.(x[[1:u_dofs...]]))
    for i in 1:199 #lastindex(x_list)
        dirichletbc = get_DirichletBC(4000.0*(i/100)) #lastindex(x_list)))
        fe_spaces = get_fe_spaces(model,dirichletbc)
        U = fe_spaces.U
        x = 𝛟_u*x_list[i][[1:m_k...]]+𝛟_𝜑*x_list[i][[1+m_k:2*m_k...]]
        uh_red = FEFunction(U,x)
        _x = _X[:,i]
        uh = FEFunction(U,_x)
        Error_u = abs.(_x[[1:u_dofs...]] - x[[1:u_dofs...]])./maximum(abs.(x[[1:u_dofs...]]))
        Error_𝜑 = abs.(_x[[u_dofs+1:u_dofs+𝜑_dofs...]] - x[[u_dofs+1:u_dofs+𝜑_dofs...]])./maximum(abs.(x[[u_dofs+1:u_dofs+𝜑_dofs...]]))
        Error_rel = vcat(Error_u,Error_𝜑)
        error_rel = FEFunction(U_,Error_rel)
        Error = abs.(_x - x)
        error = FEFunction(U_,Error)
        Λ = i/(100-1) #lastindex(x_list)-1)
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        Λ_ = i
        pvd[Λ_] = createvtk(
            Ω,
            "data/sims/EM_CM_test_2/Red/Result_withError_POD-DEIM_u_phi_sep"* "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
            cellfields=["u"=>uh[1], "phi"=>uh[2], "u_red"=>uh_red[1], "phi_red"=>uh_red[2],"error_u"=>error[1], "error_phi"=>error[2], "error_u_rel"=>error_rel[1], "error_phi_rel"=>error_rel[2]]
        )
        print("\r  $Λ_  ")
    end
    vtk_save(pvd)
end