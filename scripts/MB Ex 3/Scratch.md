# Attempt (Failed) to code a Gridap assembler that incorporated the conditions to apply ECM 

```julia
##

#region ECM Gridap Functions

function Gridap.FESpaces.assemble_matrix!(mat,a::SparseMatrixAssembler,matdata,z,w)
    LinearAlgebra.fillstored!(mat,zero(eltype(mat)))
    assemble_matrix_add!(mat,a,matdata,z,w)
end

function Gridap.FESpaces.assemble_matrix_add!(mat,a::SparseMatrixAssembler,matdata,z,w)
    numeric_loop_matrix!(mat,a,matdata,z,w)
    Gridap.FESpaces.create_from_nz(mat)
end

function Gridap.FESpaces.numeric_loop_matrix!(A,a::SparseMatrixAssembler,matdata,z,w)
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
            Gridap.FESpaces._numeric_loop_matrix!(A,caches,cellmat,cellidsrows,cellidscols,z,w)
            h+=1
        end
        # println("Count 1 = $h")
    end
    A
end

@noinline function Gridap.FESpaces._numeric_loop_matrix!(mat,caches,cell_vals,cell_rows,cell_cols,z,w)
    cells = zip(z,w)
    add_cache, vals_cache, rows_cache, cols_cache = caches
    add! = AddEntriesMap(+)
    for cell in cells
    rows = getindex!(rows_cache,cell_rows,cell[1])
    cols = getindex!(cols_cache,cell_cols,cell[1])
    vals = getindex!(vals_cache,cell_vals,cell[1])
    evaluate!(add_cache,add!,mat,cell[2]*vals,rows,cols)
    end
end

function Gridap.FESpaces.allocate_matrix(a::SparseMatrixAssembler,matdata,z,w)
    m1 = Gridap.FESpaces.nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
    symbolic_loop_matrix!(m1,a,matdata,z,w)
    m2 = Gridap.FESpaces.nz_allocation(m1)
    symbolic_loop_matrix!(m2,a,matdata,z,w)
    m3 = Gridap.FESpaces.create_from_nz(m2)
    m3
end

function Gridap.FESpaces.symbolic_loop_matrix!(A,a::SparseMatrixAssembler,matdata,z,w)
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
        Gridap.FESpaces._symbolic_loop_matrix!(A,caches,cellidsrows,cellidscols,mat1,z,w)
      end
    end
    A
end

@noinline function Gridap.FESpaces._symbolic_loop_matrix!(
  A, caches, cell_rows, cell_cols, mat1, z, w
)
    cells = zip(z,w)
    touch_cache, rows_cache, cols_cache = caches
    touch! = TouchEntriesMap()
    for cell in cells
    rows = getindex!(rows_cache,cell_rows,cell[1])
    cols = getindex!(cols_cache,cell_cols,cell[1])
    evaluate!(touch_cache,touch!,A,mat1,rows,cols)
    end
end

function Gridap.FESpaces.assemble_vector!(b,a::SparseMatrixAssembler,vecdata,z,w)
    fill!(b,zero(eltype(b)))
    assemble_vector_add!(b,a,vecdata,z,w)
end

function Gridap.FESpaces.assemble_vector_add!(b,a::SparseMatrixAssembler,vecdata,z,w)
    numeric_loop_vector!(b,a,vecdata,z,w)
    Gridap.FESpaces.create_from_nz(b)
end

function Gridap.FESpaces.numeric_loop_vector!(b,a::SparseMatrixAssembler,vecdata,z,w)
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
        Gridap.FESpaces._numeric_loop_vector!(b,caches,cellvec,cellids,z,w)
      end
    end
    b
end

@noinline function Gridap.FESpaces._numeric_loop_vector!(
  vec, caches, cell_vals, cell_rows, z, w
)
    cells = zip(z,w)
    add_cache, vals_cache, rows_cache = caches
    @assert length(cell_vals) == length(cell_rows)
    add! = AddEntriesMap(+)
    for cell in cells
    rows = getindex!(rows_cache,cell_rows,cell[1])
    vals = getindex!(vals_cache,cell_vals,cell[1])
    evaluate!(add_cache,add!,vec,cell[2]*vals,rows)
    end
end


function Gridap.FESpaces.allocate_vector(a::SparseMatrixAssembler,vecdata,z,w)
    v1 = Gridap.FESpaces.nz_counter(get_vector_builder(a),(get_rows(a),))
    symbolic_loop_vector!(v1,a,vecdata,z,w)
    v2 = Gridap.FESpaces.nz_allocation(v1)
    symbolic_loop_vector!(v2,a,vecdata,z,w) 
    v3 = Gridap.FESpaces.create_from_nz(v2)
    v3
end

function Gridap.FESpaces.symbolic_loop_vector!(b,a::SparseMatrixAssembler,vecdata,z,w)
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
        Gridap.FESpaces._symbolic_loop_vector!(b,caches,cellids,vec1,z,w)
      end
    end
    b
end


@noinline function Gridap.FESpaces._symbolic_loop_vector!(
  A, caches, cell_rows, vec1, z, w
  )
  cells = zip(z,w)
  touch_cache, rows_cache = caches
  touch! = TouchEntriesMap()
  for cell in cells
    rows = getindex!(rows_cache,cell_rows,cell[1])
    evaluate!(touch_cache,touch!,A,vec1,rows)
  end
end

#endregion

##
```
