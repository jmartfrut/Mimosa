
using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using GridapGmsh

model = GmshDiscreteModel("data/models/"*"CircularMembrane5.msh") # GmshDiscreteModel("data/models/PlateBeame10S_BC.msh") # CartesianDiscreteModel((0,1,0,1),(4,4))

D = num_cell_dims(model)
order = 2
reffe = LagrangianRefFE(VectorValue{D,Float64},HEX,order)
V = FESpace(model,reffe)

cell_dof_ids = get_cell_dof_ids(V)

trian = get_triangulation(V)
basis = get_dof_basis(reffe)
cmaps = collect1d(get_cell_map(trian))
cell_nodes = map(m -> evaluate(m,basis.nodes)[basis.dof_to_node], cmaps)

topo = get_grid_topology(model)
d_to_cell_to_dface = [Geometry.get_faces(topo,D,d) for d in 0:D]
d_to_dface_to_pindex = [get_cell_permutations(topo,d) for d in 0:D]
cell_conformity = V.metadata

cell_to_nodes = Vector{Vector{Int}}(undef,num_cells(topo))
d_to_dface_to_nodes = [ Vector{Vector{Int}}(undef,num_faces(topo,d)) for d in 0:D]
node_to_dofs = Vector{Int}[]

n_nodes = 0
for cell in 1:num_cells(topo)
    dofs = cell_dof_ids[cell]
    ctype = cell_conformity.cell_ctype[cell]
    cell_to_nodes[cell] = Int[]
    for d in 0:D
        ldface_to_ldof = cell_conformity.d_ctype_ldface_own_ldofs[d+1][ctype]
        dfaces = d_to_cell_to_dface[d+1][cell]
        for (ldface,dface) in enumerate(dfaces)
            pindex = d_to_dface_to_pindex[d+1][cell][ldface]
            perm = cell_conformity.ctype_lface_pindex_pdofs[ctype][ldface + cell_conformity.d_ctype_offset[d+1][ctype]][pindex]
            ldof = ldface_to_ldof[ldface][perm]
            lnodes = unique(basis.dof_to_node[ldof])
            if !isassigned(d_to_dface_to_nodes[d+1],dface)
                n_lnodes = length(lnodes)
                d_to_dface_to_nodes[d+1][dface] = (n_nodes+1):(n_nodes+n_lnodes)
                append!(node_to_dofs,[Int[] for n in 1:n_lnodes])
                n_nodes += n_lnodes
            
                nodes = d_to_dface_to_nodes[d+1][dface]
                for (lnode,node) in zip(lnodes,nodes)
                    ids = findall(n -> n == lnode, basis.dof_to_node)
                    append!(node_to_dofs[node],dofs[ids])
                end
                append!(cell_to_nodes[cell],nodes)
            end
        end
    end
end

findall(x -> any(!isone,x), get_cell_permutations(topo))

