using Gmsh: Gmsh, gmsh

# function generateCube(corner_point,l,n,lc)
corner_point,l,n,lc = [0,0,0],2e-4,10,1.0
gmsh.initialize()
geo = gmsh.model.geo
L = l/2
corner_point_list = [[corner_point[1], corner_point[2], corner_point[3]],
    [corner_point[1]+L, corner_point[2], corner_point[3]],
    [corner_point[1]+L, corner_point[2]+L, corner_point[3]],
    [corner_point[1],corner_point[2]+L,corner_point[3]]]
rec_list = []
for j in 1:4
    p = []
    corner_point = corner_point_list[j]
    append!(p,geo.addPoint(corner_point[1], corner_point[2], corner_point[3], lc, -1))
    append!(p,geo.addPoint(corner_point[1]+L, corner_point[2], corner_point[3], lc, -1))
    append!(p,geo.addPoint(corner_point[1]+L, corner_point[2]+L, corner_point[3], lc, -1))
    append!(p,geo.addPoint(corner_point[1],corner_point[2]+L,corner_point[3], lc, -1))
    l = []
    for i in 1:3
        append!(l,geo.addLine(p[i],p[i+1],-1))
    end
    append!(l,geo.addLine(p[4],p[1],-1))
    rec_loop = geo.addCurveLoop(l,-1)
    rec = geo.addPlaneSurface([rec_loop],-1)
    geo.synchronize()
    lines = gmsh.model.getBoundary([2,rec])
    for line in lines
        geo.mesh.setTransfiniteCurve(line[2], (n/2)+1 )
    end
    surfaces = gmsh.model.getEntities(2)
    for surface in surfaces
        geo.mesh.setTransfiniteSurface(surface[2])
        geo.mesh.setRecombine(2,surface[2])
    end
    push!(rec_list,(2,rec))
end
prism1 = geo.extrude(rec_list,0.0,0.0,l,[n],[1],true)
geo.synchronize()
global ct = 1
for surf in prism1
    point_list = []
    line_list = []
    lines = gmsh.model.getBoundary(surf)
    for line in lines
        append!(line_list,abs(line[2]))
        points = gmsh.model.getBoundary(line)
        for point in points
            append!(point_list,abs(point[2]))
        end
    end
    # gmsh.model.addPhysicalGroup(0, point_list, -1,"surf_$ct")  
    # gmsh.model.addPhysicalGroup(1, line_list, -1, "surf_$ct")  
    gmsh.model.addPhysicalGroup(2, [surf[2]], -1, "surf_$ct")
    global ct += 1
end
vol = gmsh.model.getEntities(3)
vol_list = []
    for v in vol
        append!(vol_list,v[2])
    end
    gmsh.model.addPhysicalGroup(3, vol_list, -1, "Volume_")
geo.synchronize()
gmsh.model.mesh.generate(3)
if !("-nopopup" in ARGS)
    gmsh.fltk.run()
end
model_name = "EMCube_Coarse"
output_file = joinpath(dirname(@__FILE__), model_name*".msh")
gmsh.write(output_file)
Gmsh.finalize()
# end