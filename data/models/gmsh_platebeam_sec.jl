using Gmsh: Gmsh, gmsh
gmsh.initialize()

function generateBeamSec(L,W,T,st_pt,nl,nw,nt,lc,n_sec)
    geo = gmsh.model.geo
    #print([nl,nw,nt])
    p = []
    append!(p,geo.addPoint(st_pt[1], st_pt[2], st_pt[3], lc, -1))
    append!(p,geo.addPoint(st_pt[1]+L, st_pt[2], st_pt[3], lc, -1))
    append!(p,geo.addPoint(st_pt[1]+L, st_pt[2]+W, st_pt[3], lc, -1))
    append!(p,geo.addPoint(st_pt[1],st_pt[2]+W,st_pt[3], lc, -1))
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
        pts = gmsh.model.getBoundary(line)
        pt1 = gmsh.model.getValue(pts[1][1],pts[1][2],[])
        pt2 = gmsh.model.getValue(pts[2][1],pts[2][2],[])
        pt_dif = pt2-pt1
        if pt_dif[1]==0.0 && pt_dif[2]==0.0
            geo.mesh.setTransfiniteCurve(line[2], nt )
            C1 = 1
        elseif pt_dif[2]==0.0 && pt_dif[3]==0.0
            geo.mesh.setTransfiniteCurve(line[2], nl )
            C2=2
        else
            geo.mesh.setTransfiniteCurve(line[2], nw )
            c3=3
        end
    end
    surfaces = gmsh.model.getEntities(2)
    for surface in surfaces
        geo.mesh.setTransfiniteSurface(surface[2])
        geo.mesh.setRecombine(2,surface[2])
    end


    prism1 = geo.extrude([(2,rec)],0.0,0.0,T,[nt-1],[1],true)
    prism2 = geo.extrude(prism1[1],0.0,0.0,T,[nt-1],[1],true)
    geo.synchronize()

    lines = gmsh.model.getBoundary(prism1[1])
    lines_list = []
    point_list = []
    for line in lines
        append!(lines_list,abs(line[2]))
        println(line)
        points = gmsh.model.getBoundary(line)
        println(points)
        pt1 = gmsh.model.getValue(points[1][1],points[1][2],[])
        pt2 = gmsh.model.getValue(points[2][1],points[2][2],[])
        for point in points
            append!(point_list,point[2])
            if pt1[1]-pt2[1] == 0.0 && pt1[3]-pt2[3] == 0.0 && pt1[1] == 0.0
                pt3 = gmsh.model.getValue(point[1],point[2],[])
                if pt3[2] == 0.0
                    gmsh.model.addPhysicalGroup(point[1], [point[2]], -1,"point_zy")
                    gmsh.model.addPhysicalGroup(point[1]+1, [], -1,"point_zy")
                    gmsh.model.addPhysicalGroup(point[1]+2, [], -1,"point_zy")
                else
                    gmsh.model.addPhysicalGroup(point[1], [point[2]], -1,"point_z")
                    gmsh.model.addPhysicalGroup(point[1]+1, [], -1,"point_z")
                    gmsh.model.addPhysicalGroup(point[1]+1, [], -1,"point_z")
                end 
            end
        end
    end
    gmsh.model.addPhysicalGroup(0, point_list, -1,"midsurf_$n_sec")  
    gmsh.model.addPhysicalGroup(1, lines_list, -1, "midsurf_$n_sec")  
    gmsh.model.addPhysicalGroup(2, [prism1[1][2]], -1, "midsurf_$n_sec")  

    lines = gmsh.model.getBoundary(prism2[1])
    lines_list = []
    point_list = []
    for line in lines
        append!(lines_list,abs(line[2]))
        points = gmsh.model.getBoundary(line)
        for point in points
            append!(point_list,point[2])
        end
    end
    gmsh.model.addPhysicalGroup(0, point_list, -1,"topsurf_$n_sec")  
    gmsh.model.addPhysicalGroup(1, lines_list, -1,"topsurf_$n_sec")
    gmsh.model.addPhysicalGroup(2, [prism2[1][2]], -1,"topsurf_$n_sec")

    lines = gmsh.model.getBoundary([2,rec])
    lines_list = []
    point_list = []
    for line in lines
        append!(lines_list,abs(line[2]))
        points = gmsh.model.getBoundary(line)
        for point in points
            append!(point_list,point[2])
        end
    end
    gmsh.model.addPhysicalGroup(0, point_list,  -1,"bottomsurf_$n_sec")  
    gmsh.model.addPhysicalGroup(1, lines_list, -1,"bottomsurf_$n_sec")  
    gmsh.model.addPhysicalGroup(2, [2,rec], -1,"bottomsurf_$n_sec")

    lines = gmsh.model.getBoundary(prism1[6])
    append!(lines,gmsh.model.getBoundary(prism2[6]))
    lines_list = []
    point_list = []
    for line in lines
        append!(lines_list,abs(line[2]))
        points = gmsh.model.getBoundary(line)
        for point in points
            append!(point_list,point[2])
        end
    end
    gmsh.model.addPhysicalGroup(0, point_list,  -1, "fixedup_$n_sec")  
    gmsh.model.addPhysicalGroup(1, lines_list, -1, "fixedup_$n_sec")  
    gmsh.model.addPhysicalGroup(2, [prism1[6][2], prism2[6][2]], -1, "fixedup_$n_sec") 

    gmsh.model.addPhysicalGroup(3, [prism1[2][2],prism2[2][2]], n_sec, "Volume_$n_sec") 
end

# parameters
L=100e-3;      # beam length
W=8e-3;       # beam width
T=0.4e-3;     # beam thickness


# const nl=40; # X element size
# const nw=10; # Y element size
# const nt=4; # Z element size


nl=9; # X element size: 20 for 4s and 8 for 10S
nw=8; # Y element size
nt=2; # Z element size

# fract = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
fract = [1.0]
# fract = [0.25,0.25,0.25]

model_name = "PlateBeame1S_BC"

lc = 1.0; # characteristic length for meshing
function generateBeam(L,W,T,nl,nw,nt,lc,fract,model_name)
    # append!(fract,1 - sum(fract)) #used to add a final section
    gmsh.model.add(model_name)
    n_sec = length(fract)
    st_pt = [0 0 0]
    for i in 1:n_sec
        nsec = i
        L_sec = L*fract[i]
        #nl = ceil(L_sec/2)
        # nw = ceil(W/nw)
        println("Block $i")
        generateBeamSec(L_sec,W,T,st_pt,nl,nw,nt,lc,nsec)
        st_pt = [st_pt[1]+L_sec 0 0]
    end
    gmsh.model.mesh.generate(3)
    output_file = joinpath(dirname(@__FILE__), model_name*".msh")
    gmsh.write(output_file)
    if !("-nopopup" in ARGS)
        gmsh.fltk.run()
    end
    Gmsh.finalize()
end

generateBeam(L,W,T,nl,nw,nt,lc,fract,model_name)