using Gmsh: Gmsh, gmsh

function generateBeamSec(D,d,L,n_sec_t,ct_pt,nl,nw,nt,ntan,lc,n_sec,model_name)
    R = D/2
    r = d/2
    geo = gmsh.model.geo
    #print([nl,nw,nt])
    p = []
    append!(p,geo.addPoint(ct_pt[1], ct_pt[2]+r, ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1], ct_pt[2]+R, ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1]+L, ct_pt[2]+R, ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1]+L,ct_pt[2]+r,ct_pt[3], lc, -1))
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

    rec = (0,rec)
    for i in 1:n_sec_t
        prism1 = geo.revolve(([2,rec[2]]), ct_pt[1], ct_pt[2], ct_pt[3], 1, 0, 0, pi/(n_sec_t/2), [ntan], [], true)
        rec = prism1[1]
    end
    
    geo.synchronize()
    # gmsh.model.mesh.generate(3)

    # output_file = joinpath(dirname(@__FILE__), model_name*".msh")
    # gmsh.write(output_file)
    # if !("-nopopup" in ARGS)
    #     gmsh.fltk.run()
    # end
    # Gmsh.finalize()
end

function physicalG(D,d)
    surfaces = gmsh.model.getEntities(2)
    i = 1
    j = 1
    k = 1
    l = 1
    int_surf = []
    ext_surf = []
    fixed_surf = []
    for surface in surfaces
        lines = gmsh.model.getBoundary(surface)
        lines_list = []
        point_list = []
        R_list = []
        X_list = []
        for line in lines
            append!(lines_list,abs(line[2]))
            points = gmsh.model.getBoundary(line)
            for point in points
                pt1 = gmsh.model.getValue(point[1],point[2],[])
                R = sqrt(pt1[2]^2 + pt1[3]^2)*1000
                push!(R_list,R)
                push!(X_list,pt1[1])
                append!(point_list,point[2])
            end
        end
        if R_list==[d/2,d/2,d/2,d/2,d/2,d/2,d/2,d/2]*1000
            gmsh.model.addPhysicalGroup(0, point_list, -1,"int_surf_$i")  
            gmsh.model.addPhysicalGroup(1, lines_list, -1, "int_surf_$i")  
            gmsh.model.addPhysicalGroup(2, [surface[2]], -1, "int_surf_$i")
            push!(int_surf,"int_surf_$i")
            i = i + 1
        elseif R_list==[D/2,D/2,D/2,D/2,D/2,D/2,D/2,D/2]*1000
            gmsh.model.addPhysicalGroup(0, point_list, -1,"ext_surf_$j")
            gmsh.model.addPhysicalGroup(1, lines_list, -1, "ext_surf_$j")  
            gmsh.model.addPhysicalGroup(2, [surface[2]], -1, "ext_surf_$j")
            push!(ext_surf,"ext_surf_$j")
            j = j + 1
        elseif X_list==[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            gmsh.model.addPhysicalGroup(0, point_list, -1,"fixed_surf_$k")  
            gmsh.model.addPhysicalGroup(1, lines_list, -1, "fixed_surf_$k")  
            gmsh.model.addPhysicalGroup(2, [surface[2]], -1, "fixed_surf_$k")
            push!(fixed_surf,"fixed_surf_$k")
            k = k + 1
        else
            gmsh.model.addPhysicalGroup(0, point_list, -1,"surf_$l")  
            gmsh.model.addPhysicalGroup(1, lines_list, -1, "surf_$l")  
            gmsh.model.addPhysicalGroup(2, [surface[2]], -1, "surf_$l")
            l = l + 1
        end
    end
    
    vol = gmsh.model.getEntities(3)

    vol_list = []
    for v in vol
        append!(vol_list,v[2])
    end
    gmsh.model.addPhysicalGroup(3, vol_list, -1, "Volume_")
    return int_surf, ext_surf, fixed_surf
end

# parameters
function main()
    gmsh.initialize()

    L=100e-3;      # beam length
    D=1.5e-3;       # beam width
    d=1e-3;     # beam thickness
    ct_pt = [0.0,0.0,0.0]
    n_sec_t = 4
    # const nl=40; # X element size
    # const nw=10; # Y element size
    # const nt=4; # Z element size
    
    
    nl=8; # X element size
    nw=3; # Y element size
    nt=2; # Z element size - not relevant in tube model
    ntan = 3
    
    lc = 2.0e-3; # characteristic length for meshing
    
    L_conf = [0.25,0.25,0.25,0.25]
    L_ = length(L_conf)
    model_name = "TubeBeam_V2_SecT_$n_sec_t-SecL_$L_"
    
    global Sec = 1
    for i in L_conf
        L_ = L*i
        generateBeamSec(D,d,L_,n_sec_t,ct_pt,nl,nw,nt,ntan,lc,Sec,model_name)
        ct_pt[1] = ct_pt[1] + L_
        global Sec += 1
        # if !("-nopopup" in ARGS)
        #     gmsh.fltk.run()
        # end
    end
    int_surf_list, ext_surf_list, fixed_surf_list = physicalG(D,d)
    gmsh.model.mesh.generate(3)
    println(int_surf_list)
    println(ext_surf_list)
    println(fixed_surf_list)  
    if !("-nopopup" in ARGS)
        gmsh.fltk.run()
    end
    output_file = joinpath(dirname(@__FILE__), model_name*".msh")
    gmsh.write(output_file)
    Gmsh.finalize() 
end