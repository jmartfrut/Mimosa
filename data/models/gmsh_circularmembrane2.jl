using Gmsh: Gmsh, gmsh

function Arc_Sec(R,r,θ_st,θ,ct_pt,nr,ntan,lc)
    geo = gmsh.model.geo
    #print([nl,nw,nt])
    p = []
    append!(p,geo.addPoint(ct_pt[1] + r*cos(θ_st), ct_pt[2] + r*sin(θ_st), ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1] + R*cos(θ_st), ct_pt[2] + R*sin(θ_st), ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1] + R*cos(θ_st + θ), ct_pt[2] + R*sin(θ_st + θ), ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1] + r*cos(θ_st + θ),ct_pt[2] + r*sin(θ_st + θ),ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1],ct_pt[2],ct_pt[3], lc, -1))
    l = []
    append!(l,geo.addLine(p[1],p[2],-1))
    append!(l,geo.addCircleArc(p[2],p[5],p[3],-1))
    append!(l,geo.addLine(p[3],p[4],-1))
    append!(l,geo.addCircleArc(p[4],p[5],p[1],-1))
    loop = geo.addCurveLoop(l,-1)
    rec = geo.addPlaneSurface([loop],-1)
    geo.mesh.setTransfiniteCurve(l[1], nr )
    geo.mesh.setTransfiniteCurve(l[2], ntan )
    geo.mesh.setTransfiniteCurve(l[3], nr )
    geo.mesh.setTransfiniteCurve(l[4], ntan )
    geo.synchronize()
    geo.mesh.setTransfiniteSurface(rec)
    geo.mesh.setRecombine(2,rec)
    geo.synchronize()
    return rec
end

function Center_Sec(R,θ_st,θ,ct_pt,nr,ntan,lc)
    geo = gmsh.model.geo
    #print([nl,nw,nt])
    p = []
    append!(p,geo.addPoint(ct_pt[1],ct_pt[2],ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1] + R*cos(θ_st), ct_pt[2] + R*sin(θ_st), ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1] + R*cos(θ_st + θ), ct_pt[2] + R*sin(θ_st + θ), ct_pt[3], lc, -1))
    l = []
    append!(l,geo.addLine(p[1],p[3],-1))
    append!(l,geo.addCircleArc(p[3],p[1],p[2],-1))
    append!(l,geo.addLine(p[2],p[1],-1))
    loop = geo.addCurveLoop(l,-1)
    rec = geo.addPlaneSurface([loop],-1)
    geo.mesh.setTransfiniteCurve(l[1], ntan )
    geo.mesh.setTransfiniteCurve(l[2], ntan )
    geo.mesh.setTransfiniteCurve(l[3], ntan )
    geo.synchronize()
    geo.mesh.setTransfiniteSurface(rec,"Left",p)
    geo.mesh.setRecombine(2,rec)
    return rec
end

function trans_constraint()
    geo = gmsh.model.geo
    geo.synchronize()
    surfaces = gmsh.model.getEntities(2)
    for surface in surfaces
        # println(surface)
        geo.mesh.setTransfiniteSurface(surface[2])
        geo.mesh.setRecombine(2,surface[2])
    end
end

function extrude_bi(i,T,nt)
    geo = gmsh.model.geo
    geo.synchronize()
    geo.extrude([(2,i)],0.0,0.0,T,[nt],[1],true)
    geo.extrude([(2,i)],0.0,0.0,-T,[nt],[1],true)
    geo.synchronize()
end

function physicalG(L,D)
    surfaces = gmsh.model.getEntities(2)
    i = 1
    j = 1
    k = 1
    l = 1
    bottom_surf = []
    mid_surf = []
    top_surf = []
    for surface in surfaces
        lines = gmsh.model.getBoundary(surface)
        lines_list = []
        point_list = []
        Z_list = []
        for line in lines
            append!(lines_list,abs(line[2]))
            points = gmsh.model.getBoundary(line)
            for point in points
                pt1 = gmsh.model.getValue(point[1],point[2],[])
                if pt1[1] == 0.0 && pt1[2] == 0.0 && pt1[3] == -L
                    gmsh.model.addPhysicalGroup(point[1], [point[2]], -1,"point_xyz")
                    gmsh.model.addPhysicalGroup(point[1]+1, [], -1,"point_xyz")
                    gmsh.model.addPhysicalGroup(point[1]+2, [], -1,"point_xyz")
                elseif pt1[1] == 0.0 && pt1[2] == 0.0 && pt1[3] == L
                    gmsh.model.addPhysicalGroup(point[1], [point[2]], -1,"point_xy")
                    gmsh.model.addPhysicalGroup(point[1]+1, [], -1,"point_xy")
                    gmsh.model.addPhysicalGroup(point[1]+2, [], -1,"point_xy")
                elseif pt1[3] == 0.0 && -1e-3<pt1[1] && pt1[1]<1e-3 &&  (1-1e-3)*(D/2)<pt1[2] && pt1[2]<(D/2)*(1+1e-3)
                    gmsh.model.addPhysicalGroup(point[1], [point[2]], -1,"point_x")
                    gmsh.model.addPhysicalGroup(point[1]+1, [], -1,"point_x")
                    gmsh.model.addPhysicalGroup(point[1]+2, [], -1,"point_x")
                end 
                append!(point_list,point[2])
                push!(Z_list,pt1[3])
            end
        end
        Z_avg = sum(Z_list)/length(Z_list)
        println("Look! = $((1-1e-3)*L*(-1))  or $(L*(1+1e-3)*(-1))")
        println("Look! = $((1-1e-3)*L*(-1)>Z_avg)  or $(Z_avg<L*(1+1e-3)*(-1))")
        println("Z_avg = $Z_avg")
        if Z_avg==0.0
            gmsh.model.addPhysicalGroup(0, point_list, -1,"mid_surf_$i")  
            gmsh.model.addPhysicalGroup(1, lines_list, -1, "mid_surf_$i")  
            gmsh.model.addPhysicalGroup(2, [surface[2]], -1, "mid_surf_$i")
            push!(bottom_surf,"mid_surf_$i")
            i = i + 1
        elseif (1-1e-3)*L<Z_avg && Z_avg<L*(1+1e-3)
            gmsh.model.addPhysicalGroup(0, point_list, -1,"top_surf_$j")
            gmsh.model.addPhysicalGroup(1, lines_list, -1, "top_surf_$j")  
            gmsh.model.addPhysicalGroup(2, [surface[2]], -1, "top_surf_$j")
            push!(mid_surf,"top_surf_$j")
            j = j + 1
        elseif (1-1e-3)*L*(-1)>Z_avg && Z_avg>L*(1+1e-3)*(-1)
            gmsh.model.addPhysicalGroup(0, point_list, -1,"bottom_surf_$k")  
            gmsh.model.addPhysicalGroup(1, lines_list, -1, "bottom_surf_$k")  
            gmsh.model.addPhysicalGroup(2, [surface[2]], -1, "bottom_surf_$k")
            push!(top_surf,"bottom_surf_$k")
            k = k + 1
        else
            gmsh.model.addPhysicalGroup(0, point_list, -1,"surf_$l")  
            gmsh.model.addPhysicalGroup(1, lines_list, -1, "surf_$l")  
            gmsh.model.addPhysicalGroup(2, [surface[2]], -1, "surf_$l")
            l = l + 1
        end
        println(k)
    end
    
    vol = gmsh.model.getEntities(3)

    vol_list = []
    for v in vol
        append!(vol_list,v[2])
    end
    gmsh.model.addPhysicalGroup(3, vol_list, -1, "Volume_")
    return     bottom_surf, mid_surf, top_surf
end

function run()
    gmsh.initialize()
    # ct_pt,nt,nr,ntan,lc = [0.0,0.0,0.0],1,5,7,2.0e-4 # "CircularMembrane3"
    # ct_pt,nt,nr,ntan,lc = [0.0,0.0,0.0],1,10,21,2.0e-4 # "CircularMembrane4"
    ct_pt,nt,nr,ntan,lc = [0.0,0.0,0.0],1,5,21,2.0e-4 # "CircularMembrane5"
    T = 0.4e-3
    surf_list = []
    n_sec_t = 4
    n_sec_r = 4
    Δθ = (2*pi)/n_sec_t
    R_ = 50.0e-3
    ΔR = R_/n_sec_r
    r = 0.0
    R = ΔR
    for i in 1:n_sec_r
        θ_st = 0.0
        for j in 1:n_sec_t
            if i == 1
                push!(surf_list,Center_Sec(R,θ_st,Δθ,ct_pt,nr,ntan,lc))
            else
                push!(surf_list,Arc_Sec(R,r,θ_st,Δθ,ct_pt,nr,ntan,lc))
            end
            θ_st += Δθ
        end
        r = R
        R += ΔR
    end
    # trans_constraint()
    for i in surf_list
        extrude_bi(i,T,nt)
    end 
    bottom_surf, mid_surf, top_surf = physicalG(T,R_*2)
    # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
    
    gmsh.model.mesh.generate(2)
    # gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.model.mesh.recombine()
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.set_order(2)
    if !("-nopopup" in ARGS)
        gmsh.fltk.run()
    end
    model_name = "CircularMembrane5"
    output_file = joinpath(dirname(@__FILE__), model_name*".msh")
    gmsh.write(output_file)
    Gmsh.finalize() 
end