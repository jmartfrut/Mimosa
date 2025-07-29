using Gmsh: Gmsh, gmsh

gmsh.initialize()
geo = gmsh.model.geo
lc = 1.0

function Rec_1_2(corner_point,lenght_,width)
    L = lenght_
    w = width
    p = []
    append!(p,geo.addPoint(corner_point[1], corner_point[2], corner_point[3], lc, -1))
    append!(p,geo.addPoint(corner_point[1]+L, corner_point[2], corner_point[3], lc, -1))
    append!(p,geo.addPoint(corner_point[1]+L, corner_point[2]+w, corner_point[3], lc, -1))
    append!(p,geo.addPoint(corner_point[1],corner_point[2]+w,corner_point[3], lc, -1))
    l = []
    for i in 1:3
        append!(l,geo.addLine(p[i],p[i+1],-1))
    end
    append!(l,geo.addLine(p[4],p[1],-1))
    rec_loop1 = geo.addCurveLoop(l,-1)
    p = []
    append!(p,geo.addPoint(corner_point[1], corner_point[2], corner_point[3], lc, -1))
    append!(p,geo.addPoint(corner_point[1]+L, corner_point[2], corner_point[3], lc, -1))
    append!(p,geo.addPoint(corner_point[1]+L, corner_point[2]-w, corner_point[3], lc, -1))
    append!(p,geo.addPoint(corner_point[1],corner_point[2]-w,corner_point[3], lc, -1))
    l = []
    for i in 1:3
        append!(l,geo.addLine(p[i],p[i+1],-1))
    end
    append!(l,geo.addLine(p[4],p[1],-1))
    rec_loop2 = geo.addCurveLoop(l,-1)
    geo.synchronize()
    return rec_loop1,rec_loop2
end

function Rec_0(corner_point,lenght_,width)
    L = lenght_
    w = width
    p = []
    append!(p,geo.addPoint(corner_point[1], corner_point[2], corner_point[3], lc, -1))
    append!(p,geo.addPoint(corner_point[1]+w, corner_point[2], corner_point[3], lc, -1))
    append!(p,geo.addPoint(corner_point[1]+w, corner_point[2]+L, corner_point[3], lc, -1))
    append!(p,geo.addPoint(corner_point[1],corner_point[2]+L,corner_point[3], lc, -1))
    l = []
    for i in 1:3
        append!(l,geo.addLine(p[i],p[i+1],-1))
    end
    append!(l,geo.addLine(p[4],p[1],-1))
    rec_loop1 = geo.addCurveLoop(l,-1)
    geo.synchronize()
    return rec_loop1
end

lenght_,width = 0.1,0.0005
corner_point_list_1_2 = []
separation_1_2 = 0.01
sections_1 = 5
total_width = (sections_1-1)*separation_1_2+2*width
for i in 1:sections_1
    push!(corner_point_list_1_2,[0,separation_1_2*(i-1),0])  
end
corner_point_list_0 = [[0-width,0-width,0],[0+lenght_,0-width,0]]
for corner_point in corner_point_list_1_2
    Rec_1_2(corner_point,lenght_,width)
end
Rec_0(corner_point_list_0[1],total_width,width)
Rec_0(corner_point_list_0[2],total_width,width)

corner_point_list_3 = []
sections_2 = 5
separation_3 = 0.02
for i in 1:sections_1
    for j in 1:sections_2
        if iseven(i)
            push!(corner_point_list_3,[separation_3*(j-1),separation_1_2*(i-1)+width,0])
        else
            if i<sections_1
                push!(corner_point_list_3,[separation_3*(j)-0.5*separation_3,separation_1_2*(i-1)+width,0])
            end
        end
    end
end

for corner_point in corner_point_list_3
    Rec_0(corner_point,separation_1_2-2*width,width)
end



if !("-nopopup" in ARGS)
    gmsh.fltk.run()
end
Gmsh.finalize()