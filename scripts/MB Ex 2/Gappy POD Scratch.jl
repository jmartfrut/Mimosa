X = fe_spaces.Uu.space.cell_dofs_ids

X_ = []
for x in X
    push!(X_,[[x[i],x[i+27],x[i+54]] for i in 1:Int32(length(x)/3)])
end

X = reduce(vcat,X_)
X = reduce(vcat,X)
Points = []
push!(Points,unique(X))

X = fe_spaces.Uφ.space.cell_dofs_ids

X_ = []
for x in X
    push!(X_,[[x[i]] for i in 1:Int32(length(x))])
end
 
X = reduce(vcat,X_)


push!(Points,unique(X))

dirichlet_list = Vector{Bool}()
for i in 1:lastindex(Points[1])
    if Points[1][i][1]<0 || Points[1][i][2]<0 ||Points[1][i][3]<0 ||Points[2][i][1]<0
        push!(dirichlet_list,true)
    else
        push!(dirichlet_list,false)
    end
end
deleteat!(Points[1],dirichlet_list)
deleteat!(Points[2],dirichlet_list)

Points[2] = [i.+36147 for i in Points[2]]



fe_spaces.Uu.space.fe_basis.trian.grid.reffes[1].reffe.dofs.nodes


function POD_Gappy_Matrices_split(k,l,m,n,p,U_x_u,U_x_𝜑,U_R_nl_u,U_R_nl_𝜑)
    # Get nodes to dofs mapping
    model, Ω, dΩ = get_trian_and_measure()
    dirichletbc = get_DirichletBC(0.0)
    fe_spaces = get_fe_spaces(model,dirichletbc)    
    
    X_ = []
    for x in X
        push!(X_,[[x[i],x[i+27],x[i+54]] for i in 1:Int32(length(x)/3)])
    end
    X = reduce(vcat,X_)
    Points = []
    push!(Points,unique(X))
    X = fe_spaces.Uφ.space.cell_dofs_ids
    X_ = []
    for x in X
        push!(X_,[[x[i]] for i in 1:Int32(length(x))])
    end
    X = reduce(vcat,X_)
    push!(Points,unique(X))
    dirichlet_list = Vector{Bool}()
    for i in 1:lastindex(Points[1])
        if Points[1][i][1]<0 || Points[1][i][2]<0 ||Points[1][i][3]<0 ||Points[2][i][1]<0
            push!(dirichlet_list,true)
        else
            push!(dirichlet_list,false)
        end
    end
    deleteat!(Points[1],dirichlet_list)
    deleteat!(Points[2],dirichlet_list)
    # Prepare basis
    𝛟_u = U_x_u[:,[1:k...]]
    𝛟_𝜑 = U_x_𝜑[:,[1:m...]]
    _n = n
    𝛀_u = U_R_nl_u[[1:u_dofs...],:]
    𝛀_𝜑 = U_R_nl_𝜑[[u_dofs+1:u_dofs+𝜑_dofs...],:]
    # Point selection algorith
    ec = [copy(I(u_dofs)),copy(I(𝜑_dofs))]
    n_phys = 2
    𝛀c = [𝛀_u,𝛀_𝜑]
    rc = [l,n]
    g = min(l,n,p)
    s = zeros(g)
    w = ceil(g/p)
    s[1] = floor(w*p/g)
    qc = [zeros(Int32,g),zeros(Int32,g)]
    for c in 1:n_phys
        qc[c][1] = Int32(floor(rc[c]/g))
    end
    Sᶜ = Vector{Matrix{Float64}}()
    for c in 1:n_phys
        push!(Sᶜ, copy(𝛀c[c][:,[1:qc[c][1]...]]))
    end
    N = []
    Qc = []
    Zc = []
    dof_ids_c = [[],[]]
    for i in 1:g
        println("Greedy iter = $i")
        for j in 1:s[i]
            nᶜ_max = [0,0]
            for c in 1:n_phys
                nᶜ = []
                for point in Points[c]
                    sum_ = 0
                    for Sqc in eachcol(Sᶜ[c]) 
                        sum_ += norm(Sqc[point])^2
                    end
                    push!(nᶜ,sum_)
                end
                nᶜ_max[c] = argmax(nᶜ)
            end
            n_ = []
            for i in 1:lastindex(Points[1])
                sum_ = 0
                for c in 1:n_phys
                    for Sqc in eachcol(Sᶜ[c])
                        sum_ += norm(Sqc[Points[c][i]])^2 / norm(Sqc[nᶜ_max[c]])^2
                    end
                end
                push!(n_,sum_)
            end
            n = argmax(n_)
            for c in 1:n_phys
                if i == 1
                    push!(Zc, copy(ec[c][:,Points[c][n]]))
                else
                    Zc[c] = hcat(Zc[c],ec[c][:,Points[c][n]])
                end
                append!(dof_ids_c[c],Points[c][n])
                deleteat!(Points[c],n)
            end
            push!(N,n)
        end
        for c in 1:n_phys
            if i == 1
                push!(Qc, qc[c][i])
            else
                for j in 1:qc[c][i]
                    pinv_ = pinv(Zc[c]'*𝛀c[c][:,[1:Qc[c]...]])
                    U_upd = pinv_*(Zc[c]'*𝛀c[c][:,Qc[c]+j])
                    if j<=qc[c][i-1]
                    Sᶜ[c][:,j] = 𝛀c[c][:,Qc[c]+j]- 𝛀c[c][:,[1:Qc[c]...]]*U_upd
                    else
                        Sᶜ[c] = hcat(Sᶜ[c],𝛀c[c][:,Qc[c]+j]- 𝛀c[c][:,[1:Qc[c]...]]*U_upd)
                        println(1)
                    end
                end
                Qc[c] += qc[c][i]
            end
        end
        if w==1 && i<= mod(p,g)
            s[i+1] += s[i] + 1
        elseif i < g
            s[i+1] += s[1]
        end
        for c in  1:n_phys
            if i <= mod(rc[c],g)
                qc[c][i+1] = qc[c][i] + 1
            elseif i < g
                qc[c][i+1] = qc[c][1]
            end
        end
    end
    # Fimal results
    𝛀_u = U_R_nl_u[:,[1:l...]]
    𝛀_𝜑 = U_R_nl_𝜑[:,[1:_n...]]
    𝛀 = hcat(𝛀_u,𝛀_𝜑)
    _,a = size(Zc[1])
    D_x_u = vcat(Zc[1],zeros(𝜑_dofs,a))
    _,b = size(Zc[2])
    D_x_𝜑 = vcat(zeros(u_dofs,b),Zc[2])
    Z = hcat(D_x_u,D_x_𝜑)
    pinv_Z𝛀 = pinv(Z'*𝛀)
    return Zc, dof_ids_c, N, pinv_Z𝛀, Z, 𝛀, 𝛟_𝜑, 𝛟_u
end

k,l,m,n,p = 10,75,10,35,25
Zc, dof_ids_c, N, pinv_Z𝛀, Z, 𝛀, 𝛟_𝜑, 𝛟_u = POD_Gappy_Matrices_split(k,l,m,n,p,U_x_u,U_x_𝜑,U_R_u,U_R_𝜑)
argmax(U_R_𝜑)
norm(pinv_Z𝛀)
scatter(dof_ids_c[1])
scatter(dof_ids_c[2])
scatter(N)
N_ = N
for i in 2:length(N)
    if N_[i] >= N_[i-1]
        N_[i] = N[i] + 1
    end
end

Z'*𝛀

Zc[2]

_,a = size(Zc[1])
D_x_u = vcat(Zc[1],zeros(𝜑_dofs,a))

_,b = size(Zc[2])
D_x_𝜑 = vcat(zeros(u_dofs,b),Zc[2])

maximum(abs.(D_R-𝛀*pinv_Z𝛀*Z'*D_R))