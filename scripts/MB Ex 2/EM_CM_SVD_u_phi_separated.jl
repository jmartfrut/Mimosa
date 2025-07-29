using CSV
using DataFrames
using LinearAlgebra
using Plots
using JLD2
using Base.Threads

𝜑_dofs = 30226
u_dofs = 152409

D_x = []
D_R_nl= []


# for i in 0:2
#     file_name = "data/csv/EM_CM_test_2/MaterialModel"*"$i"*"/x_.csv"
#     _X = CSV.File(file_name) |> Tables.matrix
#     push!(D_x,_X)
#     file_name = "data/csv/EM_CM_test_2/MaterialModel"*"$i"*"/R_nl_.csv"
#     _R_nl = CSV.File(file_name) |> Tables.matrix
#     push!(D_R_nl,_R_nl)
# end


for i in 0:2
    file_name = "data/csv/EM_CM_test_3/MaterialModel"*"$i"*"/x_.csv"
    _X = CSV.File(file_name) |> Tables.matrix
    push!(D_x,_X)
    file_name = "data/csv/EM_CM_test_3/MaterialModel"*"$i"*"/R_nl_.csv"
    _R_nl = CSV.File(file_name) |> Tables.matrix
    push!(D_R_nl,_R_nl)
end

D_x = reduce(hcat,D_x)
D_R_nl = reduce(hcat,D_R_nl)

total_dofs, n_s = size(D_x)

D_x_u = D_x[[1:u_dofs...],:]
D_x_𝜑 = D_x[[u_dofs+1:u_dofs+𝜑_dofs...],:]

D_x_u = vcat(D_x_u,zeros(𝜑_dofs,n_s))
D_x_𝜑 = vcat(zeros(u_dofs,n_s),D_x_𝜑)


D_R_nl_u = D_R_nl[[1:u_dofs...],:]
D_R_nl_𝜑 = D_R_nl[[u_dofs+1:u_dofs+𝜑_dofs...],:]

D_R_nl_u = vcat(D_R_nl_u,zeros(𝜑_dofs,n_s))
D_R_nl_𝜑 = vcat(zeros(u_dofs,n_s),D_R_nl_𝜑)

U_x_u, σ_i_x_u, V_x_u = svd(D_x_u)
U_R_nl_u, σ_i_R_nl_u, V_R_nl_u = svd(D_R_nl_u)

U_x_𝜑, σ_i_x_𝜑, V_x_𝜑 = svd(D_x_𝜑)
U_R_nl_𝜑, σ_i_R_nl_𝜑, V_R_nl_𝜑 = svd(D_R_nl_𝜑)

σ_i_x_rel_u = σ_i_x_u./σ_i_x_u[1]
σ_i_R_nl_rel_u = σ_i_R_nl_u./σ_i_R_nl_u[1]

σ_i_x_rel_𝜑 = σ_i_x_𝜑./σ_i_x_𝜑[1]
σ_i_R_nl_rel_𝜑 = σ_i_R_nl_𝜑./σ_i_R_nl_𝜑[1]

plotlyjs()
plot(
    [σ_i_x_rel_u[[1:200...]],σ_i_R_nl_rel_u[[1:200...]],σ_i_x_rel_𝜑[[1:200...]],σ_i_R_nl_rel_𝜑[[1:200...]]],
    yscale=:log10,
    label = ["Dₓ_u" "Dᵣ_u" "Dₓ_phi" "Dᵣ_phi"],
    ylims = (10^-float(20), 1),
    yticks=[10^-float(i*4) for i in 0:5]
)

Plots.savefig("C:/Users/mjbarillas/Documents/LaTeX/POD_DEIM_Notes/Figures/CapturedSigma.pdf")

m_k = nothing
c = 1
for σ_x_rel in σ_i_R_nl_rel_u
    if σ_x_rel<1e-13
        m_k = c
        println(m_k)
        break
    end
    c += 1
end

m_k = 75

function POD_DEIM_Matrices(m_k)
    𝛟_u = U_x_u[:,[1:m_k...]]
    𝛀_u = U_R_nl_u[:,[1:m_k...]]
    
    𝛟_𝜑 = U_x_𝜑[:,[1:m_k...]]
    𝛀_𝜑 = U_R_nl_𝜑[:,[1:m_k...]]
    
    q_u = Vector{Int}(undef,m_k)
    W_𝛀 = copy(𝛀_u)
    e = copy(I(u_dofs+𝜑_dofs))
    q_u[1] = argmax(abs.(W_𝛀[:,1]))
    Z_u = copy(e[:,q_u[1]])
    𝛀 = W_𝛀[:,1]
    
    for s in 2:m_k
        R_red = inv(Z_u'*𝛀)*Z_u'*W_𝛀[:,s]
        println(size(R_red))
        r = W_𝛀[:,s] - 𝛀*R_red
        println("Max residual from greedy = $(maximum(abs.(r)))")
        q_u[s] = argmax(abs.(r))
        Z_u = hcat(Z_u,e[:,q_u[s]])
        𝛀 = hcat(𝛀,W_𝛀[:,s])
    end
    
    Z = Z_u
    
    Zᵀ𝛀_inv_u = inv(Z_u'*𝛀_u)
    
    q_𝜑 = Vector{Int}(undef,m_k)
    W_𝛀 = copy(𝛀_𝜑)
    e = copy(I(u_dofs+𝜑_dofs))
    q_𝜑[1] = argmax(abs.(W_𝛀[:,1]))
    Z_𝜑 = copy(e[:,q_𝜑[1]])
    𝛀 = W_𝛀[:,1]
    
    for s in 2:m_k
        R_red = inv(Z_𝜑'*𝛀)*Z_𝜑'*W_𝛀[:,s]
        println(size(R_red))
        r = W_𝛀[:,s] - 𝛀*R_red
        println("Max residual from greedy = $(maximum(abs.(r)))")
        q_𝜑[s] = argmax(abs.(r))
        Z_𝜑 = hcat(Z_𝜑,e[:,q_𝜑[s]])
        𝛀 = hcat(𝛀,W_𝛀[:,s])
    end
    
    Zᵀ𝛀_inv_𝜑 = inv(Z_𝜑'*𝛀_𝜑)
    
    Z = hcat(Z,Z_𝜑)
    𝛀 = hcat(𝛀_u,𝛀_𝜑)
    Zᵀ𝛀_inv = inv(Z'*𝛀)
    
    jldsave("scripts/MB Ex 2/CM_DEIM_matrices_u_phi_sep.jld2", 𝛟_u=𝛟_u, 𝛟_𝜑=𝛟_𝜑, 𝛀_u=𝛀_u, 
        𝛀_𝜑=𝛀_𝜑, 𝛀=𝛀, Z_u=Z_u, Z_𝜑=Z_𝜑, Z=Z, Zᵀ𝛀_inv_u=Zᵀ𝛀_inv_u, Zᵀ𝛀_inv_𝜑=Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv=Zᵀ𝛀_inv, q_u=q_u, q_𝜑=q_𝜑
        )  
end

function POD_DEIM_Matrices(k,l,m,n)
    𝛟_u = U_x_u[:,[1:k...]]
    𝛀_u = U_R_nl_u[:,[1:l...]]
    
    𝛟_𝜑 = U_x_𝜑[:,[1:m...]]
    𝛀_𝜑 = U_R_nl_𝜑[:,[1:n...]]
    
    q_u = Vector{Int}(undef,l)
    W_𝛀 = copy(𝛀_u)
    e = copy(I(u_dofs+𝜑_dofs))
    q_u[1] = argmax(abs.(W_𝛀[:,1]))
    Z_u = copy(e[:,q_u[1]])
    𝛀 = W_𝛀[:,1]
    
    for s in 2:l
        R_red = inv(Z_u'*𝛀)*Z_u'*W_𝛀[:,s]
        println(size(R_red))
        r = W_𝛀[:,s] - 𝛀*R_red
        println("Max residual from greedy = $(maximum(abs.(r)))")
        q_u[s] = argmax(abs.(r))
        Z_u = hcat(Z_u,e[:,q_u[s]])
        𝛀 = hcat(𝛀,W_𝛀[:,s])
    end
    
    Z = Z_u
    
    Zᵀ𝛀_inv_u = inv(Z_u'*𝛀_u)
    
    q_𝜑 = Vector{Int}(undef,n)
    W_𝛀 = copy(𝛀_𝜑)
    e = copy(I(u_dofs+𝜑_dofs))
    q_𝜑[1] = argmax(abs.(W_𝛀[:,1]))
    Z_𝜑 = copy(e[:,q_𝜑[1]])
    𝛀 = W_𝛀[:,1]
    
    for s in 2:n
        R_red = inv(Z_𝜑'*𝛀)*Z_𝜑'*W_𝛀[:,s]
        println(size(R_red))
        r = W_𝛀[:,s] - 𝛀*R_red
        println("Max residual from greedy = $(maximum(abs.(r)))")
        q_𝜑[s] = argmax(abs.(r))
        Z_𝜑 = hcat(Z_𝜑,e[:,q_𝜑[s]])
        𝛀 = hcat(𝛀,W_𝛀[:,s])
    end
    
    Zᵀ𝛀_inv_𝜑 = inv(Z_𝜑'*𝛀_𝜑)
    
    Z = hcat(Z,Z_𝜑)
    𝛀 = hcat(𝛀_u,𝛀_𝜑)
    Zᵀ𝛀_inv = inv(Z'*𝛀)
    
    jldsave("scripts/MB Ex 2/CM_DEIM_matrices_u_phi_sep.jld2", 𝛟_u=𝛟_u, 𝛟_𝜑=𝛟_𝜑, 𝛀_u=𝛀_u, 
        𝛀_𝜑=𝛀_𝜑, 𝛀=𝛀, Z_u=Z_u, Z_𝜑=Z_𝜑, Z=Z, Zᵀ𝛀_inv_u=Zᵀ𝛀_inv_u, Zᵀ𝛀_inv_𝜑=Zᵀ𝛀_inv_𝜑, Zᵀ𝛀_inv=Zᵀ𝛀_inv, q_u=q_u, q_𝜑=q_𝜑
        )  
end


function POD_DEIM_Matrices(k,l,m,n)
    𝛟_u = U_x_u[:,[1:k...]]
    𝛀_u = U_R_nl_u[:,[1:l...]]
    
    𝛟_𝜑 = U_x_𝜑[:,[1:m...]]
    𝛀_𝜑 = U_R_nl_𝜑[:,[1:n...]]

    𝛀 = hcat(𝛀_u,𝛀_𝜑)
    
    q = Vector{Int}(undef,l+n)
    W_𝛀 = copy(𝛀)
    e = copy(I(u_dofs+𝜑_dofs))
    q[1] = argmax(abs.(W_𝛀[:,1]))
    Z = copy(e[:,q[1]])
    𝛀 = W_𝛀[:,1]
    
    for s in 2:l+n
        R_red = inv(Z'*𝛀)*Z'*W_𝛀[:,s]
        println(size(R_red))
        r = W_𝛀[:,s] - 𝛀*R_red
        println("Max residual from greedy = $(maximum(abs.(r)))")
        q[s] = argmax(abs.(r))
        Z = hcat(Z,e[:,q[s]])
        𝛀 = hcat(𝛀,W_𝛀[:,s])
    end
    
    Zᵀ𝛀_inv = inv(Z'*𝛀)
    
    jldsave("scripts/MB Ex 2/CM_DEIM_matrices_u_phi_sep.jld2", 𝛟_u=𝛟_u, 𝛟_𝜑=𝛟_𝜑,
        𝛀=𝛀, Z=Z, Zᵀ𝛀_inv=Zᵀ𝛀_inv, q_u=q[[1:l...]], q_𝜑=q[[l+1:n+l...]]
        )  
end

k,l,m,n = 10,10,10,10 # Usually works
k,l,m,n = 65,10,35,10
k,l,m,n = 65,10,35,3

function POD_DEIM_Matrices_test(l1,l2,n1,n2)
    Error = Matrix{Float64}(undef,l2-l1+1,n2-n1+1)
    for l in l1:l2
        println("----------------------------")
        println("Row = $l of $l2")
        println("----------------------------")
        println("----------------------------")
        @threads for n in n1:n2
            
            𝛀_u = U_R_nl_u[:,[1:l...]]
            𝛀_𝜑 = U_R_nl_𝜑[:,[1:n...]]
            
            q_u = Vector{Int}(undef,l)
            W_𝛀 = copy(𝛀_u)
            e = copy(I(u_dofs+𝜑_dofs))
            q_u[1] = argmax(abs.(W_𝛀[:,1]))
            Z_u = copy(e[:,q_u[1]])
            𝛀 = W_𝛀[:,1]
            
            for s in 2:l
                R_red = inv(Z_u'*𝛀)*Z_u'*W_𝛀[:,s]
                # println(size(R_red))
                r = W_𝛀[:,s] - 𝛀*R_red
                # println("Max residual from greedy = $(maximum(abs.(r)))")
                q_u[s] = argmax(abs.(r))
                Z_u = hcat(Z_u,e[:,q_u[s]])
                𝛀 = hcat(𝛀,W_𝛀[:,s])
            end
            
            Z = Z_u
            
            # Zᵀ𝛀_inv_u = inv(Z_u'*𝛀_u)
            
            q_𝜑 = Vector{Int}(undef,n)
            W_𝛀 = copy(𝛀_𝜑)
            e = copy(I(u_dofs+𝜑_dofs))
            q_𝜑[1] = argmax(abs.(W_𝛀[:,1]))
            Z_𝜑 = copy(e[:,q_𝜑[1]])
            𝛀 = W_𝛀[:,1]
            
            for s in 2:n
                R_red = inv(Z_𝜑'*𝛀)*Z_𝜑'*W_𝛀[:,s]
                # println(size(R_red))
                r = W_𝛀[:,s] - 𝛀*R_red
                # println("Max residual from greedy = $(maximum(abs.(r)))")
                q_𝜑[s] = argmax(abs.(r))
                Z_𝜑 = hcat(Z_𝜑,e[:,q_𝜑[s]])
                𝛀 = hcat(𝛀,W_𝛀[:,s])
            end
            
            # Zᵀ𝛀_inv_𝜑 = inv(Z_𝜑'*𝛀_𝜑)
            
            Z = hcat(Z,Z_𝜑)
            𝛀 = hcat(𝛀_u,𝛀_𝜑)
            Zᵀ𝛀_inv = inv(Z'*𝛀)
            Error[l-l1+1,n-n1+1] = maximum(abs.(D_R_nl-𝛀*Zᵀ𝛀_inv*Z'*D_R_nl))
            println("Column = $n of $n2")
        end
    end
    return Error
end

(l1,l2,n1,n2) = 1,150,1,15
(l1,l2,n1,n2) = 1,15,1,15

Error = POD_DEIM_Matrices_test(l1,l2,n1,n2)
heatmap(Error,colorbar_scales=:log10)
gr()
plotlyjs()
𝜑_dof_=1
plot(Error[:,𝜑_dof_],yscale=:log10,label="𝜑_dof_ = $𝜑_dof_",xlabel="u_dofs",
ylabel="Error", title="Error = maximum(abs.(D_R_nl-𝛀*Zᵀ𝛀_inv*Z'*D_R_nl))", legend_position=(0.8,0.9),)
𝜑_dof_=2
plot!(Error[:,𝜑_dof_],yscale=:log10,label="𝜑_dof_ = $𝜑_dof_")
𝜑_dof_=5
plot!(Error[:,𝜑_dof_],yscale=:log10,label="𝜑_dof_ = $𝜑_dof_")
𝜑_dof_=10
plot!(Error[:,𝜑_dof_],yscale=:log10,label="𝜑_dof_ = $𝜑_dof_")
𝜑_dof_=15
plot!(Error[:,𝜑_dof_],yscale=:log10,label="𝜑_dof_ = $𝜑_dof_")

plot(Error[75,:],yscale=:log10)
jldsave("scripts/MB Ex 2/CM_DEIM_Error_u_phi_sep.jld2", Error=Error)

(l1,l2,n1,n2) = 100,100,1,10
Error_ = POD_DEIM_Matrices_test(l1,l2,n1,n2)
plot(Error_[1,:],yscale=:log10)

Error = load("scripts/MB Ex 2/CM_DEIM_Error_u_phi_sep.jld2")
Error = Error["Error"]

Plots.savefig("C:/Users/mjbarillas/Documents/LaTeX/POD_DEIM_Notes/Figures/DEIM_Error.pdf")