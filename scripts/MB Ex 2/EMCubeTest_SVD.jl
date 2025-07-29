using CSV
using DataFrames
using LinearAlgebra
using Plots
using JLD2

# λ_list = [333288.89, 416611.11, 499933.33, 583255.55, 666577.77]
# μ_list = [66.67, 83.34, 100.01, 116.67, 133.34]

D_x = []
D_R_nl= []
for i in 1:1
    # λ = λ_list[i]
    # μ = μ_list[i]
    file_name = "data/csv/EMCube_test/x_"*"MaterialModel0"*".csv"
    _X = CSV.File(file_name) |> Tables.matrix
    push!(D_x,_X)
    file_name = "data/csv/EMCube_test/R_nl_"*"MaterialModel0"*".csv"
    _R_nl = CSV.File(file_name) |> Tables.matrix
    push!(D_R_nl,_R_nl)
end

D_x = reduce(hcat,D_x)
D_R_nl = reduce(hcat,D_R_nl)

U_x, σ_i_x, V_x = svd(D_x)
U_R_nl, σ_i_R_nl, V_R_nl = svd(D_R_nl)

σ_i_x_rel = σ_i_x./σ_i_x[1]
σ_i_R_nl_rel = σ_i_R_nl./σ_i_R_nl[1]

n, _ = size(U_x)

plot(
    [σ_i_x_rel[[1:100...]],σ_i_R_nl_rel[[1:100...]]],
    yscale=:log10,
    label = ["Dₓ","Dᵣ"],
    ylims = (10^-float(20), 1),
    yticks=[10^-float(i*4) for i in 0:5]
)

m_k = nothing
count = 1
for σ_x_rel in σ_i_x_rel
    if σ_x_rel<1e-11
        m_k = count
        break
    end
    count += 1
end

𝛟 = U_x[:,[1:m_k...]]
𝛀 = U_R_nl[:,[1:m_k...]]

q = Vector{Int}(undef,m_k)
W_𝛀 = copy(𝛀)
e = copy(I(n))
q[1] = argmax(W_𝛀[1])
Z = copy(e[:,q[1]])
𝛀 = W_𝛀[:,1]

for s in 2:m_k
    R_red = inv(Z'*𝛀)*Z'*W_𝛀[:,s]
    println(size(R_red))
    r = W_𝛀[:,s] - 𝛀*R_red
    q[s] = argmax(abs.(r))
    Z = hcat(Z,e[:,q[s]])
    𝛀 = hcat(𝛀,W_𝛀[:,s])
end

Zᵀ𝛀_inv = inv(Z'*𝛀)

Err_DEIM = D_R_nl-𝛀*Zᵀ𝛀_inv*Z'*D_R_nl

Err_DEIM_max = maximum(Err_DEIM)

M_DEIM = 𝛟'*𝛀*Zᵀ𝛀_inv

jldsave("scripts/MB Ex 2/DEIM_matrices.jld2", 𝛟=𝛟, M_DEIM=M_DEIM, Z=Z, 𝛀=𝛀)