using CSV
using Tables
using DataFrames

St,Sl,pot_list,N_rand = 4,4,[2000,3000,4000,5000],200
conf_list = CSV.File("data/csv/EM_TB_ST$(St)_SL$(Sl)_Conf0.csv") |> Tables.matrix
conf_list_ = CSV.File("data/csv/EM_TB_ST$(St)_SL$(Sl)_ConfRand.csv") |> Tables.matrix

mkpath("scripts/Scratch/TubeBeam_St$(St)_Sl$(Sl)")

conf_list = hcat(conf_list,conf_list_[:,[1:N_rand...]])
for conf in eachcol(conf_list)
    destination = "scripts/Scratch/TubeBeam_St$(St)_Sl$(Sl)/Conf-$conf"
    mkpath(destination)
    for pot in pot_list
        Λ = pot/5000
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        problemName = "TubeBeam"
        problemName = problemName*"_ϕ5000.0"*"_St$St"*"_St$Sl"
        for s in conf
            problemName = problemName*"_$s"
        end
        file_name = "data/csv/" * problemName * "/_Λ_" * Λstring * ".csv"
        cp(file_name,destination* "/ϕ$pot.csv",force=true)
    end
    
end

conf_list_test = CSV.File("data/csv/EM_TB_St4_Sl4_Phi2000_Random/EM_TB_ST4_SL4_ConfRand.csv") |> Tables.matrix
conf_list_test = conf_list_test[:,[1000-31:1000-2...]];
for conf in eachcol(conf_list_test)
    destination = "scripts/Scratch/TubeBeam_St$(St)_Sl$(Sl)/Conf-$conf"
    mkpath(destination)
    for pot in pot_list
        Λ = pot/5000
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        problemName = "TubeBeam"
        problemName = problemName*"_ϕ5000.0"*"_St$St"*"_St$Sl"
        for s in conf
            problemName = problemName*"_$s"
        end
        file_name = "data/csv/" * problemName * "/_Λ_" * Λstring * ".csv"
        try
            cp(file_name,destination* "/ϕ$pot.csv",force=true)
        catch
            println(destination* "/ϕ$pot.csv")
        end
    end
    
end

include("C:/Users/mjbarillas/Documents/GitHub/Mimosa/scripts/MB Ex/kPCA/TB/ex2_a_kPCA_TB_ST4SL2.jl")

pot, parts = 2000, 4
X, X_ = ReadData2(pot,parts)

conf_list = []
for i in [0,1], j in [0,1], k in [0,1], l in [0,1], m in [0,1], n in [0,1], o in [0,1], p in [0,1]
    push!(conf_list,[i,j,k,l,m,n,o,p])
end
conf_list = reduce(hcat,conf_list)
Sl = 2
i = 1
for conf in eachcol(conf_list)
    destination = "scripts/Scratch/TubeBeam_St$(St)_Sl$(Sl)/Conf-$conf"
    mkpath(destination)

    df = DataFrame(x = X[:,i])
    CSV.write(destination*"/φ2000.csv",df)
    
end