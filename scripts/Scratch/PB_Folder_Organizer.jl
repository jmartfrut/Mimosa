using CSV
using Tables

n = 340
C_n = CSV.File("data/csv/Config_PB_10S_40ref_1000rand.csv")|> Tables.matrix
C_n = Int.(C_n[:,[1:n...]])
S = 10
pot_10 = 50000.0

for conf in eachcol(C_n)
    destination = "scripts/Scratch/PlateBeam_S$S/Conf-$conf"
    mkpath(destination)
    for pot in pot_list
        Λ = pot/5000
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        diffstrat = "autodiff"
        problemName = "PB-S$S-O2-/$diffstrat/Yeoh/PL"
        for s in conf
            problemName = problemName*"_$s"
        end
        problemName = problemName*"_ϕ$pot_10"
        file_name = "data/csv/" * problemName * "/_Λ_" * Λstring * ".csv"
        cp(file_name,destination* "/ϕ$pot.csv",force=true)
    end
    
end

C_test = CSV.File("data/csv/Config_PB_10S_40ref_1000rand.csv")|> Tables.matrix
C_test = Int.(C_test[:,[540-29:540...]])

for conf in eachcol(C_test)
    destination = "scripts/Scratch/PlateBeam_S$S/Conf-$conf"
    mkpath(destination)
    for pot in pot_list
        Λ = pot/5000
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        diffstrat = "autodiff"
        problemName = "PB-S$S-O2-/$diffstrat/Yeoh/PL"
        for s in conf
            problemName = problemName*"_$s"
        end
        problemName = problemName*"_ϕ$pot_10"
        file_name = "data/csv/" * problemName * "/_Λ_" * Λstring * ".csv"
        cp(file_name,destination* "/ϕ$pot.csv",force=true)
    end
    
end

S = 4
pot_10 = 50000.0

Conf = []
for i1 in 0:1, i2 in 0:1, i3 in 0:1, i4 in 0:1, i5 in 0:1, i6 in 0:1, i7 in 0:1, i8 in 0:1
    push!(Conf,[i1,i2,i3,i4,i5,i6,i7,i8])
end
C_n = reduce(hcat,Conf)

pot_list = [5000]

for conf in eachcol(C_n)
    destination = "scripts/Scratch/PlateBeam_S$S/Conf-$conf"
    mkpath(destination)
    for pot in pot_list
        Λ = pot/5000
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        diffstrat = "autodiff"
        problemName = "PB-S$S-O2-/$diffstrat/Yeoh/PL"
        for s in conf
            problemName = problemName*"_$s"
        end
        problemName = problemName*"_ϕ$pot_10"
        file_name = "data/csv/" * problemName * "/_Λ_" * Λstring * ".csv"
        cp(file_name,destination* "/ϕ$pot.csv",force=true)
    end
    
end