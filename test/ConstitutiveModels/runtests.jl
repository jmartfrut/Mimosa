using Test

@testset "ConstitutiveModels" verbose=true begin

    include("MechanicalModelsTests.jl")

    include("ThermalModelsTests.jl")

end
