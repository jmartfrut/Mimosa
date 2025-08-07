include("ConstitutiveModelsTests.jl")


δθ = 1.1

Cv = 1.1
θr = 2.2
α  = 3.3

@testset "ThermalModels" verbose=true begin

    @testset "ThermalModel" begin
        model = ThermalModel(Cv=Cv, θr=θr, α=α)
        test_constitutive_model_derivatives(model, δθ, rtol=1e-14)
    end

end
