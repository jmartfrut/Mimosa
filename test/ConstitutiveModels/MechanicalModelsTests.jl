include("ConstitutiveModelsTests.jl")


F = TensorValue(0.01+1.0, 0.02, 0.03, 0.04, 0.05+1.0, 0.06, 0.07, 0.08, 0.09+1.0)
C = F' * F

μ = 5.832e4
λ = 10*μ

@testset "MechanicalModels" verbose=true begin

    @testset "LinearElasticity3D" begin
        model = LinearElasticity3D(λ=λ, μ=μ)
        test_constitutive_model_derivatives(model, F; rtol=1e-14)
    end

    @testset "NeoHookean3D" begin
        model = NeoHookean3D(λ=λ, μ=μ)
        test_constitutive_model_derivatives(model, F; rtol=1e-14)
    end

    @testset "MoneyRivlin3D" begin
        model = MoneyRivlin3D(λ=λ, μ1=μ, μ2=μ)
        test_constitutive_model_derivatives(model, F; rtol=1e-14)
    end

end   # @testset "MechanicalModels
