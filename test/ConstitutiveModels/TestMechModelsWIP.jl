using Mimosa
using Gridap.TensorValues
using Gridap.Arrays
using Test

import Base:isapprox

function isapprox(A::TensorValue, B::TensorValue; kwargs...)
    isapprox(get_array(A), get_array(B); kwargs...)
end

F = TensorValue{3,3,Float64}(1,2,3,4,5,6,7,8,9)

C1 = 1.2e6
C2 = 1.3e7
C3 = 1.4e8

μ = 5.23e6
λ = 6.78e7

@testset "MechanicalModels" begin

    @testset "Yeoh" begin
        model = Yeoh3D(C1,C2,C3)

        ΨA, dΨA, ddΨA = model(DerivativeStrategy{:analytic}())
        ΨD, dΨD, ddΨD = model(DerivativeStrategy{:autodiff}())

        @test ΨA(F) ≈ ΨD(F)
        @test dΨA(F) ≈ dΨD(F)
        @test ddΨA(F) ≈ ddΨD(F)  # TODO: This test is failing. The analytical derivatives are wrong...
    end

    @testset "ComposedMechanicalModel" begin
        neo = NeoHookean3D(μ=μ,λ=λ)  # TODO: Check neo-Hookean derivatives...
        yeoh = Yeoh3D(C1,C2,C3)
        model = neo + yeoh
        
        ΨA, dΨA, ddΨA = model(DerivativeStrategy{:analytic}())
        ΨD, dΨD, ddΨD = model(DerivativeStrategy{:autodiff}())

        @test ΨA(F) ≈ ΨD(F)
        @test dΨA(F) ≈ dΨD(F)
        @test ddΨA(F) ≈ ddΨD(F)
    end
end;
