using Mimosa.ConstitutiveModels
using Gridap.Arrays
using Gridap.TensorValues
using Test

import Base:isapprox

function isapprox(A::TensorValue, B::AbstractArray; kwargs...)
    isapprox(get_array(A), B; kwargs...)
end

function isapprox(A::AbstractArray, B::TensorValue; kwargs...)
    isapprox(A, get_array(B); kwargs...)
end

function isapprox(A::TensorValue, B::TensorValue; kwargs...)
    isapprox(get_array(A), get_array(B); kwargs...)
end


"""
    testMechanicalModelDerivatives

Check the analytical expressions match the numerical ones from autodiff.
The check is performed for the energy density, its first derivative (i.e. stress) and the sedond derivative (Jacobian).

# Arguments
- `model::ConstitutiveModel`
- `kinematic`: The kinematic definition, F or C
- `args`: Extra arguments to pass to the physical model, such as `StressTensor{:SecondPiola}()`
- `debug=nothing`: it can be `:energy`, `:gradient` or `:jacobian`
- `kwargs`: Extra named arguments to pass to isapprox, such as `rtol=1e-12`
"""
function test_constitutive_model_derivatives(model::ConstitutiveModel, kinematic, args...; debug=nothing, kwargs...)
    Ψ_A, ∂Ψ_A, ∂∂Ψ_A = model(DerivativeStrategy{:analytic}(), args...)
    Ψ_D, ∂Ψ_D, ∂∂Ψ_D = model(DerivativeStrategy{:autodiff}(), args...)
    @test isapprox(Ψ_A(kinematic), Ψ_D(kinematic); kwargs...)
    @test isapprox(∂Ψ_A(kinematic), ∂Ψ_D(kinematic); kwargs...)
    @test isapprox(∂∂Ψ_A(kinematic), ∂∂Ψ_D(kinematic); kwargs...)
    if debug == :energy
        @show Ψ_A(kinematic)
        @show Ψ_D(kinematic)
    elseif debug == :gradient
        @show ∂Ψ_A(kinematic)
        @show ∂Ψ_D(kinematic)
    elseif debug == :jacobian
        @show ∂∂Ψ_A(kinematic)
        @show ∂∂Ψ_D(kinematic)
    end
end
