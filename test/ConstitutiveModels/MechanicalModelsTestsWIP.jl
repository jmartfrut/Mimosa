using .MechanicalModelsTests


using Gridap.Arrays
using ForwardDiff
using Mimosa.TensorAlgebra
function _getKinematic(::Mechano)
  F(∇u) = one(∇u) + ∇u
  J(F) = det(F)
  H(F) = J(F) * inv(F)'
  return (F, H, J)
end
struct IncompressibleNeoHookean3D_2dP <: Mechano
  μ::Float64
end
function (obj::IncompressibleNeoHookean3D_2dP)(::DerivativeStrategy{:autodiff})
    Ψ(Ce)      = obj.μ / 2 * tr(Ce) * (det(Ce))^(-1 / 3)
    Se(Ce)     = 2 * ForwardDiff.gradient(Ce -> Ψ(Ce), get_array(Ce))
    ∂Se∂Ce(Ce) = ForwardDiff.jacobian(Ce -> Se(Ce), get_array(Ce))
    return (Ψ, Se, ∂Se∂Ce)
end
function (obj::IncompressibleNeoHookean3D_2dP)(::DerivativeStrategy{:analytic})
    _, H, J = _getKinematic(obj)
    μ = obj.μ
    I3__ = I3()
    Ψ(Ce) = μ / 2 * tr(Ce) * (det(Ce))^(-1 / 3)
    ∂Ψ∂Ce(Ce) =  μ / 2 * I3__ * (det(Ce))^(-1 / 3)
    ∂Ψ∂dCe(Ce) = - μ / 6 * tr(Ce) * (det(Ce))^(-4 / 3)
    Se(Ce)  = 2 * (∂Ψ∂Ce(Ce) + ∂Ψ∂dCe(Ce) * H(Ce))
    ∂2Ψ∂CedCe(Ce) =  - μ / 6 * I3__ * (det(Ce))^(-4 / 3)
    ∂2Ψ∂2dCe(Ce) =  2*μ / 9 * tr(Ce) * (det(Ce))^(-7 / 3)
    ∂Se∂Ce(Ce) = 2 *  (∂2Ψ∂2dCe(Ce) * (H(Ce) ⊗ H(Ce)) + ∂2Ψ∂CedCe(Ce) ⊗ H(Ce) + H(Ce) ⊗ ∂2Ψ∂CedCe(Ce) + ∂Ψ∂dCe(Ce) * ×ᵢ⁴(Ce))
    return (Ψ, Se, ∂Se∂Ce)
end


@testset "IncompressibleNeoHookean3D_2dP" begin
    μ = 5.832e4
    model = IncompressibleNeoHookean3D_2dP(μ)
    testMechanicalModelDerivatives(model, C; tolerance=1e-14)
end

@testset "IncompressibleNeoHookean3D" begin
    μ = 5.832e4
    model = IncompressibleNeoHookean3D(μ)
    testMechanicalModelDerivatives(model, F, StressTensor{:FirstPiola}(); tolerance=1e-14)
    testMechanicalModelDerivatives(model, C, StressTensor{:SecondPiola}(); tolerance=1e-14)
end
