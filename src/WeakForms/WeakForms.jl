module WeakForms

using Gridap
using ..TensorAlgebra

export CouplingStrategy
export residual_EM
export jacobian_EM
export residual_M
export jacobian_M
export residual_TEM
export jacobian_TEM
export residual_TM
export jacobian_TM
export mass_term

# Coupling management
# ===================
struct CouplingStrategy{Kind} end



# ===================
# Mechanics
# ===================

function residual_M(u, v, ∂Ψu, dΩ)
    ∫(∇(v)' ⊙ (∂Ψu ∘ (∇(u)')))dΩ
end

function jacobian_M(u, du, v, ∂Ψuu, dΩ)
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (∇(u)')) ⊙ (∇(du)')))dΩ
end


function mass_term(u, v, Coeff, dΩ)
    ∫(Coeff* (u⋅v))dΩ
end


# ===================
# ThermoElectroMech
# ===================

# Stagered strategy
# -----------------

function residual_TEM(::CouplingStrategy{:staggered_M}, (u, φ, θ), v, ∂Ψu, dΩ)
    return ∫(∇(v)' ⊙ (∂Ψu ∘ (∇(u)', ∇(φ), θ)))dΩ
end

function residual_TEM(::CouplingStrategy{:staggered_E}, (u, φ, θ), vφ, ∂Ψφ, dΩ)
    return ∫(∇(vφ)' ⋅ (∂Ψφ ∘ (∇(u)', ∇(φ), θ)))dΩ
end

function residual_TEM(::CouplingStrategy{:staggered_T}, (u, φ, θ), vθ, κ, dΩ)
    return ∫(κ * ∇(θ) ⋅ ∇(vθ))dΩ
end


function jacobian_TEM(::CouplingStrategy{:staggered_M}, (u, φ, θ), du, v, ∂Ψuu, dΩ)
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (∇(u)', ∇(φ), θ)) ⊙ (∇(du)')))dΩ
end

function jacobian_TEM(::CouplingStrategy{:staggered_E}, (u, φ, θ), dφ, vφ, ∂Ψφφ, dΩ)
    ∫(∇(vφ) ⋅ ((∂Ψφφ ∘ (∇(u)', ∇(φ), θ)) ⋅ ∇(dφ)))dΩ
end

function jacobian_TEM(::CouplingStrategy{:staggered_T}, dθ, vθ, κ, dΩ)
    ∫(κ * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian_TEM(::CouplingStrategy{:staggered_T}, (u, φ, θ), dθ, vθ, κ, dΩ)
    ∫((κ ∘ (u, φ, θ)) * ∇(dθ) ⋅ ∇(vθ))dΩ
end

# Monolithic strategy
# -------------------

function residual_TEM(::CouplingStrategy{:monolithic}, (u, φ, θ), (v, vφ, vθ), (∂Ψu, ∂Ψφ), κ, dΩ)
    residual_TEM(CouplingStrategy{:staggered_M}(), (u, φ, θ), v, ∂Ψu, dΩ) +
    residual_TEM(CouplingStrategy{:staggered_E}(), (u, φ, θ), vφ, ∂Ψφ, dΩ) +
    residual_TEM(CouplingStrategy{:staggered_T}(), (u, φ, θ), vθ, κ, dΩ)
end

function jacobian_TEM(::CouplingStrategy{:monolithic}, (u, φ, θ), (du, dφ, dθ), (v, vφ, vθ), (∂Ψuu, ∂Ψφφ, ∂Ψφu, ∂Ψuθ, ∂Ψφθ), κ, dΩ)
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (∇(u)', ∇(φ), θ)) ⊙ (∇(du)')))dΩ +
    ∫(∇(dφ) ⋅ ((∂Ψφu ∘ (∇(u)', ∇(φ), θ)) ⊙ (∇(v)')))dΩ +
    ∫(∇(v)' ⊙ (∂Ψuθ ∘ (∇(u)', ∇(φ), θ)) * dθ)dΩ +
    ∫(∇(vφ) ⋅ ((∂Ψφu ∘ (∇(u)', ∇(φ), θ)) ⊙ (∇(du)')))dΩ +
    ∫(∇(vφ) ⋅ ((∂Ψφφ ∘ (∇(u)', ∇(φ), θ)) ⋅ ∇(dφ)))dΩ +
    ∫(∇(vφ) ⋅ ((∂Ψφθ ∘ (∇(u)', ∇(φ), θ)) * dθ))dΩ +
    ∫(κ * ∇(dθ) ⋅ ∇(vθ))dΩ
end



# ===================
# ThermoMech
# ===================

# Stagered strategy
# -----------------
function residual_TM(::CouplingStrategy{:staggered_M}, (u, θ), v, ∂Ψu, dΩ)
    return ∫(∇(v)' ⊙ (∂Ψu ∘ (∇(u)', θ)))dΩ
end

function residual_TM(::CouplingStrategy{:staggered_T}, (u, θ), vθ, κ, dΩ)
    return ∫(κ * ∇(θ) ⋅ ∇(vθ))dΩ
end


function jacobian_TM(::CouplingStrategy{:staggered_M}, (u, θ), du, v, ∂Ψuu, dΩ)
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (∇(u)', θ)) ⊙ (∇(du)')))dΩ
end


function jacobian_TM(::CouplingStrategy{:staggered_T}, dθ, vθ, κ, dΩ)
    ∫(κ * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian_TM(::CouplingStrategy{:staggered_T}, (u, θ), dθ, vθ, κ, dΩ)
    ∫((κ ∘ (u, θ)) * ∇(dθ) ⋅ ∇(vθ))dΩ
end


# Monolithic strategy
# -------------------

function residual_TM(::CouplingStrategy{:monolithic}, (u, θ), (v, vθ), ∂Ψu, κ, dΩ)
    residual_TM(CouplingStrategy{:staggered_M}(), (u, θ), v, ∂Ψu, dΩ) +
    residual_TM(CouplingStrategy{:staggered_T}(), (u, θ), vθ, κ, dΩ)
end

function jacobian_TM(::CouplingStrategy{:monolithic}, (u, θ), (du, dθ), (v, vθ), (∂Ψuu, ∂Ψuθ), κ, dΩ)
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (∇(u)', θ)) ⊙ (∇(du)')))dΩ +
    ∫(∇(v)' ⊙ (∂Ψuθ ∘ (∇(u)', θ)) * dθ)dΩ +
    ∫(κ * ∇(dθ) ⋅ ∇(vθ))dΩ
end

# ===================
# ElectroMechanics
# ===================

# Stagered strategy
# -----------------

function residual_EM(::CouplingStrategy{:staggered_M}, (u, φ), v, ∂Ψu, dΩ)
    ∫((∇(v)' ⊙ (∂Ψu ∘ (∇(u)', ∇(φ)))))dΩ
end

function residual_EM(::CouplingStrategy{:staggered_E}, (u, φ), vφ, ∂Ψφ, dΩ)
    ∫((∇(vφ) ⋅ (∂Ψφ ∘ (∇(u)', ∇(φ)))))dΩ
end

function const_body_force((u, φ), v, f₀, dΩ)
    ∫(f₀⋅v)dΩ
end

function jacobian_EM(::CouplingStrategy{:staggered_M}, (u, φ), du, v, ∂Ψuu, dΩ)
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (∇(u)', ∇(φ))) ⊙ (∇(du)')))dΩ
end

function jacobian_EM(::CouplingStrategy{:staggered_E}, (u, φ), dφ, vφ, ∂Ψφφ, dΩ)
    ∫(∇(vφ)' ⋅ ((∂Ψφφ ∘ (∇(u)', ∇(φ))) ⋅ ∇(dφ)))dΩ
end


# Monolithic strategy
# -------------------

function residual_EM(::CouplingStrategy{:monolithic}, (u, φ), (v, vφ), (∂Ψu, ∂Ψφ), dΩ)
    residual_EM(CouplingStrategy{:staggered_M}(), (u, φ), v, ∂Ψu, dΩ) +
    residual_EM(CouplingStrategy{:staggered_E}(), (u, φ), vφ, ∂Ψφ, dΩ)
end

function residual_EM(::CouplingStrategy{:monolithic}, (u, φ), (v, vφ), (∂Ψu, ∂Ψφ), f₀, dΩ)
    residual_EM(CouplingStrategy{:staggered_M}(), (u, φ), v, ∂Ψu, dΩ) +
    residual_EM(CouplingStrategy{:staggered_E}(), (u, φ), vφ, ∂Ψφ, dΩ) -
    const_body_force((u, φ), v, f₀, dΩ)
end

function residual_EM(::CouplingStrategy{:monolithic}, (u, φ), (v, vφ), (∂Ψu, ∂Ψφ), f₀::Nothing, dΩ)
    residual_EM(CouplingStrategy{:staggered_M}(), (u, φ), v, ∂Ψu, dΩ) +
    residual_EM(CouplingStrategy{:staggered_E}(), (u, φ), vφ, ∂Ψφ, dΩ)
end


function jacobian_EM(::CouplingStrategy{:monolithic}, (u, φ), (du, dφ), (v, vφ), (∂Ψuu, ∂Ψφu, ∂Ψφφ), dΩ)
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (∇(u)', ∇(φ))) ⊙ (∇(du)')))dΩ +
    ∫(∇(dφ) ⋅ ((∂Ψφu ∘ (∇(u)', ∇(φ))) ⊙ (∇(v)')))dΩ +
    ∫(∇(vφ) ⋅ ((∂Ψφu ∘ (∇(u)', ∇(φ))) ⊙ (∇(du)')))dΩ +
    ∫(∇(vφ) ⋅ ((∂Ψφφ ∘ (∇(u)', ∇(φ))) ⋅ ∇(dφ)))dΩ
end


end