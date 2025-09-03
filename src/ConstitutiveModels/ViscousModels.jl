

using ..TensorAlgebra
using ..TensorAlgebra: _δδ_μ_3D
using ..TensorAlgebra: _δδ_λ_3D
using ..TensorAlgebra: _δδ_μ_2D
using ..TensorAlgebra: _δδ_λ_2D
using ..TensorAlgebra: I3
using ..TensorAlgebra: I9

using .ConstitutiveModels
using .ConstitutiveModels: ViscousIncompressible


"""
  _getKinematic(::Visco, ∇u, Uv⁻¹)

Compute the kinematics of a viscous model.

# Arguments
- `::Visco`: A viscous model
- `∇u`: The deformation gradient at the considered time step
- `Uv⁻¹`: The inverse of the viscous strain at the considered time step

# Returns
- `F`
- `C`
- `Ce`
"""
function _getKinematic(::Visco, ∇u, Uv⁻¹)
  F = one(∇u) + ∇u
  C = F' * F
  Ce = Uv⁻¹ * C * Uv⁻¹
  return (F, C, Ce)
end


"""
  ViscousStrain(Ce::SMatrix, C::SMatrix)::SMatrix
  
Get viscous Uv and its inverse.

# Arguments
- `Ce`
- `C`

# Return
- `Ue`
- `Uv`
- `invUv`
"""
function ViscousStrain(Ce, C)
  Ue = msqrt(Ce)
  Ue_C_Ue = Ue * C * Ue
  invUe = inv(Ue)
  Uv = invUe * msqrt(Ue_C_Ue) * invUe
  invUv = inv(Uv)
  return Ue, Uv, invUv
end


"""
  return_mapping_algorithm!

Compute the elastic Cauchy deformation tensor and the incompressibility condition.

# Arguments
- `obj::ViscousIncompressible`: The viscous model
- `Δt::Float64`: Time step
- `Se_::Function`: Elastic 2nd Piola-Kirchhoff stress (function of C)    
- `∂Se_∂Ce_::Function`: Derivatives of elastic 2nd Piola-Kirchhoff stress (function of C)  
- `F`: Deformation gradient
- `Ce_trial`: Elastic right Green-Cauchy at intermediate statep
- `Ce`: Elastic right Green-Cauchy deformation tensor
- `λα`: incompressibility constraint (Lagrange multiplier)

# Return
- `Ce`
- `λα`
"""
function return_mapping_algorithm!(obj::ViscousIncompressible, Δt::Float64,
                            Se_::Function, ∂Se∂Ce_::Function,
                            F, Ce_trial, Ce, λα)
  γα = obj.τ / (obj.τ + Δt)
  Se_trial = Se_(Ce_trial)
  res, ∂res = JacobianReturnMapping(γα, Ce, Se_(Ce), Se_trial, ∂Se∂Ce_(Ce), F, λα)
  maxiter = 20
  tol = 1e-6
  for _ in 1:maxiter
    #---------- Update -----------#
    Δu = -∂res \ res[:]
    Ce += reshape(Δu[1:end-1], 3, 3)
    λα += Δu[end]
    #---- Residual and jacobian ---------#
    res, ∂res = JacobianReturnMapping(γα, Ce, Se_(Ce), Se_trial, ∂Se∂Ce_(Ce), F, λα)
    #---- Monitor convergence ---------#
    if norm(res) < tol
      break
    end
  end
  return Ce, λα
end


"""
Residual of the return mapping algorithm and 
its Jacobian with respect to {Ce,λα} for 
incompressible case

# Arguments

# Return
- `res`
- `∂res`
"""
function JacobianReturnMapping(γα, Ce, Se, Se_trial, ∂Se∂Ce, F, λα)
    detCe = det(Ce)
    Ge = cof(Ce)
    #--------------------------------
    # Residual
    #--------------------------------   
    res1 = Se - γα * Se_trial - (1-γα) * λα * Ge
    res2 = detCe - (det(F))^2
    #--------------------------------
    # Derivatives of residual
    #--------------------------------
    ∂res1_∂Ce = ∂Se∂Ce - (1-γα) * λα * Cross_I4_A(Ce)
    ∂res1_∂λα = -(1-γα) * Ge
    ∂res2_∂Ce = Ge
    res = [res1[:]; res2]
    ∂res = MMatrix{10,10}(zeros(10, 10))
    ∂res[1:9, 1:9] = ∂res1_∂Ce
    ∂res[1:9, 10] = ∂res1_∂λα[:]
    ∂res[10, 1:9] = (∂res2_∂Ce[:])'
    return res, ∂res
end


"""
  ViscousPiola(Se::Function, Ce::SMatrix, invUv::SMatrix, F::SMatrix)::SMatrix

Viscous 1st Piola-Kirchhoff stress

# Arguments
- `Se` Elastic Piola (function of C)
- `Ce` Elastic right Green-Cauchy deformation tensor
- `invUv` Inverse of viscous strain
- `F` Deformation gradient

# Return
- `Pα::SMatrix`
"""
function ViscousPiola(Se::Function, Ce::SMatrix, invUv::SMatrix, F::SMatrix)
    Sα = invUv * Se(Ce) * invUv
    F * Sα
end


"""
  ∂Ce_∂C(::ViscousIncompressible, γα, ∂Se_∂Ce_, invUvn, Ce, Ce_trial, λα, F)

Tangent operator of Ce for the incompressible case

# Arguments
- `::ViscousIncompressible` The viscous model
- `γα`: Characteristic time τα / (τα + Δt)
- `∂Se∂Ce_`: Function of C
- ...

# Return
- `∂Ce∂C`
"""
function ∂Ce_∂C(::ViscousIncompressible, γα, ∂Se∂Ce_, invUvn, Ce, Ce_trial, λα, F)
    C = F' * F
    G = cof(C)
    Ge = cof(Ce)
    ∂Se∂Ce = ∂Se∂Ce_(Ce)
    ∂Se∂Ce_trial = ∂Se∂Ce_(Ce_trial)
    ∂Ce_trial_∂C = Outer_13_24(invUvn, invUvn)
    #------------------------------------------
    # Derivative of return mapping with respect to Ce and λα
    #------------------------------------------   
    K11 = ∂Se∂Ce - (1-γα) * λα * Cross_I4_A(Ce)
    K12 = -(1-γα) * Ge
    K21 = Ge
    #------------------------------------------
    # Derivative of return mapping with respect to C
    #------------------------------------------   
    F1 = γα * ∂Se∂Ce_trial * ∂Ce_trial_∂C
    F2 = G
    #------------------------------------------
    # Derivative of {Ce,λα} with respect to C
    #------------------------------------------   
    K = MMatrix{10,10}(zeros(10, 10))
    K[1:9, 1:9] = K11
    K[1:9, 10] = K12[:]
    K[10, 1:9] = (K21[:])'
    F = [F1; (F2[:])']
    ∂u∂C = K \ F
    ∂Ce∂C = ∂u∂C[1:9, 1:9]
    return ∂Ce∂C
end


"""
Tangent operator of Ce for at fixed Uv
"""
function ∂Ce_∂C_Uvfixed(invUv)
  return Outer_13_24(invUv, invUv)
end


"""
∂Ce∂(Uv^{-1})
"""
function ∂Ce_∂invUv(C, invU)
  invU_C = invU * C
  Id = SMatrix{3,3}(I)
  Outer_13_24(invU_C, Id) + Outer_13_24(Id, invU_C)
end


"""
  ViscousTangentOperator

Tangent operator for the incompressible case

# Arguments
- `obj::ViscousIncompressible`
- `Δt::Float64`: Time step
- `Se_::Function`: Function of C
- `∂Se∂Ce_::Function`: Function of C
- `F`: Deformation tensor
- `Ce_trial`: Right Green-Cauchy deformation tensor at intermediate step
- `Ce`: Right Green-Cauchy deformation tensor at curent step
- `invUv`
- `invUvn`
- `λα`

# Return
- `Cv::SMatrix{9,9}`: A fourth-order tensor in flattened notation
"""
function ViscousTangentOperator(obj::ViscousIncompressible, Δt::Float64,
                  Se_::Function, ∂Se∂Ce_::Function,
                  F, Ce_trial, Ce, invUv, invUvn, λα)
  # -----------------------------------------
  # Characteristic time
  #------------------------------------------
  γα = obj.τ / (obj.τ + Δt)
  #------------------------------------------
  # Extract τv, Δt, μv
  #------------------------------------------  
  C = F' * F
  DCe_DC = ∂Ce_∂C(obj, γα, ∂Se∂Ce_, invUvn, Ce, Ce_trial, λα, F)
  DCe_DC_Uvfixed = ∂Ce_∂C_Uvfixed(invUv)
  DCe_DinvUv = ∂Ce_∂invUv(C, invUv)
  DinvUv_DC = inv(DCe_DinvUv) * (DCe_DC - DCe_DC_Uvfixed)
  I2 = SMatrix{3,3}(I)
  DCDF = Outer_13_24(F', I2) + Outer_14_23(I2, F')
  #------------------------------------------
  # 0.5*δC_{Uvfixed}:DSe[ΔC]
  #------------------------------------------
  C1 = 0.5 * DCe_DC_Uvfixed' * ∂Se∂Ce_(Ce) * DCe_DC
  #------------------------------------------
  # Se:0.5*(DUv^{-1}[ΔC]*δC*Uv^{-1} + Uv^{-1}*δC*DUv^{-1}[ΔC])
  #------------------------------------------
  invUv_Se = invUv * Se_(Ce)
  C2 = 0.5 * (Contraction_IP_JPKL(invUv_Se, DinvUv_DC) +
              Contraction_IP_PJKL(invUv_Se, DinvUv_DC))
  #------------------------------------------
  # Sv:(D(δC_{Uvfixed})[ΔC])
  #------------------------------------------
  Sv = invUv_Se * invUv
  C3 = Outer_13_24(Sv, I2)
  #------------------------------------------
  # Total Contribution
  #------------------------------------------
  Cv = DCDF' * (C1 + C2) * DCDF + C3
  return Cv
end


"""
  First Piola-Kirchhoff for the incompressible case

# Arguments
- `obj::ViscousIncompressible`: The visous model
- `Δt`: Current time step
- `Se_`: Elastic 2nd Piola (function of C)
- `∂Se∂Ce_`: 2nd Piola Derivatives (function of C)
- `∇u_`: Current deformation gradient
- `∇un_`: Previous deformation gradient
- `stateVars`: State variables (Uvα and λα)

# Return
- `Pα::Gridap.TensorValues.TensorValue`
"""
function Piola(obj::ViscousIncompressible, Δt::Float64,
                Se_::Function, ∂Se∂Ce_::Function,
                ∇u_::TensorValue, ∇un_::TensorValue, stateVars::VectorValue)
  state_vars = get_array(stateVars)
  Uvn = SMatrix{3,3}(state_vars[1:9])
  λαn = state_vars[10]
  ∇u = get_array(∇u_)
  ∇un = get_array(∇un_)
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn  = inv(Uvn)
  F, C, Ceᵗʳ = _getKinematic(obj, ∇u, invUvn)
  _, _, Cen  = _getKinematic(obj, ∇un, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, _ = return_mapping_algorithm!(obj, Δt, Se_, ∂Se∂Ce_, F, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Get invUv and Sα
  #------------------------------------------
  _, _, invUv = ViscousStrain(Ce, C)
  Pα = ViscousPiola(Se_, Ce, invUv, F)
  #------------------------------------------
  # Tangent operator
  #------------------------------------------
  return TensorValue(Pα)
end


"""
Visco-Elastic model for incompressible case

# Arguments
- `obj::ViscousIncompressible`: The visous model
- `Δt`: Current time step
- `Se_`: Elastic 2nd Piola (function of C)
- `∂Se∂Ce_`: 2nd Piola Derivatives (function of C)
- `∇u_`: Current deformation gradient
- `∇un_`: Previous deformation gradient
- `stateVars`: State variables (Uvα and λα)

# Return
- `Cα::Gridap.TensorValues.TensorValue`
"""
function Tangent(obj::ViscousIncompressible, Δt::Float64,
                 Se_::Function, ∂Se∂Ce_::Function,
                 ∇u_::TensorValue, ∇un_::TensorValue, stateVars::VectorValue)
  state_vars = get_array(stateVars)
  Uvn = SMatrix{3,3}(state_vars[1:9])
  λαn = state_vars[10]
  ∇u = get_array(∇u_)
  ∇un = get_array(∇un_)
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn  = inv(Uvn)
  F, C, Ceᵗʳ = _getKinematic(obj, ∇u, invUvn)
  _, _, Cen  = _getKinematic(obj, ∇un, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, λα = return_mapping_algorithm!(obj, Δt, Se_, ∂Se∂Ce_, F, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Get invUv and Sα
  #------------------------------------------
  _, _, invUv = ViscousStrain(Ce, C)
  #------------------------------------------
  # Tangent operator
  #------------------------------------------
  Cα = ViscousTangentOperator(obj, Δt, Se_, ∂Se∂Ce_, F, Ceᵗʳ, Ce, invUv, invUvn, λα)
  return TensorValue(Cα)
end


"""
    Return mapping for the incompressible case

    # Arguments
    - `::ViscousIncompressible`
    - `Δt::Float64`: Time step
    - `Se_::Function`: Elastic Piola (function of C)
    - `∂Se∂Ce_::Function`: Piola Derivatives (function of C)
    - `∇u_::TensorValue`
    - `∇un_::TensorValue`
    - `stateVars::VectorValue`: State variables (10-component vector gathering Uvα and λα)

    # Return
    - `::bool`: indicates whether the state variables should be updated
    - `::VectorValue`: State variables at new time
"""
function ReturnMapping(obj::ViscousIncompressible, Δt::Float64,
                       Se_::Function, ∂Se∂Ce_::Function,
                       ∇u_::TensorValue, ∇un_::TensorValue, stateVars::VectorValue)
  state_vars = get_array(stateVars)
  Uvn = SMatrix{3,3}(state_vars[1:9])
  λαn = state_vars[10]
  ∇u = get_array(∇u_)
  ∇un = get_array(∇un_)
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn  = inv(Uvn)
  F, C, Ceᵗʳ = _getKinematic(obj, ∇u, invUvn)
  _, _, Cen  = _getKinematic(obj, ∇un, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, λα = return_mapping_algorithm!(obj, Δt, Se_, ∂Se∂Ce_, F, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Get Uv and λα
  #------------------------------------------
  _, Uv, _ = ViscousStrain(Ce, C)
  Cell_ = [Uv[:]; λα]
  return true, VectorValue(Cell_)
end


