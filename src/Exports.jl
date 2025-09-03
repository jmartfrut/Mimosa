
macro publish(mod, name)
  quote
    using Mimosa.$mod: $name
    export $name
  end
end


@publish TensorAlgebra (*)
@publish TensorAlgebra (×ᵢ⁴)
@publish TensorAlgebra (⊗₁₂³)
@publish TensorAlgebra (⊗₁₃²)
@publish TensorAlgebra (⊗₁²³)
@publish TensorAlgebra (⊗₁₃²⁴)
@publish TensorAlgebra (⊗₁₂³⁴)
@publish TensorAlgebra (⊗₁²)


@publish ConstitutiveModels logreg
@publish ConstitutiveModels DerivativeStrategy
@publish ConstitutiveModels LinearElasticity3D
@publish ConstitutiveModels NeoHookean3D
@publish ConstitutiveModels MoneyRivlin3D
@publish ConstitutiveModels ThermalModel
@publish ConstitutiveModels IdealDielectric
@publish ConstitutiveModels ElectroMech
@publish ConstitutiveModels ThermoElectroMech
@publish ConstitutiveModels ThermoMech
@publish ConstitutiveModels Visco
@publish ConstitutiveModels ViscoElastic
@publish ConstitutiveModels ViscousIncompressible
@publish ConstitutiveModels GeneralizedMaxwell
@publish ConstitutiveModels IncompressibleNeoHookean3D
@publish ConstitutiveModels initializeStateVariables
@publish ConstitutiveModels updateStateVariables!

@publish WeakForms CouplingStrategy
@publish WeakForms residual_EM
@publish WeakForms jacobian_EM
@publish WeakForms residual_M
@publish WeakForms jacobian_M
@publish WeakForms residual_TEM
@publish WeakForms jacobian_TEM
@publish WeakForms  residual_TM
@publish WeakForms  jacobian_TM

@publish BoundaryConditions  DirichletBC
@publish BoundaryConditions  MultiFieldBoundaryCondition

@publish Drivers Problem
@publish Drivers execute
@publish Drivers ElectroMechProblem
@publish Drivers MechanicalProblem
@publish Drivers ThermoElectroMechProblem
@publish Drivers get_problem

@publish Solvers IterativeSolver
# @publish LinearSolvers solve
# @publish LinearSolvers solve!


