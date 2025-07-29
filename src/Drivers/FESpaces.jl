# ===================
# ElectroMechProblem
# ===================

function get_FE_spaces(::ElectroMechProblem{:monolithic},
    model,
    order::Int64,
    bconds::MultiFieldBoundaryCondition; constraint=nothing)

    # Reference FE
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    reffeφ = ReferenceFE(lagrangian, Float64, order)

    # Test FE Spaces
    # println(bconds.BoundaryCondition[1].tags)
    # println(bconds.BoundaryCondition[1].masks)
    Vu = TestFESpace(model, reffeu, dirichlet_tags=bconds.BoundaryCondition[1].tags,dirichlet_masks=bconds.BoundaryCondition[1].masks, conformity=:H1)
    # Vu = TestFESpace(model, reffeu, dirichlet_tags=["point_zy","point_z","fixedup_1"],dirichlet_masks=[(true,true,true),(true,true,false),(true,false,false)], conformity=:H1)
    # println(Vu)
    Vφ = TestFESpace(model, reffeφ, dirichlet_tags=bconds.BoundaryCondition[2].tags, conformity=:H1)

    # Trial FE Spaces
    # println(map(f -> f(1.0), bconds.BoundaryCondition[1].values))
    Uu = TrialFESpace(Vu,  map(f -> f(1.0), bconds.BoundaryCondition[1].values)) #[VectorValue(0.0, 0.0, 0.0),VectorValue(0.0, 0.0, 0.0),VectorValue(0.0, 0.0, 0.0)]) # map(f -> f(1.0), bconds.BoundaryCondition[1].values))
    Uφ = TrialFESpace(Vφ, map(f -> f(1.0), bconds.BoundaryCondition[2].values))

    # Multifield FE Spaces
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([Uu, Uφ])

    return @ntuple Vu Vφ Uu Uφ V U
end

function get_FE_spaces!(::ElectroMechProblem{:monolithic},
    fe_spaces,
    bconds::MultiFieldBoundaryCondition, Λ)

    @unpack Vu, Vφ = fe_spaces

    # Trial FE Spaces
    Uu = TrialFESpace(Vu, map(f -> f(1.0), bconds.BoundaryCondition[1].values)) #[VectorValue(0.0, 0.0, 0.0),VectorValue(0.0, 0.0, 0.0),VectorValue(0.0, 0.0, 0.0)])  # map(f -> f(1.0), bconds.BoundaryCondition[1].values))
    Uφ = TrialFESpace(Vφ, map(f -> f(Λ), bconds.BoundaryCondition[2].values))

    # Multifield FE Spaces
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([Uu, Uφ])

    fe_spaces = @ntuple Vu Vφ Uu Uφ V U
end

# ========================
# ThermoElectroMechProblem
# ========================

function get_FE_spaces(::ThermoElectroMechProblem{:monolithic},
    model,
    order::Int64,
    bconds::MultiFieldBoundaryCondition; constraint=nothing)

    # Reference FE
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    reffeφ = ReferenceFE(lagrangian, Float64, order)
    reffeθ = ReferenceFE(lagrangian, Float64, 1)

    # Test FE Spaces
    Vu = TestFESpace(model, reffeu, dirichlet_tags=bconds.BoundaryCondition[1].tags, conformity=:H1)
    Vφ = TestFESpace(model, reffeφ, dirichlet_tags=bconds.BoundaryCondition[2].tags, conformity=:H1)
    Vθ = TestFESpace(model, reffeθ, dirichlet_tags=bconds.BoundaryCondition[3].tags, conformity=:H1)

    # Trial FE Spaces
    Uu = TrialFESpace(Vu, map(f -> f(1.0), bconds.BoundaryCondition[1].values))
    Uφ = TrialFESpace(Vφ, map(f -> f(1.0), bconds.BoundaryCondition[2].values))
    Uθ = TrialFESpace(Vθ, map(f -> f(1.0), bconds.BoundaryCondition[3].values))

    # Multifield FE Spaces
    V = MultiFieldFESpace([Vu, Vφ, Vθ])
    U = MultiFieldFESpace([Uu, Uφ, Uθ])

    return @ntuple Vu Vφ Vθ Uu Uφ Uθ V U
end

function get_FE_spaces!(::ThermoElectroMechProblem{:monolithic},
    fe_spaces,
    bconds::MultiFieldBoundaryCondition, Λ)

    @unpack Vu, Vφ, Vθ = fe_spaces

    # Trial FE Spaces
    Uu = TrialFESpace(Vu, map(f -> f(Λ), bconds.BoundaryCondition[1].values))
    Uφ = TrialFESpace(Vφ, map(f -> f(Λ), bconds.BoundaryCondition[2].values))
    Uθ = TrialFESpace(Vθ, map(f -> f(Λ), bconds.BoundaryCondition[3].values))

    # Multifield FE Spaces
    V = MultiFieldFESpace([Vu, Vφ, Vθ])
    U = MultiFieldFESpace([Uu, Uφ, Uθ])

    fe_spaces = @ntuple Vu Vφ Vθ Uu Uφ Uθ V U
end


# ========================
# MechanicalProblem
# ========================

function get_FE_spaces(::MechanicalProblem,
    model,
    order::Int64,
    bconds::DirichletBC; constraint=nothing)

    # Reference FE
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)

    # Test FE Spaces
    V = TestFESpace(model, reffeu, dirichlet_tags=bconds.tags, conformity=:H1)

    # Trial FE Spaces
    U = TrialFESpace(V, map(f -> f(1.0), bconds.values))

    return @ntuple V U
end

function get_FE_spaces!(::MechanicalProblem,
    fe_spaces,
    bconds::DirichletBC, Λ)

    @unpack V = fe_spaces

    # Trial FE Spaces
    U = TrialFESpace(V, map(f -> f(Λ), bconds.values))

    fe_spaces = @ntuple V U
end




# @generated function residual_Neumann(model, tags_::Vector{String}, degree)
#     str = ""
#     for i in eachindex(tags_)
#     Γ= BoundaryTriangulation(model, tags=tags_[i])
#     dΓ= Measure(Γ, degree)
#     str *= "∫(v⋅neumannbc[i]) dΓ"
#     end
#     Meta.parse(str)

# end