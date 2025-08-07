using Mimosa
using Test

@testset "MimosaTests" verbose=true begin
    
    include("ConstitutiveModels/runtests.jl")

    include("TensorAlgebra/runtests.jl")

end;  # @testset "MimosaTests"
