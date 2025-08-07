using Gridap.Arrays
using Gridap.TensorValues
using Test

n_times = 1e6

A = TensorValue(1, 2, 3, 4, 5, 6, 7, 8, 9)
S = get_array(A)

@testset "TensorAlgebra" verbose=true begin

@testset "TensorValueSum" begin
    for _ in 1:n_times
       A+A 
    end
    @test true
end

@testset "SMatrixSum" begin
    for _ in 1:n_times
        S+S
    end
    @test true
end

@testset "TensorValueProd" begin
    for _ in 1:n_times
        A ⋅ A
    end
    @test true
end

@testset "SMatrixProd" begin
    for _ in 1:n_times
        S*S
    end
    @test true
end

end   # @testset "TensorAlgebra"
