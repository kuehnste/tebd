using Test

include("../src/tensorfunctions.jl")
include("../src/MPS_OBC.jl")
include("../src/operators.jl")

@testset "Random MPS and basis states" begin
    # Use left canonical form
    for N = 10:2:20
        mps_real = random_mps_obc(N, 5, 3, Float64)
        mps_complex = random_mps_obc(N, 5, 3, ComplexF64)
        mps_real = svd_compress_mps(mps_real, 0, 1E-12, true, direction=:left)
        mps_complex = svd_compress_mps(mps_complex, 0, 1E-12, true, direction=:left)
        @test isapprox(inner_product(mps_real, mps_real), 1.0)
        @test isapprox(inner_product(mps_complex, mps_complex), 1.0 + 0.0im)
    end
    # Use right canonical form
    for N = 10:2:20
        mps_real = random_mps_obc(N, 5, 3, Float64)
        mps_complex = random_mps_obc(N, 5, 3, ComplexF64)
        mps_real = svd_compress_mps(mps_real, 0, 1E-12, true, direction=:right)
        mps_complex = svd_compress_mps(mps_complex, 0, 1E-12, true, direction=:right)
        @test isapprox(inner_product(mps_real, mps_real), 1.0)
        @test isapprox(inner_product(mps_complex, mps_complex), 1.0 + 0.0im)
    end

    # Invalid configurations which should produce an error
    configuration = [0; 1; 2; 1; 2]
    @test_throws ArgumentError basis_state_obc(configuration)
    configuration = [1; 1; 3; 1; 2]
    @test_throws ArgumentError basis_state_obc(configuration)

    # Some valid configurations
    v = Vector{Vector{Float64}}(undef, 3)
    v[1] = [1.0; 0; 0]
    v[2] = [0; 1.0; 0]
    v[3] = [0; 0; 1.0]
    for N = 4:10
        configuration = rand([1; 2; 3], N)
        mps = basis_state_obc(configuration, 3)
        @test isapprox(inner_product(mps, mps), 1.0)
    end
end

@testset "Identity operator" begin
    for N = 10:2:20
        for d = 2:5
            mps_complex = random_mps_obc(N, 5, d, ComplexF64)
            mps_complex = svd_compress_mps(mps_complex, 0, 1E-12, true, direction = :right)
            id_mpo = get_identity_mpo(N, d)
            mps_operator_applied = apply_operator(id_mpo, mps_complex)
            @test isapprox(inner_product(mps_complex, mps_operator_applied), 1.0 + 0.0im)
        end
    end
end

@testset "Canonical form" begin
    mps = random_mps_obc(10, 5, 2, ComplexF64)
    mps_left_gauged = svd_compress_mps(mps, 0, 1E-12, true, direction = :left)
    mps_right_gauged = svd_compress_mps(mps, 0, 1E-12, true, direction = :right)
    # Check the left canonical gauge
    for i = 1:length(mps_left_gauged)
        Dr = size(mps_left_gauged[i], 2)
        res = contract_tensors(conj(mps_left_gauged[i]), [1; 3], mps_left_gauged[i], [1; 3])
        @test isapprox(res, Matrix((1.0 + 0.0im) * I, Dr, Dr))
    end
    # Check the right canonical gauge
    for i = 1:length(mps_right_gauged)
        Dl = size(mps_right_gauged[i], 1)
        res = contract_tensors(conj(mps_right_gauged[i]), [2; 3], mps_right_gauged[i], [2; 3])
        @test isapprox(res, Matrix((1.0 + 0.0im) * I, Dl, Dl))
    end
end

@testset "Inplace operator application" begin
    # Check that we get an error, if we try to overwrite a real MPS with a complex MPO
    mps = random_mps_obc(10, 9, 2, Float64)
    H =  get_heisenberg_mpo(10, 1, 1)
    @test_throws InexactError apply_operator!(H, mps)

    # Compute result of the contraction and store in a new MPS, compare it with result of inplace contraction
    for N = 10:2:20
        # Prepare a random MPS and gauge it
        mps = random_mps_obc(N, 15, 2, ComplexF64)
        mps = svd_compress_mps(mps, 0, 1E-12, true, direction = :left)
        # Some MPOs for testing
        H = get_ising_mpo(N, 1.0, 0.9)
        # Apply the Hamiltonian generating a copy
        mps_new = apply_operator(H, mps)
        mps_new = svd_compress_mps(mps_new, 0, 1E-12, true, direction = :left)
        # Now in place
        apply_operator!(H, mps)
        mps = svd_compress_mps(mps, 0, 1E-12, true, direction = :left)
        # Check the expectation values
        @test isapprox(inner_product(mps, mps_new), 1.0 + 0.0im)
    end
end

@testset "Basis states and contracting virtual indices of an MPS" begin
    # Some experiments with product states
    v = Vector{Vector{Float64}}(undef, 3)
    v[1] = [1.0; 0; 0]
    v[2] = [0; 1.0; 0]
    v[3] = [0; 0; 1.0]
    for N = 4:10
        # A MPS representing a simple basis state
        configuration = rand([1; 2; 3], N)
        mps = basis_state_obc(configuration, 3)
        # Construct the corresponding state vector
        state_vector = 1
        for i = 1:N
            state_vector = kron(state_vector, v[configuration[i]])
        end
        # Compare to the result from the MPS
        state_vector_mps = contract_virtual_indices(mps)
        @test isapprox(state_vector' * state_vector_mps, 1.0)
    end
end

@testset "Contracting virtual indices of an MPO" begin
    # Test with the identity
    id_mpo = get_identity_mpo(5, 2)
    id_mpo_matrix = contract_virtual_indices(id_mpo)
    @test isapprox(id_mpo_matrix, Matrix(1.0I, 2^5, 2^5))

    # A simple instance of the Ising Hamiltonian
    J = 0.9
    lambda = 1.1
    ising_mpo = get_ising_mpo(4, J, lambda)
    Id, X, _, Z = get_pauli_matrices()
    Hising = -J * (kron(X, X, Id, Id) + kron(Id, X, X, Id) + kron(Id, Id, X, X)) - lambda * (kron(Z, Id, Id, Id) + kron(Id, Z, Id, Id) + kron(Id, Id, Z, Id) + kron(Id, Id, Id, Z))
    Hising_mpo = contract_virtual_indices(ising_mpo)
    @test isapprox(Hising, Hising_mpo)
end

@testset "MPO decomposition" begin
    # Check if erroneous inputs are detected
    H = zeros(4, 5)
    @test_throws ArgumentError decompose_into_mpo(H, 2)
    H = zeros(5, 5)
    @test_throws ArgumentError decompose_into_mpo(H, [2; 3])
    # All local dimensions the same    
    for i = 1:5
        A = rand(2, 2)
        B = rand(2, 2)
        C = rand(2, 2)
        H = kron(A, B, C)
        mpo = decompose_into_mpo(H, 2)
        H_mpo = contract_virtual_indices(mpo)
        @test isapprox(H, H_mpo)
    end
    # Different local dimensions
    for i = 1:10
        A = rand(2, 2)
        B = rand(4, 4)
        C = rand(3, 3)
        H = kron(A, B, C)
        mpo = decompose_into_mpo(H, [2; 4; 3])
        H_mpo = contract_virtual_indices(mpo)
        @test isapprox(H, H_mpo)
    end
end
