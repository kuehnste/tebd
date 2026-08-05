using Test
using LinearAlgebra
using MatrixProductStates

@testset "Random MPS and basis states" begin
    # Use left canonical form
    for N = 10:2:20
        mps_real = random_mps_obc(N, 5, 3, Float64)
        mps_complex = random_mps_obc(N, 5, 3, ComplexF64)
        gaugeMPS!(mps_real, :left, true)
        gaugeMPS!(mps_complex, :left, true)
        @test isapprox(calculate_overlap(mps_real, mps_real), 1.0)
        @test isapprox(calculate_overlap(mps_complex, mps_complex), 1.0 + 0.0im)
    end
    # Use right canonical form
    for N = 10:2:20
        mps_real = random_mps_obc(N, 5, 3, Float64)
        mps_complex = random_mps_obc(N, 5, 3, ComplexF64)
        gaugeMPS!(mps_real, :right, true)
        gaugeMPS!(mps_complex, :right, true)
        @test isapprox(calculate_overlap(mps_real, mps_real), 1.0)
        @test isapprox(calculate_overlap(mps_complex, mps_complex), 1.0 + 0.0im)
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
        @test isapprox(calculate_overlap(mps, mps), 1.0)
    end
end

@testset "Identity operator" begin
    for N = 10:2:20
        for d = 2:5
            mps_complex = random_mps_obc(N, 5, d, ComplexF64)
            gaugeMPS!(mps_complex, :left, true)
            id_mpo = getIdentityMPO(N, d)
            mps_operator_applied = apply_operator(id_mpo, mps_complex)
            @test isapprox(calculate_overlap(mps_complex, mps_operator_applied), 1.0 + 0.0im)
        end
    end
end

@testset "Canonical form" begin
    mps = random_mps_obc(10, 5, 2, ComplexF64)
    mps_left_gauged = gaugeMPS(mps, :left, true)
    mps_right_gauged = gaugeMPS(mps, :right, true)
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
    H = getHeisenbergMPO(10, 1, 1)
    @test_throws InexactError apply_operator!(H, mps)

    # Compute result of the contraction and store in a new MPS, compare it with result of inplace contraction
    for N = 10:2:20
        # Prepare a random MPS and gauge it
        mps = random_mps_obc(N, 15, 2, ComplexF64)
        gaugeMPS!(mps, :left, true)
        # Some MPOs for testing
        H = getIsingMPO(N, 1.0, 0.9)
        # Apply the Hamiltonian generating a copy
        mps_new = apply_operator(H, mps)
        gaugeMPS!(mps_new, :left, true)
        # Now in place
        apply_operator!(H, mps)
        gaugeMPS!(mps, :left, true)
        # Check the expectation values
        @test isapprox(calculate_overlap(mps, mps_new), 1.0 + 0.0im)
    end
end

@testset "Summing MPSs" begin
    for N = 10:2:20
        # Prepare a random MPS and gauge it
        mps = random_mps_obc(N, 15, 2, ComplexF64)
        gaugeMPS!(mps, :left, true)
        # Sum it with itself
        mps2 = sum_states(mps, mps)
        # Check the expectation values
        @test isapprox(calculate_overlap(mps2, mps), 2.0)
        @test isapprox(calculate_overlap(mps2, mps2), 4.0)
    end
end

@testset "Summing MPOs" begin
    for N = 10:2:20
        # Prepare a random MPS and gauge it
        mps = random_mps_obc(N, 15, 2, ComplexF64)
        gaugeMPS!(mps, :left, true)
        # Some MPOs for testing
        id_mpo = getIdentityMPO(N, 2)
        H = getIsingMPO(N, 1.0, 0.9)
        # Compute sums thereof
        O1 = sum_operators(id_mpo, H)
        O2 = sum_operators(H, H)
        # Check the expectation values
        @test isapprox(expectation_value(mps, O1), 1.0 + expectation_value(mps, H))
        @test isapprox(expectation_value(mps, O2), 2.0 * expectation_value(mps, H))
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

    # Now try with the GHZ state on 10 sites
    all_zeros = basis_state_obc(ones(Int64, 10))
    all_ones = basis_state_obc(2 * ones(Int64, 10))
    mps_ghz = sum_states(all_zeros, all_ones)
    gaugeMPS!(mps_ghz, :left, true)
    v1 = zeros(2^10)
    v1[1] = 1
    v2 = zeros(2^10)
    v2[end] = 1
    state_vector_ghz = 1 / sqrt(2) * (v1 + v2)
    state_vector_mps = contract_virtual_indices(mps_ghz)
    @test isapprox(state_vector_ghz' * state_vector_mps, 1.0)
end

@testset "Contracting virtual indices of an MPO" begin
    # Test with the identity
    id_mpo = getIdentityMPO(5, 2)
    id_mpo_matrix = contract_virtual_indices(id_mpo)
    @test isapprox(id_mpo_matrix, Matrix(1.0I, 2^5, 2^5))

    # A simple instance of the Ising Hamiltonian
    J = 0.9
    lambda = 1.1
    ising_mpo = getIsingMPO(4, J, lambda)
    Id, X, _, Z = getPauliMatrices()
    Hising = -J * (kron(X, X, Id, Id) + kron(Id, X, X, Id) + kron(Id, Id, X, X)) - lambda * (kron(Z, Id, Id, Id) + kron(Id, Z, Id, Id) + kron(Id, Id, Z, Id) + kron(Id, Id, Id, Z))
    Hising_mpo = contract_virtual_indices(ising_mpo)
    @test isapprox(Hising, Hising_mpo)
end

@testset "Test entropy computation" begin
    # A product state should always have vanishing entropy
    mps = random_mps_obc(10, 1, 2)
    gaugeMPS!(mps, :left, true)
    for i = 1:length(mps)
        @test abs(compute_entropy(mps, i)) < 1E-12
    end
    @test abs(compute_entropy(mps)) < 1E-12

    # Now prepare the GHZ state
    all_zeros = basis_state_obc(ones(Int64, 10))
    all_ones = basis_state_obc(2 * ones(Int64, 10))
    mps = sum_states(all_zeros, all_ones)
    gaugeMPS!(mps, :left, true)
    for i = 1:length(mps)-1
        @test isapprox(compute_entropy(mps, i), 1.0)
    end

    # Compare the entropy computation with the default argument to what one would expect
    mps = random_mps_obc(11, 4, 3)
    entropy1 = compute_entropy(mps)
    entropy2 = compute_entropy(mps, 6)
    @test isapprox(entropy1, entropy2)

    mps = random_mps_obc(8, 6, 3)
    entropy1 = compute_entropy(mps)
    entropy2 = compute_entropy(mps, 4)
    @test isapprox(entropy1, entropy2)
end

@testset "Compressing an MPS via SVD" begin
    mps = basis_state_obc([1; 2; 1; 2; 1; 2])
    @test_throws AssertionError svd_compress_mps(mps, 0, 0.0)

    # Generate an empty state corresponding to a 0
    mps = random_mps_obc(4, 1, 2)
    mps[1] = 0.0 * mps[1]
    # Build an equal weight superposition of all basis states which is a simple product state
    for i1 = 1:2
        for i2 = 1:2
            for i3 = 1:2
                for i4 = 1:2
                    tmp = basis_state_obc([i1; i2; i3; i4], 2)
                    mps = sum_states(mps, tmp)
                end
            end
        end
    end
    # Now we compress the MPS, which is exactly represented by a MPS with bond dimension one
    res = svd_compress_mps(mps, 1)
    res2 = svd_compress_mps(mps, 0, 1E-6)
    res3 = svd_compress_mps(mps, 1, 1E-6)
    # The norm will be 16, as we added all 16 configurations without normalizing the result
    @test isapprox(calculate_overlap(mps, res), 16.0)
    @test isapprox(calculate_overlap(mps, res2), 16.0)
    @test isapprox(calculate_overlap(mps, res3), 16.0)
end

@testset "Sampling from an MPS" begin
    # Some experiments with product states for which the outcome has to be deterministic
    for N = 4:6
        configuration = rand([1; 2; 3], N)
        mps = basis_state_obc(configuration, 3)
        for i = 1:10
            sample = sample_from_mps(mps)
            @test isequal(sample, configuration)
        end
    end
    # Now prepare the GHZ state
    all_zeros = basis_state_obc(ones(Int64, 4))
    all_ones = basis_state_obc(2 * ones(Int64, 4))
    mps_ghz = sum_states(all_zeros, all_ones)
    gaugeMPS!(mps_ghz, :left, true)
    prob = zeros(2)
    nsamples = 100000
    # Check the outcome for the first 10 samples
    for i = 1:10
        sample = sample_from_mps(mps_ghz)
        @test (sample == [1; 1; 1; 1] || sample == [2; 2; 2; 2])
        if sample == [1; 1; 1; 1]
            prob[1] += 1
        else
            prob[2] += 1
        end
    end
    # For efficiency do not check the other ones anymore
    for i = 11:nsamples
        sample = sample_from_mps(mps_ghz)
        if sample == [1; 1; 1; 1]
            prob[1] += 1
        else
            prob[2] += 1
        end
    end
    @test (sum(abs.(prob ./ nsamples - 0.5 * ones(2))) < 1E-2)
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
