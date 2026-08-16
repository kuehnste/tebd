############################################################################
# Find the ground state of the Ising Hamiltonian and get the magnetization #
############################################################################
include("../src/tensorfunctions.jl")
include("../src/MPS_OBC.jl")
include("../src/observables_and_evolution.jl")
include("../src/operators.jl")

let

    ##################################################################
    # Parameters of the model
    ##################################################################

    # Number of Spins
    N = 20
    # Bond Dimension
    D = 20
    # Physical dimension
    d = 2
    # Desired accuracy in the SVD compression
    acc = 1E-10
    # The number of time steps we take
    nsteps = 200
    # The size of each time step
    dt = 5E-2

    # Coupling between nearest neighbors
    J = 1.0
    # Coupling to external field
    lambda = 1.0

    ##################################################################
    # Construct the odd-even time evolution operators
    ##################################################################

    # The required Pauli matrices for building the evolution MPOs
    Id, X, _, Z = get_pauli_matrices()

    # Prepare the matrix representation for the different exponentials appearing in the odd-even decomposition in imaginary time
    Uloc_left_boundary = exp(-dt * (-J * kron(X, X) - lambda * kron(Z, Id) - 0.5 * lambda * kron(Id, Z)))
    Uloc_bulk = exp(-dt * (-J * kron(X, X) - 0.5 * lambda * (kron(Z, Id) + kron(Id, Z))))
    Uloc_right_boundary = exp(-dt * (-J * kron(X, X) - 0.5 * lambda * kron(Z, Id) - lambda * kron(Id, Z)))

    # Find the mpo representation for the local terms
    mpo_Uloc_left_boundary = decompose_into_mpo(Uloc_left_boundary, d)
    mpo_Uloc_bulk = decompose_into_mpo(Uloc_bulk, d)
    mpo_Uloc_right_boundary = decompose_into_mpo(Uloc_right_boundary, d)

    # Build the MPO for evolving the odd sites
    Uodd, Ueven = u_odd_even_mpo(N, mpo_Uloc_left_boundary, mpo_Uloc_right_boundary, mpo_Uloc_bulk)

    # Prepare the matrix representation for the different Hamiltonian terms
    Hloc_left_boundary = -J * kron(X, X) - lambda * kron(Z, Id) - 0.5 * lambda * kron(Id, Z)
    Hloc_bulk = -J * kron(X, X) - 0.5 * lambda * (kron(Z, Id) + kron(Id, Z))
    Hloc_right_boundary = -J * kron(X, X) - 0.5 * lambda * kron(Z, Id) - lambda * kron(Id, Z)

    # Find the mpo representation for the local terms
    mpo_Hloc_left_boundary = decompose_into_mpo(Hloc_left_boundary, d)
    mpo_Hloc_bulk = decompose_into_mpo(Hloc_bulk, d)
    mpo_Hloc_right_boundary = decompose_into_mpo(Hloc_right_boundary, d)

    ##################################################################
    # Running the actual evolution
    ##################################################################

    # Evolve in imaginary time starting from a random state
    psi = random_mps_obc(N, D, d, Float64)
    # To normalize and remove unnecessary parameters, we put the MPS in left and right canonical form
    psi = svd_compress_mps(psi, 0, 1E-15, true, direction=:left)
    psi = svd_compress_mps(psi, 0, 1E-15, direction=:right)
    for i = 1:nsteps
        # Apply the evolution operator using a first-order Suzuki Trotter approximation
        psi = apply_operator(Uodd, psi)
        psi = apply_operator(Ueven, psi)
        # Put the MPS in right canonical gauge
        psi = svd_compress_mps(psi, 0, 1E-15, direction=:right)
        # SVD compress and renormalize
        psi = svd_compress_mps(psi, D, acc, true, direction=:left)
        println("Step: ", i, ":  E = ", real(expectation_energy(psi, mpo_Hloc_left_boundary, mpo_Hloc_right_boundary, mpo_Hloc_bulk)))
    end
    # Compute the observables at the end
    E0 = expectation_energy(psi, mpo_Hloc_left_boundary, mpo_Hloc_right_boundary, mpo_Hloc_bulk)
    Sz = expectation_single_body_operator(psi, Z)

    # Print the results
    println(" ")
    println("Groundstate energy:            ", E0)
    println("Groundstate energy density:    ", E0 / N)
    println("Total spin:                    ", Sz)
    println(" ")

    # For comparison, we try with the MPO version
    Hmpo = get_ising_mpo(N, J, lambda)
    # Get the MPO for the total spin
    tot_z_mpo = get_total_pauli_z_mpo(N)
    println("Groundstate energy with MPO:   ", expectation_value(psi, Hmpo))
    println("Total spin with MPO:           ", expectation_value(psi, tot_z_mpo))

    nothing
end