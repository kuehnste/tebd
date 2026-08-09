############################################################################
# Find the ground state of the Ising Hamiltonian and get the magnetization #
############################################################################
include("../src/tensorfunctions.jl")
include("../src/MPS_OBC.jl")
include("../src/observables_and_evolution.jl")
include("../src/operators.jl")

"""
    get_maximum_bond_dimension(mps::MPS)

Get the maximum bond dimension of an MPS, i.e. maximum virtual dimension of the MPS tensors
"""
function maximum_bond_dimension(mps::MPS)
    Dmax = 0
    for i=1:length(mps)
        Dloc = max(size(mps[i], 1), size(mps[i], 2))
        if Dloc > Dmax
            Dmax = Dloc
        end
    end
    return Dmax
end

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
    acc = 1E-8
    # The number of time steps we take
    nsteps = 25
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
    Uloc_left_boundary = exp(-1.0im * dt * (-J * kron(X, X) - lambda * kron(Z, Id) - 0.5 * lambda * kron(Id, Z)))
    Uloc_bulk = exp(-1.0im * dt * (-J * kron(X, X) - 0.5 * lambda * (kron(Z, Id) + kron(Id, Z))))
    Uloc_right_boundary = exp(-1.0im * dt * (-J * kron(X, X) - 0.5 * lambda * kron(Z, Id) - lambda * kron(Id, Z)))

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
    # Energy and total Pauli-Z as MPO
    Hmpo = get_ising_mpo(N, J, lambda)
    tot_z_mpo = get_total_pauli_z_mpo(N)

    # Evolve in real starting from a random product state
    psi = random_mps_obc(N, 1, d, Float64)
    # To normalize and remove unnecessary parameters, we put the MPS in left and right canonical form
    psi = svd_compress_mps(psi, 0, 1E-15, true, direction=:left)
    psi = svd_compress_mps(psi, 0, 1E-15, direction=:right)
    for i = 1:nsteps
        # Apply the evolution operator using a first-order Suzuki Trotter approximation
        psi = apply_operator(Uodd, psi)
        psi = apply_operator(Ueven, psi)
        # Compress
        psi = svd_compress_mps(psi, 0, acc, direction=:left)
        println("Step: ", i, ": Dmax = ", maximum_bond_dimension(psi), ", E = ", real(expectation_value(psi, Hmpo)), ", Sz = ", real(expectation_value(psi, tot_z_mpo)))
    end

    nothing
end