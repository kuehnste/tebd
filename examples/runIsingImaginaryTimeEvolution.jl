############################################################################
# Find the ground state of the Ising Hamiltonian and get the magnetization #
############################################################################
include("../src/tensorfunctions.jl")
include("../src/MPS_OBC.jl")
include("../src/operators.jl")

function get_energy(mps::MPS, H_tensors_left_boundary::Vector{<:AbstractArray{<:Number,4}}, H_tensors_right_boundary::Vector{<:AbstractArray{<:Number,4}}, H_tensors_bulk::Vector{<:AbstractArray{<:Number,4}})
    # Initialize the energy
    energy = 0
    # Compute the energy by contracting the local two-body operators into the MPS and evaluating the inner product with the original MPS
    N = length(mps)
    for i=1:(N-1)
        # Get a copy of the original MPS
        mps_tmp = deepcopy(mps)
        # Apply the local operator
        if i==1
            mps_tmp[i] = contract_local_tensor(H_tensors_left_boundary[1], mps_tmp[i])
            mps_tmp[i+1] = contract_local_tensor(H_tensors_left_boundary[2], mps_tmp[i+1])
        elseif i==(N-1)
            mps_tmp[i] = contract_local_tensor(H_tensors_right_boundary[1], mps_tmp[i])
            mps_tmp[i+1] = contract_local_tensor(H_tensors_right_boundary[2], mps_tmp[i+1])
        else
            mps_tmp[i] = contract_local_tensor(H_tensors_bulk[1], mps_tmp[i])
            mps_tmp[i+1] = contract_local_tensor(H_tensors_bulk[2], mps_tmp[i+1])
        end
        # Get the inner product
        energy += inner_product(mps, mps_tmp)
    end
    return energy
end

"""
    get_expectation_single_body_operator(mps::MPS, op::Matrix{<:Number})

Compute the expectation value of a single-body operator summed over the
sites of an MPS.

The operator op is applied independently to each site of mps, and the
resulting overlaps with the original MPS are summed. Specifically, this
computes
    \\sum_i \\langle \\mathrm{MPS} | O_i | \\mathrm{MPS} \\rangle,
where O_i denotes the operator op acting on site i.
"""
function get_expectation_single_body_operator(mps::MPS, op::Matrix{<:Number})
    # Initialize the expectation value
    expectation_value = 0
    # Reshape the matrix into a MPO style tensor
    d = size(op, 1)
    op_local = reshape(op, (1, 1, d, d))
    # Compute the expected value by contracting the local two-body operators into the MPS and evaluating the inner product with the original MPS
    N = length(mps)
    for i=1:N
        # Get a copy of the original MPS
        mps_tmp = deepcopy(mps)
        # Apply the local operator
        mps_tmp[i] = contract_local_tensor(op_local, mps_tmp[i])
        # Get the inner product
        expectation_value += inner_product(mps, mps_tmp)
    end
    return expectation_value
end

let

    ##################################################################
    # Parameters of the model and some basic operators
    ##################################################################

    # Number of Spins
    N = 20
    # Bond Dimension
    D = 20
    # Physical dimension
    d = 2
    # Desired Accuracy
    acc = 1E-10
    # The number of time steps we take
    nsteps = 200
    # The size of each time step
    dt = 5E-2

    # Coupling between nearest neighbors
    J = 1.0
    #J = 0.0
    # Coupling to external field
    lambda = 1.0
    # Generate Ising Hamiltonian H = -J * sum_{i=1}^{N-1} X^i X^i+1 - lambda * sum_{i=1}^N Z^i.
    H = get_ising_mpo(N, J, lambda)
    # Get the MPO for the total spin
    Sz = get_total_pauli_z_mpo(N)

    ##################################################################
    # Construct the odd-even time evolution operators
    ##################################################################

    #=
    In the following we prepare the time evolution operators in MPO form using an
    odd-even decomposition for the evolution operator. Since the Ising Hamiltonian is
    of the form H = \sum_i h_{i,i+1} we can approximate the matrix exponential using 
    a Suzuki-Trotter decomposition as follows
    exp(-dt * H) ≈ \prod_{i even} exp(-dt * h_{i,i+1}) \prod_{i odd} exp(-dt * h_{i,i+1}).
    Since all terms starting at an odd (even) site commute, we can find an MPO
    representation by decomposing one of the local terms exp(-dt * h_{i,i+1}) into an MPO,
    and using the same tensors at every site. Special care has to be taken on the
    boundaries. 
    =#

    # The required Pauli matrices for building the evolution MPOs
    Id, X, _, Z = get_pauli_matrices()

    # Prepare the matrix representation for the different exponentials appearing in the odd-even decomposition
    Uloc_left_boundary = exp(-dt * (-J * kron(X, X) - lambda * kron(Z, Id) - 0.5 * lambda * kron(Id, Z)))
    Uloc_center = exp(-dt * (-J * kron(X, X) - 0.5 * lambda * (kron(Z, Id) + kron(Id, Z))))
    Uloc_right_boundary = exp(-dt * (-J * kron(X, X) - 0.5 * lambda * kron(Z, Id) - lambda * kron(Id, Z)))

    # The identity operator as MPO tensor
    id_tensor = reshape(Id, (1, 1, 2, 2))

    # Find the mpo representation for the local terms
    mpo_Uloc_left_boundary = decompose_into_mpo(Uloc_left_boundary, d)
    mpo_Uloc_center = decompose_into_mpo(Uloc_center, d)
    mpo_Uloc_right_boundary = decompose_into_mpo(Uloc_right_boundary, d)

    # Build the MPO for evolving the odd sites
    Uodd = MPO{Float64}(undef, N)
    for i = 1:2:(N-1)
        if i == 1
            Uodd[i] = mpo_Uloc_left_boundary[1]
            Uodd[i+1] = mpo_Uloc_left_boundary[2]
        elseif i == N - 1
            Uodd[i] = mpo_Uloc_right_boundary[1]
            Uodd[i+1] = mpo_Uloc_right_boundary[2]
        else
            Uodd[i] = mpo_Uloc_center[1]
            Uodd[i+1] = mpo_Uloc_center[2]
        end
    end
    if isodd(N)
        Uodd[N] = id_tensor
    end

    # Build the MPO for evolving the even sites
    Ueven = MPO{Float64}(undef, N)
    Ueven[1] = id_tensor
    for i = 2:2:(N-1)
        if i == N - 1
            Ueven[i] = mpo_Uloc_right_boundary[1]
            Ueven[i+1] = mpo_Uloc_right_boundary[2]
        else
            Ueven[i] = mpo_Uloc_center[1]
            Ueven[i+1] = mpo_Uloc_center[2]
        end
    end
    if iseven(N)
        Ueven[N] = id_tensor
    end

    #=
    Similarly, we can construct the Hamiltonian, this time it is even exact.

    Note that there are more efficient ways to directly obtain the Hamiltonian as an MPO, see references in the lectures.
    =#

    # Prepare the matrix representation for the different exponentials appearing in the odd-even decomposition
    Hloc_left_boundary = -J * kron(X, X) - lambda * kron(Z, Id) - 0.5 * lambda * kron(Id, Z)
    Hloc_bulk = -J * kron(X, X) - 0.5 * lambda * (kron(Z, Id) + kron(Id, Z))
    Hloc_right_boundary = -J * kron(X, X) - 0.5 * lambda * kron(Z, Id) - lambda * kron(Id, Z)

    # The identity operator as MPO tensor
    id_tensor = reshape(Id, (1, 1, 2, 2))

    # Find the mpo representation for the local terms
    mpo_Hloc_left_boundary = decompose_into_mpo(Hloc_left_boundary, d)
    mpo_Hloc_bulk = decompose_into_mpo(Hloc_bulk, d)
    mpo_Hloc_right_boundary = decompose_into_mpo(Hloc_right_boundary, d)

    # Build the MPO for evolving the odd sites
    Hodd = MPO{Float64}(undef, N)
    for i = 1:2:(N-1)
        if i == 1
            Hodd[i] = mpo_Hloc_left_boundary[1]
            Hodd[i+1] = mpo_Hloc_left_boundary[2]
        elseif i == N - 1
            Hodd[i] = mpo_Hloc_right_boundary[1]
            Hodd[i+1] = mpo_Hloc_right_boundary[2]
        else
            Hodd[i] = mpo_Hloc_bulk[1]
            Hodd[i+1] = mpo_Hloc_bulk[2]
        end
    end
    if isodd(N)
        Hodd[N] = id_tensor
    end

    # Build the MPO for evolving the even sites
    Heven = MPO{Float64}(undef, N)
    Heven[1] = id_tensor
    for i = 2:2:(N-1)
        if i == N - 1
            Heven[i] = mpo_Hloc_right_boundary[1]
            Heven[i+1] = mpo_Hloc_right_boundary[2]
        else
            Heven[i] = mpo_Hloc_bulk[1]
            Heven[i+1] = mpo_Hloc_bulk[2]
        end
    end
    if iseven(N)
        Heven[N] = id_tensor
    end

    println("Hodd = ", Hodd)
    println("Heven= ", Heven)

    ##################################################################
    # Running the actual evolution
    ##################################################################

    # Evolve in imaginary time starting from a random, normalized state
    psi = random_mps_obc(N, D, d, Float64)
    # To normalize and remove unnecessary parameters, we put the MPS in left and right canonical form
    psi = svd_compress_mps(psi, 0, 1E-15, true, direction=:left)
    psi = svd_compress_mps(psi, 0, 1E-15, direction=:right)
    for i = 1:nsteps
        println("Step: ", i)
        # Apply the evolution operators
        psi = apply_operator(Uodd, psi)
        psi = apply_operator(Ueven, psi)
        # Compress and renormalize
        psi = svd_compress_mps(psi, D, acc, true, direction=:left)
    end
    E0 = expectation_value(psi, H)
    E02 = get_energy(psi, mpo_Hloc_left_boundary, mpo_Hloc_right_boundary, mpo_Hloc_bulk)


    # Print the results
    println(" ")
    println("Groundstate energy:            ", E0)
    println("Groundstate energy:            ", E02)
    println("Groundstate energy density:    ", E0 / N)
    println("Total spin:                    ", expectation_value(psi, Sz))
    println("Total spin:                    ", get_expectation_single_body_operator(psi, Z))

    nothing
end