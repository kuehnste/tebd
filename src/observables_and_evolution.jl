"""
    get_energy(mps::MPS, H_tensors_left_boundary::Vector{<:AbstractArray{<:Number,4}}, H_tensors_right_boundary::Vector{<:AbstractArray{<:Number,4}}, H_tensors_bulk::Vector{<:AbstractArray{<:Number,4}})

Compute the expectation value of a nearest-neighbor Hamiltonian for an MPS.

The Hamiltonian is represented as a sum of local two-body operators acting on neighboring sites. Separate MPO-style tensors have to be provided for the left boundary bond, bulk bonds, and right boundary bond. For each nearest-neighbor pair, the corresponding two local operator tensors are contracted into the MPS and the overlap with the original MPS is evaluated. The resulting contributions are summed to obtain the total energy.
"""
function expectation_energy(mps::MPS, H_tensors_left_boundary::Vector{<:AbstractArray{<:Number,4}}, H_tensors_right_boundary::Vector{<:AbstractArray{<:Number,4}}, H_tensors_bulk::Vector{<:AbstractArray{<:Number,4}})
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
    expectation_single_body_operator(mps::MPS, op::Matrix{<:Number})

Compute the expectation value of a single-body operator summed over the
sites of an MPS.

The operator op is applied independently to each site of mps, and the
resulting overlaps with the original MPS are summed. Specifically, this
computes
    \\sum_i \\langle \\mathrm{MPS} | O_i | \\mathrm{MPS} \\rangle,
where O_i denotes the operator op acting on site i.
"""
function expectation_single_body_operator(mps::MPS, op::Matrix{<:Number})
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
        # Compute the inner product
        expectation_value += inner_product(mps, mps_tmp)
    end
    return expectation_value
end

"""
    u_odd_even_mpo(N::Int, mpo_Uloc_left_boundary, mpo_Uloc_right_boundary, mpo_Uloc_bulk)

Construct the odd- and even-bond MPOs for a nearest-neighbor time-evolution operator on an N-site chain.

The nearest-neighbor evolution operator is decomposed into two sets of non-overlapping two-site operators. Uodd contains the operators acting on bonds (1,2), (3,4), ..., while Ueven contains the operators acting on bonds (2,3), (4,5), .... Identity operators are inserted on sites that are not part of a given bond. Different local MPO tensors can be specified for the left boundary, right boundary, and bulk bonds to account for position-dependent interactions.
"""
function u_odd_even_mpo(N::Int, mpo_Uloc_left_boundary::Vector{<:AbstractArray{<:Number,4}}, mpo_Uloc_right_boundary::Vector{<:AbstractArray{<:Number,4}}, mpo_Uloc_bulk::Vector{<:AbstractArray{<:Number,4}})
    # The identity operator as MPO tensor
    d = size(mpo_Uloc_right_boundary[2],4)
    id_tensor = reshape(Matrix{Float64}(I, d, d), (1, 1, d, d))
    # Build the MPO for evolving the odd sites
    Uodd = MPO{ComplexF64}(undef, N)
    for i = 1:2:(N-1)
        if i == 1
            Uodd[i] = mpo_Uloc_left_boundary[1]
            Uodd[i+1] = mpo_Uloc_left_boundary[2]
        elseif i == N - 1
            Uodd[i] = mpo_Uloc_right_boundary[1]
            Uodd[i+1] = mpo_Uloc_right_boundary[2]
        else
            Uodd[i] = mpo_Uloc_bulk[1]
            Uodd[i+1] = mpo_Uloc_bulk[2]
        end
    end
    if isodd(N)
        Uodd[N] = id_tensor
    end

    # Build the MPO for evolving the even sites
    Ueven = MPO{ComplexF64}(undef, N)
    Ueven[1] = id_tensor
    for i = 2:2:(N-1)
        if i == N - 1
            Ueven[i] = mpo_Uloc_right_boundary[1]
            Ueven[i+1] = mpo_Uloc_right_boundary[2]
        else
            Ueven[i] = mpo_Uloc_bulk[1]
            Ueven[i+1] = mpo_Uloc_bulk[2]
        end
    end
    if iseven(N)
        Ueven[N] = id_tensor
    end

    return Uodd, Ueven
end