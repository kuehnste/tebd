##########################################################################
#       Some useful models and observables to get started with           #
##########################################################################

using LinearAlgebra

"""
    get_pauli_matrices()

Returns the Pauli matrices including the identity as dense matrices.
"""
function get_pauli_matrices()
    Id = [1.0 0; 0.0 1.0]
    X = [0.0 1.0; 1.0 0.0]
    Y = [0.0 -1.0im; 1.0im 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    return Id, X, Y, Z
end


"""
    get_heisenberg_mpo(N::Int, J::Real, lambda::Real)::MPO{ComplexF64}
    
Construct a Matrix Product Operator representation of the one-dimensional Heisenberg Hamiltonian defined as.

H = J * sum_{i} (X^{i} X^{i+1} +Y^{i} Y^{i+1} + Z^{i} Z^{i+1}) + λ * sum_{i} X^{i}

where `N` is the number of sites, `J` is the nearest-neighbor coupling strength, and `lambda` is the strength of the transverse field. Both `J` and `lambda` must be real-valued parameters.
"""
function get_heisenberg_mpo(N::Int, J::Real, lambda::Real)::MPO{ComplexF64}
    # Provide the Pauli matrices
    Id, X, Y, Z =  get_pauli_matrices()
    # Initialize single tensor as complex array and the MPO as an array holding those
    first_tensor = zeros(ComplexF64, 1, 5, 2, 2)
    tensor = zeros(ComplexF64, 5, 5, 2, 2)
    last_tensor = zeros(ComplexF64, 5, 1, 2, 2)
    HeisenbergMPO = MPO{ComplexF64}(undef, N)

    # First MPO tensor
    first_tensor[1, 1, :, :] = -lambda * Z
    first_tensor[1, 2, :, :] = -J * X
    first_tensor[1, 3, :, :] = -J * Y
    first_tensor[1, 4, :, :] = -J * Z
    first_tensor[1, 5, :, :] = Id

    # Tensors in the center of the MPO
    tensor[1, 1, :, :] = Id
    tensor[2, 1, :, :] = X
    tensor[3, 1, :, :] = Y
    tensor[4, 1, :, :] = Z
    tensor[5, 1, :, :] = -lambda * Z
    tensor[5, 2, :, :] = -J * X
    tensor[5, 3, :, :] = -J * Y
    tensor[5, 4, :, :] = -J * Z
    tensor[5, 5, :, :] = Id

    # Last MPO tensor
    last_tensor[1, 1, :, :] = Id
    last_tensor[2, 1, :, :] = X
    last_tensor[3, 1, :, :] = Y
    last_tensor[4, 1, :, :] = Z
    last_tensor[5, 1, :, :] = -lambda * Z

    # Fill the MPO
    HeisenbergMPO[1] = first_tensor
    for i = 2:(N-1)
        HeisenbergMPO[i] = tensor
    end
    HeisenbergMPO[N] = last_tensor

    return HeisenbergMPO
end


"""
    get_ising_mpo(N::Int, J::Real, lambda::Real)::MPO{Float64}
    
Construct an MPO representation of the one-dimensional transverse-field Ising Hamiltonian, defined as

H = -J * sum_{i} X^{i} X^{i+1} - lambda * sum_{i} Z^{i}

where `N` is the number of sites, `J` is the nearest-neighbor interaction strength, and `lambda` is the transverse field strength. Both `J` and `lambda` must be real-valued parameters.
"""
function get_ising_mpo(N::Int, J::Real, lambda::Real)::MPO{Float64}
    # Provide the Pauli matrices
    Id, X, _, Z =  get_pauli_matrices()
    # Initialize single tensor as complex array and the MPO as an array holding those
    first_tensor = zeros(Float64, 1, 5, 2, 2)
    tensor = zeros(Float64, 5, 5, 2, 2)
    last_tensor = zeros(Float64, 5, 1, 2, 2)
    IsingMPO = MPO{Float64}(undef, N)

    # First MPO tensor
    first_tensor[1, 1, :, :] = Id
    first_tensor[1, 2, :, :] = -J * X
    first_tensor[1, 3, :, :] = -lambda * Z

    # Tensors in the center of the MPO
    tensor[1, 1, :, :] = Id
    tensor[1, 2, :, :] = -J * X
    tensor[1, 3, :, :] = -lambda * Z
    tensor[2, 3, :, :] = X
    tensor[3, 3, :, :] = Id

    # Last MPO tensor
    last_tensor[1, 1, :, :] = -lambda * Z
    last_tensor[2, 1, :, :] = X
    last_tensor[3, 1, :, :] = Id

    # Fill the MPO
    IsingMPO[1] = first_tensor
    for i = 2:(N-1)
        IsingMPO[i] = tensor
    end
    IsingMPO[N] = last_tensor

    return IsingMPO
end


"""
    get_total_spin_mpo(N::Int)::MPO{Float64}
    
Construct a Matrix Product Operator representation of the total z-component of spin. The operator is defined as

S^z = sum_{i)Z^i

where `N` is the number of lattice sites.
"""
function get_total_pauli_z_mpo(N::Int)::MPO{Float64}
    # Provide the Pauli matrices
    Id, X, Y, Z =  get_pauli_matrices()
    # Initialize the result
    TotalSpinMPO = MPO{Float64}(undef, N)

    # Initialize single tensor as complex array and the MPO as an array holding those
    first_tensor = zeros(Float64, 1, 2, 2, 2)
    tensor = zeros(Float64, 2, 2, 2, 2)
    last_tensor = zeros(Float64, 2, 1, 2, 2)

    # First MPO tensor
    first_tensor[1, 1, :, :] = Id
    first_tensor[1, 2, :, :] = Z

    # Tensors in the center of the MPO
    tensor[1, 1, :, :] = Id
    tensor[1, 2, :, :] = Z
    tensor[2, 2, :, :] = Id

    # Last MPO tensor
    last_tensor[1, 1, :, :] = Z
    last_tensor[2, 1, :, :] = Id

    # Fill the MPO
    TotalSpinMPO[1] = first_tensor
    for i = 2:(N-1)
        TotalSpinMPO[i] = tensor
    end
    TotalSpinMPO[N] = last_tensor

    return TotalSpinMPO
end


"""
    get_identity_mpo(N::Int, d::Int, tensortype::Type{T}=Float64)::MPO{T} where T
    
Constructs the Matrix Product Operator representation of the identity operator on an `N`-site Hilbert space. The local Hilbert space dimension of each site is assumed to be `d`. 
"""
function get_identity_mpo(N::Int, d::Int, tensortype::Type{T}=Float64)::MPO{T} where T
    # Initialize single tensor as complex array and the MPO as an array holding those
    tensor = reshape(Matrix{T}(I, d, d), (1, 1, d, d))
    IdentityMPO = MPO{T}(undef, N)

    # Fill the MPO
    for i = 1:N
        IdentityMPO[i] = tensor
    end

    return IdentityMPO
end


