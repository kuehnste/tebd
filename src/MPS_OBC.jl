##########################################################################
#   	Basic functionality for doing MPS calculations with OBC     	#
##########################################################################
using LinearAlgebra

# Define sites as 3 legged tensors filled with elements of some type
Site{T} = Array{T,3}
# An MPS is an array of Sites of a certain type
MPS{T} = Vector{Site{T}}
# Define operators as 4 legged tensors filled with elements of some type
Operator{T} = Array{T,4}
# A MPO is an array of Operators
MPO{T} = Vector{Operator{T}}

"""
    Base.show(io::IO, mps::MPS)

Display a human-readable summary of a Matrix Product State

The printed representation provides a compact overview of the Matrix Product State, including relevant structural information such as the number of sites, physical dimensions and bond dimensions.
"""
function Base.show(io::IO, mps::MPS)
    N = length(mps)
    println(io, "MPS with $N sites and tensors")
    for i=1:N
        println(" - Site [$i]: (D_l, D_r, d) = (", size(mps[i], 1), ", ", size(mps[i], 2), ", ", size(mps[i], 3), ")")
    end
end


"""
    Base.show(io::IO, mpo::MPO)

Display a human-readable summary of a Matrix Product Operator

The printed representation provides a compact overview of the Matrix Product operators, including relevant structural information such as the number of sites, physical dimensions and bond dimensions.
"""
function Base.show(io::IO, mpo::MPO)
    N = length(mpo)
    print(io, "MPO with $N sites and tensors")
    for i=1:N
        println(" - Site [$i]: (D_l, D_r, d_o, d_i) = (", size(mpo[i], 1), ", ", size(mpo[i], 2), ", ", size(mpo[i], 3), ", ", size(mpo[i], 4), ")")
    end
end


"""
    random_mps_obc(N::Int, D::Int, d, tensortype::Type{T}=ComplexF64)::MPS{T} where T
    
Generates a random Matrix Product State with open boundary conditions.
N: Number of sites
D: Bond dimension
d: Vector of physical dimensions, or a single integer dimension. If the latter is given, it is assumed that all physical dimensions are the same, if sites with varying dimensions are required a vector of integers of length N has to be provided
Tensor convention: The MPS tensors are rank-3 tensors. The first two indices correspond to the virtual (bond) indices, while the third index corresponds to the physical index.
   3
   |
1--A--2
"""
function random_mps_obc(N::Int, D::Int, d, tensortype::Type{T}=ComplexF64)::MPS{T} where {T}
    # Ensure the the system size is at least two
    @assert(N > 1)

    # Initialize
    mps = Array{Site{T}}(undef, N)
    dim = Vector{Int64}(undef, N)
    if isa(d, Number)
        # Single number, I assume all dimensions are the same
        dim = d * ones(Int64, N)
    else
        # Make sure that the input is really a vector
        dim = vec(d)
        # Make sure input has valid length
        @assert(length(dim) == N)
    end

    # Left boundary tensor (row vector)
    mps[1] = rand(tensortype, 1, D, dim[1])

    # Right boundary tensor (column vector)
    mps[N] = rand(tensortype, D, 1, dim[N])

    # Tensors in between
    for i = 2:(N-1)
        mps[i] = rand(tensortype, D, D, dim[i])
    end

    return mps
end

"""
    basis_state_obc(configuration::Vector{<:Int}, d::Int=2)::MPS

Constructs a product-state MPS corresponding to the basis state |configuration> on N sites.

The configuration vector must contain N integers, each with a value between 1 and d. The value configuration[i] specifies that site i is initialized in the corresponding canonical basis state of the local Hilbert space of dimension d.
"""
function basis_state_obc(configuration::Vector{<:Int}, d::Int=2)::MPS{Float64}
    # Some error checking
    if any(x -> (x < 1 || x > d), configuration)
        throw(ArgumentError("configuration must contain integer elements in the range from 1 to d, got d=$(repr(d)), configuration=$(repr(configuration))"))
    end
    # Generate the MPS
    N = length(configuration)
    psi = MPS{Float64}(undef, N)
    # The individual tensors corresponding to the different basis states
    tensors = Vector{Array{Float64,3}}(undef, d)
    for i = 1:d
        tmp = zeros(Float64, 1, 1, d)
        tmp[1, 1, i] = 1.0
        tensors[i] = tmp
    end
    # Fill the MPS with the tensors corresponding to the local basis state
    for i = 1:N
        psi[i] = tensors[configuration[i]]
    end
    return psi
end

"""
    inner_product(mps1::MPS, mps2::MPS)::Number

Compute the inner product <mps1|mps2> of two matrix product states.

Both MPS must have the same number of sites and compatible physical dimensions.
"""
function inner_product(mps1::MPS, mps2::MPS)::Number
    # Make sure, that the two MPS have the same length
    N1 = length(mps1)
    N2 = length(mps2)
    @assert(N1 == N2)

    # Now compute their inner product
    ip = ones(Float64, 1, 1)
    for i = 1:N1
        ip = contract_tensors(ip, [2], mps2[i], [1])
        ip = contract_tensors(conj(mps1[i]), [1; 3], ip, [1; 3])
    end

    return ip[1]
end

"""
    svd_compress_mps(mps::MPS, Dmax::Int, tol::Real=0.0, unit_normalize::Bool=false; direction::Symbol = :left)::MPS

Compresses a Matrix Product State (MPS) by applying singular value decompositions at each bond and truncating the resulting singular values. 

The truncation can be controlled by either a maximum bond dimension or a singular value threshold: 
- If `Dmax > 0` and `tol == 0.0`, at most `Dmax` singular values are kept at each bond, resulting in an MPS with maximum bond dimension `Dmax`. 
- If `tol > 0` and `Dmax == 0`, all singular values larger than `tol` are kept. 
- If both `Dmax > 0` and `tol > 0` are specified, at most `Dmax` singular values larger than `tol` are kept at each bond. 
"""
function svd_compress_mps(mps::MPS, Dmax::Int, tol::Real=0.0, unit_normalize::Bool=false; direction::Symbol=:left)::MPS
    # One of the two parameters has to be larger than zero
    @assert((Dmax > 0) || (tol > 0))
    @assert direction in (:left, :right)
    if direction == :left
        svd_sweep_left(mps, Dmax, tol, unit_normalize)
    else
        svd_sweep_right(mps, Dmax, tol, unit_normalize)
    end
end

"""
    svd_sweep_left(mps::MPS, Dmax::Int, tol::Real=0.0, unit_normalize::Bool=false)::MPS

Perform a left-to-right SVD compression sweep of an MPS.

The sweep absorbs the singular values into the right tensor and places the truncated left singular vectors on the left tensor. The resulting MPS is left-canonical up to the first site. If unit_normalize is set to true, the norm factor is discarded and the first tensor is also in left canonical gauge.

`Dmax` limits the maximum bond dimension.
`tol` discards singular values below the threshold.
"""
function svd_sweep_left(mps::MPS, Dmax::Int, tol::Real=0.0, unit_normalize::Bool=false)::MPS
    # Extract the length and prepare a result
    N = length(mps)
    res = deepcopy(mps)
    Dnew = 0
    for i = 1:(N-1)
        Dl1, _, d1 = size(res[i])
        _, Dr2, d2 = size(res[i+1])
        tmp = contract_tensors(res[i], [2], res[i+1], [1])
        tmp = reshape(tmp, (Dl1 * d1, Dr2 * d2))
        U, S, V = svd!(tmp)
        M = Diagonal(S) * V'
        # Now truncate if necessary
        if Dmax > 0 && !(tol > 0)
            Dnew = min(Dmax, length(S))
            U = U[:, 1:Dnew]
            M = M[1:Dnew, :]
        else
            ind = findall(x -> x > tol, S)
            if Dmax > 0 && length(ind) > Dmax
                ind = ind[1:Dmax]
            end
            U = U[:, ind]
            M = M[ind, :]
            Dnew = length(ind)
        end
        # Reshape and set new tensors
        res[i] = permutedims(reshape(U, (Dl1, d1, Dnew)), (1, 3, 2))
        res[i+1] = reshape(M, (Dnew, Dr2, d2))
    end
    # We apply an SVD to last site and drop the norm factor if we want to normalize
    if unit_normalize
        Dl1, Dr1, d1 = size(res[N])
        tmp = permutedims(res[N], (1, 3, 2))
        tmp = reshape(tmp, (Dl1 * d1, Dr1))
        U, S, V = svd!(tmp)
        res[N] = permutedims(reshape(U, (Dl1, d1, Dr1)), (1, 3, 2))
    end
    return res
end

"""
    svd_sweep_right(mps::MPS, Dmax::Int, tol::Real=0.0, unit_normalize::Bool=false)::MPS

Perform a right-to-left SVD compression sweep of an MPS.

The sweep absorbs the singular values into the left tensor and places the truncated right singular vectors on the right tensor. The resulting MPS is right-canonical up to the first site. If unit_normalize is set to true, the norm factor is discarded and the first tensor is also in right canonical gauge.

`Dmax` limits the maximum bond dimension.
`tol` discards singular values below the threshold.
"""
function svd_sweep_right(mps::MPS, Dmax::Int, tol::Real=0.0, unit_normalize::Bool=false)::MPS
    # Extract the length and prepare a result
    N = length(mps)
    res = deepcopy(mps)
    Dnew = 0
    for i = N:-1:2
        Dl1, _, d1 = size(res[i-1])
        _, Dr2, d2 = size(res[i])
        tmp = contract_tensors(res[i-1], [2], res[i], [1])
        tmp = reshape(tmp, (Dl1 * d1, Dr2 * d2))
        U, S, V = svd!(tmp)
        M = U * Diagonal(S)
        # Now truncate if necessary
        if Dmax > 0 && !(tol > 0)
            Dnew = min(Dmax, length(S))
            V = V[:, 1:Dnew]'
            M = M[:, 1:Dnew]
        else
            ind = findall(x -> x > tol, S)
            if Dmax > 0 && length(ind) > Dmax
                ind = ind[1:Dmax]
            end
            V = V[:, ind]'
            M = M[:, ind]
            Dnew = length(ind)
        end
        # Reshape and set new tensors
        res[i] = reshape(V, (Dnew, Dr2, d2))
        res[i-1] = permutedims(reshape(M, (Dl1, d1, Dnew)), (1, 3, 2))
    end
    # We apply an SVD to last site and drop the norm factor if we want to normalize
    if unit_normalize
        Dl1, Dr1, d1 = size(res[1])
        tmp = reshape(res[1], (Dl1, Dr1 * d1))
        U, S, V = svd!(tmp)
        res[1] = reshape(V', (Dl1, Dr1, d1))
    end
    return res
end


"""
    contract_virtual_indices(mps::MPS)::Vector{<:Number}

Given an MPS contract the virtual indices such that one obtains a dense vector. The indices are ordered such that they are compatible with the standard Julia kronecker product.

Warning: the object constructed will have exponential memory requirements in terms of the number of sites, use with care!
"""
function contract_virtual_indices(mps::MPS)::Vector{<:Number}
    N = length(mps)

    # Since we deal with open boundary conditions, we drop the dummy indices one on the left (right) boundary for the first (last) tensor manually. We start from the right to have the physical indices in the order compatible with Julia's kronecker product
    res = mps[N][:, 1, :]
    for i = (N-1):-1:2
        res = contract_tensors(res, [ndims(res) - 1], mps[i], [2])
    end
    res = contract_tensors(res, [ndims(res) - 1], mps[1][1, :, :], [1])

    # Now reshape the result accordingly
    res = reshape(res, prod(size(res)))

    return res
end


"""
    contract_virtual_indices(mps::MPO)::Matrix{<:Number}

Given an MPO contract the virtual indices such that one obtains a dense matrix. The indices are ordered such that they are compatible with the standard Julia kronecker product.

Warning: the object constructed will have exponential memory requirements in terms of the number of sites, use with care!
"""
function contract_virtual_indices(mpo::MPO)::Matrix{<:Number}
    N = length(mpo)

    # Since we deal with open boundary conditions, we drop the dummy indices one on the left (right) boundary for the first (last) tensor manually. We start from the right to have the physical indices in the order compatible with Julia's kronecker product
    res = mpo[N][:, 1, :, :]
    for i = (N-1):-1:2
        res = contract_tensors(res, [ndims(res) - 2], mpo[i], [2])
    end
    res = contract_tensors(res, [ndims(res) - 2], mpo[1][1, :, :, :], [1])

    # Reshuffle the indices to be compatible with Julia's standard kronecker product
    res = permutedims(res, [collect(1:2:ndims(res)); collect(2:2:ndims(res))])

    # Now reshape the result accordingly
    dims = size(res)
    dr = prod(dims[1:N])
    dc = prod(dims[(N+1):end])
    res = reshape(res, (dr, dc))

    return res
end

"""
    contract_local_tensor(op::Operator, site::Site)

Contract a local MPO tensor op with the corresponding MPS site tensor site.

The contraction connects the fourth index of op with the third index of site, then permutes and reshapes the resulting tensor into a three-index tensor suitable for use as an MPS site.
"""
function contract_local_tensor(op::Operator, site::Site)
    temp = contract_tensors(op, [4], site, [3])
    temp = permutedims(temp, (1, 4, 2, 5, 3))
    dim1, dim2, dim3, dim4, dim5 = size(temp)
    return reshape(temp, (dim1 * dim2, dim3 * dim4, dim5))
end


"""
    apply_operator(operator::MPO{T1}, mps::MPS{T2})::MPS where {T1,T2}
    
Apply an operator given as MPO to an MPS. The resulting MPS will have a bond dimension that is the product of the bond dimensions of the MPS and the MPO.
"""
function apply_operator(operator::MPO{T1}, mps::MPS{T2})::MPS where {T1,T2}
    N1 = length(mps)
    N2 = length(operator)
    @assert(N1 == N2)

    # Generate a new MPO of the correct type
    Tres = Base.return_types(*, (T1, T2))[1]
    res = MPS{Tres}(undef, N1)

    # Apply the MPO to the MPS and generate new MPS
    for i = 1:N1
        res[i] = contract_local_tensor(operator[i], mps[i])
    end
    return res
end


"""
    apply_operator!(operator::MPO, mps::MPS)
    
Apply an operator given as MPO to an MPS. The resulting MPS will have a bond dimension that is the product of the bond dimensions of the MPS and the MPO. The input will be overwritten by the result. This requires that the type of the elements of the input MPS is able to accommodate the result of multiplying the MPO tensors into the MPS tensors (e.g. applying a complex MPO to a real MPS cannot be done in place as the result will be complex)
"""
function apply_operator!(operator::MPO, mps::MPS)
    N1 = length(mps)
    N2 = length(operator)
    @assert(N1 == N2)

    # Contract the tensors for each site
    for i = 1:N1
        mps[i] = contract_local_tensor(operator[i], mps[i])
    end
end


"""
    decompose_into_mpo(M::Matrix{T}, d::Vector{Int}) where T <:Number

Given a many-body operator represented as a dense matrix, decomposes it into a Matrix Product Operator.

The input operator is assumed to follow the index ordering convention of Julia's built-in kron function. The vector d specifies the local Hilbert space dimensions.
"""
function decompose_into_mpo(M::Matrix{T}, d::Vector{Int})::MPO{T} where {T<:Number}
    N = length(d)
    dim = prod(d)
    if size(M) != (prod(d), prod(d))
        throw(ArgumentError("matrix not compatible with specified local dimensions, got size(M)=$(repr(size(M))), d=$(repr(d))"))
    end
    # The MPO holding the result
    res = MPO{T}(undef, N)
    # Reshape the matrix and rearrange the indices, such that the row and column index for each site are adjacent and add dummy indices 1 at the boundaries    
    A = reshape(M, (1, reverse(d)..., reverse(d)..., 1))
    ind = collect(Iterators.flatten(zip(collect(N:-1:1), collect(2N:-1:(N+1)))))
    A = permutedims(A, (1, ind .+ 1..., 2 * length(d) + 2))
    # Now split it into tensors using an SVD
    Dl = 1
    for i = 1:(N-1)
        # Take the first three indices together
        dims = size(A)
        A = reshape(A, (prod(dims[1:3]), prod(dims[4:end])))
        # SVD the matrix representation
        U, S, V = svd(A)
        A = diagm(S) * V'
        Dr = size(U, 2)
        # Extract the tensor and the remaining part
        U = reshape(U, (Dl, d[i], d[i], Dr))
        res[i] = permutedims(U, (1, 4, 2, 3))
        A = reshape(A, (Dr, dims[4:end]...))
        Dl = Dr
    end
    res[N] = permutedims(A, (1, 4, 2, 3))

    return res
end


"""
    decompose_into_mpo(M::Matrix{T}, d::Int) where T <:Number

Given a many-body operator represented as a dense matrix, decomposes it into a Matrix Product Operator.

The input operator is assumed to follow the index ordering convention of Julia's built-in kron function. The number d specifies the local Hilbert space dimension.

# Examples
Decomposing a simple two-site operator made up from sum of Pauli terms into MPO form.
```julia-repl
julia>  Id = [1.0 0; 0.0 1.0]
julia>  X = [0.0 1.0; 1.0 0.0]
julia>  Z = [1.0 0.0; 0.0 -1.0]
julia>  H = kron(X,X) + kron(Id,Z) + kron(Z,Id)
julia>  mpo = decompose_into_mpo(H, 2)
```
"""
function decompose_into_mpo(M::Matrix{T}, d::Int)::MPO{T} where {T<:Number}
    # Some error checks
    dl, dr = size(M)
    if dl != dr
        throw(ArgumentError("local dimensions must all be the same, obtained a matrix with dimensions $(repr((dl,dr)))"))
    end
    N = Int(round(log(d, dl)))
    # Call the more general function allowing for varying local dimensions
    return decompose_into_mpo(M, d * ones(Int64, N))
end

"""
   expectation_value(mps::MPS, mpo::MPO)::Number
   
Given a MPS and a MPO compute the expectation value <mps|mpo|mps>.
"""
function expectation_value(mps::MPS, mpo::MPO)::Number
    # Make sure, that the two MPS have the same length
    N1 = length(mps)
    N2 = length(mpo)
    @assert(N1 == N2)

    # Contract zipper like
    val = ones(Float64, 1, 1, 1)
    for i = 1:N1
        val = contract_tensors(val, [1], conj(mps[i]), [1])
        val = contract_tensors(val, [1; 4], mpo[i], [1; 3])
        val = contract_tensors(val, [1; 4], mps[i], [1; 3])
    end

    return val[1]
end