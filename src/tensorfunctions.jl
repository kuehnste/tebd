###########################################################################
#       Basic functionality for contracting and manipulating tensors      #
###########################################################################

using LinearAlgebra


"""
    contract_tensors(T1, cT1::Vector{Int}, T2, cT2::Vector{Int}, ordering::Vector{Int}=Vector{Int64}())

Contracts two tensors `T1` and `T2` over the specified indices.

The vectors `cT1` and `cT2` specify which indices are contracted. Their elements correspond pairwise, i.e. `cT1[i]` is contracted with `cT2[i]`. The contracted indices must have matching dimensions. 

Internally, the tensors are permuted and reshaped into matrices so that the contraction can be performed efficiently as a matrix multiplication. The result is then reshaped back into a tensor.

# Arguments
- `T1`: First tensor.
- `cT1`: Indices of `T1` to be contracted.
- `T2`: Second tensor.
- `cT2`: Indices of `T2` to be contracted.
- `ordering`: Optional permutation specifying the order of the output
  tensor indices. If omitted, the remaining (uncontracted) indices of
  `T1` are followed by those of `T2`.

# Example

Suppose `T1` has indices `(i,j,k)` and `T2` has indices `(k,j,l,m)`.
Then

```julia
C = contract_tensors(T1, [2,3], T2, [2,1])

contracts

T1 index 2 (j) with T2 index 2 (j), and
T1 index 3 (k) with T2 index 1 (k),

producing a tensor with indices (i,l,m).
"""
function contract_tensors(T1, cT1::Vector{Int}, T2, cT2::Vector{Int}, ordering::Vector{Int}=Vector{Int64}())
    # Make sure the number of indices to contract is the same
    @assert(length(cT1) == length(cT2))

    # Determine size of the tensors
    dimT1 = size(T1)
    dimT2 = size(T2)

    # Make sure the dimensions along the contracted indices are equal
    if (dimT1[cT1] != dimT2[cT2])
        error("Dimensions along the contracted indices must be equal, failed to contract the indices")
    end

    # Get the uncontracted indices of T1 and T2
    uT1 = setdiff(collect(1:ndims(T1)), cT1)
    uT2 = setdiff(collect(1:ndims(T2)), cT2)

    # Bring the contracted indices of T1 to the end
    T1 = permutedims(T1, (uT1..., cT1...))
    # Bring the contracted indices of T2 to the front
    T2 = permutedims(T2, (cT2..., uT2...))

    # Fuse all contracted and uncontracted indices to obtain a matrix with two indices
    il = prod(dimT1[uT1])
    ir = prod(dimT2[uT2])
    T1 = reshape(T1, (il, prod(dimT1[cT1])))

    T2 = reshape(T2, (prod(dimT2[cT2]), ir))
    # Actual contraction as matrix multiplication
    res = T1 * T2
  
    # Expand to individual indices
    res = reshape(res, (dimT1[uT1]..., dimT2[uT2]...))

    # Permute the order if required
    if (!isempty(ordering))
        res = permutedims(res, (ordering...,))
    end

    return res
end


