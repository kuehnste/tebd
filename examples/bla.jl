include("../src/tensorfunctions.jl")
include("../src/MPS_OBC.jl")
include("../src/operators.jl")

A = [1 2;
     3 4]

B = [5 6;
     7 8]

M = kron(A,B)

mpo = decompose_into_mpo(M,2)