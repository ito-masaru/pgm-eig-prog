"""
eig_prog_base.jl
"""

using LinearAlgebra
using Printf
using CPUTime
import Base: zero, reshape

# include source files
include("algebra.jl")
include("projected_gradient.jl")

