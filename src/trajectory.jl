# -*- coding: utf-8 -*-
# %%
# Copyright 2020-2021 John T. Foster
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
include("bspline.jl")
using LinearAlgebra, IterativeSolvers, Plots
export construct_spline_matrix, reconstruct_trajectory, construct_helix, L2error, main, construct_helix_tangents, construct_non_helix

function construct_helix_tangents(n::Integer = 100)
    """
    Helper function which outputs an nx3 array of form [cos.(x)' sin.(x)'  linspace(0,1,n+1)] 
    """
    t = LinRange(0, 4*π, n+1)
    arr = zeros(Float64, (length(t), 3))
    arr[:, 1] = cos.(t)
    arr[:, 2] = sin.(t)
    arr[:, 3] = LinRange(0, 4*π, n+1)
    arr
end

function construct_helix(n::Integer = 100)
    """
    Helper function which outputs an nx3 array of form [cos.(x)' sin.(x)'  linspace(0,1,n+1)] 
    """
    t = LinRange(0, 4*π, n+1)
    arr = zeros(Float64, (length(t), 3))
    arr[:, 1] = sin.(t)
    arr[:, 2] = -cos.(t)
    arr[:, 3] = LinRange(0, 4*π, n+1)
    arr
end

function create_knot_vector(Qk::Array{<:Real}, p::Integer = 3)
    ū = create_ūk(Qk)
    n = length(Qk[:,1])
    m = n + p + 1
    kv = zeros(m)
    for j = 2:(n - p + 1)
        kv[j+p] = sum(ū[j:(j+p-1)]) / float(p)
    end
    kv[(end - p):end] .= 1
    kv
end

function reconstruct_control_points(Q::Array{<:Real}, u::Vector{<:Real}, p::Integer=3)
    P = zeros(size(Q))
    P[1,:] .= 0
    for j=1:size(Q,2)
        for i=2:size(Q,1) #Iterate through the columns
            P[i,j] = Q[i,j]*(u[i+p+1] - u[i+1])/p - P[i-1,j]
        end
    end
    P
end

#From definition, m (Num Knots) = n (Num points) + p (Polynomial Degree) + 1
function reconstruct_trajectory(tangents::Array{<:Real}, p::Integer=3)
    kv = create_knot_vector(tangents, p) #Knot vector U' must be constructed with p-1 since it is for the derivatives of our basis
    ū = create_ūk(tangents) # n = m - p - 1
    basis = BSplineBasis(kv, p, k=3)
    N = construct_spline_matrix(basis, ū, length(kv), p)
    tangent_control_points = hcat(map(col -> lsmr(N, col), eachcol(tangents))...)
    curve = BSplineCurve(basis, tangent_control_points)
    return curve
end

function construct_spline_matrix(basis::BSplineBasis, samples::Vector{<:Real}, num_knots::Integer, p::Integer=3)
    rows, cols = length(samples), num_knots - p - 1
    t = eltype(samples)
    N = zeros(t, (rows, cols))
    N[1, 1] = 1
    N[end, end] = 1
    for i in 2:rows-1
        evals = basis(samples[i])
        column = find_knot_span(basis, samples[i])
        N[i,column-p:column] = evals[1,:]#evals[2, :][evals[2, :] .> 0]
    end
    N
end

function L2error(C::BSplineCurve, tangents::Array{<:Real})
    """
    Returns a vector with the cumulative sum of the L2 error for each dimension.
    """
    evals = evaluate(C, length(tangents[:,1]))
    error = zeros(length(tangents[:,1]))
    for (i, u) in enumerate(evals[:,3]')
        index = findall(x-> x>=u, tangents[:,3])[1] #Find index of tangents vec closest to z axis of our control point
        error[i] = abs(tangents[index,2] - evals[i,2]) + abs(tangents[index,3] - evals[i,3])
    end
    return error
end

function main(n::Integer=100, p::Integer=3)
    Q = construct_helix(n)#[0:.05:4 sin.(0:.05:4)]#construct_non_helix(n) # Textbook Example: 
    Curve = reconstruct_trajectory(Q, p)
    plot(Curve, label="Bspline")
    plot!(tuple(eachcol(Q)...), label="Original Helix")
end
