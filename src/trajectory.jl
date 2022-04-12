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
using LinearAlgebra, IterativeSolvers
export construct_spline_matrix, reconstruct_trajectory, construct_helix, L2error

function construct_helix(n::Integer = 100)
    """
    Helper function which outputs an nx3 array of form [cos.(x)' sin.(x)'  linspace(0,1,n+1)] 
    """
    t = LinRange(0, 4*π, n+1)
    arr = zeros(Float64, (length(t), 3))
    arr[:, 1] = cos.(t)
    arr[:, 2] = sin.(t)
    arr[:, 3] = LinRange(0, 1, n+1)
    arr
end

function create_knot_vector(Qk::Vector{<:Float64}, p::Integer = 3)
    ū = create_ūk(Qk)
    n = length(ū)
    m = length(ū) + p + 1
    kv = zeros(m)
    for j = 2:(n - p)
        kv[j+p] = sum(ū[j:(j+p-1)]) / float(p)
    end
    kv[(end - p):end] .= 1
    kv
end

#Knots = m
#From definition, m = n + p + 1
#set m = length(tangents) so the linear algebra works out
#then set n (num samples) = m - p - 1
function reconstruct_trajectory(tangents::Matrix{<:Float64}, p::Integer=3)
    control_points = zeros(size(tangents))
    control_points[:,1] = reconstruct_trajectory_1d(tangents[:,1], p)
    control_points[:,2] = reconstruct_trajectory_1d(tangents[:,2], p)
    control_points[:,3] = tangents[:,3] #Don't reconstruct the domain
    kv = create_knot_vector(tangents[:,1], p)
    basis = BSplineBasis(kv, p, k=2)
    curve = BSplineCurve(basis, control_points)
    return curve
end

function reconstruct_trajectory_1d(tangents::Vector{<:Float64}, p::Integer=3)
    kv = create_knot_vector(tangents, p)
    ū = create_ūk(tangents) # n = m - p - 1
    basis = BSplineBasis(kv, p, k=2)
    N, Nprime = construct_spline_matrix(basis, ū, length(kv), p)
    control_points = lsmr(N, tangents)
    return control_points
end

function construct_spline_matrix(basis::BSplineBasis, samples::Vector{<:Float64}, num_knots::Integer, p::Integer=3)
    rows, cols = length(samples), num_knots - p - 1
    N, Nprime = zeros(Float64, (rows, cols)), zeros(Float64, (rows, cols))
    N[1, 1] = 1
    N[end, end] = 1
    Nprime[1, 1] = 1
    Nprime[end, end] = 1
    for i in p:rows
        evals = basis(samples[i])
        column = find_knot_span(basis, samples[i])
        N[i,column-p:column] = evals[1,:]#evals[1, :][evals[1, :] .> 0]
        Nprime[i,column-p:column] = evals[2,:]#evals[2, :][evals[2, :] .> 0]
    end
    N, Nprime
end

function L2error(C::BSplineCurve, tangents::Matrix{<:Float64})
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
    Q = construct_helix(n)
    Curve = reconstruct_trajectory(Q, p)
    Q, Curve
end
