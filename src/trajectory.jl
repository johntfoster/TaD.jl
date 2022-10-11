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
export reconstruct_trajectory, create_knot_vector

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
