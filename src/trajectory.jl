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
using LinearAlgebra, Plots
export main, construct_spline_matrix, reconstruct_trajectory, generate_helix_test_tangents, generate_test_set

function return_zero_rows(A)
    row_zeros = mapslices(is_row_zero, A, dims = [1]) .* cumsum(ones(size(A, 1)))'
    row_zeros = row_zeros[row_zeros .> 0]
    return row_zeros
end

function is_row_zero(A)
    if sum(A[A .!= 0]) == 0
        return true
    end
    return false
end

function create_ūk(Qk::Vector{<:Float64})#, p::Integer = 3, num_samples::Integer = length(Qk) - p - 1) # n = m - p - 1, p = 3 
    """
    According to chord length method defined on pg 364
    """
    Qk_copy = copy(Qk)
    Qk_shift = copy(Qk)
    popfirst!(Qk_shift)
    pop!(Qk_copy)
    ūk = cumsum(broadcast(abs, Qk_shift - Qk_copy)) / sum(broadcast(abs, Qk_shift - Qk_copy))
    insert!(ūk, 1, 0)
    ūk
end

function create_knot_vector(Qk::Vector{<:Float64}, p::Integer = 3)
    ū = create_ūk(Qk)
    m = length(Qk)
    kv = zeros(m)
    n = m - p - 1
    d = (m + 1) / (n - p + 1)
    for j = 2:(n - p)
        i = floor(Int, j * d)
        α = j * d - i
        kv[j + p] = (1 - α) * ū[i - 1] + α * ū[i]
    end
    kv[(end - p):end] .= 1
    kv#, ū
end

function generate_helix_test_tangents(
    r::Real = 1,
    c::Real = 1,
    number_of_points::Integer = 100,
)
    t = LinRange(0, 1, number_of_points)
    theta = t .* 100
    dx = r .* cos.(theta)
    dy = r .* sin.(theta)
    dz = c
    arr = zeros(Float64, (length(t), 3))
    arr[:, 1] = dx
    arr[:, 2] = dy
    arr[:, 3] = c * t
    arr
end

function pow(x,y=2)
    return x^y
end

function generate_test_set((f1, f2), num::Integer = 100)
    t = LinRange(0, 1, num+1)
    dx = f1.(t)
    dy = f2.(t)
    dz = 1
    arr = zeros(Float64, (length(t), 3))
    arr[:, 1] = dx
    arr[:, 2] = dy
    arr[:, 3] = dz * t
    arr
end

function reconstruct_trajectory(
    tangents::Matrix{<:Float64},
    number_of_control_points::Integer = (length(tangents) ÷ 2),
)
    control_points = similar(tangents)
    control_points[:, 1] = reconstruct_trajectory_1d(tangents[:, 1])
    control_points[:, 2] = reconstruct_trajectory_1d(tangents[:, 2])
    control_points[:, 3] = tangents[:, 3]
    kv = create_knot_vector(tangents[:, 1])
    basis = BSplineBasis(kv, 3, k = 2)
    curve = BSplineCurve(basis, control_points)
    return curve
end

function reconstruct_trajectory_1d(tangents::Vector{<:Float64}, number_of_control_points::Integer=(length(tangents) ÷ 2))
    kv = create_knot_vector(tangents)
    ū = create_ūk(tangents) # n = m - p - 1
    basis = BSplineBasis(kv, 3, k=2)
    N, Nprime = construct_spline_matrix(basis, ū, length(kv))
    T = Nprime'*Nprime
    #print("\n", return_zero_rows(T))
    A = pinv(T)
    B = (Nprime' * tangents)
    control_points = A*B #N*(A*B)#idrs(Nprime, tangents), Nprime*idrs(Nprime, tangents)#
end

function construct_spline_matrix(basis::BSplineBasis, samples::Vector{<:Float64}, num_knots::Integer)
    rows, cols = length(samples), num_knots
    N, Nprime = zeros(Float64, (rows, cols)), zeros(Float64, (rows, cols))
    N[1, 1] = 1
    N[2, 1] = 1
    Nprime[1, 1] = 1
    Nprime[2, 1] = 1
    for i in 3:1:rows-1
        evals = basis(samples[i])
        N[i,find_knot_span(basis, samples[i])-2:find_knot_span(basis, samples[i])+1] =evals[1,:]
        Nprime[i,find_knot_span(basis, samples[i])-2:find_knot_span(basis, samples[i])+1] = evals[2,:]
        #N[i,i-2:i+1] =evals[1,:]
        #Nprime[i,i-2:i+1] = evals[2,:]
    end
    N, Nprime
end

function main()
    Q = generate_helix_test_tangents(1, 1, 1000)
    #Q = round.(Q, digits=10)
    Curve = reconstruct_trajectory(Q)
    plot(Curve)
end
