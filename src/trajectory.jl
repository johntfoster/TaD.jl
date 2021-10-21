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
export reconstruct_trajectory_1d, reconstruct_trajectory_3d, generate_helix_test_tangents

function generate_helix_test_tangents(r::Real=1, c::Real=1, number_of_points::Integer=100)
    t = LinRange(0, 100, number_of_points)
    dx = -r .* sin.(t) 
    dy = r .* cos.(t)
    dz = c
    arr = zeros(Float64, (length(t), 3))
    arr[:, 1] = dx
    arr[:, 2] = dy
    arr[:, 3] = c*t
    arr
end

function create_knot_vector(number_of_data_points::Integer, 
                            number_of_control_points::Integer=(number_of_data_points ÷ 2), 
                            p::Integer=3,
                            Δū::Real=30) 
    ū = collect(StepRange(0, Δū, number_of_data_points * Δū)) / (number_of_data_points * Δū)
    d = (number_of_data_points + 1) / (number_of_control_points - p + 1)
    kv = zeros(number_of_control_points + p + 2)
    for j in 1:(number_of_control_points - p)
        i = floor(Int, j * d)
        α = j * d - i
        kv[j + p] = (1 - α) * ū[i - 1] + α * ū[i]
    end
    kv[end-4:end] .= 1
    kv
end

function reconstruct_trajectory_3d(tangents::Matrix{<:Float64}, number_of_control_points::Integer=(length(tangents) ÷ 2))
  arr = tangents
  for i in 1:1:2
    control_points, arr[:,i] = reconstruct_trajectory_1d(tangents[:,i])
  end
  fig = plot((arr[:,1], arr[:,2], arr[:,3]))
  savefig(fig, joinpath(@__DIR__, "../examples/reconstructed_helix.png"))
end
function reconstruct_trajectory_1d(tangents::Vector{<:Float64}, number_of_control_points::Integer=(length(tangents) ÷ 2))
  kv = create_knot_vector(length(tangents), number_of_control_points)
  knot_space_samples = 0:1/(length(tangents)-1):1
  basis = BSplineBasis(kv, 3, k=1)
  N, Nprime = construct_spline_matrix(basis, collect(knot_space_samples), length(kv))
  A = pinv(Nprime' * Nprime)
  B = (Nprime' * tangents)
  control_points, curve = A*B, Nprime*(A*B)
end

function construct_spline_matrix(basis::BSplineBasis, samples::Vector{<:Float64}, num_knots::Integer)
  rows, cols = length(samples), num_knots
  N, Nprime = zeros(Float64, (rows, cols)), zeros(Float64, (rows, cols))
  for i in 1:1:rows-1
    evals = basis(samples[i])
    N[i,find_knot_span(basis, samples[i])-2:find_knot_span(basis, samples[i])+1] =evals[1,:]
    Nprime[i,find_knot_span(basis, samples[i])-2:find_knot_span(basis, samples[i])+1] = evals[2,:]
  end
  N, Nprime
end


function predict(x)
end
function loss(x)
end
