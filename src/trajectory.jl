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
function generate_helix_test_tangents(r::Real=1, c::Real=1, number_of_points::Integer=100)
    t = LinRange(0, 100, number_of_points)
    dx = -r .* sin.(t) 
    dy = r .* cos.(t)
    dz = c  
    arr = zeros(Float64, (length(t), 3))
    arr[:, 1] = dx
    arr[:, 2] = dy
    arr[:, 3] .= dz
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
    kv
end

function predict(control_points::Array{<:AbstractFloat,2})

end

function loss(x)
end
