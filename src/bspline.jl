# Copyright 2020 John T. Foster
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
import Plots: plot

# %%
#
function augment_knot_vector(knot_vector::AbstractVector, p::Integer)

    number_of_knots = length(knot_vector) + 2 * p

    open_knot_vector = Vector{eltype(knot_vector)}(undef, number_of_knots)

    for i in 1:number_of_knots
        if i ≤ p
            open_knot_vector[i] = knot_vector[1]
        elseif p < i < length(knot_vector) + p
            open_knot_vector[i] = knot_vector[i - p]
        else
            open_knot_vector[i] = knot_vector[end]
        end
    end
    open_knot_vector
end

struct BSplineBasis{R <: Real, S <: AbstractVector{R}, T <: Integer} <: AbstractVector{R}
    knot_vector::S
    order::T
    function BSplineBasis(knot_vector::AbstractVector{<:Real}, p::Integer) where {T}
        if (all(knot_vector[1:(p+1)] .== knot_vector[1]) &&
            all(knot_vector[(end - p):end] .== knot_vector[end]))
            new{eltype(knot_vector), typeof(knot_vector), typeof(p)}(knot_vector, p)
        else
            new{eltype(knot_vector), typeof(knot_vector), typeof(p)}(augment_knot_vector(knot_vector, p), p)
        end
    end
end

BSplineBasis(knot_range::UnitRange{<:Int}, p::Integer) = BSplineBasis(collect(knot_range), p)

Base.length(b::BSplineBasis) = length(b.knot_vector)
Base.size(b::BSplineBasis) = (length(b),)
Base.getindex(b::BSplineBasis, i) = b.knot_vector[i]

# %%
function find_knot_span(b::BSplineBasis, u::Real)
    p = b.order
    n = length(b.knot_vector) - p - 1
    if isapprox(u, b.knot_vector[n+1], atol=1e-15)
        return n
    else
        low = p
        high = n + 1
        mid = (low + high) ÷ 2
        while (u < b.knot_vector[mid] || u ≥ b.knot_vector[mid+1])
            if (u < b.knot_vector[mid])
                high = mid
            else
                low = mid
            end
            mid = (low + high) ÷ 2
        end
        return mid
    end
end

function evaluate_basis_functions(b::BSplineBasis, u::Real, knot_span_index::Integer)

    i = knot_span_index
    p = b.order

    N = zeros(Float64, p + 1)
    left = zeros(Float64, p)
    right = zeros(Float64, p)

    N[1] = 1.0
    for j in 1:p
        left[j] = u - b.knot_vector[i+1-j]
        right[j] = b.knot_vector[i+j] - u
        saved = 0.0
        for r in 0:(j-1)
            temp = N[r+1] / (right[r+1] + left[j-r])
            N[r+1] = saved + right[r+1] * temp
            saved = left[j-r] * temp
        end
        N[j+1] = saved;
    end
    return N
end

function evaluate_basis_functions(b::BSplineBasis, u::Real)
    i = find_knot_span(b, u)
    evaluate_basis_functions(b, u, i)
end


function plot(b::BSplineBasis, num::Integer=100, show_legend::Bool=false)

    p = b.order

    start = b.knot_vector[1]
    stop = b.knot_vector[end - p]

    x = LinRange(start, stop, num)

    N = zeros(Float64, (length(x), length(b.knot_vector) - p - 1))

    for (j, u) in enumerate(x)
        i = find_knot_span(b, u)
        N[j, (i-p):i] = evaluate_basis_functions(b, u, i)
    end

    labels = Array{String}(undef, 1, size(N)[2])
    for i in 1:(size(N)[2])
        labels[1, i] = "N_{$i,$p}"
    end
    plot(x, N, xaxis="knots", label=labels, legend=show_legend)
end

# %%
struct BSplineCurve{S <: Real, T <: AbstractArray{S, 2}} <: AbstractArray{S, 2}
    basis::BSplineBasis
    control_points::T
end

Base.length(c::BSplineCurve) = length(c.basis)
Base.size(c::BSplineCurve) = (length(c.basis),)
Base.getindex(c::BSplineCurve, i) = c.basis.knot_vector[i]
