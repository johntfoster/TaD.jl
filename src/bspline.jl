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
import RecipesBase: @recipe, @series

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
    derivative_order::T
    function BSplineBasis(knot_vector::AbstractVector{<:Real}, p::Integer, k::Integer=0) where {T}
        if (all(knot_vector[1:(p+1)] .== knot_vector[1]) &&
            all(knot_vector[(end - p):end] .== knot_vector[end]))
            new{eltype(knot_vector), typeof(knot_vector), typeof(p)}(knot_vector, p, k)
        else
            new{eltype(knot_vector), typeof(knot_vector), typeof(p)}(augment_knot_vector(knot_vector, p), p, k)
        end
    end
end

BSplineBasis(knot_range::UnitRange{<:Integer}, p::Integer, k::Integer=0) = BSplineBasis(collect(knot_range), p, k)

Base.length(b::BSplineBasis) = length(b.knot_vector)
Base.size(b::BSplineBasis) = (length(b),)
Base.getindex(b::BSplineBasis, i) = b.knot_vector[i]

derivative(b::BSplineBasis) = BSplineBasis(b.knot_vector, b.order, b.derivative_order + 1)

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

# %%
function evaluate(b::BSplineBasis, u::Real, knot_span_index::Integer)

    i = knot_span_index
    p = b.order
    derivative_order = b.derivative_order

    N = zeros(Float64, (p+1, p+1))
    left = zeros(Float64, p)
    right = zeros(Float64, p)

    N[1, 1] = 1.0
    for j in 1:p
        left[j] = u - b.knot_vector[i+1-j]
        right[j] = b.knot_vector[i+j] - u
        saved = 0.0
        for r in 0:(j-1)
            N[j+1, r+1] = (right[r+1] + left[j-r])
            temp = N[r+1, j] / N[j+1, r+1]
            N[r+1, j+1] = saved + right[r+1] * temp
            saved = left[j-r] * temp
        end
        N[j+1, j+1] = saved;
    end

    derivatives = zeros(Float64, (derivative_order+1, p+1))
    derivatives[1, :] = N[:, p+1]

    a = zeros(Float64, (2, p+1))

    for r in 0:p
        a[1, 1] = 1.0
        s1 = 0
        s2 = 1

        for k in 1:derivative_order
            d = 0.0
            rk = r - k
            pk = p - k

            if r ≥ k
                a[s2+1, 1] = a[s1+1, 1] / N[pk+2, rk+1]
                d = a[s2+1, 1] * N[rk+1, pk+1]
            end

            j1 = rk ≥ -1 ? 1 : -rk

            j2 = r-1 ≤ pk ? k - 1 : p - r

            for j in j1:j2
                a[s2+1, j+1] = (a[s1+1, j+1] - a[s1+1, j]) / N[pk+2, rk+j+1]
                d += a[s2+1, j+1] * N[rk+j+1, pk+1]
            end

            if r ≤ pk
                a[s2+1, k+1] = -a[s1+1, k] / N[pk+2, r+1]
                d += a[s2+1, k+1] * N[r+1, pk+1]
            end
            derivatives[k+1, r+1] = d
            j = s1
            s1 = s2
            s2 = j
        end
    end

    r = p
    for k in 1:derivative_order
        for j in 0:p
            derivatives[k+1, j+1] *= r
        end
        r *= (p + 1 - k)
    end

    return derivatives
end

function evaluate(b::BSplineBasis, u::Real)
    i = find_knot_span(b, u)
    evaluate(b, u, i)
end


# %%
function default_range(b::BSplineBasis, num::Integer=100)
    p = b.order
    start = b.knot_vector[1]
    stop = b.knot_vector[end - p]
    LinRange(start, stop, num)
end


@recipe function f(b::BSplineBasis, x=default_range(b); basis_function_index=nothing)

    xguide --> "Knots"

    p = b.order
    k = b.derivative_order
    number_of_basis_functions = length(b.knot_vector) - p - 1

    if basis_function_index == nothing
        bfi = 1:number_of_basis_functions
    else
        bfi = basis_function_index
    end

    N = zeros(Float64, (length(x), number_of_basis_functions))

    for (j, u) in enumerate(x)
        i = find_knot_span(b, u)
        N[j, (i-p):i] = evaluate(b, u, i)[k+1, :]
    end

    x, N[:, bfi]
end



# %%
struct BSplineCurve{S <: Real, T <: AbstractArray{S, 2}} <: AbstractVector{S}
    basis::BSplineBasis
    control_points::T
end

Base.length(c::BSplineCurve) = length(c.basis)
Base.size(c::BSplineCurve) = (length(c.basis),)
Base.getindex(c::BSplineCurve, i) = c.basis.knot_vector[i]

@recipe function f(c::BSplineCurve)
    x = default_range(c.basis)
    curve = zeros(Float64, (length(x), size(c.control_points)[2]))

    label --> ""

    @series begin
        seriestype := :path 
        primary := false
        linecolor := :lightgray
        markercolor := :red
        markershape := :circle
        tuple(eachcol(c.control_points)...)
    end

    for (j, u) in enumerate(x)
        i = find_knot_span(c.basis, u)
        curve[j, :] = sum(evaluate(c.basis, u, i)[1, :] .* c.control_points[(i-p):i, :], dims=1)
    end

    tuple(eachcol(curve)...)
end


# %%
# using Plots
# kv = [0, 0, 0, 1, 1, 1.]
# control_points = [0 0 0; 0.5 1.0 0.25; 1.0 0 0]
# p = 2;

# b = BSplineBasis(kv, p)
# plot(b)
# plot(derivative(b))

# c = BSplineCurve(b, control_points)
# plot(c)
