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
export reconstruct_trajectory, create_knot_vector, asc, mcm, fit_bspline, error, printerror, reconstruct_synthetic, example_helix, asc_tangents

function error(orig, reconstruct)
    error = map((col)->norm(reconstruct[:,col]-orig[:,col], Inf), [1 2 3])
    return error
end

function printerror(orig, reconstruct)
    return "$(round.(error(orig, reconstruct), digits=3))"
end

function rk4(∇f, x, δ)
    return (∇f.(x) + 4*∇f.(x.+δ/2) + ∇f.(x.+δ))*δ/6
end

function asc_tangents(MD, θ, ϕ)
    λ_E   = sin.(ϕ) .* sin.(θ)
    λ_N   = sin.(ϕ) .* cos.(θ)
    λ_TVD = cos.(ϕ)
    return [λ_E λ_N λ_TVD] #x y z
end

function reconstruct_synthetic(f::Function, df::Function, N::Int64; period::Float64=1.0, method="ASC")
    dS = period / N
    del_arc_length(t) = t > 0 ? sqrt.(sum(df(t).^2)) : 0
    trajectory = copy(hcat(f.(LinRange(dS,period,N))...)')
    derivatives = hcat(df.(LinRange(dS,period,N))...)'
    φp = del_arc_length.(LinRange(dS,period,N)) #\varphi
    MD = cumsum(φp * dS)
    tangents = derivatives ./ φp
    if method == "RK4"
        reconstruct = copy(hcat(f(0), hcat(cumsum(rk4(df, LinRange(dS,period-dS,N-1), dS))...) .+ f(0))')
    elseif method == "MCM"
        reconstruct = mcm(MD, tangents, init=f(dS))
    else
        reconstruct = asc(MD, tangents, init=f(dS))
    end
    return reconstruct, trajectory
end

function construct_z_matrix(s, h, u)
    Z = zeros(Float64, (length(s)-2, length(s)-2)) # Z = v1 to vn-1
    Z[1,1] = (u[2] + h[1] + (h[1]*h[1])/h[2])
    Z[1,2] = (h[2] - (h[1]*h[1])/h[2]) #Z[1,1]z1 + Z[1,2]z2 = v1
    for i=2:size(Z,1)-1
        Z[i, i-1] = h[i-1]
        Z[i, i] = u[i]
        Z[i, i+1] = h[i]
    end
    Z[end,end-1] = (h[end-1] - (h[end]*h[end])/(h[end-1])) #
    Z[end,end] = (u[end] + h[end] + (h[end]*h[end])/(h[end-1])) #h,u = n-1 long
    return Z
end

function spline_element_approx(λ, z, h, index)
    A = λ[index]
    B = (λ[index+1] - λ[index])/h[index] - (z[index+1]*h[index])/6 - (z[index]*h[index])/3
    C = (z[index])/2
    D = (z[index+1] - z[index])/(6*h[index])
    T = A*h[index] + (B*h[index]^2)/2 + (C*h[index]^3)/3  + (D*h[index]^4)/4
    return T
end

function calc_h(MD)
    hs = ones(length(MD)-1) #i=0,...,n-1
    hs[1] = MD[2] - MD[1]
    for i=2:length(hs)
        hs[i] = MD[i]-MD[i-1]
    end
    return hs
end

function calc_u(h)
    us = ones(length(h)-1) #i=1,...,n-1
    us[1] = h[1]
    for i=2:length(us)
        us[i] = 2*(h[i]+h[i-1])
    end
    return us
end

function asc(MD, λ; init::Vector{<:AbstractFloat} = ones(0))
    h = calc_h(MD) 
    u = calc_u(h)
    mapcolu = (col)->6*((col[3:end]- col[2:end-1]) - (col[2:end-1] - col[1:end-2]))./h[1:end-1]
    v = hcat(map(mapcolu, eachcol(λ))...)
    Zmat = construct_z_matrix(MD,h,u)
    z = (Zmat' * Zmat) \ (Zmat' * v) #Zmat' # Solve Az = v -> inv(Zmat'*Zmat)*Zmat'*v = z
    z = hcat(map((col) -> pushfirst!(z[:,col], (z[1,col] + h[1]*(z[1,col] - z[2,col])/h[2])), [1;2;3])...)
    z = hcat(map((col) -> append!(z[:,col], (z[end-1,col] + h[end]*(z[end-1,col] - z[end-2,col])/h[end-1])), [1;2;3])...)
    E = cumsum(map(x->spline_element_approx(λ[:,1],z[:,1],h,x), 1:1:(length(MD)-1)))
    N = cumsum(map(x->spline_element_approx(λ[:,2],z[:,2],h,x), 1:1:(length(MD)-1)))
    TVD = cumsum(map(x->spline_element_approx(λ[:,3],z[:,3],h,x), 1:1:(length(MD)-1)))
    if length(init) > 0
        E = E .+ init[1]
        pushfirst!(E, init[1])
        N = N .+ init[2]
        pushfirst!(N, init[2])
        TVD = TVD .+ init[3]
        pushfirst!(TVD, init[3])
    end
    return hcat(E, N, TVD)
end

function mcm(MD, λ; init::Vector{<:AbstractFloat} = ones(0)) h = TaD.calc_h(MD)
    u = TaD.calc_u(h)
    mapcolu = (col)->6*((col[3:end]- col[2:end-1]) - (col[2:end-1] - col[1:end-2]))./h[1:end-1]
    v = hcat(map(mapcolu, eachcol(λ))...)
    Zmat = construct_z_matrix(MD,h,u)
    z = (Zmat' * Zmat) \ (Zmat' * v) #Zmat' # Solve Az = v -> inv(Zmat'*Zmat)*Zmat'*v = z
    z = hcat(map((col) -> pushfirst!(z[:,col], 0), [1;2;3])...) #Natural Spline Boundary Conditions
    z = hcat(map((col) -> append!(z[:,col], 0), [1;2;3])...)
    E = cumsum(map(x->spline_element_approx(λ[:,1],z[:,1],h,x), 1:1:(length(MD)-1)))
    N = cumsum(map(x->spline_element_approx(λ[:,2],z[:,2],h,x), 1:1:(length(MD)-1)))
    TVD = cumsum(map(x->spline_element_approx(λ[:,3],z[:,3],h,x), 1:1:(length(MD)-1)))
    if length(init) > 0
        E = E .+ init[1]
        pushfirst!(E, init[1])
        N = N .+ init[2]
        pushfirst!(N, init[2])
        TVD = TVD .+ init[3]#(init[3] - TVD[1])
        pushfirst!(TVD, init[3])
    end
    return hcat(E, N, TVD)
    
end

function create_knot_vector(Qk::Array{<:AbstractFloat}, p::Integer = 3)
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

function fit_bspline(Qk::Array{<:AbstractFloat}; p::Integer=3)
    kv = create_knot_vector(Qk, p)
    ū = create_ūk(Qk)
    basis = BSplineBasis(kv, p, k=3)
    N = construct_spline_matrix(basis, ū, length(kv), p)
    tangent_control_points = hcat(map(col -> lsmr(N, col), eachcol(Qk))...)
    curve = BSplineCurve(basis, tangent_control_points)
    return curve
end


function reconstruct_trajectory(MD::Vector{<:AbstractFloat}, λ::Array{<:AbstractFloat}; init::Vector{<:AbstractFloat} = [], p::Integer=3, method="ASC")
    if method=="MCM"
        Qk = mcm(MD, λ, init=init)
    else
        Qk = asc(MD, λ, init=init)
    end
    curve = fit_bspline(Qk, p=p)
    return curve
end

function construct_spline_matrix(basis::BSplineBasis, samples::Vector{<:AbstractFloat}, num_knots::Integer, p::Integer=3)
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

function example_helix(N=1000, method="ASC"; δ=2, ζ=3)
    period=4π
    dS = period / N
    helix(t) = [ζ*sin(t); δ*cos(t); t]
    helix_ders(t) = [ζ*cos(t); -δ*sin(t); 1]
    reconstruct, trajectory = reconstruct_synthetic(helix, helix_ders, N; period=period, method=method)
    reconstruct = fit_bspline(reconstruct)
    original = fit_bspline(trajectory)
    return reconstruct, original
end

