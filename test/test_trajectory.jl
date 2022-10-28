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
using Test
using TaD
using LinearAlgebra
import TaD: evaluate, asc, reconstruct_trajectory, fit_bspline

@testset "reconstruct trajectory of helix in 3D" begin
    function helix(;Ω1::Tuple=(0.,4*π), Ω2::Tuple=(0.,4*π), n::Int64=100, α::Tuple=(1.,1.,1.))
        """
        Constructs nx3 Matrix [α1cos.(θ) α2sin.(θ), α3LinRange(Ω, n+1)]
        """
        arr = zeros(Float64, (n+1, 3))
        arr[:, 1] = α[1]*cos.(LinRange(Ω1[1], Ω1[2], n+1))
        arr[:, 2] = α[2]*sin.(LinRange(Ω2[1], Ω2[2], n+1))
        arr[:, 3] = α[3]*LinRange(0, Ω1[2]-Ω1[1], n+1)
        arr
    end
    λ = helix(n=1000) #Helix Derivative
    Λ = helix(Ω1=(-π/2, 4*π-π/2), Ω2=(π/2, 4*π+π/2), n=1000, α=(1,-1,1)) # Original Helix
    tup = asc(λ[:,3], λ, [0.0,-1.,0.])
    Curve = fit_bspline(hcat(tup[1], tup[2], λ[:,3]))
    @test norm(evaluate(Curve, length(λ[:,1])) - Λ) < .01
end

@testset "Abughaban test data" begin
    function asc_tangents(MD, θ, ϕ)
        λ_E   = sin.(ϕ) .* sin.(θ)
        λ_N   = sin.(ϕ) .* cos.(θ)
        λ_TVD = cos.(ϕ)
        return [λ_E λ_N λ_TVD] #x y z
    end
    MD = collect(5000.:100.:5900.)
    θ, ϕ = (2π/360) .* collect(0:5:45), (2π/360) .* collect(0:10:90)
    λ = asc_tangents(MD, θ, ϕ)
    Curve = reconstruct_trajectory(MD, λ, init=[0., 0., 5000.], p=3)
    N = [0.00 8.69 34.29 75.46 130.05 195.24 267.75 344.03 420.52 493.85]
    @test norm(evaluate(Curve, length(MD))[:,2] - N') < 1
end
