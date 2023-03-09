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
import TaD: evaluate, asc, reconstruct_trajectory, fit_bspline, reconstruct_synthetic

@testset "Helix in 3D" begin
    N = 1000
    period=4π
    dS = period / N
    ζ, δ = 1, 1
    helix(t) = [ζ*sin(t); δ*cos(t); t]
    helix_ders(t) = [ζ*cos(t); -δ*sin(t); 1]
    reconstruct, trajectory = reconstruct_synthetic(helix, helix_ders, N; period=period, method="ASC")
    BsplineReconstruct = fit_bspline(reconstruct)
    BsplineActual = fit_bspline(trajectory)
    @test norm(evaluate(BsplineReconstruct, size(reconstruct, 1)) - evaluate(BsplineActual, size(reconstruct, 1)), Inf) < .1
end

@testset "Scaled Helix in 3D" begin
    N = 1000
    period=4π
    dS = period / N
    ζ, δ = 3, 2
    helix(t) = [ζ*sin(t); δ*cos(t); t]
    helix_ders(t) = [ζ*cos(t); -δ*sin(t); 1]
    reconstruct, trajectory = reconstruct_synthetic(helix, helix_ders, N; period=period, method="ASC")
    BsplineReconstruct = fit_bspline(reconstruct)
    BsplineActual = fit_bspline(trajectory)
    @test norm(evaluate(BsplineReconstruct, size(reconstruct, 1)) - evaluate(BsplineActual, size(reconstruct, 1)), Inf) < 1
end

@testset "Synthetic Trajectory" begin
    N = 500
    period= 1.0
    dS = period / N
    α, ζ, η, δ = 6000, 300, 80, 3000
    synthetic(t) = [α * t^2; ζ * t * sin(η*t); δ*(2*t-t^2)]
    synthetic_ders(t) = [2 * α * t; ζ*sin(η*t) + η*ζ*t*cos(η*t); δ*(2-2*t)]
    reconstruct, trajectory = reconstruct_synthetic(synthetic, synthetic_ders, N; period=period, method="ASC")
    trajectory = copy(hcat(synthetic.(LinRange(0,period, N))...)')
    BsplineReconstruct = fit_bspline(reconstruct)
    BsplineActual = fit_bspline(trajectory)
    @test norm(evaluate(BsplineReconstruct, size(reconstruct, 1)) - evaluate(BsplineActual, size(reconstruct, 1)), Inf) < 30
end

@testset "Abughaban Sample Calculation" begin
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
