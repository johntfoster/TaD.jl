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
import TaD: construct_helix_tangents, evaluate

@testset "reconstruct trajectory of helix in 3D" begin
    λ = helix(n=1000) #Helix Derivative
    Λ = helix(Ω1=(-π/2, 4*π-π/2), Ω2=(π/2, 4*π+π/2), n=1000, α=(1,-1,1)) # Original Helix
    tup = asc(λ[:,3], λ, [0.0,-1.,0.])
    Curve = reconstruct_trajectory(hcat(tup[1], tup[2], λ[:,3]))
    @test norm(evaluate(Curve, length(λ[:,1])) - Λ) < .01
end

@testset "Abughaban test data" begin
    MD = collect(5000:100:5900)
    θ, ϕ = to_radian(collect(0:5:45)), to_radian(collect(0:10:90))
    λ = asc_tangents(MD, θ, ϕ)
    tup = asc(MD, λ, [0., 0., 5000.])
    Curve = reconstruct_trajectory(hcat(tup...))
    N = [0.00 8.69 34.29 75.46 130.05 195.24 267.75 344.03 420.52 493.85]
    @test norm(evaluate(Curve, length(MD))[:,2] - N') < 1
end
