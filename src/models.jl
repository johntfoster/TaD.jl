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
import DifferentialEquations: ODEProblem, solve
using LinearAlgebra

"""
    soft_string_drag(dF, F, p, s)

Defines the soft string ordinary differential equation for the axial force in
a drillstring while tripping in/out of a well, i.e.

\$\\frac{dF_t}{ds} = -w_{bp} \\hat{t}_3 + \\mu w_c,\$

where

\$w_{bp} = A g (\\rho_p - \\rho_m),\$

and

\$w_c = \\sqrt{\\left(F_t \\kappa + w_{bp} \\hat{n}_3\\right)^2 + \\left(w_{bp} \\hat{b}_3 \\right)^2}.\$

\$\\hat{t}, \\hat{n}, \\hat{b}\$ are the tangential, normal, and bi-normal vectors 
in the Frenet frame that follows the drillstring trajectory.  From the position 
vector along trajectory curve defined by the B-Splines \$\\vec{r} = \\vec{r}(s)\$ 
these are defined as

\$\\hat{t} = \\frac{\\vec{r}^\\prime}{\\left\\Vert \\vec{r}^\\prime \\right\\Vert},\$

\$\\hat{b} = \\frac{\\vec{r}^{\\prime\\prime}}{\\left\\Vert \\vec{r}^{\\prime\\prime} \\right\\Vert},\$

\$\\hat{n} = \\hat{t} \\times \\hat{b},\$

and finally, \$\\kappa\$ is the curvature defined by

\$\\kappa = \\frac{\\left\\Vert \\vec{r}^{\\prime} \\times \\vec{r}^{\\prime\\prime} \\right\\Vert}{\\left\\Vert \\vec{r}^\\prime \\right\\Vert^3}\$.

\$\\mu\$ is the friction factor, \$\\rho_p\$ and \$\\rho_m\$ are the densities of the 
drill pipe and drilling mud respectively, \$A\$ is the cross-sectional area of 
the pipe, and \$g\$ is acceration due to gravity.

# Arguments
- `p::Vector{<:Number}`: Vector that defines the parameters of the problem.  
  Must be in the following order:

  p = [μ, rₒ, rᵢ, g, ρₛ, ρₘ, c]

  where

  μ - friction factor
  rₒ - outer radius of drill pipe
  rᵢ - inner radius of drill pipe
  g - acceleration due to gravity
  ρₛ - density of drill pipe
  ρₘ - density of drilling mud
  c - `BSplineCurve` that defines the wellbore trajectory
"""
function soft_string_drag(dF,F,p,s)
  Fₜ, = F
  μ, rₒ, rᵢ, g, ρₛ, ρₘ, c = p

  w_dp = π / 4 * (rₒ ^ 2 - rᵢ ^ 2)  * g * (ρₛ- ρₘ)
    
  curves = c(s)

  r⃗′   = curves[2,:]
  r⃗′′  = curves[3,:]
  
  t̂ = r⃗′ / norm(r⃗′)
  n̂ = r⃗′′ / norm(r⃗′′)
  b̂ = t̂ × n̂
    
  κ = norm(r⃗′ × r⃗′′) / norm(r⃗′) ^ 3
    
  wₛ = sqrt((Fₜ * κ + w_dp * n̂[3])^ 2 + (w_dp * b̂[3]) ^ 2)
    
  dF[1] =  -w_dp * t̂[3] + μ * wₛ
end

function stiff_string_drag!(dF,F,p,s)
  Fₜ, Fₙ, F_b = F
  μ, E, rₒ , rᵢ, g, ρₛ, ρₘ, c = p

  EI = E * π * (rₒ ^ 4 - rᵢ ^ 4) / 64
  w_dp = π / 4 * (rₒ ^ 2 - rᵢ ^ 2)  * g * (ρₛ- ρₘ)

  curves = c(s)

  r⃗′    = curves[2,:]
  r⃗′′   = curves[3,:]
  r⃗′′′  = curves[4,:]
  
  t̂ = r⃗′ / norm(r⃗′)
  b̂ = (r⃗′ × r⃗′′) /  norm(r⃗′ × r⃗′′)
  n̂ = b̂ × t̂
    
  κ = norm(r⃗′ × r⃗′′) / norm(r⃗′) ^ 3
  τ = r⃗′ ⋅ (r⃗′′ × r⃗′′′) / norm(r⃗′ × r⃗′′) ^ 2

  # c⃗′′′ = (r⃗′′′ * norm(r⃗′) ^ 3 / 
          # ((r⃗′ ⋅ r⃗′) * (r⃗′ ⋅ r⃗′′′ + r⃗′′ ⋅ r⃗′′) - (r⃗′ ⋅ r⃗′′) ^ 2))
  # κ′ = c⃗′′′ ⋅ n̂

    
  # a = Fₙ+ EI * κ′
  # b = -F_b - EI * κ * τ
  a = Fₜ * κ + w_dp * n̂[3]
  b = w_dp * b̂[3]
  wₛ = sqrt(a ^ 2 + b ^ 2) # / μ / rₒ
  θ = atan(b, a)
    
  dF[1] =  κ * Fₙ - w_dp * t̂[3] + μ * wₛ
  dF[2] = -κ * Fₜ - w_dp * n̂[3] - wₛ * cos(θ) - F_b * τ
  dF[3] =  τ * Fₙ - w_dp * b̂[3] + wₛ * sin(θ) 
end

export stiff_string_drag!, soft_string_drag

# %%
#
