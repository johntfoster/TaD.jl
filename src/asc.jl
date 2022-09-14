using Revise
using Plots
using LinearAlgebra, IterativeSolvers, Plots

function construct_z_matrix(s, h, u)
    Z = zeros(Float64, (length(s)-2, length(s)-2)) # Z = v1 to vn-1
    Z[1,1] = (u[2] + h[1] + (h[1]*h[1])/h[2])
    Z[1,2] = (h[1] - (h[1]*h[1])/h[2]) #Z[1,1]z1 + Z[1,2]z2 = v1
    for i=2:size(Z,1)-1
        Z[i, i-1] = h[i-1]
        Z[i, i] = u[i]
        Z[i, i+1] = h[i]
    end
    Z[end,end-1] = (h[end-2] - (h[end-1]*h[end-1])/(h[end-2]))
    Z[end,end] = (u[end-1] + h[end-1] + (h[end-1]*h[end-1])/(h[end-2]))
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

function spline_element_function(x, s, λ, z, h, index)
    A = λ[index]
    B = (λ[index+1] - λ[index])/h[index] - (z[index+1]*h[index])/6 - (z[index]*h[index])/3
    C = (z[index])/2
    D = (z[index+1] - z[index])/6*h[index]
    T(x) = A + B*(x - s[index]) + C*(x - s[index])^2 + D*(x - s[index])^3
    return T(x)
end

function find_span(x, s)
    return s[ s .> x][1]
end

function helix_tangents(MD, θ, ϕ)
    λ_E   = cos.(θ)
    λ_N   = sin.(θ)
    λ_TVD = (MD[2]-MD[1]).*ones(length(MD))#cos.(ϕ)
    return [λ_E λ_N λ_TVD] #x y z
end

function asc_tangents(MD, θ, ϕ)
    λ_E   = sin.(ϕ) .* sin.(θ)
    λ_N   = sin.(ϕ) .* cos.(θ)
    λ_TVD = cos.(ϕ)
    return [λ_E λ_N λ_TVD] #x y z
end

function to_radian(θ)
    #if  ((( (θ[end]) / π ) % .33333) == 0) | ((((θ[end]) / π) % .25) == 0)
    if θ[end] > 22
        return (2*π/360).*θ
    else
        return θ
    end
end


function asc(MD, λ)
    h = (MD[2]-MD[1])*ones(length(MD)-1) #Uniform
    u = 2*(h[2]+h[1])*ones(length(h)) #Uniform
    mapcolu = (col)->6*((col[3:end]- col[2:end-1]) - (col[2:end-1] - col[1:end-2]))./h[1:end-1]
    v = hcat(map(mapcolu, eachcol(λ))...)
    Zmat = construct_z_matrix(MD,h,u)
    z = inv(Zmat'*Zmat)*Zmat'*v # Solve Az = v -> inv(Zmat'*Zmat)*Zmat'*v = z
    z = hcat(map((col) -> pushfirst!(z[:,col], (z[1,col] - h[1]*(z[2,col] - z[1,col])/h[2])), [1;2;3])...)
    z = hcat(map((col) -> append!(z[:,col], (z[end-1,col] - h[1]*(z[end-1,col] - z[end-2,col])/h[2])), [1;2;3])...)
    E = cumsum(map(x->spline_element_approx(λ[:,1],z[:,1],h,x), 1:1:(length(MD)-1)))
    N = cumsum(map(x->spline_element_approx(λ[:,2],z[:,2],h,x), 1:1:(length(MD)-1)))
    TVD = cumsum(map(x->spline_element_approx(λ[:,3],z[:,3],h,x), 1:1:(length(MD)-1)))
    pushfirst!(E, 0)
    pushfirst!(N, 0)
    TVD = TVD .+ MD[1]
    pushfirst!(TVD, MD[1])
    return tuple(N, E, TVD)
end

function plot_results(asc, tangents)
    plot(tuple(asc[1], asc[2], asc[3]), zaxis=(:flip))
    plot!(tuple(tangents[:,1], tangents[:,2], 1:1:length(tangents[:,1])))
end

function validate_toy()
    MD = collect(5000:100:5900)
    θ, ϕ = collect(0:5:45), collect(0:10:90)
    θ = to_radian(θ)
    ϕ = to_radian(ϕ)
    λ = asc_tangents(MD, θ, ϕ)
    tup = asc(MD, λ)
    return tup, λ
end

function validate_helix()
    s = collect(1:1:100)
    θ, ϕ = range(0,4*pi,length(s)), range(0, 4*pi,length(s))
    λ = helix_tangents(s, θ, ϕ)
    tup = asc(s, λ)
    return (tup[1]./maximum(tup[1]), tup[2]./maximum(tup[2]), tup[3]), λ
end

#plot(tuple(col1, col2, col3)...)
#title!("Reconstruct")
#savefig("reconstruct.png")
#plot!(tuple(cos.(θ).*cos.(ϕ), -cos.(ϕ).*sin.(θ), sin.(ϕ))...)
#s = collect(0:30:30000)[1:end-1] #1000 samples
#λ = construct_tangents(s, θ, ϕ)
#h = ones(length(s)-1) #Uniform
#u = ones(length(h)-1) #Uniform
#mapcolu = (col)->6*((col[3:end]- col[2:end-1]) + (col[2:end-1] - col[1:end-2]))./h
#v = hcat(map(mapcolu, eachcol(λ))...)

#Zmat = construct_z_matrix(s,h,u)
#solver = (b)->gmres(Zmat, b)
#z = hcat(map(solver, eachcol(v))...)
## Z0 = Z1 - h0(z2-z1)/h1
# Zn = Zn-1 - h0(zn-2-zn-1)/h1
#map((col) -> pushfirst!(z[:,col], (z[1,col] - h[1]*(z[2,col] - z[1,col])/h[2])), [1;2;3])
#map((col) -> append!(z[:,col], (z[end-1,col] - h[1]*(z[end-1,col] - z[end-2,col])/h[2])), [1;2;3])


#col1 = map(x->spline_element_approx(λ[:,1],z[:,1],h,x), 1:1:length(s)-2)
#col2 = map(x->spline_element_approx(λ[:,2],z[:,2],h,x), 1:1:length(s)-2)
#col3 = map(x->spline_element_approx(λ[:,3],z[:,3],h,x), 1:1:length(s)-2)
#plot(tuple(col2, col3, col1))
#plotlyjs()
#plot(tuple(col1, col2, col3)...)
#title!("Reconstruct")
#savefig("reconstruct.png")
#plot!(tuple(cos.(θ).*cos.(ϕ), -cos.(ϕ).*sin.(θ), sin.(ϕ))...)
#title!("Integral")
#savefig("integral.png")
#plot(tuple(λ[:,1], λ[:,2], λ[:,3])...)
#title!("Tangents")
#savefig("tangents.png")
