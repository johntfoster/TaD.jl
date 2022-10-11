using LinearAlgebra, IterativeSolvers

export asc

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

function asc(MD, λ, init::Vector{Float64} = [])
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
    if length(init) > 0
        E = E .+ init[1]
        pushfirst!(E, init[1])
        N = N .+ init[2]
        pushfirst!(N, init[2])
        TVD = TVD .+ MD[1]
        pushfirst!(TVD, init[3])
    end
    return tuple(E, N, TVD)
end
