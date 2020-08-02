struct ConstantLayer
    w
end

ConstantLayer(N::Integer) = ConstantLayer(zeros(N))

(m::ConstantLayer)(x) = m.w

Flux.@functor ConstantLayer

mutable struct A2TNetwork
  base::Chain
  attn::Chain
  solutions::Array{Function} # function takes in state and outputs vector of action values
end

function (m::A2TNetwork)(input)
    b = m.base(input) #output is (Na, B)
    w = m.attn(input) #output is (Nt+1, B)
    B, Nt = size(input, 2), size(w,1) - 1
    if size(b,1) == 1
        # Collapse qs (form 3D) to 2D when action space is 1D.
        qs = Zygote.ignore(() -> vcat([mapslices(s, input; dims=1) for s in m.solutions]...)) #output is (B, Nt)
        return sum(w[1:Nt, :].*qs, dims=1) .+ w[Nt+1:Nt+1, :] .* b
    else
        qs = Zygote.ignore(() -> [hcat([s(input[:,i]) for s in m.solutions]...) for i=1:B]) #output is Bx(Na, Nt)
        return Flux.stack(qs .* Flux.unstack(w[1:Nt, :], 2), 2) .+ w[Nt+1:Nt+1, :] .* b
    end
end

Flux.@functor A2TNetwork

Flux.trainable(m::A2TNetwork) = (m.base, m.attn)

function Base.iterate(m::A2TNetwork, i=1)
    i > length(m.base.layers) + length(m.attn.layers) && return nothing
    if i <= length(m.base.layers)
        return (m.base[i], i+1)
    elseif i <= length(m.base.layers) + length(m.attn.layers)
        return (m.attn[i - length(m.base.layers)], i+1)
    end
end

function Base.deepcopy(m::A2TNetwork)
  A2TNetwork(deepcopy(m.base), deepcopy(m.attn), deepcopy(m.solutions))
end

## A2T Network with state transform
mutable struct A2TSTNetwork
  base::Chain
  attn::Chain
  strans::Chain
  solutions::Array{Any} # function takes in state and outputs vector of action values
end

function (m::A2TSTNetwork)(input)
    tinput = m.strans(input) # output is (Ni, B)
    b = m.base(tinput) #output is (Na, B)
    w = m.attn(tinput) #output is (Nt+1, B)
    B, Nt = size(input, 2), size(w,1) - 1
    qs = [hcat([s(tinput[:,i]) for s in m.solutions]...) for i=1:B] #output is Bx(Na, Nt)
    Flux.stack(qs .* Flux.unstack(w[1:Nt, :], 2), 2) .+ w[Nt+1:Nt+1, :] .* b
end

Flux.@functor A2TSTNetwork

Flux.trainable(m::A2TSTNetwork) = (m.base, m.attn, m.strans)

function Base.iterate(m::A2TSTNetwork, i=1)
    i > length(m.base.layers) + length(m.attn.layers) && return nothing
    if i <= length(m.base.layers)
        return (m.base[i], i+1)
    elseif i <= length(m.base.layers) + length(m.attn.layers)
        return (m.attn[i - length(m.base.layers)], i+1)
    elseif i <= length(m.base.layers) + length(m.attn.layers) + length(m.strans.layers)
        return (m.strans[i - length(m.base.layers) - length(m.attn.layers)], i+1)
    end
end


