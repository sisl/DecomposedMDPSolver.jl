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
    Na, B, Nt  = size(b,1), size(b, 2), size(w,1) - 1

    qs = Array{Float32}(undef, Nt, Na, B)
    Zygote.ignore(() -> begin
        for i=1:B, s=1:Nt
            qs[s, :, i] .= m.solutions[s](input[:, i])
        end
    end)
    sum(qs .* Flux.unsqueeze(w[1:Nt, :], 2), dims=1)[1,:,:] .+ w[Nt+1:Nt+1, :] .* b
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
  A2TNetwork(deepcopy(m.base), deepcopy(m.attn), m.solutions)
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
    Na, B, Nt  = size(b,1), size(b, 2), size(w,1) - 1
    qs = Zygote.Buffer(Array{Float32}(undef, Nt, Na, B))
    for i=1:B, s=1:Nt
        qs[s, :, i] = m.solutions[s](input[:, i])
    end
    sum(copy(qs) .* Flux.unsqueeze(w[1:Nt, :], 2), dims=1)[1,:,:] .+ w[Nt+1:Nt+1, :] .* b
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


