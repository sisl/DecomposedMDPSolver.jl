struct ConstantLayer
    w
end

ConstantLayer(N::Integer) = ConstantLayer(zeros(N))

(m::ConstantLayer)(x) = m.w

Flux.@functor ConstantLayer

mutable struct A2TNetwork
  base::Chain
  attn::Chain
  solutions::Array{Any} # function takes in state and outputs vector of action values
  ignore_solution_computation::Bool
end

# By default, ignore the computation of the solutions
A2TNetwork(base, attn, solutions) = A2TNetwork(base, attn, solutions, true)

function (m::A2TNetwork)(input)
    b = m.base(input) #output is (Na, B)
    w = m.attn(input) #output is (Nt+1, B)
    Na, B, Nt  = size(b,1), size(b, 2), size(w,1) - 1

    if m.ignore_solution_computation
        qs = Array{Float32}(undef, Nt, Na, B)
        Zygote.ignore(() -> begin
            for i=1:B, s=1:Nt
                qs[s, :, i] .= m.solutions[s](input[:, i])
            end
        end)
    else
        qs = Flux.stack([m.solutions[s](input) for s in 1:Nt], 1)
    end
    sum(qs .* Flux.unsqueeze(w[1:Nt, :], 2), dims=1)[1,:,:] .+ w[Nt+1:Nt+1, :] .* b
end

Flux.@functor A2TNetwork

Flux.trainable(m::A2TNetwork) = (m.base, m.attn, (!m.ignore_solution_computation ? m.solutions : ())...)

function Base.iterate(m::A2TNetwork, i=1)
    layers = [m.base.layers..., m.attn.layers...]
    if !m.ignore_solution_computation
        for s in m.solutions
            for l in s
                push!(layers, l)
            end
        end
    end
    i > length(layers) && return nothing
    return (layers[i], i+1)
end

## A2T Network with state transform
mutable struct A2TSTNetwork
  base::Chain
  attn::Chain
  strans::Array{Chain}
  solutions::Array{Any} # function takes in state and outputs vector of action values
end

function (m::A2TSTNetwork)(input)
    b = m.base(input) #output is (Na, B)
    w = m.attn(input) #output is (Nt+1, B)
    Na, B, Nt  = size(b,1), size(b, 2), size(w,1) - 1
    qs = Zygote.Buffer(Array{Float32}(undef, Nt, Na, B))
    for s=1:Nt
        tinput = m.strans[s](input)
        for i=1:B
            qs[s, :, i] = m.solutions[s](tinput[:, i])
        end
    end
    sum(copy(qs) .* Flux.unsqueeze(w[1:Nt, :], 2), dims=1)[1,:,:] .+ w[Nt+1:Nt+1, :] .* b
end

Flux.@functor A2TSTNetwork

Flux.trainable(m::A2TSTNetwork) = (m.base, m.attn, m.strans...)

function Base.iterate(m::A2TSTNetwork, i=1)
    layers = [m.base.layers..., m.attn.layers...]
    for net in m.strans
        push!(layers, net.layers...)
    end
    i > length(layers) && return nothing
    return (layers[i], i+1)
end

## A2T network with fine tuning

mutable struct A2TFTNetwork
  base::Chain
  attn::Chain
  solutions::Array{Any} # function takes in state and outputs vector of action values
  finetune::Array{Chain} # fine tune network, one for each soluion
end

function (m::A2TFTNetwork)(input) # input is (Ni, B)
    Ni = size(input, 1)
    b = m.base(input) #output is (Na, B)
    w = m.attn(input) #output is (Nt+1, B)
    Na, B, Nt  = size(b,1), size(b, 2), size(w,1) - 1

    qs = Array{Float32}(undef, Nt, Na + Ni, B)
    Zygote.ignore(() -> begin
        for i=1:B, s=1:Nt
            qs[s, 1:Na, i] .= m.solutions[s](input[:, i])
            qs[s, Na+1:Na+Ni, i] .= input[:, i]
        end
    end)

    ft = Zygote.Buffer(Array{Float32}(undef, Nt, Na, B))
    for s=1:Nt
        ft[s, :, :] = m.finetune[s](qs[s, :, :])
    end

    sum(copy(ft) .* Flux.unsqueeze(w[1:Nt, :], 2), dims=1)[1,:,:] .+ w[Nt+1:Nt+1, :] .* b
end

Flux.@functor A2TFTNetwork

Flux.trainable(m::A2TFTNetwork) = (m.base, m.attn, m.finetune...)

function Base.iterate(m::A2TFTNetwork, i=1)
    layers = [m.base.layers..., m.attn.layers...]
    for net in m.finetune
        push!(layers, net.layers...)
    end
    i > length(layers) && return nothing
    return (layers[i], i+1)
end


## A Fine-tune network with choice of number of layers to freeze

mutable struct FTNetwork
  net::Chain
  train_layers::Array{Int}
end

(m::FTNetwork)(input) = m.net(input)
Flux.@functor FTNetwork

Flux.trainable(m::FTNetwork) = m.net.layers[m.train_layers]

function Base.iterate(m::FTNetwork, i=1)
    i > length(m.train_layers) && return nothing
    return (m.net.layers[m.train_layers[i]], i+1)
end


