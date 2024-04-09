#! /usr/bin/env julia

using Printf
using TOML
using Flux
using Comonicon
using ProgressMeter
using Statistics
using Plots
plotly()

if Flux.GPU_BACKEND == "AMDGPU"
    using AMDGPU
elseif Flux.GPU_BACKEND == "CUDA"
    using CUDA
end

# OKLAB {{{
const XYZ_M1::Matrix{Float32} = transpose([
    0.4124 0.3576 0.1805;
    0.2126 0.7152 0.0722;
    0.0193 0.1192 0.9505
])
const OKLAB_M1::Matrix{Float32} = [
    0.8189330101 0.0329845436 0.0482003018;
    0.3618667424 0.9293118715 0.2643662691;
    -0.1288597137 0.0361456387 0.6338517070
]
const OKLAB_M2::Matrix{Float32} = [
    0.2104542553 1.9779984951 0.0259040371;
    0.7936177850 -2.4285922050 0.7827717662;
    -0.0040720468 0.4505937099 -0.8086757660
]

const SRGBEOTF_ALPHA::Float32 = 0.055;
const SRGBEOTF_GAMMA::Float32 = 2.4;
const SRGBEOTF_PHI::Float32 = 12.92;
const SRGBEOTF_CHI::Float32 = 0.04045;
const SRGBEOTF_CHI_INV::Float32 = 0.0031308;

function srgb_eotf(n::Float32)::Float32
    if n <= SRGBEOTF_CHI
        return n / SRGBEOTF_PHI
    else
        return ((n + SRGBEOTF_ALPHA) / (1 + SRGBEOTF_ALPHA))^(SRGBEOTF_GAMMA)
    end
end

function srgb_eotf_inverse(n::Float32)::Float32
    if n <= SRGBEOTF_CHI_INV
        return n * SRGBEOTF_PHI
    else
        return (1 + SRGBEOTF_ALPHA) * (n^(1 / SRGBEOTF_GAMMA)) - SRGBEOTF_ALPHA
    end
end

function srgb_to_oklab(arr::AbstractArray{Float32})::Array{Float32}
    return .∛(srgb_eotf.(arr) * (XYZ_M1 * OKLAB_M1)) * OKLAB_M2
end

function oklab_to_srgb(arr::AbstractArray{Float32})::Array{Float32}
    return srgb_eotf_inverse.((arr * inv(OKLAB_M2)) .^ 3 * (inv(OKLAB_M1) * inv(XYZ_M1)))
end
# }}}

function train(network, source::AbstractArray, dest::AbstractArray; epochs::Int64=1000, rate::Float64=0.001)::Vector
    optim = Flux.setup(AdamW(rate), network)
    meter = Progress(epochs, dt=0.5, showspeed=true)
    losses = []
    for _ in 1:epochs
        loss, grads = Flux.withgradient(m -> mean(abs.(m(source) .- dest)), network)
        if !isfinite(loss)
            break
        end
        Flux.update!(optim, network, grads[1])
        push!(losses, loss)
        next!(meter, showvalues=[(:loss, loss)])
    end
    return losses
end

@main function main(measured_data::String)
    data = TOML.parsefile(measured_data)
    latent_dist::Array{Float32} = Array{Float32}(undef, length(data["data"][1]["latent_dist"]), length(data["data"]), length(data["data"][1]["latent_dist"][1]))
    rgb::Array{Float32} = Array{Float32}(undef, length(data["data"]), length(data["data"][1]["rgb"]))
    for (n, batch) in enumerate(data["data"])
        ld = cat(batch["latent_dist"]...; dims=3)
        latent_dist[:, n, :] = permutedims(ld, (3, 2, 1))
        rgb[n, :] = batch["rgb"]
    end
    oklab = srgb_to_oklab(rgb)

    device = gpu
    dtype = f32

    network = Flux.Chain(
                  Flux.Dense(4 => 4, swish),
                  Flux.Dense(4 => 3, x -> x),
              ) |> device |> dtype

    trim = 5
    latent_dist = latent_dist[1+trim:size(latent_dist)[1]-trim, :, :]
    colors = oklab

    performance = train(
        network,
        permutedims(latent_dist, (3, 2, 1)) |> device |> dtype,
        permutedims(colors, (2, 1)) |> device |> dtype;
        epochs=200_000,
        rate=2e-3
    )

    for param in network |> cpu
        println()
        println("[")
        for row in eachrow(param.weight)
            print("    [")
            for f in row
                @printf("%.6f, ", f)
            end
            println("],")
        end
        println("]")
        print("[")
        for f in param.bias
            @printf("%.6f, ", f)
        end
        println("]")
        println()
    end

    plot(performance, ylabel="loss", xlabel="epochs", ylims=(0, 0.1), show=true)
end
