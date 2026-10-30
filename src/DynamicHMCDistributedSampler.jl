"""
Placeholder for a short summary about DynamicHMCDistributedSampler.
"""
module DynamicHMCDistributedSampler

export distributed_sampler, collect_started_chains, load_mcmc

using ArgCheck: @argcheck
import Dates
using DynamicHMC
import JLD2
using LoggingExtras: with_logger, FileLogger
import Random

struct DistributedSampler{P<:AbstractString,L,K}
    path::P
    ∇ℓ::L
    N::Int
    kw_generator::K
end

function distributed_sampler(path::AbstractString, ∇ℓ, N::Int; kw_generator = (;))
    d, b = splitdir(path)
    @argcheck !isempty(d) "Directory part of $(path) is empty"
    @argcheck !isempty(b) "Basename part of $(path) is empty"
    e = splitext(b)[2]
    @argcheck isempty(e) "Path $(path) has a nonempty extension $(e)"
    mkpath(d)
    DistributedSampler(path, ∇ℓ, N, kw_generator)
end

function _filenames(path::AbstractString, i::Int)
    common = path * "_chain_$i"
    log_path = common * ".log"
    mcmc_path = common * ".jld2"
    (; log_path, mcmc_path)
end

make_keywords(kw_generator::Function) = kw_generator()

make_keywords(kw::NamedTuple) = kw

function (sampler::DistributedSampler)(i::Int)
    (; path, ∇ℓ, N, kw_generator) = sampler
    (; log_path, mcmc_path) = _filenames(path, i)
    @argcheck !isfile(mcmc_path) "Won't overwrite $(mcmc_path), remove"
    @argcheck !isfile(log_path) "Won't overwrite $(log_path), remove"
    open(log_path, "w") do io
        println(io, "Starting MCMC chain $i at ", Dates.now())
    end
    try
        with_logger(FileLogger(log_path; append = true)) do
            mcmc = mcmc_with_warmup(Random.default_rng(), copy(∇ℓ), N;
                                    make_keywords(kw_generator)...)
            JLD2.save_object(mcmc_path, mcmc)
        end
        true
    catch e
        bt = catch_backtrace()
        open(log_path, "a") do io
            showerror(io, e, bt)
        end
        false
    end
end

function collect_started_chains(path::AbstractString)
    chains = Vector{@NamedTuple{log_path::String, mcmc_path::Union{String,Nothing}}}()
    i = 1
    while true
        (; log_path, mcmc_path) = _filenames(path, i)
        isfile(log_path) || break
        if !isfile(mcmc_path)
            mcmc_path = nothing
        end
        push!(chains, (; log_path, mcmc_path))
        i += 1
    end
    chains
end

function load_mcmc(chain)
    (; mcmc_path) = chain
    if mcmc_path ≡ nothing
        nothing
    else
        JLD2.load_object(mcmc_path)
    end
end

end # module
