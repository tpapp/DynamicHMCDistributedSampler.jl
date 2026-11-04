"""
$(DocStringExtensions.README)
"""
module DynamicHMCDistributedSampler

export distributed_sampler, collect_started_chains, load_mcmc

using ArgCheck: @argcheck
import Dates
using DocStringExtensions
using DynamicHMC
import JLD2
using LoggingExtras: with_logger, FileLogger
import Random

struct DistributedSampler{P<:AbstractString,L,K} <: Function
    path::P
    ∇ℓ::L
    N::Int
    kw_generator::K
end

"""
$(SIGNATURES)

Set up a *sampler*, which is a callable taking integer arguments `<index>` to

1. run `DynamicHMC.mcmc_with_warmup`,
2. save the *result* in `path_chain_<index>.jld2`,
3. save the *log* in `path_chain_<index>.log`

for each. In case there is an error, the message is written to the log file, and no
`jld` file will be created. The callable returns a boolean for success.

`path` should be of the form `directory/basename`, without extensions. It is created by
this function.

`∇ℓ` is a differentiable log density, conforming to the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) API.

`N` is the number of samples.

The keyword argument `kw_generator` can be a `NamedTuple`, or a function that returns a
`NamedTuple` of keywords, with no argument. These are passed to `mcmc_with_warmup`.
"""
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

"""
$(SIGNATURES)

Collect all chains that have a log file. Stop at the first index which does not have one.

Return a vector of `(; log_path, mcmc_path)`, where `mcmc_path ≡ nothing` indicates
missing MCMC results, which means that MCMC errored or has not terminated yet.
"""
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

"""
$(SIGNATURES)

Load the MCMC results, ie the output from `DynamicHMC.mcmc_with_warmup`. Return
`nothing` when there are no results, which indicates that the either the chain has not
terminated, or an error.
"""
function load_mcmc(chain)
    (; mcmc_path) = chain
    if mcmc_path ≡ nothing
        nothing
    else
        JLD2.load_object(mcmc_path)
    end
end

end # module
