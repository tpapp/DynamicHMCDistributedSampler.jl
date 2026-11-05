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

struct DistributedSampler{P<:AbstractString,M,K,C,L,S} <: Function
    path::P
    model::M
    N::Int
    kw_generator::K
    copy_model::C
    prepare_logdensity::L
    collect_statistics::S
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

`model`: see below.

`N` is the number of samples.

The keyword argument `kw_generator` can be a `NamedTuple`, or a function that returns a
`NamedTuple` of keywords, with no argument. These are passed to `mcmc_with_warmup`.

For the other keyword arguments, see below.

# Model interface

`model` is used in the following way:

1. first, it is copied for each worker using `copy_model`.

2. `prepare_logdensity` is called on the copy, should yield a differentiable log
density, conforming to the
[LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) API.

3. `collect_statistics` is called on the copy, the result is appended as
`model_statistics` to the MCMC results.
"""
function distributed_sampler(path::AbstractString, model, N::Int;
                             kw_generator = (;),
                             copy_model = identity,
                             prepare_logdensity = identity,
                             collect_statistics = () -> nothing)
    d, b = splitdir(path)
    @argcheck !isempty(d) "Directory part of $(path) is empty"
    @argcheck !isempty(b) "Basename part of $(path) is empty"
    e = splitext(b)[2]
    @argcheck isempty(e) "Path $(path) has a nonempty extension $(e)"
    mkpath(d)
    DistributedSampler(path, model, N, kw_generator,
                       copy_model, prepare_logdensity, collect_statistics)
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
    (; path, model, N, kw_generator, copy_model, prepare_logdensity,
     collect_statistics) = sampler
    (; log_path, mcmc_path) = _filenames(path, i)
    @argcheck !isfile(mcmc_path) "Won't overwrite $(mcmc_path), remove"
    @argcheck !isfile(log_path) "Won't overwrite $(log_path), remove"
    with_logger(FileLogger(log_path; append = true)) do
        @info "Starting MCMC chain $i" at = Dates.now()
        try
            _model = copy_model(model)
            _∇ℓ = prepare_logdensity(_model)
            kw = merge(make_keywords(kw_generator),
                       (reporter = LogProgressReport(; chain_id = i),))
            mcmc = mcmc_with_warmup(Random.default_rng(), _∇ℓ, N; kw...)
            _mcmc = merge(mcmc, (model_statistics = collect_statistics(_model),))
            @info "Finished MCMC chain $i" at = Dates.now()
            JLD2.save_object(mcmc_path, _mcmc)
            true
        catch e
            bt = catch_backtrace()
            open(log_path, "a") do io
                showerror(io, e, bt)
            end
            false
        end
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
