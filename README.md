# DynamicHMCDistributedSampler.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![build](https://github.com/tpapp/DynamicHMCDistributedSampler.jl/workflows/CI/badge.svg)](https://github.com/tpapp/DynamicHMCDistributedSampler.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/tpapp/DynamicHMCDistributedSampler.jl/branch/master/graph/badge.svg?token=ey9qzQDxN4)](https://codecov.io/gh/tpapp/DynamicHMCDistributedSampler.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A tiny Julia package for running MCMC chains with DynamicHMC.

## Example

```julia
path = "/my_project/mcmc/modelvariant"

# run the calculation
sampler = distributed_sampler(, ∇ℓ, 1000)
map(sampler, 1:5) # or use pmap with Distributed

# analyze the results
chains = collect_started_chains(path)
pool_posterior_matrices(filter(!isnothing, map(load_mcmc, chains)))
```

The intended use case running computations on a *server* or a *cluster*, saving the logs, syncing the chains to a local machine with tools like [rsync](https://rsync.samba.org/), and then proceed with the analysis locally.

The output from MCMC is saved and loaded using [JLD2.jl](https://github.com/JuliaIO/JLD2.jl). Log files are text files.

## Related tools

When running computations on a server, [Ntfy.jl](https://github.com/cjdoris/Ntfy.jl) is very useful for getting a message about the chains finishing:

``` julia
using Ntfy
let topic = "my topic"
    try
        successes = pmap(sampler, 1:5)
        ntfy(topic, "MCMC done at $(sampler.path), $(sum(successes))/$(length(successes)) chains finished")
    catch
        ntfy(topic, "MCMC terminated with an error")
    end
end
```

[rsync](https://rsync.samba.org/) can be used to sync results from a server:

```bash
rsync -mavz -f'+ **/' -f'+ **/*.jld2' -f'+ **/*.log' -f'- **'  \
      remote-server://my_project/mcmc ./

```
