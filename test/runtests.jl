using DynamicHMCDistributedSampler
using Test, LogDensityTestSuite, DynamicHMC

using JET
@testset "static analysis with JET.jl" begin
    JET.test_package(DynamicHMCDistributedSampler, target_modules=(DynamicHMCDistributedSampler,))
end

@testset "concistency test" begin
    path = joinpath(@__DIR__, "results", "tag")
    sampler = distributed_sampler(path, StandardMultivariateNormal(2), 100;
                                  copy_model = m -> (; model = m, stats = 1),
                                  prepare_logdensity = m -> m.model,
                                  collect_statistics = m -> m.stats)
    @test all(map(sampler, 1:3))
    for i in 1:3
        @test isfile(path * "_chain_$i.jld2")
        log = read(path * "_chain_$i.jld2", String)
        occursin("Starting MCMC chain $i", log)
        occursin("MCMC progress", log)
        occursin("Finished MCMC chain $i", log)
    end

    # remove second chain and test results without it
    rm(path * "_chain_2.jld2")
    chains = collect_started_chains(path)
    @test map(c -> typeof(c.mcmc_path), chains) == [String, Nothing, String]
    @test size(pool_posterior_matrices(filter(!isnothing, map(load_mcmc, chains)))) == (2, 200)

    # cleanup for local testing
    rm(splitdir(path)[1]; force = true, recursive = true)
end

@testset "QA with Aqua" begin
    import Aqua
    Aqua.test_all(DynamicHMCDistributedSampler)
end
