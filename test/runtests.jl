using DynamicHMCDistributedSampler
using Test, LogDensityTestSuite, DynamicHMC

using JET
@testset "static analysis with JET.jl" begin
    JET.test_package(DynamicHMCDistributedSampler, target_modules=(DynamicHMCDistributedSampler,))
end

Base.copy(s::StandardMultivariateNormal) = s # piracy for testing

@testset "concistency test" begin
    path = joinpath(@__DIR__, "results", "tag")
    sampler = distributed_sampler(path, StandardMultivariateNormal(2), 100)
    @test all(map(sampler, 1:3))
    for i in 1:3
        @test isfile(path * "_chain_$i.jld2")
        @test isfile(path * "_chain_$i.log")
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
