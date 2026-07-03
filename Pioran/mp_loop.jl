#= 
Run with: julia mp_loop.jl ./data_folder/
or with MPI: mpirun -n 4 julia mp_loop.jl ./data_folder/
=#

# used for parallel processing trough a folder

using MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

using Pioran, Distributions, Statistics, Random, DelimitedFiles, PyCall

ultranest = pyimport("ultranest")
rng = MersenneTwister(123 + rank) 

path_to_data = ARGS[1]
all_files = filter(x -> endswith(x, ".txt"), readdir(path_to_data, join=true))


my_files = [all_files[i] for i in 1:length(all_files) if (i-1) % size == rank]

if rank == 0
    println("Totaal aantal bestanden: ", length(all_files))
    println("Verdeeld over $size cores.")
end

for filename in my_files
    # Print status per core
    println("Rank $rank start met: $filename")
    
    fname = replace(split(filename, "/")[end], ".txt" => "_single")
    # Create output directory for this file
    dir = "inference/SANE/Quarters_same_prior/" * fname
    plot_path = dir * "/plots/"
    if !ispath(dir) mkpath(dir) end

    # Data laden
    A = readdlm(filename, comments = true, comment_char = '#')
    t_all, y_all, yerr_all = A[:, 1], A[:, 2], A[:, 3]
    t, y, yerr, x̄, va = extract_subset(rng, dir * "/" * fname, t_all, y_all, yerr_all)

    f_min, f_max = 1 / (t[end] - t[1]), 1 / minimum(diff(t)) / 2
    f0, fM = f_min / 20.0, f_max * 20.0
    min_f_b, max_f_b = f0 * 4.0, fM / 4.0
    μᵥ, σᵥ = -1.5, 1.0; μₙ, σₙ² = 2μᵥ, 2(σᵥ)^2; σₙ = sqrt(σₙ²)
    
    basis_function = "DRWCelerite"
    n_components = 20
    model = SingleBendingPowerLaw

    function GP_model(params)
        α₁, f₁, α₂, variance, ν, μ = params
        σ² = ν .* yerr .^ 2 ./ y .^ 2
        yn = log.(y)
        𝓟 = model(α₁, f₁, α₂)
        𝓡 = approx(𝓟, f_min, f_max, n_components, variance, basis_function = basis_function)
        f_gp = ScalableGP(μ, 𝓡)
        return f_gp(t, σ²), yn
    end

    logl(pars) = logpdf(GP_model(pars)...)

    function prior_transform(cube)
        α₁ = quantile(Uniform(0.0, 3.0), cube[1])
        f₁ = quantile(LogUniform(min_f_b, max_f_b), cube[2])
        α₂ = quantile(Uniform(α₁, 6.0), cube[3])
        variance = quantile(LogNormal(μₙ, σₙ), cube[4])
        ν = quantile(Gamma(2, 0.5), cube[5])
        μ = quantile(Normal(x̄, 5 * sqrt(va)), cube[6])
        return [α₁, f₁, α₂, variance, ν, μ]
    end

    paramnames = ["α₁", "f₁", "α₂", "variance", "ν", "μ"]


    sampler = ultranest.ReactiveNestedSampler(
        paramnames, logl, 
        resume = true, 
        transform = prior_transform, 
        log_dir = dir, 
        vectorized = false 
    )
    
    results = sampler.run()
    
    sampler.print_results()
    sampler.plot()
    

    if rank == 0 || size > 1
        println("Rank $rank start posterior checks voor: $fname")

        sample_file = dir * "/chains/equal_weighted_post.txt"
        
        if isfile(sample_file)
            samples = readdlm(sample_file, skipstart = 1)
            
            paramnames_split = Dict(
                "psd" => ["α₁", "f₁", "α₂"],
                "norm" => "variance",
                "scale_err" => "ν",
                "log_transform" => "c",
                "mean" => "μ"
            )
            
            GP_model_ppc(t_p, y_p, σ_p, params) = GP_model(params)[1]

            try
                run_posterior_predict_checks(
                    samples, 
                    paramnames, 
                    paramnames_split, 
                    t, y, yerr, 
                    model, 
                    GP_model_ppc, 
                    true; # verbose
                    path = plot_path, 
                    basis_function = basis_function, 
                    n_components = n_components, 
                    plots = ["corner", "psd", "trace"]
                )
                println("Rank $rank: Plots succesvol opgeslagen in $plot_path")
            catch e
                println("Rank $rank: Fout bij posterior checks voor $fname: $e")
            end
        else
            println("Rank $rank: Kan sample file niet vinden voor $fname, checks overgeslagen.")
        end
    end
    
    println("Rank $rank is klaar met: $filename")
end

MPI.Finalize()