################################################################################
# Bayesian estimation of heart rate dynamics
#
# This script runs the estimation of heart rate trajectories for a single input file
# specified as a command line argument.
#
# Usage: julia script.jl <input_file_path> <output_file_path>
#
# The script relies primarily on the open source package PointProcessInference.
################################################################################

using PointProcessInference 
const PPI = PointProcessInference
using Random
using Statistics
using CSV
using DataFrames
using Distributions

function generate_hr(data, IT, tau, theta, Nr, Nd, verbose)
    # Building a prior from basic data stats
    RR = diff(data)
    valid_mask = (RR .> 0) .& isfinite.(RR) #New
    RR = RR[valid_mask] #New
    ss = fit_mle(Gamma, 1 ./ RR)
    α1 = αind = ss.α
    β1 = βind = 1/ss.θ

    # Setting the sampling frequency (Fs=3Hz)
    N = round(Int64, (maximum(data) - minimum(data))*3)

    # Create a container for the IT sampled trajectories, each of length N
    M = zeros(N, IT+1)

    # Sample each of the IT trajectories
    for it=1:IT
        if verbose>0
            println("Iteration $it")
        end

        # Main function call to GMC inference routine
        F = PPI.inference(data; 
            T0=minimum(data), 
            N=N, 
            title="HR", 
            samples=1:1:Nr, 
            α1=α1, 
            β1=β1, 
            αind=αind, 
            βind=βind, 
            τ=tau, 
            Π=Exponential(theta), 
            verbose=false
        )

        X = F.ψ  # extract the obtained values 
        X = X[Nd:end,:]  # discard the first Nd realisations
        M[:,it+1] = mean(X, dims=1) * 60  # take the mean value of the remaining realisations

        if it==1  # make a copy of the resulting time indices
            t_index = collect(F.breaks)
            t_mean = (t_index[1:end-1] + t_index[2:end])/2
            M[:,it] = t_mean
        end
    end

    return M
end 

function process_single_file(input_filepath, output_filepath; IT=3, tau=1, theta=1, Nr=20000, Nd=5000, verbose=1)
    """
    Process a single file for heart rate estimation
    
    Parameters
    ----------
    input_filepath : String
        Path to the input CSV file containing R-peaks data
    output_filepath : String
        Path where the output CSV file will be saved
    IT : Int
        Number of sampled trajectories extracted via the Gibbs sampler
    tau : Int
        Random walk step-size in Metropolis-Hastings step
    theta : Float64
        Hyperparameter of prior of gamma
    Nr : Int
        Number of runs of the Gibbs sampler per estimated trajectory
    Nd : Int
        Number of runs of the Gibbs sampler discarded before calculating average
    verbose : Int
        Verbosity level
    """
    Random.seed!(1234)  # set fixed seed for reproducibility
    
    if verbose > 0
        println("\nProcessing file: $input_filepath")
    end

    # Load and process the data
    data = CSV.read(input_filepath, DataFrame; header=false)[:,1]
    M = generate_hr(data, IT, tau, theta, Nr, Nd, verbose)

    # Save results
    #CSV.write(output_filepath, DataFrame(M, :auto), header=false)
    # Replace with this code to manually create columns
    df = DataFrame()
    for i in 1:size(M, 2)
        df[!, Symbol("Column_$i")] = M[:, i]
    end
CSV.write(output_filepath, df, header=false)
    println("Estimation saved in $output_filepath")
    
    return M
end

function main()
    if length(ARGS) != 2
        println("Usage: julia script.jl <input_file_path> <output_file_path>")
        exit(1)
    end

    input_file = ARGS[1]
    output_file = ARGS[2]

    if !isfile(input_file)
        println("Error: Input file '$input_file' does not exist")
        exit(1)
    end

    # Create output directory if it doesn't exist
    output_dir = dirname(output_file)
    if !isempty(output_dir) && !isdir(output_dir)
        mkpath(output_dir)
    end

    try
        process_single_file(input_file, output_file)
    catch e
        println("Error processing file: ", e)
        exit(1)
    end
end

# Run the script
main()