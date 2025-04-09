# TO RUN THIS FILE, MUST HAVE JULIA WITH Pkg.add("PointProcessInference")
# Note: Hardcoded path to jvm.dll in calculate_HRentropy_single.py

import os
import subprocess

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the data directory and subfolders
data_dir = os.path.join(script_dir, "..", "data")  # Go up one level, then into "data"
acq_dir = os.path.join(data_dir, "acq")

# Define output folders
output_folders = {
    "rpeaks": os.path.join(data_dir, "rpeaks"),
    "hr_estimations": os.path.join(data_dir, "hr_estimations"),
    "bayes": os.path.join(data_dir, "bayes"),
    "freq": os.path.join(data_dir, "freq"),
    "hr_entropy": os.path.join(data_dir, "hr_entropy"),
    "entropy_plots": os.path.join(data_dir, "entropy_plots"),
}

# Create output folders if they don't exist
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# List all subject numbers (adjust as needed)
subjects = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"]
condition = ["experiment"]


for sub in subjects:
    for cond in condition:
        print(f"Processing subject {sub} {cond}")

        # Define input and output file paths
        acq_file = os.path.join(acq_dir, f"ID{sub}_{cond}.acq")
        rpeak_file = os.path.join(output_folders["rpeaks"], f"ID{sub}_{cond}_rpeak.csv")
        hr_estimation_file = os.path.join(output_folders["hr_estimations"], f"ID{sub}_{cond}_hr_estimation.csv")
        hr_bayes_file = os.path.join(output_folders["bayes"], f"ID{sub}_{cond}_hr_estimation_bayes.csv")
        hr_freq_file = os.path.join(output_folders["freq"], f"ID{sub}_{cond}_hr_estimation_freq.csv")
        entropy_file = os.path.join(output_folders["hr_entropy"], f"ID{sub}_{cond}_entropy.csv")
        plot_file =  os.path.join(output_folders["entropy_plots"], f"ID{sub}_{cond}_entropy_plot.png")

        # Step 1: Convert R Data Frame to R-peaks CSV
        subprocess.run([
            "python", os.path.join(script_dir, "convert_acq_to_rpeaks.py"),
            acq_file,
            rpeak_file
        ])

        # Step 2: Run Julia inference
        subprocess.run([
            "julia", "--project=.", os.path.join(script_dir, "gmc_inference_single.jl"),
            rpeak_file,
            hr_estimation_file
        ])

        # Step 3: Generate Bayesian and Frequentist HR estimates along with Plot
        subprocess.run([
            "python", os.path.join(script_dir, "generate_HR_single.py"),
            hr_estimation_file,
            hr_bayes_file,
            hr_freq_file,
            plot_file
        ])

        # Step 4: Calculate Entropy from Bayesian data
        subprocess.run([
            "python", os.path.join(script_dir, "calculate_HRentropy_single.py"),
            hr_bayes_file,
            entropy_file
        ])

print("All subjects processed!")