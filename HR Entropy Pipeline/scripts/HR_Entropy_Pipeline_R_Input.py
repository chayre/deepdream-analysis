import os
import subprocess
import pandas as pd
import tempfile

# Define the correct script directory
script_dir = r"C:\Users\CAyre\Documents\Coding\deepdream-analysis\deepdream-analysis\HR Entropy Pipeline\scripts"
    

def process_dataframe(df):
    """Process the given dataframe through the HR entropy pipeline."""
    
    # Filter only section 1
    #df_section1 = df[df['section'] == 1]

    # Create a temporary CSV file to store the DataFrame
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_input:
        df.to_csv(temp_input.name, index=False)
        input_csv = temp_input.name

    print(f"Input CSV: {input_csv}")

    # Step 1: Convert to R-peaks
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_rpeak:
        rpeak_csv = temp_rpeak.name

    result = subprocess.run(
        ["python", os.path.join(script_dir, "convert_R_df_to_rpeaks.py"), input_csv, rpeak_csv],
        capture_output=True, text=True
    )
    print(result.stdout, result.stderr)
    print(f"Rpeak CSV: {rpeak_csv}")

    # Step 2: Julia inference
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_hr_estimation:
        hr_estimation_csv = temp_hr_estimation.name

    result = subprocess.run(
        ["julia", "--project=" + script_dir, os.path.join(script_dir, "gmc_inference_single.jl"), rpeak_csv, hr_estimation_csv],
        capture_output=True, text=True
    )
    print(result.stdout, result.stderr)

    # Step 3: Generate HR estimates
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_hr_bayes:
        hr_bayes_csv = temp_hr_bayes.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_hr_freq:
        hr_freq_csv = temp_hr_freq.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_plot:
        plot_file = temp_plot.name

    result = subprocess.run(
        ["python", os.path.join(script_dir, "generate_HR_single.py"), hr_estimation_csv, hr_bayes_csv, hr_freq_csv, plot_file],
        capture_output=True, text=True
    )
    print(result.stdout, result.stderr)

    # Step 4: Calculate Entropy
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_entropy:
        entropy_csv = temp_entropy.name

    result = subprocess.run(
        ["python", os.path.join(script_dir, "calculate_HRentropy_single.py"), hr_bayes_csv, entropy_csv],
        capture_output=True, text=True
    )

    # Read entropy value
    entropy_df = pd.read_csv(entropy_csv)
    entropy_value = entropy_df.mean()

    return entropy_value
