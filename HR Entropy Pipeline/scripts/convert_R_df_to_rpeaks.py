import sys
import pandas as pd
import numpy as np

def main(input_file, output_file):
    # Read the CSV file
    try:
        # Load CSV
        df = pd.read_csv(input_file)

        print("CSV Data Preview:")
        print(df.head())

    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        sys.exit(1)
    
    # Extract and process R-peaks
    try:
        # Use the 'ECG R-R' column directly from your CSV
        r_peaks = df['ECG R-R']
        
        # Remove leading zeros if they exist
        first_non_zero_index = r_peaks[r_peaks != 0].first_valid_index()
        if first_non_zero_index is not None:
            r_peaks = r_peaks[first_non_zero_index:]
        
        # Identify changes in R-R values to find true peaks
        indices = r_peaks != r_peaks.shift(1)
        indices = indices[indices == True].index[1:]

        segments = []
        for i in range(len(indices) - 1):
            start_idx = indices[i]
            end_idx = indices[i + 1] - 1
            segment = r_peaks.loc[start_idx:end_idx]
            segments.append(segment)
        
        # Calculate cumulative sum of first value of each segment
        true_r_peaks = pd.Series(np.array([x.values[0] for x in segments]).cumsum())
    
        # Save to output file
        true_r_peaks.to_csv(output_file, index=False, header=False)
        print(f"R-peaks saved to {output_file}")
        
    except KeyError:
        print("Column 'ECG R-R' not found in the input data.")
    except Exception as e:
        print(f"Error processing R-peaks: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file.csv> <output_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)