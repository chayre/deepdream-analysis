import sys
import pandas as pd
import numpy as np
import bioread

def main(input_file, output_file):
    # Read the .acq file
    try:
        acq_data = bioread.read_file(input_file)
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        sys.exit(1)
    
    # Convert to DataFrame
    try:
        df = pd.DataFrame({channel.name: channel.data for channel in acq_data.channels})
        # if you want to have immendiate step .csv filed save change the output_file string to something else
        # df.to_csv(output_file, index=False)
        # print(f"Data saved to {output_file}")
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)
    
    # Extract and process R-peaks
    try:
        r_peaks = df['ECG R-R']
        first_non_zero_index = r_peaks[r_peaks != 0].first_valid_index()
        r_peaks = r_peaks[first_non_zero_index:]
        
        # Identify true R-peaks
        indices = r_peaks != r_peaks.shift(1)
        indices = indices[indices == True].index[1:]

        segments = []
        for i in range(len(indices) - 1):
            start_idx = indices[i]
            end_idx = indices[i + 1] - 1
            segment = r_peaks.loc[start_idx:end_idx]
            segments.append(segment)
        
        # Save only the first value of each segment
        true_r_peaks = pd.Series(np.array([x.values[0] for x in segments]).cumsum())
    
        true_r_peaks.to_csv(output_file, index=False, header=False)
        print(f"R-peaks saved to {output_file}")
    except KeyError:
        print("Column 'ECG R-R' not found in the input data.")
    except Exception as e:
        print(f"Error processing R-peaks: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file.acq> <output_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
