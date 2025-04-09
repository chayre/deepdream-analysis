"""
Bayesian and Frequentist Estimation of Heart Rate Dynamics

This script takes an input .csv file with sequences of inter-beat intervals,
and generates two output files with Bayesian and frequentist heart rate
trajectories.

Modified to accept command-line arguments.
"""

import sys
import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import seaborn as sns
import os

def hr_interpolate(L_data, fs=1):
    """
    Function for aligning data via interpolation.
    """
    L_out = []
    for k in range(len(L_data)):
        data = L_data[k]
        T = data.index

        # Create common time index
        time_ix = np.arange(np.ceil(T[0]), np.floor(T[-1]), 1 / fs)
        df_out = pd.DataFrame(index=time_ix, columns=data.columns)
        df_out.index.name = 'Time'

        # Interpolate data
        for c in data.columns:
            D = data[c].values
            F = interpolate.interp1d(x=T, y=D, kind='cubic')
            df_out.loc[time_ix, c] = F(time_ix)
        L_out.append(df_out)

    return L_out


def frequentist_hr(filenames, verbose=1):
    """
    Frequentist HR estimation.
    """
    L_data = []

    for f in filenames:
        if verbose > 0:
            name = f.split('/')[-1].split('.')[0]
            print(f'\nRunning frequentist estimation for {name}')

        # Load and transform
        S = pd.read_csv(f, header=None, sep=',')[0]
        rr = S.diff().iloc[1:]
        if rr.min() < 0:
            raise ValueError(f'No useful data found in file {f}')

        HR = 60 / rr.values
        time = S.values[:-1] + rr.values / 2
        df = pd.DataFrame(data=HR, index=time)
        df.index.name = 'Time'
        L_data.append(df)

    hr_aligned = hr_interpolate(L_data)
    return hr_aligned


def bayesian_hr(filenames, IT=3, theta=1., tau=1., Nr=20000, Nd=5000, rol=9, w_type='triang', dec=3, fs=1, verbose=1):
    """
    Bayesian HR estimation.
    """
    L_data = []

    for f in filenames:
        data = pd.read_csv(f, sep=',', header=None).values

        df = pd.DataFrame(data=data[:, 1:], index=data[:, 0])
        df.index.name = 'Time'
        df = df.rolling(rol, center=True, win_type=w_type).mean().dropna()
        df = df.iloc[::dec]
        L_data.append(df)

    hr_aligned = hr_interpolate(L_data, fs)
    return hr_aligned


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python script.py <input_file> <bayes_output> <freq_output> <plot_path>")
        print("Example: python script.py input.csv bayes.csv freq.csv data/plots/ddvr_01_baseline.png")
        sys.exit(1)

    input_file = sys.argv[1]
    bayes_output = sys.argv[2]
    freq_output = sys.argv[3]
    plot_path = sys.argv[4]  # Full path including filename

    try:
        # Create plot directory if needed
        plot_dir = os.path.dirname(plot_path)
        os.makedirs(plot_dir, exist_ok=True)

        # Load input file
        filenames = [input_file]

        # Bayesian and frequentist estimation
        bayes_hrs = bayesian_hr(filenames, IT=3)
        freq_hrs = frequentist_hr(filenames)

        # Save results
        bayes_hrs[0].to_csv(bayes_output, index=True)
        freq_hrs[0].to_csv(freq_output, index=True)

        # =====================================================================
        # Plotting with Custom Path
        # =====================================================================
        sns.set_style('whitegrid')
        sns.set_palette('Set1')

        plt.figure(figsize=(10, 6))
        plt.plot(freq_hrs[0].index, freq_hrs[0].values, linewidth=1.5, color='black')
        X = bayes_hrs[0].reset_index().melt(id_vars='Time', var_name='Run', value_name='HR')
        sns.lineplot(data=X, x='Time', y='HR', errorbar='sd', linewidth=1.5)
        
        # Extract base filename for title
        plot_title = os.path.splitext(os.path.basename(plot_path))[0]
        plt.title(f"Heart Rate Estimation - {plot_title}")
        plt.legend(['Frequentist', 'Bayesian'])
        
        # Save to specified path
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {os.path.abspath(plot_path)}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
