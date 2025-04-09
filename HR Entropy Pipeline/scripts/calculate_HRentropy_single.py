"""
Calculation of Bayesian estimators of HR markers

This script takes a collection of estimated HR trajectories as inputs and calculates HR entropy.
"""

import sys
import os
import numpy as np
import pandas as pd
import jpype as jp
from jpype import JArray, JByte

# Start JVM
script_dir = os.path.dirname(os.path.abspath(__file__))
vmm_jar = os.path.join(script_dir, "vmm.jar")
trove_jar = os.path.join(script_dir, "trove.jar")
classpath = f"-Djava.class.path={vmm_jar};{trove_jar}"

if not os.path.exists(vmm_jar) or not os.path.exists(trove_jar):
    raise FileNotFoundError("Required JAR files not found. Make sure vmm.jar and trove.jar are in the script's directory.")

# HARDCODED
if not jp.isJVMStarted():
    jp.startJVM(r"C:\Program Files\Java\jdk-23\bin\server\jvm.dll", '-ea', '-Xmx8192m', classpath)
# HARDCODED

# Java helpers
String = jp.JPackage('java.lang').String


def javify(py_str, ab_dict):
    byte_array = JArray(JByte)([ab_dict[s] for s in py_str])
    return String(byte_array)


def quantize_df(data):
    """
    Binarize data according to its mean value.
    """
    data = data - data.mean()
    R = (data > 0).astype(int)
    return R


def ctw_entropy(X, Y=None, vmm_order=30):
    """
    Calculate HR entropy using the CTW algorithm.
    """
    if Y is None:
        Y = X

    Yq = quantize_df(Y)
    alphabet = set(Yq.iloc[:, 0])
    ab_size = len(alphabet)
    ab_dict = {cc: i for cc, i in zip(sorted(alphabet), range(ab_size))}

    # Initialize and train the probabilistic model
    vmm = jp.JPackage('vmm.algs').DCTWPredictor()
    vmm.init(ab_size, vmm_order)
    for e in Yq.columns:
        vmm.learn(javify(Yq[e], ab_dict))

    # Evaluate the model on testing data X
    res = pd.Series(index=X.columns, dtype=float)
    Xq = quantize_df(X)
    for c in Xq.columns:
        data = Xq[c].values
        if len(data) == 0:
            res.loc[c] = np.nan
        else:
            res.loc[c] = vmm.logEval(javify(data, ab_dict)) / len(data)

    return res


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_file.csv> <output_file.csv> [<ctw_order=30>]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    ctw_order = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    try:
        # Load Bayesian estimations
        data = pd.read_csv(input_file, index_col=0)

        # Calculate HR entropy
        data_diff = data.diff().dropna()
        h = ctw_entropy(data_diff, vmm_order=ctw_order)
        mean_h = h.mean()

        # Save entropy calculations
        h.to_csv(output_file, index=True, header=["Entropy"])
        print(f"HR entropy saved to {output_file}")
        print(f"Average HR entropy: {mean_h}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
