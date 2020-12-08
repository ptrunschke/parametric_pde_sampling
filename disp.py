# coding: utf-8
import argparse, os, re

import yaml, numpy as np

from compute_samples import load_parameters


if __name__=='__main__':
    descr = """Display the problem parameter and some statistics."""
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('PROBLEM', help='Path to the directory containing the problem specification (PROBLEM/parameters.json).')
    args = parser.parse_args()

    problemDir = args.PROBLEM
    if not os.path.isdir(problemDir):
        raise IOError(f"PROBLEM '{problemDir}' is not a directory")

    problemInfo = load_parameters(problemDir)

    print("="*35 + " Problem " + "="*36)
    print(yaml.dump(problemInfo), end='')

    ys = []
    ks = []
    us = []
    regex = re.compile(r"(\d+\.npz$)")
    for fileName in os.listdir(problemDir):
        if not regex.match(fileName):
            continue
        z = np.load(f"{problemDir}/{fileName}")
        ys.append(z['ys'])
        ks.append(z['ks']) 
        us.append(z['us']) 

    if len(ys) > 0:
        ys = np.concatenate(ys)
        ks = np.concatenate(ks)
        us = np.concatenate(us)
        assert len(ys) == len(ks) == len(us)

        print('-'*80)
        mean = np.mean(ys, axis=0)
        var = np.var(ys, axis=0)
        mask = mean != 0

        print(f"""ys:
  shape: {ys.shape}
  Mean(ys) \u2208 [{np.min(mean): .0e}, {np.max(mean): .0e}]
  Var(ys)  \u2208 [{np.min(var): .0e}, {np.max(var): .0e}]
  Maximum relative variance: {100*np.max(var[mask]/abs(mean[mask])):.0f}%""")

        for name in ["ks", "us"]:
            arr = globals()[name]
            mean = np.mean(arr, axis=0)
            var = np.var(arr, axis=0)
            mask = mean != 0
            print(f"""{name}:
  shape: {arr.shape}  
  Mean({name}(x)) \u2208 [{np.min(mean): .0e}, {np.max(mean): .0e}]
  Var({name}(x))  \u2208 [{np.min(var): .0e}, {np.max(var): .0e}]
  Maximum relative variance: {100*np.max(var[mask]/abs(mean[mask])):.0f}%""")
    print("="*80)
