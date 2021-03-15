# Usage


## Computing new samples

First create a new directoty (in the following referred to as `PROBLEM_DIRECTORY`)
containing a file `parameters.json`. This file must have the following format:
```json
    {
        "problem": {
            "name": "darcy"
        },
        "fe": {
            "degree": 1,
            "mesh": 50
        },
        "expansion": {
            "mean": 1.0,
            "scale": 0.6,
            "size": 20,
            "decay rate": 2.0
        },
        "sampling": {
            "distribution": "uniform",
            "strategy": "sobol"
        }
    }
```
Then execute the following command:
```bash
$ compute_samples.py PROBLEM_DIRECTORY            # draw new samples for the problem and store them in PROBLEM_DIRECTORY
```


## Reconstructing from existing samples

To reconstruct a function from existing samples execute the following commands in order:
```bash
$ compute_orthogonalization.py PROBLEM_DIRECTORY  # compute the stiffness matrix and its sparse Cholesky factorization and store them in PROBLEM_DIRECTORY
$ compute_reconstruction.py PROBLEM_DIRECTORY     # compute the vmc reconstruction of the problem and store it in PROBLEM_DIRECTORY
```


## Computing functionals of existing samples

The script `compute_functional.py` applies a linear functional to the samples and stores the result in `PROBLEM_DIRECTORY`.
These values can be used to compute the l1SALSA or weighted l1 recovery as done in Compressed Sensing Petrov Galerkin.



# Notes

For Darcy with lognormal coefficient you need to choose a proper scale s.t. the resulting PDE-Operator is still coercive.
For any decay > 2.0 you can choose 6/pi**2 since this is the reciprocal of sum(1/n**2 for n in range(1, infty)).
For decay == 1 and size == 20 you can use 0.2779522965244017051270673922432532848716135458434908413850884393.
