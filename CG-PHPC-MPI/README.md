

HOWTO COMPILE AND RUN
=====================

Requirements : 

- a recent compiler (like gcc or intel)
- a BLAS library (like openblas or intel MKL)
- a MPI and openmp library (like openmpi)

compile on SCITAS clusters for the MPI version :

```
$ module load gcc openblas openmpi
$ git clone https://github.com/Vuillecard/phpc_submission_2020.git
$ cd CG-PHPC-MPI
$ make
```

run on a SCITAS cluster via a batch file :

```
$ sbatch script.batch
```

You should see this output (timing is indicative) :

```
$ srun ./conjugategradient lap2D_5pt_n100.mtx 
size of matrix = 10000 x 10000
Call cgsolver() on matrix size (10000 x 10000)
	[STEP 488] residual = 1.103472E-10
Time for CG = 36.269389 [s]
```

The given example is a 5-points stencil for the 2D Laplace problem. The matrix is in sparse format. 

