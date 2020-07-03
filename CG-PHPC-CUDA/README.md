PHPC - CONJUGATE GRADIENT PROJECT CUDA VERSION

HOWTO COMPILE AND RUN
=====================
File :

- *main_cg.cu* : compute the cuda version with dense matrix
- *main_cg_sparse.cu* : compute the cuda version with sparse matrix 

Requirements : 

- a recent compiler (like gcc or intel)
- a CUDA library (lika cuda)

compile on SCITAS clusters :

```
$ module load gcc openblas
$ git clone ssh://git@c4science.ch/source/CG-PHPC.git
$ cd CG-PHPC-CUDA
$ make
```

to run on a laptop :

```
$ ./cg_gpu lap2D_5pt_n100.mtx
```
You should see this output (timing is indicative) :

```
$ srun ./conjugategradient lap2D_5pt_n100.mtx 
size of matrix = 10000 x 10000
Call cgsolver() on matrix size (10000 x 10000)
	[STEP 488] residual = 1.103472E-10
Time for GPU CG = 36.269389 [s]
```
Note: 
- you can change the block size and thread per block in the code directly 
- To change from dense to sparse you to change OBJS in the makefile 

The given example is a 5-points stencil for the 2D Laplace problem. The matrix is in sparse format.

The matrix format is [Matrix Market format (.mtx extension)](https://sparse.tamu.edu/). You can use other matrices there or create your own. 

