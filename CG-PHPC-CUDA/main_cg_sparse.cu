#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdbool.h>

extern "C" {
#include "util.h"
#include "mmio_wrapper.h"

#include "parameters.h"
#include "second.h"
}

#define NUM_THREADS 256
#define NUM_BLOCKS 32

// Macros to simplify kernels
#define THREAD_ID threadIdx.x+blockIdx.x*blockDim.x
#define THREAD_COUNT gridDim.x*blockDim.x

// Solver parameters - relative tolerance and maximum iterations
#define epsilon 1e-10
#define IMAX 10000
// Error function

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

double * init_source_term(int n, double h){
    double * f;
    int i;
    f  = (double*) malloc(n*sizeof(double*));

    for(i = 0; i < n; i++) {
        f[i] = (double)i * -2. * M_PI * M_PI * sin(10.*M_PI*i*h) * sin(10.*M_PI*i*h);
    }
    return f;
}

// use to keep the partial result in the dot product
double* partial_result;

/*
 * Kernel function to compute the partial dot product
 * The partial result is stored in partial.
 */
__global__ void vecdot_partial(int n, double* vec1, double* vec2, double* partial)
{
    __shared__ double tmp[NUM_THREADS];
    tmp[threadIdx.x] = 0;

    for (int i=THREAD_ID; i<n; i+=THREAD_COUNT)
    {
        tmp[threadIdx.x] += vec1[i]*vec2[i];
    }

    for (int i=blockDim.x/2;i>=1;i = i/2)
    {
        __syncthreads();
        if (threadIdx.x < i)
        {
            tmp[threadIdx.x] += tmp[i + threadIdx.x];
        }

    }

    if (threadIdx.x == 0)
    {
        partial[blockIdx.x] = tmp[0];
    }

}

/*
* Kernel function to reduces the output of the vecdot_partial kernel to a single value.
* The result is stored in result.
*/
__global__ void vecdot_reduce(double* partial, double* result)
{
    __shared__ double tmp[NUM_BLOCKS];

    if (threadIdx.x < NUM_BLOCKS)
    {
        tmp[threadIdx.x] = partial[threadIdx.x];
    }else
    {
        tmp[threadIdx.x] = 0;
    }

    for (int i=blockDim.x/2;i>=1;i = i/2)
    {
        __syncthreads();
        if (threadIdx.x < i)
        {
            tmp[threadIdx.x] += tmp[i + threadIdx.x];
        }

    }
    if (threadIdx.x == 0)
    {
        *result = tmp[0];
    }

}

/*
 * Kernel function to reduces the output of the vecdot_partial kernel to a single value.
 * The result is stored in result.
 */
void dot(int n, double* vec1, double* vec2, double* result)
{
    dim3 BlockDim(NUM_THREADS);
    dim3 GridDim(NUM_BLOCKS);

    vecdot_partial<<<GridDim,BlockDim>>>(n, vec1, vec2, partial_result);
    vecdot_reduce<<<1,NUM_BLOCKS>>>(partial_result, result);
}

/*
 * Compute a simple scalar division on the device
 */
__global__ void scalardiv(double* num, double* den, double* result)
{
    if(threadIdx.x==0 && blockIdx.x==0)
        *result = (*num)/(*den);
}

/*
 * Computes r= a*x+y for vectors x and y, and scalar a.
 * n is the size of the vector
 */
__global__ void axpy(int n, double* a, double* x, double* y, double* r)
{
    for (int i=THREAD_ID; i<n; i+=THREAD_COUNT)
        r[i] = y[i] + (*a)*x[i];
}

/*
 * Computes y= y-a*x forcvectors x and y, and scalar a.
 * n is the size of the vector
 */
__global__ void ymax(int n, double* a, double* x, double* y)
{
    for (int i=THREAD_ID; i<n; i+=THREAD_COUNT)
        y[i] = y[i] - (*a)*x[i];
}

/*
 * matrice vector multiplication for sparse matrix using csr format
 */
__global__ void spmv_csr( int num_rows ,int* ptr , int* indices , double * data , double * x , double * y)
{

    for( int row=THREAD_ID; row<num_rows ; row+=THREAD_COUNT )
    {
        y[row] = 0;
        int row_start = ptr[ row ];
        int row_end = ptr[ row +1];
        for (int jj = row_start ; jj < row_end ; jj++)
        {
            y[row] += data[jj-1]*x[ indices[jj-1]-1];
        }
    }
}


int main ( int argc, char **argv ) {

	//double * host_A;
	double * host_x;
	double * host_b;

	double t1,t2;

    double * host_val = NULL;
    int * host_Irn = NULL;
    int * host_Jcn = NULL;
    int host_N;
    int host_nz;
    const char * element_type ="d";
    int symmetrize=1;

	int m_rows,n_cols;
	struct size_m sA;
	double h;

	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
	else
	{
        //host_A = read_mat(argv[1]);
        sA = get_size(argv[1]);
        printf("Matrix loaded from file %s\n",argv[1]);
        printf("Rows = %d \n",sA.m);
        printf("Cols = %d \n",sA.n);
	}

    if (loadMMSparseMatrix(argv[1], *element_type, true, &host_N, &host_N, &host_nz, &host_val, &host_Irn, &host_Jcn, symmetrize)){
        fprintf (stderr, "!!!! loadMMSparseMatrix FAILED\n");
        return EXIT_FAILURE;
    }else{
        printf("Matrix loaded from file %s\n",argv[1]);
        printf("N = %d \n",host_N);
        printf("nz = %d \n",host_nz);
        printf("val[0] = %f \n",host_val[0]);
    }

	m_rows = sA.m;
	n_cols = sA.n;

	h = 1./(double)n_cols;
	host_b = init_source_term(n_cols,h);

	// allocate initial guess :
	host_x = (double*) malloc(n_cols * sizeof(double));
    for (int i = 0; i < n_cols; ++i) {
        host_x[i] = 0.0;
    }


    // Allocate memorie for CG solver
    double *gpu_val ;
    int *gpu_Irn,*gpu_Jcn;
    double *gpu_x,*gpu_b;

    double *gpu_r,*gpu_p,*gpu_Ap;
    double *gpu_alpha , *gpu_beta , *gpu_r_sqrd_old , *gpu_r_sqrd_new;

    double snew ;

    gpuErrchk(cudaMalloc( (void**)& gpu_val, host_nz* sizeof(double)));
    gpuErrchk(cudaMalloc( (void**)& gpu_Irn, (host_N+1)* sizeof(int)));
    gpuErrchk(cudaMalloc( (void**)& gpu_Jcn, host_nz* sizeof(int)));

    gpuErrchk(cudaMalloc( (void**)& gpu_x, n_cols* sizeof(double)));
    gpuErrchk(cudaMalloc( (void**)& gpu_b, n_cols* sizeof(double)));

    gpuErrchk(cudaMalloc( (void**)& gpu_r, n_cols* sizeof(double)));
    gpuErrchk(cudaMalloc( (void**)& gpu_p, n_cols* sizeof(double)));
    gpuErrchk(cudaMalloc( (void**)& gpu_Ap, m_rows* sizeof(double)));

    gpuErrchk(cudaMalloc((void**)& gpu_alpha, sizeof(double)));
    gpuErrchk(cudaMalloc((void**)& gpu_beta, sizeof(double)));
    gpuErrchk(cudaMalloc((void**)& gpu_r_sqrd_old, sizeof(double)));
    gpuErrchk(cudaMalloc((void**)& gpu_r_sqrd_new, sizeof(double)));

    gpuErrchk(cudaMalloc((void**)& partial_result, NUM_BLOCKS* sizeof(double)));

    // maybe (void*)
    gpuErrchk(cudaMemcpy(gpu_val, host_val, host_nz*sizeof(double) ,cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_Irn, host_Irn, (host_N+1)*sizeof(int) ,cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_Jcn, host_Jcn, host_nz*sizeof(int) ,cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(gpu_x,host_x, n_cols*sizeof(double) ,cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_b, host_b, n_cols*sizeof(double) ,cudaMemcpyHostToDevice));

    // Dimensions of blocks and grid on the GPU
    dim3 BlockDim(NUM_THREADS);
    dim3 GridDim(NUM_BLOCKS);

    // start the timer
    t1 = second();

    // begin the cg algorithm
    // r=b
    gpuErrchk(cudaMemcpy(gpu_r, host_b, n_cols*sizeof(double) ,cudaMemcpyHostToDevice));
    // p=r
    gpuErrchk(cudaMemcpy(gpu_p, gpu_r, n_cols*sizeof(double) ,cudaMemcpyDeviceToDevice));
    // rsold = r*r
    dot(n_cols, gpu_r ,gpu_r, gpu_r_sqrd_old);

    gpuErrchk(cudaMemcpy(&snew, gpu_r_sqrd_old, sizeof(double)  ,cudaMemcpyDeviceToHost));

    for (int j = 0; j < IMAX ; ++j) {

        // Ap
        spmv_csr<<<GridDim,BlockDim>>>(host_N,gpu_Irn,gpu_Jcn,gpu_val,gpu_p,gpu_Ap);

        // alpha = rsold / (p' * Ap);
        dot(n_cols,gpu_p,gpu_Ap,gpu_alpha);
        scalardiv<<<1,1>>>(gpu_r_sqrd_old, gpu_alpha, gpu_alpha);

        // x = x + alpha * p;
        axpy<<<GridDim,BlockDim>>>(n_cols, gpu_alpha, gpu_p, gpu_x, gpu_x);

        // r = r - alpha * Ap;
        ymax<<<GridDim,BlockDim>>>(n_cols, gpu_alpha, gpu_Ap , gpu_r);

        //rsnew = r' * r;
        dot(n_cols,gpu_r,gpu_r,gpu_r_sqrd_new);

        // put it in the host device inorder to compare with the tolerance
        cudaMemcpy(&snew, gpu_r_sqrd_new, sizeof(double) ,cudaMemcpyDeviceToHost);

        /*
        if ( (j%1000) == 0)
        printf("residual = %E\n",snew);
        */

        if(snew < epsilon*epsilon)
        {
            printf("\t[STEP %d] residual = %E\n",j,sqrt(snew));
            break;
        }
        // p = r + (rsnew / rsold) * p;
        scalardiv<<<1,1>>>(gpu_r_sqrd_new, gpu_r_sqrd_old, gpu_beta);
        axpy<<<GridDim,BlockDim>>>(n_cols, gpu_beta, gpu_p, gpu_r, gpu_p);

        //rsold = rsnew;
        cudaMemcpy(gpu_r_sqrd_old, gpu_r_sqrd_new, sizeof(double) ,cudaMemcpyDeviceToDevice);
    }
    // get the results
    cudaMemcpy(host_x, gpu_x, n_cols*sizeof(double) ,cudaMemcpyDeviceToHost);

    // end the timer
    t2 = second();

	printf("Time for GPU CG (sparse solver)  = %f [s]\n",(t2-t1));

	// verification test


	// clean
    cudaFree(gpu_val);
    cudaFree(gpu_Irn);
    cudaFree(gpu_Jcn);
    cudaFree(gpu_x);
    cudaFree(gpu_b);
    cudaFree(gpu_r);
    cudaFree(gpu_p);
    cudaFree(gpu_Ap);
    cudaFree(gpu_alpha);
    cudaFree(gpu_beta);
    cudaFree(gpu_r_sqrd_old);
    cudaFree(gpu_r_sqrd_new);

	//free(host_A);
	free(host_b);
	free(host_x);


//	free(x0);

	return 0;
}



