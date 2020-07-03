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

// tolerance and maximum iterations
#define epsilon 1e-10
#define IMAX 1000

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
        tmp[threadIdx.x] = partial[threadIdx.x];
    else
        tmp[threadIdx.x] = 0;

    for (int i=blockDim.x/2;i>=1;i = i/2) {
        __syncthreads();
        if (threadIdx.x < i)
            tmp[threadIdx.x] += tmp[i + threadIdx.x];
    }

    if (threadIdx.x == 0)
        *result = tmp[0];
}

/*
 * Function to perform the dot product
 */
void dot(int n, double* vec1, double* vec2, double* result)
{
    dim3 BlockDim(NUM_THREADS);
    dim3 GridDim(NUM_BLOCKS);

    // call the kernel function
    vecdot_partial<<<GridDim,BlockDim>>>(n, vec1, vec2, partial_result);
    vecdot_reduce<<<1,NUM_BLOCKS>>>(partial_result, result);
}

/*
 * Kernel function to perform a matrice vector multiplication
 */
__global__ void mat_vec_mul_kernel(double *device_Mat, double *device_Vect,int matRowSize, int vlength, double *device_ResVect)
{
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    int tindex=tidx+gridDim.x*NUM_BLOCKS*tidy;

    if(tindex<matRowSize)
    {
        int i;int m=tindex*vlength;
        device_ResVect[tindex]=0.00;
        for(i=0;i<vlength;i++)
            device_ResVect[tindex]+=device_Mat[m+i]*device_Vect[i];
    }

    __syncthreads();

}


/*function to launch kernel*/
void mat_vec_mul(double *device_Mat, double *device_Vect,int matRowSize, int vlength,double *device_ResVect)
{
    int max=NUM_BLOCKS*NUM_BLOCKS;
    int BlocksPerGrid=matRowSize/max+1;
    dim3 dimBlock(NUM_BLOCKS,NUM_BLOCKS);
    if(matRowSize%max==0)BlocksPerGrid--;
    dim3 dimGrid(1,BlocksPerGrid);

    mat_vec_mul_kernel<<<dimGrid,dimBlock>>>(device_Mat,device_Vect,matRowSize,vlength,device_ResVect);
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


int main ( int argc, char **argv ) {

	double * host_A;
	double * host_x;
	double * host_b;

	double t1,t2;

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
        host_A = read_mat(argv[1]);
        sA = get_size(argv[1]);
        printf("Matrix loaded from file %s\n",argv[1]);
        printf("Rows = %d \n",sA.m);
        printf("Cols = %d \n",sA.n);
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
    double *gpu_A,*gpu_x,*gpu_b;

    double *gpu_r,*gpu_p,*gpu_Ap;
    double *gpu_alpha , *gpu_beta , *gpu_r_sqrd_old , *gpu_r_sqrd_new;

    int iter = 0;
    double snew ;

    gpuErrchk(cudaMalloc((void**)& gpu_A, m_rows*n_cols* sizeof(double)));
    gpuErrchk(cudaMalloc((void**)& gpu_x, n_cols* sizeof(double)));
    gpuErrchk(cudaMalloc((void**)& gpu_b, n_cols* sizeof(double)));
    gpuErrchk(cudaMalloc((void**)& gpu_r, n_cols* sizeof(double)));
    gpuErrchk(cudaMalloc((void**)& gpu_p, n_cols* sizeof(double)));
    gpuErrchk(cudaMalloc((void**)& gpu_Ap, m_rows* sizeof(double)));
    gpuErrchk(cudaMalloc((void**)& gpu_alpha, sizeof(double)));
    gpuErrchk(cudaMalloc((void**)& gpu_beta, sizeof(double)));
    gpuErrchk(cudaMalloc((void**)& gpu_r_sqrd_old, sizeof(double)));
    gpuErrchk(cudaMalloc((void**)& gpu_r_sqrd_new, sizeof(double)));
    gpuErrchk(cudaMalloc((void**)& partial_result, sizeof(double)*NUM_BLOCKS));

    // Initialise variable in the gpu :
    gpuErrchk(cudaMemcpy(gpu_A, host_A, m_rows*n_cols*sizeof(double) ,cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_x,host_x, n_cols*sizeof(double) ,cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_b, host_b, n_cols*sizeof(double) ,cudaMemcpyHostToDevice));

    // Dimensions of blocks and grid on the GPU :
    dim3 BlockDim(NUM_THREADS);
    dim3 GridDim(NUM_BLOCKS);

    //start the timer
    t1 = second();

    // begin the cg algorithm
    // r=b
    gpuErrchk(cudaMemcpy(gpu_r, host_b, n_cols*sizeof(double) ,cudaMemcpyHostToDevice));

    // p=r
    gpuErrchk(cudaMemcpy(gpu_p, gpu_r, n_cols*sizeof(double) ,cudaMemcpyDeviceToDevice));
    // rsold = r'*r
    dot(n_cols, gpu_r ,gpu_r, gpu_r_sqrd_old);

    gpuErrchk(cudaMemcpy(&snew, gpu_r_sqrd_old, sizeof(double)  ,cudaMemcpyDeviceToHost));

    for (int j = 0; j < IMAX ; ++j) {

        // compute Ap
        mat_vec_mul(gpu_A,gpu_p,m_rows,n_cols,gpu_Ap);

        // compute p.Ap
        dot(n_cols,gpu_p,gpu_Ap,gpu_alpha);

        // alpha = rsold / (p' * Ap);
        scalardiv<<<1,1>>>(gpu_r_sqrd_old, gpu_alpha, gpu_alpha);

        // x = x + alpha * p;
        axpy<<<GridDim,BlockDim>>>(n_cols, gpu_alpha, gpu_p, gpu_x, gpu_x);

        // r = r - alpha * Ap;
        ymax<<<GridDim,BlockDim>>>(n_cols, gpu_alpha, gpu_Ap , gpu_r);

        //rsnew = r.r;
        dot(n_cols,gpu_r,gpu_r,gpu_r_sqrd_new);

        // transfer rsnew from device to host to compare with the given tolerance
        cudaMemcpy(&snew, gpu_r_sqrd_new, sizeof(double) ,cudaMemcpyDeviceToHost);

        // Convergence test
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

    // Get back the results from the device to the host
    cudaMemcpy(host_x, gpu_x, n_cols*sizeof(double) ,cudaMemcpyDeviceToHost);

    // start the timer
    t2 = second();

	printf("Time for GPU CG (dense solver)  = %f [s]\n",(t2-t1));


	// free the memories for the device
	cudaFree(gpu_A);
    cudaFree(gpu_x);
    cudaFree(gpu_b);
    cudaFree(gpu_r);
    cudaFree(gpu_p);
    cudaFree(gpu_Ap);
    cudaFree(gpu_alpha);
    cudaFree(gpu_beta);
    cudaFree(gpu_r_sqrd_old);
    cudaFree(gpu_r_sqrd_new);

    // free the memories for the host
	free(host_A);
	free(host_b);
	free(host_x);


	return 0;
}


