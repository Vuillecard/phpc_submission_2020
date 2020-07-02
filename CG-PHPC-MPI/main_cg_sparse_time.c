//
// Created by pierre on 02.07.20.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdbool.h>
// MPI

#include "mmio_wrapper.h"
#include "util.h"
#include "parameters.h"
#include "cg_mpi_openmp.h"
#include "second.h"
#include <mpi.h>
#include <omp.h>

/*
Implementation of a simple CG solver using matrix in the mtx format (Matrix market)
Any matrix in that format can be used to test the code
*/
int main ( int argc, char **argv ) {

    double * b;
    int rows_size, cols_size, vector_size;
    int bloc_rows_size , bloc_A_size ,bloc_vector_size;
    double * A;
    double * x;

    // Arrays and parameters to read and store the sparse matrix
    double * val = NULL;
    int * Irn = NULL;
    int * Jcn = NULL;
    int N;
    int nz;
    const char * element_type ="d";
    int symmetrize=1;

    int NumProcs, MyRank, Root=0;
    int m,n;
    struct size_m sA;
    double h;

    double StartTime, EndTime;

    /*...Initialising MPI .......*/
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&NumProcs);
    MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);

    if(MyRank == Root) {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
            exit(1);
        } else {

            A = read_mat(argv[1]);
            sA = get_size(argv[1]);
        }

        if (loadMMSparseMatrix(argv[1], *element_type, true, &N, &N, &nz, &val, &Irn, &Jcn, symmetrize)) {
            fprintf(stderr, "!!!! loadMMSparseMatrix FAILED\n");
            return EXIT_FAILURE;
        } else {
            printf("Matrix loaded from file %s\n", argv[1]);
            printf("N = %d \n", N);
            printf("nz = %d \n", nz);
            printf("val[0] = %d \n", val[0]);
        }


        rows_size = sA.m;
        cols_size = sA.n;
        vector_size = sA.n;

        h = 1. / (double) vector_size;
        b = init_source_term(vector_size, h);

    }


    MPI_Barrier(MPI_COMM_WORLD);

    //...Broadcast Matrix and Vector size and perform input validation tests...
    MPI_Bcast(&rows_size, 1, MPI_INT, Root, MPI_COMM_WORLD);
    MPI_Bcast(&cols_size, 1, MPI_INT, Root, MPI_COMM_WORLD);
    MPI_Bcast(&vector_size, 1, MPI_INT, Root, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_INT, Root, MPI_COMM_WORLD);

    if(rows_size % NumProcs != 0){
        MPI_Finalize();
        if(MyRank == Root)
            printf("Error : Matrix cannot be evenly striped among processes");
        exit(-1);
    }


    if(MyRank != Root)
    {
        b = (double*) malloc(vector_size*sizeof(double));
        Irn = (int*) malloc((vector_size+1)*sizeof(double));
        Jcn = (int*) malloc(nz*sizeof(double));
        val = (double*) malloc(nz*sizeof(double));
    }


    /* send the vector b from root to all other processor */
    MPI_Bcast(b,vector_size,MPI_DOUBLE,Root,MPI_COMM_WORLD);
    MPI_Bcast(Jcn,nz,MPI_INT,Root,MPI_COMM_WORLD);
    MPI_Bcast(Irn,vector_size+1,MPI_INT,Root,MPI_COMM_WORLD);
    MPI_Bcast(val,nz,MPI_DOUBLE,Root,MPI_COMM_WORLD);

    bloc_rows_size = rows_size/NumProcs ;
    bloc_vector_size = bloc_rows_size ;

    /* Allocate memory for the solution vector x */
    x = (double*) malloc(vector_size* sizeof(double));
    memset(x, 0., vector_size*sizeof(double));

    StartTime = MPI_Wtime();
    cg_solver_mpi_sparse_time(Irn,Jcn,val,b,x,vector_size,bloc_rows_size,bloc_vector_size);
    EndTime = MPI_Wtime();

    if(MyRank == 0){

        printf("Time for CG (dense solver)  = %f [s]\n",(EndTime-StartTime));
        // verification test
        double *r_test;
        r_test = (double*) malloc(vector_size*sizeof(double));
        memset(r_test, 0., vector_size*sizeof(double));
        cblas_dgemv (CblasColMajor, CblasNoTrans, rows_size, cols_size, 1., A, rows_size, x, 1, 0., r_test, 1);
        cblas_daxpy(vector_size, -1., b, 1, r_test, 1);
        double res = cblas_ddot(vector_size, r_test, 1, r_test, 1);

        printf(" error estimate ||Ax-b|| %E \n",sqrt(res) );
        free(r_test);
        free(A);

    }
    free(val);
    free(Irn);
    free(Jcn);
    free(b);
    free(x);

    MPI_Finalize();
//	free(x0);

    return 0;
}
