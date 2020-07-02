//
// Created by pierre on 23.04.20.
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
    double * bloc_A ;
    double * A;
    double * x;

    int NumProcs, MyRank, Root=0;
    int m,n;
    struct size_m sA;
    double h;

    double StartTime, EndTime;
    /*...Initialising MPI .......*/
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&NumProcs);
    MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);

    if(MyRank == Root){


        if (argc < 2)
        {
            fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
            exit(1);
        }
        else
        {
            A = read_mat(argv[1]);
            sA = get_size(argv[1]);
            printf("Matrix loaded from file %s\n",argv[1]);
            printf("Rows = %d \n",sA.m);
            printf("Cols = %d \n",sA.n);
        }

        rows_size = sA.m;
        cols_size = sA.n;
        vector_size = sA.n;

        h = 1./(double)vector_size;
        //b = init_source_term(vector_size,h);
        x = (double*) malloc(vector_size* sizeof(double));

        for (int i = 0; i < vector_size ; ++i) {
            x[i] = 1.0;
        }
        b = (double*) malloc(vector_size* sizeof(double));
        memset(b, 0., vector_size*sizeof(double));
        cblas_dgemv(CblasColMajor, CblasNoTrans, rows_size, cols_size, 1., A, rows_size, x, 1, 0., b, 1);
        for (int i = 0; i < 100 ; ++i) {
            printf("b[i] = %f [s]\n",b[i]);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);

    /*...Broadcast Matrix and Vector size and perform input validation tests...*/
    MPI_Bcast(&rows_size, 1, MPI_INT, Root, MPI_COMM_WORLD);
    MPI_Bcast(&cols_size, 1, MPI_INT, Root, MPI_COMM_WORLD);
    MPI_Bcast(&vector_size, 1, MPI_INT, Root, MPI_COMM_WORLD);

    if(rows_size % NumProcs != 0){
        MPI_Finalize();
        if(MyRank == Root)
            printf("Error : Matrix cannot be evenly striped among processes");
        exit(-1);
    }


    if(MyRank != Root)
        b = (double*) malloc(vector_size*sizeof(double));
    /* send the vector b from root to all other processor */
    MPI_Bcast(b,vector_size,MPI_DOUBLE,Root,MPI_COMM_WORLD);

    bloc_rows_size = rows_size/NumProcs ;
    bloc_vector_size = bloc_rows_size ;
    bloc_A_size = bloc_rows_size*cols_size ;
    bloc_A = (double*) malloc(bloc_A_size* sizeof(double));
    MPI_Scatter(A,bloc_A_size,MPI_DOUBLE,bloc_A,bloc_A_size,MPI_DOUBLE,Root,MPI_COMM_WORLD);

    /* Allocate memory for the solution vector x */
    x = (double*) malloc(vector_size* sizeof(double));
    memset(x, 0., vector_size*sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);

    StartTime = MPI_Wtime();
    cg_solver_mpi(bloc_A,b,x,vector_size,bloc_rows_size,bloc_vector_size);
    EndTime = MPI_Wtime();



    if(MyRank == 0)
    {
        printf("Time for CG (dense solver)  = %f [s]\n",(EndTime-StartTime));

        for (int i = 0; i < 100 ; ++i) {
            printf("x[i] = %f [s]\n",x[i]);
        }

        // verification test
        double *r_test;
        r_test = (double*) malloc(vector_size*sizeof(double));
        memset(r_test, 0., vector_size*sizeof(double));
        cblas_dgemv (CblasColMajor, CblasNoTrans, rows_size, cols_size, 1., A, rows_size, x, 1, 0., r_test, 1);

        for (int i = 0; i < 100 ; ++i) {
            printf("r[i] = %f [s]\n",r_test[i]);
        }

        cblas_daxpy(vector_size, -1., b, 1, r_test, 1);

        for (int i = 0; i < 100 ; ++i) {
            printf("r_test[i] = %f [s]\n",r_test[i]);
        }

        double res = cblas_ddot(vector_size, r_test, 1, r_test, 1);

        printf("\terror estimate is ||Ax -b|| = %E\n", sqrt(res));
        free(r_test);
        free(A);
    }

    free(b);
    free(x);
    free(bloc_A);

    MPI_Finalize();
    return 0;
}