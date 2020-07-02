//
// Created by pierre on 23.04.20.
//

#ifndef CG_PHPC_COPY_CG_MPI_OPENMP_H
#define CG_PHPC_COPY_CG_MPI_OPENMP_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "util.h"
#include "parameters.h"
#include <cblas.h>
// parallel
#include <mpi.h>
#include <omp.h>

// helper function :
double * init_source_term(int n, double h);

// - dense matrix
void mat_vec_mul(double *bloc_A, double *v, double *result, int bloc_row, int vector_size);
double mpi_dot(double *v1, double *v2,int vector_size);
double dot(double *v1, double *v2, int vector_size);
void residual_calculation(double *bloc_r, double *bloc_A, double *b, double *x, int bloc_row, int vector_size, int MyRank);

// - sparse matrix
void mat_vec_mul_sparse(int *row, int *col, double *val, double *v, double *result, int bloc_row, int vector_size,int MyRank);
void residual_calculation_sparse(double *bloc_r, int *row, int *col, double *val, double *b, double *x, int bloc_row, int vector_size, int MyRank);

// conjugate gradient algorithm for dense matrix:
void cg_solver_mpi( double *bloc_A, double *b, double *x, int vector_size,int bloc_rows_size ,int bloc_vector_size );
void cg_solver_mpi_time( double *bloc_A, double *b, double *x, int vector_size,int bloc_rows_size ,int bloc_vector_size );
void cg_solver_mpi_openmp( double *bloc_A, double *b, double *x, int vector_size,int bloc_rows_size ,int bloc_vector_size );

// conjugate gradient algorithm for sparse matrix in csr format:
void cg_solver_mpi_sparse(int *Irn, int *Jcn, double *val, double *b, double *x, int vector_size, int bloc_rows_size, int bloc_vector_size);
void cg_solver_mpi_sparse_time(int *Irn, int *Jcn, double *val, double *b, double *x, int vector_size, int bloc_rows_size, int bloc_vector_size);


#endif //CG_PHPC_COPY_CG_MPI_OPENMP_H
