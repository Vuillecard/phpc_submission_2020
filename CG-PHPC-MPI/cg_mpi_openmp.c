//
// Created by pierre on 23.04.20.
//

#include "cg_mpi_openmp.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

/*
 * This file contain all the function needed to compute conjugate gradient in parallel using mpi or openmp
 */

const int MAX_ITERATIONS = 1000;
const double TOLERANCE = 1.0e-10 ;
const double NEARZERO = 1.0e-14;

/*
 * Matrix vector multiplication of bloc_A.v = results
 * vector_size: is the size of the vector v
 * bloc_row: size of the row of bloc_A
 * Note: it use openblas to compute the dot product
*/
void mat_vec_mul(double *bloc_A, double *v, double *result, int bloc_row, int vector_size)
{
    int   irow, index ;
    for(irow=0; irow<bloc_row; irow++)
    {
        index = irow * vector_size;
        result[irow] = cblas_ddot(vector_size,&bloc_A[index],1,v,1);  // dot product of ith row of bloc_A with the vector v
        //result[irow] = dot(&bloc_A[index],v, vector_size);
    }
}

/*
 * Matrix vector multiplication of bloc_A.v = results
 * vector_size: is the size of the vector v
 * bloc_row: size of the row of bloc_A
 * Note: it use openblas to compute the dot product and it use openmp to parallelise the work among the row
 */
void mat_vec_mul_openmp(double *bloc_A, double *v, double *result, int bloc_row, int vector_size)
{
    int   irow, index ;

    #pragma omp parallel for simd // dividing the work among the rows
    for(irow=0; irow<bloc_row; irow++)
    {
        index = irow * vector_size;
        result[irow] = cblas_ddot(vector_size,&bloc_A[index],1,v,1);
    }
}
/*
 * dot product between vector v1 and v2
 * vector_size: is the size of the vector v1 and v2
 * Note: it use Allreduce to share the result among the processor
 */
double mpi_dot(double *v1, double *v2,int vector_size)
{
    double result;
    double sub_result = 0.0;
    int index ;

    for ( index = 0; index < vector_size; index++)
    {
        sub_result += v1[index]*v2[index];
    }
    //sub_prod = cblas_ddot(vector_size,v1,1,v2,1);
    MPI_Allreduce(&sub_result, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return(result);
}
/*
 * dot product between vector v1 and v2
 * vector_size: is the size of the vector v1 and v2
 * Note: it use Allreduce to share the result among the processor and openmp to re-distribute the work among the process
 */
double mpi_dot_openmp(double *v1, double *v2,int vector_size) // need to pass it the buffer where to keep the result
{
    double result;

    double sub_result= 0.0;
    int index ;

    #pragma omp parallel for reduction(+:sub_result)
    for ( index = 0; index < vector_size; index++)
    {
        sub_result += v1[index]*v2[index];
    }
    MPI_Allreduce(&sub_result, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return(result);

}
/*
 * dot product between vector v1 and v2
 * vector_size: is the size of the vector v1 and v2
 * Note: it use Allreduce to share the result among the processor and openmp to re-distribute the work among the process
 */
double dot(double *v1, double *v2, int vector_size)
{
    int 	index;
    double result=0.;

    for(index=0; index<vector_size; index++)
    {
        result += v1[index]*v2[index];
    }
    return(result);
}

/*
 * Computes bloc_r = b - bloc_A*x
 * vector_size: is the size of the vector v
 * bloc_row: size of the row of bloc_A
 * MyRank: is the processor number
 * Note: it use dot to compute the dot product
 */
void residual_calculation(double *bloc_r, double *bloc_A, double *b, double *x, int bloc_row, int vector_size, int MyRank)
{

    int   irow, index, GlobalVectorIndex;
    double value;
    GlobalVectorIndex = MyRank * bloc_row;
    for(irow=0; irow<bloc_row; irow++){
        index = irow * vector_size;
        value = dot(&bloc_A[index], x, vector_size);
        bloc_r[irow] = b[GlobalVectorIndex++] - value ;
    }
}

/*
 * CG algorithm implementation using mpi and dense matrix
 */
void cg_solver_mpi( double *bloc_A, double *b, double *x, int vector_size,int bloc_rows_size ,int bloc_vector_size )
{
    // MPI initialisation
    int NumProcs, MyRank, Root=0;
    MPI_Comm_size(MPI_COMM_WORLD,&NumProcs);
    MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);

    // variable initialisation
    double r_sqrd;
    double r_sqrd_old;
    double pAp ;
    double alpha;
    double beta;
    int iter ;

    double *p;
    double *bloc_r;
    double *bloc_p;
    double *bloc_x;
    double *bloc_Ap;
    double *tmp2;

    // memories allocations
    p = (double *) malloc(vector_size*sizeof(double));
    bloc_r = (double *) malloc(bloc_vector_size*sizeof(double));
    bloc_p = (double *) malloc(bloc_vector_size*sizeof(double));
    bloc_x = (double *) malloc(bloc_vector_size*sizeof(double));
    bloc_Ap = (double *) malloc(bloc_rows_size*sizeof(double));
    tmp2 = (double *) malloc(bloc_vector_size*sizeof(double));

    // Initialisation of the conjugate gradient algorithm :

    // if we consider x = 0 then r = b
    //MPI_Scatter(b,bloc_vector_size,MPI_DOUBLE,bloc_r,bloc_vector_size,MPI_DOUBLE,Root,MPI_COMM_WORLD);

    // bloc_r = bloc_b - bloc_A*x
    residual_calculation(bloc_r, bloc_A, b, x,bloc_rows_size, vector_size , MyRank);
    // bloc_p = bloc_r
    cblas_dcopy(bloc_vector_size, bloc_r, 1, bloc_p, 1);
    // p = r
    MPI_Allgather(bloc_p, bloc_vector_size, MPI_DOUBLE, p, bloc_vector_size, MPI_DOUBLE, MPI_COMM_WORLD);
    // Compute the residual norm
    r_sqrd = mpi_dot(bloc_r, bloc_r,bloc_vector_size) ; // already use allreduce

    // begin the iterative conjugate gradient algorithm :
    for(iter = 0 ; iter < MAX_ITERATIONS ; iter++)
    {
        // stock the old residual norm to compute beta
        r_sqrd_old = r_sqrd ;

        // computing pAp :
        //   first computing bloc_Ap = bloc_A*p
        memset(bloc_Ap, 0., bloc_vector_size* sizeof(double));
        mat_vec_mul(bloc_A,p,bloc_Ap,bloc_rows_size,vector_size);
        //   then compute pAp = p*bloc_Ap
        pAp = mpi_dot(bloc_p, bloc_Ap , bloc_vector_size); // already use allreduce

        //compute alpha = ||r||²/pAp
        alpha = r_sqrd/ fmax(pAp,r_sqrd_old * NEARZERO );

        // update bloc_x = bloc_x + alpha * bloc_p;
        cblas_daxpy(bloc_vector_size, alpha, bloc_p, 1, bloc_x, 1);

        // update bloc_r = bloc_r - alpha * bloc_Ap;
        cblas_daxpy(bloc_vector_size, -alpha, bloc_Ap, 1, bloc_r, 1);

        // computing the new residual norm
        r_sqrd = mpi_dot(bloc_r, bloc_r,bloc_vector_size) ;

        // Convergence test
        if (sqrt(r_sqrd) < TOLERANCE )
        {
            if(MyRank==0)
            {
                printf("\t[STEP %d] residual = %E\n",iter,sqrt(r_sqrd));
            }

            break;
        }
        //compute alpha = ||r_new||²/||r_old||²
        beta = r_sqrd/r_sqrd_old;

        // update bloc_p = bloc_r + (r_new / r_old) * bloc_p
        cblas_dcopy(bloc_vector_size,bloc_r,1,tmp2,1);
        cblas_daxpy(bloc_vector_size, beta, bloc_p, 1, tmp2, 1);
        cblas_dcopy(bloc_vector_size,tmp2,1,bloc_p,1);

        // send p to all processor
        MPI_Allgather(bloc_p, bloc_vector_size, MPI_DOUBLE, p, bloc_vector_size, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    // send x to processor Root
    MPI_Allgather(bloc_x, bloc_vector_size , MPI_DOUBLE, x, bloc_vector_size, MPI_DOUBLE, MPI_COMM_WORLD);

    /* Debug test
    double *r;
    r = (double *) malloc(vector_size*sizeof(double));
    memset(bloc_Ap, 0., bloc_vector_size* sizeof(double));
    memset(r, 0., vector_size * sizeof(double));
    mat_vec_mul(bloc_A,x,bloc_Ap,bloc_rows_size,vector_size);
    MPI_Allgather(bloc_Ap, bloc_vector_size, MPI_DOUBLE, r, bloc_vector_size, MPI_DOUBLE, MPI_COMM_WORLD);
    cblas_daxpy(vector_size, -1., b, 1, r, 1);
    double res = cblas_ddot(vector_size, r, 1, r, 1);

    if(MyRank==0)
    printf("\terror estimate is ||Ax -b|| = %E\n", sqrt(res));

    free(r);
    */

    // free the memories :

    free(p);
    free(bloc_r);
    free(bloc_p);
    free(bloc_x);
    free(bloc_Ap);
    free(tmp2);

}
/*
 * CG algorithm for time checking implementation using mpi and dense matrix
 */
void cg_solver_mpi_time( double *bloc_A, double *b, double *x, int vector_size,int bloc_rows_size ,int bloc_vector_size )
{
    // MPI initialisation
    int NumProcs, MyRank, Root=0;
    MPI_Comm_size(MPI_COMM_WORLD,&NumProcs);
    MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);

    // Time check :
    double tot_time ;
    double start_iteration ;
    double start_mat_vec , end_mat_vec ;
    double start_dot_1 , end_dot_1 ;
    double start_reduce_1 , end_reduce_1 ;
    double start_dot_2 , end_dot_2 ;
    double start_reduce_2 , end_reduce_2 ;
    double start_allgather , end_allgather ;

    double mat_vec = 0. ;
    double dot_1 = 0. ;
    double reduce_1 = 0. ;
    double dot_2 = 0. ;
    double reduce_2 = 0. ;
    double allgather = 0. ;

    // variable initialisation
    double r_sqrd;
    double r_sqrd_old;
    double r_sqrd_bloc;
    double pAp ;
    double pAp_bloc;
    double alpha;
    double beta;
    int iter ;

    double *p;
    double *bloc_r;
    double *bloc_p;
    double *bloc_x;
    double *bloc_Ap;
    double *tmp2;

    // memories allocations
    p = (double *) malloc(vector_size*sizeof(double));
    bloc_r = (double *) malloc(bloc_vector_size*sizeof(double));
    bloc_p = (double *) malloc(bloc_vector_size*sizeof(double));
    bloc_x = (double *) malloc(bloc_vector_size*sizeof(double));
    bloc_Ap = (double *) malloc(bloc_rows_size*sizeof(double));
    tmp2 = (double *) malloc(bloc_vector_size*sizeof(double));


    // Initialisation of the conjugate gradient algorithm :

    // if we consider x = 0 then r = b
    //MPI_Scatter(b,bloc_vector_size,MPI_DOUBLE,bloc_r,bloc_vector_size,MPI_DOUBLE,Root,MPI_COMM_WORLD);

    // bloc_r = bloc_b - bloc_A*x
    residual_calculation(bloc_r, bloc_A, b, x,bloc_rows_size, vector_size , MyRank);
    // bloc_p = bloc_r
    cblas_dcopy(bloc_vector_size, bloc_r, 1, bloc_p, 1);
    // p = r
    MPI_Allgather(bloc_p, bloc_vector_size, MPI_DOUBLE, p, bloc_vector_size, MPI_DOUBLE, MPI_COMM_WORLD);

    // Compute the residual norm
    r_sqrd_bloc = dot(bloc_r, bloc_r,bloc_vector_size) ; // already use allreduce

    MPI_Allreduce(&r_sqrd_bloc, &r_sqrd, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // begin the iterative conjugate gradient algorithm
    for(iter = 0 ; iter < MAX_ITERATIONS ; iter++)
    {
        start_iteration = MPI_Wtime();

        // stock the old residual norm to compute beta
        r_sqrd_old = r_sqrd ;

        // computing pAp :
        //   first computing bloc_Ap = bloc_A*p
        start_mat_vec = MPI_Wtime();
        mat_vec_mul(bloc_A,p,bloc_Ap,bloc_rows_size,vector_size);
        end_mat_vec = MPI_Wtime();

        //   then compute pAp = p*bloc_Ap
        start_dot_1 = MPI_Wtime();
        pAp_bloc = dot(bloc_p, bloc_Ap , bloc_vector_size); // already use allreduce
        end_dot_1 = MPI_Wtime();

        start_reduce_1 = MPI_Wtime();
        MPI_Allreduce(&pAp_bloc, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        end_reduce_1 = MPI_Wtime();

        //compute alpha = ||r||²/pAp
        alpha = r_sqrd/fmax(pAp,r_sqrd_old * NEARZERO );

        // update bloc_x = bloc_x + alpha * bloc_p;
        cblas_daxpy(bloc_vector_size, alpha, bloc_p, 1, bloc_x, 1);

        // update bloc_r = bloc_r - alpha * bloc_Ap;
        cblas_daxpy(bloc_vector_size, -alpha, bloc_Ap, 1, bloc_r, 1);

        // computing the new residual norm
        start_dot_2 = MPI_Wtime();
        r_sqrd_bloc = dot(bloc_r, bloc_r,bloc_vector_size) ;
        end_dot_2 = MPI_Wtime();

        start_reduce_2 = MPI_Wtime();
        MPI_Allreduce(&r_sqrd_bloc, &r_sqrd, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        end_reduce_2 = MPI_Wtime();

        // Convergence test
        if (sqrt(r_sqrd) < TOLERANCE )
        {
            if(MyRank==0)
                printf("\t[STEP %d] residual = %E\n",iter,sqrt(r_sqrd));
            break;
        }
        //compute alpha = ||r_new||²/||r_old||²
        beta = r_sqrd/r_sqrd_old;

        // update bloc_p = bloc_r + (r_new / r_old) * bloc_p
        cblas_dcopy(bloc_vector_size,bloc_r,1,tmp2,1);
        cblas_daxpy(bloc_vector_size, beta, bloc_p, 1, tmp2, 1);
        cblas_dcopy(bloc_vector_size,tmp2,1,bloc_p,1);

        // send p to all processor
        start_allgather = MPI_Wtime();
        MPI_Allgather(bloc_p, bloc_vector_size, MPI_DOUBLE, p, bloc_vector_size, MPI_DOUBLE, MPI_COMM_WORLD);
        end_allgather = MPI_Wtime();

        tot_time =  end_allgather - start_iteration;

        mat_vec      += (end_mat_vec-start_mat_vec)/tot_time*100 ;
        dot_1        += (end_dot_1-start_dot_1)/tot_time*100 ;
        reduce_1     += (end_reduce_1-start_reduce_1)/tot_time*100;
        dot_2        += (end_dot_2-start_dot_2)/tot_time*100 ;
        reduce_2     += (end_reduce_2-start_reduce_2)/tot_time*100 ;
        allgather  +=( end_allgather -start_allgather)/tot_time*100;


    }

    printf("Processeur %d : Time for CG part mat vec = %f \n",MyRank,mat_vec/iter) ;
    printf("Processeur %d : Time for CG part dot 1 = %f \n",MyRank,dot_1/iter );
    printf("Processeur %d : Time for CG part reduce 1 = %f \n",MyRank,reduce_1/iter );
    printf("Processeur %d : Time for CG part dot 2 = %f \n",MyRank, dot_2/iter );
    printf("Processeur %d : Time for CG part reduce 2 = %f \n",MyRank, reduce_2/iter );
    printf("Processeur %d : Time for CG part allgather = %f \n",MyRank, allgather/iter);

    // send x to Root
    MPI_Gather(bloc_x, bloc_vector_size , MPI_DOUBLE, x, bloc_vector_size, MPI_DOUBLE, Root ,MPI_COMM_WORLD);

    // free the memories
    /*
    free(p);
    free(bloc_r);
    free(bloc_p);
    free(bloc_x);
    free(bloc_Ap);
    free(tmp2);
     */
}
/*
 * CG algorithm implementation using mpi, openmp and dense matrix
 */
void cg_solver_mpi_openmp( double *bloc_A, double *b, double *x, int vector_size,int bloc_rows_size ,int bloc_vector_size )
{
    // MPI initialisation
    int NumProcs, MyRank, Root=0;
    MPI_Comm_size(MPI_COMM_WORLD,&NumProcs);
    MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);

    // variable initialisation
    double r_sqrd;
    double r_sqrd_old;
    double pAp ;
    double alpha;
    double beta;
    int iter ;

    double *p;
    double *bloc_r;
    double *bloc_p;
    double *bloc_x;
    double *bloc_Ap;
    double *tmp2;

    // memories allocations
    p = (double *) malloc(vector_size*sizeof(double));
    bloc_r = (double *) malloc(bloc_vector_size*sizeof(double));
    bloc_p = (double *) malloc(bloc_vector_size*sizeof(double));
    bloc_x = (double *) malloc(bloc_vector_size*sizeof(double));
    bloc_Ap = (double *) malloc(bloc_rows_size*sizeof(double));
    tmp2 = (double *) malloc(bloc_vector_size*sizeof(double));

    // Initialisation of the conjugate gradient algorithm :

    // if we consider x = 0 then r = b
    //MPI_Scatter(b,bloc_vector_size,MPI_DOUBLE,bloc_r,bloc_vector_size,MPI_DOUBLE,Root,MPI_COMM_WORLD);

    // bloc_r = bloc_b - bloc_A*x
    residual_calculation(bloc_r, bloc_A, b, x,bloc_rows_size, vector_size , MyRank);
    // bloc_p = bloc_r
    cblas_dcopy(bloc_vector_size, bloc_r, 1, bloc_p, 1);
    // p = r
    MPI_Allgather(bloc_p, bloc_vector_size, MPI_DOUBLE, p, bloc_vector_size, MPI_DOUBLE, MPI_COMM_WORLD);
    // Compute the residual norm
    r_sqrd = mpi_dot(bloc_r, bloc_r,bloc_vector_size) ; // already use allreduce

    // begin the iterative conjugate gradient algorithm
    for(iter = 0 ; iter < MAX_ITERATIONS ; iter++)
    {
        // stock the old residual norm to compute beta
        r_sqrd_old = r_sqrd ;

        // computing pAp :
        //   first computing bloc_Ap = bloc_A*p
        memset(bloc_Ap, 0., bloc_vector_size* sizeof(double));
        mat_vec_mul_openmp(bloc_A,p,bloc_Ap,bloc_rows_size,vector_size);
        //   then compute pAp = p*bloc_Ap
        pAp = mpi_dot(bloc_p, bloc_Ap , bloc_vector_size); // already use allreduce

        //compute alpha = ||r||²/pAp
        alpha = r_sqrd/fmax(pAp,r_sqrd_old * NEARZERO );

        // update bloc_x = bloc_x + alpha * bloc_p;
        cblas_daxpy(bloc_vector_size, alpha, bloc_p, 1, bloc_x, 1);

        // update bloc_r = bloc_r - alpha * bloc_Ap;
        cblas_daxpy(bloc_vector_size, -alpha, bloc_Ap, 1, bloc_r, 1);

        // computing the new residual norm
        r_sqrd = mpi_dot(bloc_r, bloc_r,bloc_vector_size) ;

        // Convergence test
        if (sqrt(r_sqrd) < TOLERANCE )
        {
            if(MyRank==0)
                printf("\t[STEP %d] residual = %E\n",iter,sqrt(r_sqrd));
            break;
        }
        //compute alpha = ||r_new||²/||r_old||²
        beta = r_sqrd/r_sqrd_old;

        // update bloc_p = bloc_r + (r_new / r_old) * bloc_p
        cblas_dcopy(bloc_vector_size,bloc_r,1,tmp2,1);
        cblas_daxpy(bloc_vector_size, beta, bloc_p, 1, tmp2, 1);
        cblas_dcopy(bloc_vector_size,tmp2,1,bloc_p,1);

        // send p to all processor
        MPI_Allgather(bloc_p, bloc_vector_size, MPI_DOUBLE, p, bloc_vector_size, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    // send x to processor Root
    MPI_Gather(bloc_x, bloc_vector_size , MPI_DOUBLE, x, bloc_vector_size, MPI_DOUBLE,Root, MPI_COMM_WORLD);

    // free the memories
    /*
    free(p);
    free(bloc_r);
    free(bloc_p);
    free(bloc_x);
    free(bloc_Ap);
    free(tmp2);
     */
}

/*
 * Matrice vector multiplication for sparse matrix
 * row, col and val: is the sparse matrix in csr format
 * result: for the result
 * vector_size: the size of the vector to multiply
 * bloc_row : nb of row of the matrix
 * MyRank: the processor number
 */
void mat_vec_mul_sparse(int *row, int *col, double *val, double *v, double *result, int bloc_row, int vector_size,int MyRank)
{

    int start , finish ;
    int index = 0 ;
    start = MyRank*bloc_row;
    finish = (MyRank+1)*bloc_row;

    for (int i=start; i<finish; ++i) {
        result[index] = 0.0;
        for (int j=row[i]; j<row[i+1]; ++j){
            result[index] += val[j-1]*v[col[j-1]-1];
        }
        index += 1 ;
    }
}


/*
 * residual calculation for sparse matrix
 * row, col and val: is the sparse matrix in csr format
 * result: for the result
 * vector_size: the size of the vector to multiply
 * bloc_row : nb of row of the matrix
 * MyRank: the processor number
 */
void residual_calculation_sparse(double *bloc_r, int *row, int *col, double *val, double *b, double *x, int bloc_row, int vector_size, int MyRank)
{
    /*... Computes residue = b - A*x .......*/
    int start , finish ;
    int index = 0 ;

    start = MyRank*bloc_row;
    finish = (MyRank+1)*bloc_row;

    for (int i=start; i<finish; ++i) {
        bloc_r[index] = 0.0 ;
        for (int j=row[i]; j<row[i+1]; ++j){
            bloc_r[index] -= val[j-1]*x[col[j-1]-1] ;
        }
        bloc_r[index] += b[i] ;
        index += 1 ;
    }
}

/*
 * CG algorithm implementation using mpi and dense matrix
 */
void cg_solver_mpi_sparse(int *Irn, int *Jcn, double *val, double *b, double *x, int vector_size, int bloc_rows_size, int bloc_vector_size)
{
    // MPI initialisation
    int NumProcs, MyRank, Root=0;
    MPI_Comm_size(MPI_COMM_WORLD,&NumProcs);
    MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);

    // variable initialisation
    double r_sqrd;
    double r_sqrd_old;
    double pAp ;
    double alpha;
    double beta;
    int iter ;

    double *p;
    double *bloc_r;
    double *bloc_p;
    double *bloc_x;
    double *bloc_Ap;
    double *tmp2;

    // memories allocation
    p = (double *) malloc(vector_size*sizeof(double));
    bloc_r = (double *) malloc(bloc_vector_size*sizeof(double));
    bloc_p = (double *) malloc(bloc_vector_size*sizeof(double));
    bloc_x = (double *) malloc(bloc_vector_size*sizeof(double));
    bloc_Ap = (double *) malloc(bloc_rows_size*sizeof(double));
    tmp2 = (double *) malloc(bloc_vector_size*sizeof(double));

    //MPI_Scatter(b,bloc_vector_size,MPI_DOUBLE,bloc_r,bloc_vector_size,MPI_DOUBLE,Root,MPI_COMM_WORLD);

    residual_calculation_sparse(bloc_r, Irn, Jcn, val, b, x, bloc_rows_size, vector_size, MyRank);

    // bloc_p = bloc_r
    cblas_dcopy(bloc_vector_size, bloc_r, 1, bloc_p, 1);
    // p = r
    MPI_Allgather(bloc_p, bloc_vector_size, MPI_DOUBLE, p, bloc_vector_size, MPI_DOUBLE, MPI_COMM_WORLD);

    // this is r squared allreduce
    r_sqrd = mpi_dot(bloc_r, bloc_r,bloc_vector_size) ;


    for(iter = 0 ; iter < MAX_ITERATIONS ; iter++)
    {
        r_sqrd_old = r_sqrd ;
        // computing pAp :
        //   first computing bloc_Ap = bloc_A*p
        memset(bloc_Ap, 0., bloc_vector_size* sizeof(double));
        mat_vec_mul_sparse(Irn,Jcn,val,p,bloc_Ap,bloc_rows_size,vector_size,MyRank);
        //   then compute pAp = p*bloc_Ap
        pAp = mpi_dot(bloc_p, bloc_Ap , bloc_vector_size); // already use allreduce

        //compute alpha = ||r||²/pAp
        alpha = r_sqrd/fmax(pAp,r_sqrd_old * NEARZERO );

        // update bloc_x = bloc_x + alpha * bloc_p;
        cblas_daxpy(bloc_vector_size, alpha, bloc_p, 1, bloc_x, 1);

        // update bloc_r = bloc_r - alpha * bloc_Ap;
        cblas_daxpy(bloc_vector_size, -alpha, bloc_Ap, 1, bloc_r, 1);

        // computing the new residual norm
        r_sqrd = mpi_dot(bloc_r, bloc_r,bloc_vector_size) ;

        // convergence test
        if (sqrt(r_sqrd) < TOLERANCE ) {
            if(MyRank==0)
            {
                printf("\t[STEP %d] residual = %E\n",iter,sqrt(r_sqrd));
            }
            break;
        }

        //compute alpha = ||r_new||²/||r_old||²
        beta = r_sqrd/r_sqrd_old;

        // update bloc_p = bloc_r + (r_new / r_old) * bloc_p
        cblas_dcopy(bloc_vector_size,bloc_r,1,tmp2,1);
        cblas_daxpy(bloc_vector_size, beta, bloc_p, 1, tmp2, 1);
        cblas_dcopy(bloc_vector_size,tmp2,1,bloc_p,1);

        // send p to all processor
        MPI_Allgather(bloc_p, bloc_vector_size, MPI_DOUBLE, p, bloc_vector_size, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    // send x to all processor
    MPI_Allgather(bloc_x, bloc_vector_size , MPI_DOUBLE, x, bloc_vector_size, MPI_DOUBLE, MPI_COMM_WORLD);

    free(p);
    free(bloc_r);
    free(bloc_p);
    free(bloc_x);
    free(bloc_Ap);
    free(tmp2);
}

/*
 * CG algorithm implementation using mpi and dense matrix
 */
void cg_solver_mpi_sparse_time(int *Irn, int *Jcn, double *val, double *b, double *x, int vector_size, int bloc_rows_size, int bloc_vector_size)
{
    // MPI initialisation
    int NumProcs, MyRank, Root=0;
    MPI_Comm_size(MPI_COMM_WORLD,&NumProcs);
    MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);

    // Time check :
    double tot_time ;
    double start_iteration ;
    double start_mat_vec , end_mat_vec ;
    double start_dot_1 , end_dot_1 ;
    double start_reduce_1 , end_reduce_1 ;
    double start_dot_2 , end_dot_2 ;
    double start_reduce_2 , end_reduce_2 ;
    double start_allgather , end_allgather ;

    double mat_vec = 0. ;
    double dot_1 = 0. ;
    double reduce_1 = 0. ;
    double dot_2 = 0. ;
    double reduce_2 = 0. ;
    double allgather = 0. ;

    // variable initialisation
    double r_sqrd;
    double r_sqrd_bloc ;
    double r_sqrd_old;
    double pAp ;
    double pAp_bloc ;
    double alpha;
    double beta;
    int iter ;

    double *p;
    double *bloc_r;
    double *bloc_p;
    double *bloc_x;
    double *bloc_Ap;
    double *tmp2;

    // memories allocation
    p = (double *) malloc(vector_size*sizeof(double));
    bloc_r = (double *) malloc(bloc_vector_size*sizeof(double));
    bloc_p = (double *) malloc(bloc_vector_size*sizeof(double));
    bloc_x = (double *) malloc(bloc_vector_size*sizeof(double));
    bloc_Ap = (double *) malloc(bloc_rows_size*sizeof(double));
    tmp2 = (double *) malloc(bloc_vector_size*sizeof(double));

    //MPI_Scatter(b,bloc_vector_size,MPI_DOUBLE,bloc_r,bloc_vector_size,MPI_DOUBLE,Root,MPI_COMM_WORLD);

    residual_calculation_sparse(bloc_r, Irn, Jcn, val, b, x, bloc_rows_size, vector_size, MyRank);

    // bloc_p = bloc_r
    cblas_dcopy(bloc_vector_size, bloc_r, 1, bloc_p, 1);
    // p = r
    MPI_Allgather(bloc_p, bloc_vector_size, MPI_DOUBLE, p, bloc_vector_size, MPI_DOUBLE, MPI_COMM_WORLD);

    // this is r squared allreduce
    r_sqrd = mpi_dot(bloc_r, bloc_r,bloc_vector_size) ;


    for(iter = 0 ; iter < MAX_ITERATIONS ; iter++)
    {
        start_iteration = MPI_Wtime();
        r_sqrd_old = r_sqrd ;

        memset(bloc_Ap, 0., bloc_vector_size* sizeof(double));
        start_mat_vec = MPI_Wtime();
        mat_vec_mul_sparse(Irn,Jcn,val,p,bloc_Ap,bloc_rows_size,vector_size,MyRank);
        end_mat_vec = MPI_Wtime();

        start_dot_1 = MPI_Wtime();
        pAp_bloc = dot(bloc_p, bloc_Ap , bloc_vector_size);
        end_dot_1 = MPI_Wtime();

        start_reduce_1 = MPI_Wtime();
        MPI_Allreduce(&pAp_bloc, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        end_reduce_1 = MPI_Wtime();


        alpha = r_sqrd/fmax(pAp,r_sqrd_old * NEARZERO );

        cblas_daxpy(bloc_vector_size, alpha, bloc_p, 1, bloc_x, 1);

        cblas_daxpy(bloc_vector_size, -alpha, bloc_Ap, 1, bloc_r, 1);

        // computing the new residual norm
        start_dot_2 = MPI_Wtime();
        r_sqrd_bloc = dot(bloc_r, bloc_r,bloc_vector_size) ;
        end_dot_2 = MPI_Wtime();

        start_reduce_2 = MPI_Wtime();
        MPI_Allreduce(&r_sqrd_bloc, &r_sqrd, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        end_reduce_2 = MPI_Wtime();

        if (sqrt(r_sqrd) < TOLERANCE ) {
            if(MyRank==0)
                printf("\t[STEP %d] residual = %E\n",iter,sqrt(r_sqrd));

            break;
        }

        beta = r_sqrd/r_sqrd_old;

        cblas_dcopy(bloc_vector_size,bloc_r,1,tmp2,1);
        cblas_daxpy(bloc_vector_size, beta, bloc_p, 1, tmp2, 1);
        cblas_dcopy(bloc_vector_size,tmp2,1,bloc_p,1);

        start_allgather = MPI_Wtime();
        MPI_Allgather(bloc_p, bloc_vector_size, MPI_DOUBLE, p, bloc_vector_size, MPI_DOUBLE, MPI_COMM_WORLD);
        end_allgather = MPI_Wtime();

        tot_time =  end_allgather - start_iteration;

        mat_vec      += (end_mat_vec-start_mat_vec)/tot_time*100 ;
        dot_1        += (end_dot_1-start_dot_1)/tot_time*100 ;
        reduce_1     += (end_reduce_1-start_reduce_1)/tot_time*100;
        dot_2        += (end_dot_2-start_dot_2)/tot_time*100 ;
        reduce_2     += (end_reduce_2-start_reduce_2)/tot_time*100 ;
        allgather    +=( end_allgather -start_allgather)/tot_time*100;


    }

    printf("Processeur %d : Time for CG part mat vec = %f \n",MyRank,mat_vec/iter) ;
    printf("Processeur %d : Time for CG part dot 1 = %f \n",MyRank,dot_1/iter );
    printf("Processeur %d : Time for CG part reduce 1 = %f \n",MyRank,reduce_1/iter );
    printf("Processeur %d : Time for CG part dot 2 = %f \n",MyRank, dot_2/iter );
    printf("Processeur %d : Time for CG part reduce 2 = %f \n",MyRank, reduce_2/iter );
    printf("Processeur %d : Time for CG part allgather = %f \n",MyRank, allgather/iter);

    MPI_Allgather(bloc_x, bloc_vector_size , MPI_DOUBLE, x, bloc_vector_size, MPI_DOUBLE, MPI_COMM_WORLD);


    free(p);
    free(bloc_r);
    free(bloc_p);
    free(bloc_x);
    free(bloc_Ap);
    free(tmp2);
}


double * init_source_term(int n, double h)
{
    double * f;
    int i;
    f  = (double*) malloc(n*sizeof(double*));

    for(i = 0; i < n; i++) {
        f[i] = (double)i * -2. * M_PI * M_PI * sin(10.*M_PI*i*h) * sin(10.*M_PI*i*h);
    }
    return f;

}