CC=mpicc
CC2= gcc
LD=${CC}
LD2 = ${CC2}
COVFLAGS = -fprofile-arcs -ftest-coverage
INCLUDES = -I$(OPENBLAS_ROOT)/include
CFLAGS+=-Wall -pedantic -fopenmp -O3 -fPIC $(INCLUDES) #${COVFLAGS}
DBGFLAGS = -g -pg

LDFLAGS+=-lm -fopenmp -L$(OPENBLAS_ROOT)/lib -lopenblas # ${COVFLAGS} -g -pg -lopenblas
#OBJS=blas.o cg.o util.o cg_blas.o mmio.o

OBJS= mmio.o util.o second.o main_cg_sparse.o cg_mpi_openmp.o

OBJS6= mmio.o util.o second.o main_cg_sparse_time.o cg_mpi_openmp.o

OBJS2= mmio.o util.o second.o cg.o cg_blas.o

OBJS3= mmio.o util.o second.o main.o cg_mpi_openmp.o

OBJS4= mmio.o util.o second.o main_cg_time.o cg_mpi_openmp.o

OBJS5= mmio.o util.o second.o main_cg_mpi_openmp.o cg_mpi_openmp.o

all: cg_mpi_sparse cg_original cg_mpi_time cg_mpi cg_mpi_openmp cg_mpi_sparse_time

cg_mpi_sparse: $(OBJS)
	$(LD) $(OBJS) $(LDFLAGS) -o $@


cg_mpi_sparse_time: $(OBJS6)
	$(LD) $(OBJS6) $(LDFLAGS) -o $@

cg_mpi:	$(OBJS3)
	$(LD) $(OBJS3) $(LDFLAGS) -o $@

cg_mpi_time: $(OBJS4)
	$(LD) $(OBJS4) $(LDFLAGS) -o $@

cg_mpi_openmp: $(OBJS5)
	$(LD) $(OBJS5) $(LDFLAGS) -o $@
	#export OMP_NUM_THREADS=2  # use this for OpenMP

cg_original: $(OBJS2)
	$(LD2) $(OBJS2) $(LDFLAGS) -o $@


#coverage study
coverage: 
	./cg_mpi_sparse matrix3x3.mtx vector3.mtx
	gcov *.c
	lcov --capture --directory . --output-file coverage.info
	genhtml coverage.info --output-directory out_html

clean:
	rm -Rf cg_mpi_sparse cg_original cg_mpi_time cg_mpi cg_mpi_openmp cg_mpi_sparse_time *.o *~ *.gcda *.gcov *.gcno out_html coverage.info gmon.out

