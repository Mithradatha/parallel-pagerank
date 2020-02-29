#include "mpi.h"    /* mpi */
#include <stdlib.h> /* malloc */
#include <stdio.h>  /* prints */

#define DEBUG 1

/* copy contents of one array into another of equal size */
void copyArray(const double *src, const int size, double *dest)
{
    int i;
    for (i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
}

/* return index of minimum value in array */
int minIndex(const double *arr, const int size)
{
    int minx = 0;
    double min = arr[minx];

    int arrx;
    for (arrx = 0; arrx < size; arrx++)
    {
        if (arr[arrx] < min)
        {
            minx = arrx;
            min = arr[minx];
        }
    }

    return minx;
}

/* return index of maximum value in array */
int maxIndex(const double *arr, const int size)
{
    int maxx = 0;
    double max = arr[maxx];

    int arrx;
    for (arrx = 0; arrx < size; arrx++)
    {
        if (arr[arrx] > max)
        {
            maxx = arrx;
            max = arr[maxx];
        }
    }

    return maxx;
}

/* add potential linking to random page */
void dampenTransitionMatrix(
    double *A, const int rows, const int pages, const double factor)
{
    int rowx, colx;
    for (rowx = 0; rowx < rows; rowx++)
    {
        for (colx = 0; colx < pages; colx++)
        {
            A[rowx * pages + colx] *= (1 - factor);
            A[rowx * pages + colx] += (factor * (1.0 / pages));
        }
    }
}

/* populate with page linking probabilities */
double *createTranslationMatrix(
    const int pages, const int beginx, const int endx)
{
    const int rows = endx - beginx;

    double *local_A = (double *)malloc(sizeof(double) * (pages * rows));

    if (local_A == NULL)
    {
        printf("allocation failed for size %d", pages);
        exit(1);
    }

    int rowx, colx;
    for (rowx = 0; rowx < rows; rowx++)
    {
        int global_rowx = beginx + rowx;

        local_A[rowx * pages + 0] = 0.0;
        local_A[rowx * pages + (pages - 1)] = 0.0;

        if (global_rowx == 1)
            /* handle first page */
            local_A[rowx * pages + 0] = 1.0;
        else if (global_rowx == 0 || global_rowx == pages - 2)
            /* handle last page */
            local_A[rowx * pages + (pages - 1)] = 0.5;
    }

    for (colx = 1; colx < pages - 1; colx++)
    {
        int beforex = colx - 1;
        int afterx = colx + 1;

        for (rowx = 0; rowx < rows; rowx++)
        {
            int global_rowx = beginx + rowx;

            if (global_rowx == beforex || global_rowx == afterx)
                /* P(i-1) <- P(i) -> P(i+1) */
                local_A[rowx * pages + colx] = 0.5;
            else
                local_A[rowx * pages + colx] = 0.0;
        }
    }

    return local_A;
}

/* populate with rank initial guess */
initializeRankVector(double *X, const int rows, const double guess)
{
    int rowx;
    for (rowx = 0; rowx < rows; rowx++)
    {
        X[rowx] = guess;
    }
}

/* print page linking probabilites */
void printTransitionMatrix(
    const double *A, const int pages, const int precision)
{
    if (!DEBUG)
        return;

    int rowx, colx;
    for (rowx = 0; rowx < pages; rowx++)
    {
        for (colx = 0; colx < pages; colx++)
        {
            printf("%.*f ", precision, A[rowx * pages + colx]);
        }

        printf("\n");
    }

    printf("\n");
}

/* special print for page rank vector */
void printRankVector(const double *X, const int pages)
{
    if (!DEBUG)
        return;

    int page;
    for (page = 0; page < pages; page++)
    {
        printf("Page [%d] Rank: %.6f", page + 1, X[page]);
        printf("\n");
    }

    printf("\n");
}

int main(int argc, char *argv[])
{
    /* declarations */
    int master_proc_id = 0;

    /* default values */
    int pages = 10;
    double damping_factor = 0.15;
    int matvecs = 1000;

    int argx;

    /* read optional command line arguments to overide defaults */
    for (argx = 1; argx < argc; argx++)
    {
        /* optionally redefine number of pages from command line */
        if (strcmp(argv[argx], "-n") == 0)
        {
            pages = atoi(argv[++argx]);
        }
        /* optionally redefine damping factor from command line */
        else if (strcmp(argv[argx], "-q") == 0)
        {
            damping_factor = atof(argv[++argx]);
        }
        /* optionally redefine number of matvecs from command line */
        else if (strcmp(argv[argx], "-k") == 0)
        {
            matvecs = atoi(argv[++argx]);
        }
    }

    int processors, my_id, rows_per_processor;
    double initial_guess;

    /* Initialize MPI */
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

    if (my_id == master_proc_id)
    {
        /* assume no remainder */
        rows_per_processor = pages / processors;

        /* compute initial guess at page rank */
        initial_guess = 1.0 / pages;
        printf("Initial Guess: %.6f", initial_guess);
        printf("\n\n");
    }

    MPI_Bcast(&rows_per_processor, 1, MPI_INT,
              master_proc_id, MPI_COMM_WORLD);

    MPI_Bcast(&initial_guess, 1, MPI_DOUBLE,
              master_proc_id, MPI_COMM_WORLD);

    /* distribute pages */
    int beginx = my_id * rows_per_processor;
    int endx = beginx + rows_per_processor;

    printf("Processor: %d, Rows: [%d, %d)\n", my_id, beginx, endx);
    MPI_Barrier(MPI_COMM_WORLD);

    /* allocate vectors */
    double *local_A = createTranslationMatrix(pages, beginx, endx);
    double *X = (double *)malloc(sizeof(double) * pages);
    double *local_x = (double *)malloc(sizeof(double) * rows_per_processor);
    double *local_y = (double *)malloc(sizeof(double) * rows_per_processor);

    if (X == NULL || local_x == NULL || local_y == NULL)
    {
        printf("allocation failed for size %d", pages);
        exit(1);
    }

    dampenTransitionMatrix(local_A, rows_per_processor, pages, damping_factor);

    initializeRankVector(local_x, rows_per_processor, initial_guess);

    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_id == master_proc_id)
    {
        /* start timer */
        start_time = MPI_Wtime();
    }

    int iteration, rowx, colx;
    for (iteration = 0; iteration < matvecs; iteration++)
    {
        MPI_Allgather(local_x, rows_per_processor, MPI_DOUBLE,
                      X, rows_per_processor, MPI_DOUBLE,
                      MPI_COMM_WORLD);

        for (rowx = 0; rowx < rows_per_processor; rowx++)
        {
            local_y[rowx] = 0.0;
            for (colx = 0; colx < pages; colx++)
                /* update page rank guess */
                local_y[rowx] += local_A[rowx * pages + colx] * X[colx];
        }

        /* store updated rank */
        copyArray(local_y, rows_per_processor, local_x);
    }

    MPI_Gather(local_x, rows_per_processor, MPI_DOUBLE,
               X, rows_per_processor, MPI_DOUBLE,
               master_proc_id, MPI_COMM_WORLD);

    if (my_id == master_proc_id)
    {
        /* stop timer */
        end_time = MPI_Wtime();
        printf("\n");

        /* calculate min/max indicies */
        int minx = minIndex(X, pages);
        int maxx = maxIndex(X, pages);

        /* output page rank information */
        printRankVector(X, pages);
        printf("Minimum Page [%d] Rank: %.6f", minx + 1, X[minx]);
        printf("\n");
        printf("Maximum Page [%d] Rank: %.6f", maxx + 1, X[maxx]);
        printf("\n\n");

        /* output environment values */
        printf("N=%d, Q=%.2f, K=%d, P=%d",
               pages, damping_factor, matvecs, processors);
        printf("\n\n");

        /* output solution time */
        printf("runtime = %.16e", end_time - start_time);
        printf("\n\n");
    }

    MPI_Finalize();
}