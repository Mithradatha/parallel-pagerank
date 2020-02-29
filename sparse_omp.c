#include <omp.h>    /* OpenMP */
#include <stdlib.h> /* malloc */
#include <stdio.h>  /* prints */

#define DEBUG 1

/* set each element in array to 0.0 */
void initZero(double *arr, const int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        arr[i] = 0.0;
    }
}

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
void dampenTempoaryRankVector(
    double *y, const int pages, const double factor)
{
    int i;
    for (i = 0; i < pages; i++)
    {
        y[i] *= (1 - factor);
        y[i] += (factor * (1.0 / pages));
    }
}

/* populate with all non-zero entries */
void fillTransitionVector(double *A, const int pages)
{
    A[0] = 0.5;
    A[1] = 0.5;
    A[2] = 1.0;

    int i;
    for (i = 3; i < 2 * pages - 1; i++)
    {
        A[i] = 0.5;
    }
}

/* populate with number of non-zero entries per row */
void fillCountVector(int *iA, const int pages)
{
    iA[0] = 0;

    int i;
    for (i = 1; i < pages; i++)
    {
        iA[i] = iA[i - 1] + 2;
    }

    iA[pages] = 2 * pages - 1;
}

/* populate with column number for non-zero entries */
void fillIndexVector(int *jA, const int pages)
{
    jA[0] = 1;
    jA[1] = pages - 1;

    int i = 2;

    int colx;
    for (colx = 1; colx < pages - 1; colx++)
    {
        jA[i++] = colx - 1;
        jA[i++] = colx + 1;
    }

    jA[i] = pages - 2;
}

/* populate with rank initial guess */
initializeRankVector(double *X, const int pages, const double guess)
{
    int page;
    for (page = 0; page < pages; page++)
    {
        X[page] = guess;
    }
}

/* print indicies and values for vector of doubles */
void printDBLVector(const double *V, const int size)
{
    if (!DEBUG)
        return;

    int i;
    for (i = 0; i < size; i++)
    {
        printf("Index [%d] Value: %.6f", i, V[i]);
        printf("\n");
    }

    printf("\n");
}

/* print indicies and values for vector of integers */
void printINTVector(const int *V, const int size)
{
    if (!DEBUG)
        return;

    int i;
    for (i = 0; i < size; i++)
    {
        printf("Index [%d] Value: %d", i, V[i]);
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
    int master_thread_id = 0;

    /* default values */
    int pages = 10;
    double damping_factor = 0.15;
    int matvecs = 1000;
    int threads = 1;

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

        /* redefine number of threads from open mp */
#pragma omp parallel shared(threads)
    {
        int my_id = omp_get_thread_num();

        if (my_id == master_thread_id)
        {
            threads = omp_get_num_threads();
        }
    }

    /* allocate vectors */
    double *A = (double *)malloc(sizeof(double) * (2 * pages - 1));
    int *iA = (int *)malloc(sizeof(int) * (pages + 1));
    int *jA = (int *)malloc(sizeof(int) * (2 * pages - 1));
    double *X = (double *)malloc(sizeof(double) * pages);
    double *y = (double *)malloc(sizeof(double) * pages);

    if (A == NULL || iA == NULL || jA == NULL || X == NULL || y == NULL)
    {
        printf("allocation failed for size %d", pages);
        exit(1);
    }

    /* compute initial guess at page rank */
    double initial_guess = 1.0 / pages;
    printf("Initial Guess: %.6f", initial_guess);
    printf("\n\n");

    fillTransitionVector(A, pages);
    printDBLVector(A, 2 * pages - 1);

    fillCountVector(iA, pages);
    printINTVector(iA, pages + 1);

    fillIndexVector(jA, pages);
    printINTVector(jA, 2 * pages - 1);

    initializeRankVector(X, pages, initial_guess);
    initZero(y, pages);

    /* start timer for code execution time */
    double start_time = omp_get_wtime();

    int iteration, rowx, colx;
    for (iteration = 0; iteration < matvecs; iteration++)
    {
#pragma omp parallel for collapse(2)
        for (rowx = 0; rowx < pages; rowx++)
        {
            for (colx = iA[rowx]; colx < iA[rowx + 1]; colx++)
            {
                /* update page rank guess */
                y[rowx] += (A[colx] * X[jA[colx]]);
            }
        }

        dampenTempoaryRankVector(y, pages, damping_factor);

        /* store updated rank */
        copyArray(y, pages, X);
        initZero(y, pages);
    }

    /* stop timer */
    double end_time = omp_get_wtime();

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
           pages, damping_factor, matvecs, threads);
    printf("\n\n");

    /* output solution time */
    printf("runtime = %.16e", end_time - start_time);
    printf("\n\n");
}
