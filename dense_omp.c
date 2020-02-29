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
void dampenTransitionMatrix(
    double *A, const int pages, const double factor)
{
    int rowx, colx;
    for (rowx = 0; rowx < pages; rowx++)
    {
        for (colx = 0; colx < pages; colx++)
        {
            A[rowx * pages + colx] *= (1 - factor);
            A[rowx * pages + colx] += (factor * (1.0 / pages));
        }
    }
}

/* populate with page linking probabilities */
void fillTransitionMatrix(double *A, const int pages)
{
    int colx, rowx;
    for (colx = 0; colx < pages; colx++)
    {
        int beforex = colx - 1;
        int afterx = colx + 1;

        for (rowx = 0; rowx < pages; rowx++)
        {
            /* fill with placeholder */
            A[rowx * pages + colx] = 0.0;
        }

        if (beforex > -1)
            /* P(i-1) <- P(i) */
            A[beforex * pages + colx] = 0.5;
        else
        {
            /* handle first page */
            A[afterx * pages + colx] = 1.0;
            continue;
        }

        if (afterx < pages)
            /* P(i) -> P(i+1) */
            A[afterx * pages + colx] = 0.5;
        else
            /* handle last page */
            A[0 * pages + colx] = 0.5;
    }
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
    double *A = (double *)malloc(sizeof(double) * (pages * pages));
    double *X = (double *)malloc(sizeof(double) * pages);
    double *y = (double *)malloc(sizeof(double) * pages);

    if (A == NULL || X == NULL || y == NULL)
    {
        printf("allocation failed for size %d", pages);
        exit(1);
    }

    /* compute initial guess at page rank */
    double initial_guess = 1.0 / pages;
    printf("Initial Guess: %.6f", initial_guess);
    printf("\n\n");

    fillTransitionMatrix(A, pages);
    printTransitionMatrix(A, pages, 1);

    dampenTransitionMatrix(A, pages, damping_factor);
    printTransitionMatrix(A, pages, 3);

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
            for (colx = 0; colx < pages; colx++)
            {
                /* update page rank guess */
                y[rowx] += (A[rowx * pages + colx] * X[colx]);
            }
        }

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