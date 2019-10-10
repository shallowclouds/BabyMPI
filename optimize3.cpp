#include "mpi.h"
#include <math.h>
#include <stdio.h>
#define MIN(a,b)  ((a)<(b)?(a):(b))

int main(int argc, char *argv[]) {
    int count;        /* Local prime count */
    double elapsed_time; /* Parallel execution time */
    int first;        /* Index of first multiple */
    int global_count = 0; /* Global prime count */
    int high_value;   /* Highest value on this proc */
    int i;
    int id;           /* Process ID number */
    int index = 0;        /* Index of current prime */
    int low_value;    /* Lowest value on this proc */
    char *marked;       /* Portion of 2,...,'n' */
    int n;            /* Sieving from 2, ..., 'n' */
    int p;            /* Number of processes */
    int proc0_size;   /* Size of proc 0's subarray */
    int prime;        /* Current prime */
    int size;         /* Elements in 'marked' */


    // 这是我电脑的一级缓存大小，256 KB
    const int CACHE_SIZE = 256 * 1024;

    // 这里将分块大小设置为一级缓存的一半，因为考虑到可能有超线程之类的东西
    const int BLOCK_SIZE = CACHE_SIZE / 2;

    MPI_Init(&argc, &argv);

    /* Start the timer */

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();


    if (argc != 2) {
        if (!id) printf("Command line: %s <m>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }
    n = atoi(argv[1]);

    /* Figure out this process's share of the array, as
       well as the integers represented by the first and
       last array elements */

    low_value = 2 + id * (n - 1) / p;
    high_value = 1 + (id + 1) * (n - 1) / p;
    size = high_value - low_value + 1;

    // 先筛出前 sqrt(n) 个数中的所有素数
    int sqrt_size = int(sqrt(double(n))) + 1;
    // pre_primes[i] = 0 说明 i 是素数
    char* pre_mark = (char*)malloc(sqrt_size * sizeof(sqrt_size));
    // 存放前 sqrt(n) 个数中的素数
    int* primes = (int*)malloc(sqrt_size * sizeof(int));
    if(pre_mark == NULL)
    {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }
    // 先全部清零
    memset(pre_mark, 0, sqrt_size * sizeof(sqrt_size));
    // 素数个数
    int prime_count = 0;
    for(i = 2; i <= sqrt_size; i++)
    {
        // 如果 i 是素数，那么就跳过，因为它的倍数必然已经被前面的数筛过了
        if(pre_mark[i])
            continue;
        // 将素数放入数组中
        primes[prime_count++] = i;
        // i 是素数，那么它的所有倍数都不是素数
        for(int j = i * 2; j <= sqrt_size; j += i)
        {
            pre_mark[j] = 1;
        }
    }

    /* Bail out if all the primes used for sieving are
       not all held by process 0 */

    proc0_size = (n - 1) / p;

    if ((2 + proc0_size) < (int) sqrt((double) n)) {
        if (!id) printf("Too many processes\n");
        MPI_Finalize();
        exit(1);
    }

    /* Allocate this process's share of the array. */

    // 根据缓存大小进行分块
    int block_count = size / BLOCK_SIZE + (size % BLOCK_SIZE != 0);
    char** blocks = (char**)malloc(block_count * sizeof(char*));
    for(i = 0; i < block_count; i++)
    {
        blocks[i] = (char*)malloc(BLOCK_SIZE * sizeof(char));
        memset(blocks[i], 0, BLOCK_SIZE * sizeof(char));
    }
    // 每一块内进行筛选
    for(i = 0; i < block_count; i++)
    {
        int cur_prime_pos = 0;
        prime = primes[cur_prime_pos];
        int cur_block_first = i * BLOCK_SIZE;
        int cur_block_last = MIN((i + 1) * BLOCK_SIZE, size);
        do {
            if(prime * prime > cur_block_first + low_value)
            {
                first = prime * prime - cur_block_first - low_value;
            }
            else
            {
                if(!((low_value + cur_block_first) % prime))
                    first = 0;
                else
                    first = prime - ((low_value + cur_block_first) % prime);
            }
            for(int j = first; j < BLOCK_SIZE; j += prime)
                blocks[i][j] = 1;
            cur_prime_pos ++;
            prime = primes[cur_prime_pos];
        } while (prime * prime <= n && cur_prime_pos < prime_count);
    }
    // 统计每块内素数个数
    count = 0;
    for (i = 0; i < size; i++)
    {
        if (!blocks[i / BLOCK_SIZE][i % BLOCK_SIZE])
            count++;
    }
    // 释放内存
    free(pre_mark);

    for(i = 0; i < block_count; i++)
        free(blocks[i]);

    if (p > 1)
        MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM,
                   0, MPI_COMM_WORLD);
    else
        global_count = count;

    /* Stop the timer */

    elapsed_time += MPI_Wtime();


    /* Print the results */

    if (!id) {
        printf("There are %d primes less than or equal to %d\n",
               global_count, n);
        printf("SIEVE (%d) %10.6f\n", p, elapsed_time);
    }
    MPI_Finalize();
    return 0;
}
