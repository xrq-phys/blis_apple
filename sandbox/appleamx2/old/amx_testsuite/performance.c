/*
                \WAS/
    This program is designed to gather the performance data of the POWER10
    GEMM kernels in `blis/sandbox/power10`.

    By default, the performance of the kernels is gather over a set of square
    matrices. The perfromance results are reported in GFLOPS, and outputted in
    CSV format.

*/

#include "performance.h"
#include "blis.h"
#include "../../bli_sandbox.h"
#include "common.h"

#include <stdio.h>
// print kernel name
const char* get_kernel_name(int kernel_id)
{
    switch (kernel_id)
    {
        case FLOAT64    : return "bli_d_gemm";
        case FLOAT32    : return "bli_s_gemm";
        case FLOAT16    : return "bli_shgemm";
        case FLOAT16_32 : return "bli_s_shgemm";
        case INT16      : return "bli_i16gemm";
        case INT16_32   : return "bli_i32_i16gemm";
        default: printf("INCORRECT KERNEL ID\n"); exit(-1);
    }
}

// create all the performance gathering functions for each kernel
GET_PERF_API_TEMP(     d_,      bli_d_gemm, float64_t, float64_t);
GET_PERF_API_TEMP(     s_,      bli_s_gemm, float32_t, float32_t);
GET_PERF_API_TEMP(     sh,      bli_shgemm, float16_t, float16_t);
GET_PERF_API_TEMP(   s_sh,    bli_s_shgemm, float16_t, float32_t);
GET_PERF_API_TEMP(    i16,     bli_i16gemm,   int16_t,   int16_t);
GET_PERF_API_TEMP(i32_i16, bli_i32_i16gemm,   int16_t,   int32_t);


// using the DATATYPE enum, gather the performance of the respective GEMM kernel
double run_kernel(int kernel_id, int nreps, int m, int n, int k)
{
    switch (kernel_id)
    {
        case FLOAT64    : return test_d_api     (nreps, m, n, k);
        case FLOAT32    : return test_s_api     (nreps, m, n, k);
        case FLOAT16    : return test_shapi     (nreps, m, n, k);
        case FLOAT16_32 : return test_s_shapi   (nreps, m, n, k);
        case INT16      : return test_i16api    (nreps, m, n, k);
        case INT16_32   : return test_i32_i16api(nreps, m, n, k);
        default: return -1.0;
    }
}

// print the performance data in CSV format
// performance is measured in terms of GFLOPs
void print_perf_data(int m, int n, int k, double best_time)
{
    double GFLOPS = (2.0 * m * n * k) / (1e9 * best_time);
    printf("%d, %d, %d, %.2f\n", m, n, k, GFLOPS);
}

// get performance data
void get_perf(int kernel_id, int nreps, int start, int end, int inc)
{
    // csv header
    printf("%s performance\n", get_kernel_name(kernel_id));
    printf("m, n, k, GFLOPS\n");

    int m,n,k;

    // run over all problem sizes
    for (int p=start; p<=end; p+=inc)
    {
        // change here to adjust problem size
        m = p,
        n = p,
        k = p;

        double best_run_time = run_kernel(kernel_id, nreps, m, n, k);

        print_perf_data(m, n, k, best_run_time);
    }
}

int main(int argc, char *argv[])
{
    // number of times the kernel will be run
    int nreps = 5;

    // run a respective kernel
    get_perf(    FLOAT64, nreps, 20, 1200, 20);
    get_perf(    FLOAT32, nreps, 40, 2400, 40);
    get_perf(    FLOAT16, nreps, 80, 4800, 80);
    get_perf( FLOAT16_32, nreps, 40, 2400, 40);
    get_perf(      INT16, nreps, 40, 2400, 40);
    get_perf(   INT16_32, nreps, 40, 2400, 40);

    return 0;
}
