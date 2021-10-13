/*
                \WAS/
    This program is designed to test the correctness of the POWER10 GEMM 
    kernels in `blis/sandbox/power10`.

    By default, the correctness of the kernels is determined by measuring how 
    close the return value of the following function is to zero for square 
    matrix sizes.

    F(A, B, C_orig, C_ans, alpha, beta, t) =

        normf( (C_ans * t) - ((alpha * A * B + beta * C_orig) * t) )

    The function above can only be used to measure correctness if
    A, B, C_orig, and t have been randomized and normalized.

    The correctness is reported by printing the function return value along
    with the matrices' sizes.

*/


#include "blis.h"
#include "correctness.h"
#include "../../bli_sandbox.h"
#include "common.h"

#include <stdio.h>
#include <arm_neon.h>
// print kernel name
const char* get_kernel_name(int kernel_id)
{
    switch (kernel_id)
    {
        case FLOAT16 : return "bli_shgemm";
        case INT16   : return "bli_i16gemm";
        default: printf("INCORRECT KERNEL ID\n"); exit(-1);
    }
}

// normalize the vector using the forbenious norm
void normalize_vec(float *t, int n)
{
    // normalize t
    float norm_factor;
    bli_snormfv(n, t, 1, &norm_factor);
    // round up to closest power of 2
    norm_factor = 1 / (pow( 2.0, ceil( log2( norm_factor ) ) ));
    bli_sscalv(BLIS_NO_CONJUGATE, n, &norm_factor, t, 1);
}

	// Pre-conditions:
	// - a is randomized.
	// - b is randomized.
	// - c_orig is randomized.
	// Note:
	// - alpha and beta should have non-zero imaginary components in the
	//   complex cases in order to more fully exercise the implementation.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   C := beta * C_orig + alpha * transa(A) * transb(B)
	//
	// is functioning correctly if
	//
	//   normfv( v - z )
	//
	// is negligible, where
	//
	//   v = C * t
	//   z = ( beta * C_orig + alpha * transa(A) * transb(B) ) * t
	//     = beta * C_orig * t + alpha * transa(A) * transb(B) * t
	//     = beta * C_orig * t + alpha * transa(A) * w
	//     = beta * C_orig * t + z
float get_resid(
    int m, int n, int k,
    float *a, int rsa, int csa,
    float *b, int rsb, int csb,
    float *c, int rsc, int csc,
    float *c_orig,
    float *alpha, float *beta
)
{

    float t[n], v[m], w[k], z[m];
    float one = 1.0, zero = 0.0;

    bli_srandv(n, t, 1);

    // normalize so that the values are at the same precision of the input values
    normalize_vec(t, n);

    // v = C * t
    bli_sgemv(
        BLIS_NO_TRANSPOSE,
        BLIS_NO_CONJUGATE,
        m,
        n,
        &one,
        c, rsc, csc,
        t, 1,
        &zero,
        v, 1
    );

    // w = B * t
    bli_sgemv(
        BLIS_NO_TRANSPOSE,
        BLIS_NO_CONJUGATE,
        k,
        n,
        &one,
        b, rsb, csb,
        t, 1,
        &zero,
        w, 1
    );

    // z = alpha * A * w
    bli_sgemv(
        BLIS_NO_TRANSPOSE,
        BLIS_NO_CONJUGATE,
        m,
        k,
        alpha,
        a, rsa, csa,
        w, 1,
        &zero,
        z, 1
    );

    // z += beta * C_orig * t
    bli_sgemv(
        BLIS_NO_TRANSPOSE,
        BLIS_NO_CONJUGATE,
        m,
        n,
        beta,
        c_orig, rsc, csc,
        t, 1,
        &one,
        z, 1
    );

    // v = v - z
    bli_ssubv ( 
        BLIS_NO_CONJUGATE,
        m,
        z, 1,
        v, 1
    );

    // norm = normfv(v)
    float norm;
    bli_snormfv (
        m,
        v, 1,
        &norm
    );

    return norm;
}


// test to see if the result from a BLIS GEMM kernel is correct for a given m x n x k mat-mul
// assumes the matrices are of type float
// assumes the matrices were randomized and normalized
void correctness_checker(
    int m, int n, int k,
    float *a, int rsa, int csa,
    float *b, int rsb, int csb,
    float *c_orig, int rsc, int csc,
    float *c_ans,
    float alpha, float beta
)
{   
    double start, end;

    start = bli_clock();
    float resid = get_resid (
            m, n, k,
            a, rsa, csa,
            b, rsb, csb,
            c_ans, rsc, csc,
            c_orig,
            &alpha, &beta
    );
    end = bli_clock();

    printf("%d, %d, %d, %8.4le\n", m,n,k, resid);
}


// create all the correctness checking functions for each kernel
GEN_I_COR_KERNEL( sh,  bli_shgemm, float16_t, float16_t);
GEN_I_COR_KERNEL(i16, bli_i16gemm,   int16_t,   int16_t);

// using the DATATYPE enum, gather test the correctness of the respective GEMM kernel
void run_correctness_kernel(int kernel_id, int m, int n, int k)
{
    switch (kernel_id)
    {
        case FLOAT16 : shcorrectness_kernel (m, n, k); break;
        case INT16   : i16correctness_kernel(m, n, k); break;
        default: break;
    }
}

void test_correctness(int kernel_id, int start, int end, int inc)
{
    printf("%s correctness test\n", get_kernel_name(kernel_id));
    printf("m, n, k, resid\n");
    int m,n,k;
    for (int p=start; p<=end; p+=inc)
    {
        m=n=k=p;
        run_correctness_kernel(kernel_id, m, n, k);
    }
}

// correctness test for bfloat16 gemm
int main(int argc, char *argv[])
{
    test_correctness(FLOAT16, 80, 2000, 80);
    test_correctness(  INT16, 80, 2000, 80);
}
