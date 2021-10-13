// templates for generating correctness checking functions that check the correctness of GEMM kernels
// using the BLIS GEMM correctness method

#define COR_KERNEL_NAME_(ch) ch ## correctness_kernel
#define COR_KERNEL_NAME(ch) COR_KERNEL_NAME_(ch)


// correctness template for ALL types
#define GEN_I_COR_KERNEL(ch, kernel, input_t, output_t) \
void COR_KERNEL_NAME(ch) (int m, int n, int k) \
{ \
    int rsa = 1, csa = m, \
        rsb = 1, csb = k, \
        rsc = 1, csc = m; \
\
    input_t *a, *b; \
\
    output_t *c_ans, *c_orig, alpha, beta; \
\
    float *a_float, *b_float, \
          *c_ans_float, *c_orig_float; \
\
    /* buffers that will be passed into the kernel */ \
    a = (input_t *) malloc (m * k * sizeof(input_t)); \
    b = (input_t *) malloc (k * n * sizeof(input_t)); \
    c_ans  = (output_t *) malloc (m * n * sizeof(output_t)); \
    c_orig = (output_t *) malloc (m * n * sizeof(output_t)); \
\
    /* std format buffers that will be used by the correctness checker */ \
    a_float = (float *) malloc (m * k * sizeof(float)); \
    b_float = (float *) malloc (k * n * sizeof(float)); \
    c_ans_float  = (float *) malloc (m * n * sizeof(float)); \
    c_orig_float = (float *) malloc (m * n * sizeof(float)); \
\
    /* randomize matrices with float vals */ \
    bli_srandv(m*k, a_float, 1); \
    bli_srandv(k*n, b_float, 1); \
    bli_srandv(m*n, c_orig_float, 1); \
\
    /* normalize the matrices */ \
    normalize_vec(a_float, m*k); \
    normalize_vec(b_float, k*n); \
    normalize_vec(c_orig_float, m*n); \
\
    /* cast the float buffers into the buffers for the kernel */ \
    for (int i = 0; i < m*k; ++i) a[i] = a_float[i]; \
    for (int i = 0; i < k*n; ++i) b[i] = b_float[i]; \
\
    /* cast float buffers to support int values */ \
    for (int i = 0; i < m*n; ++i) c_orig[i] = c_orig_float[i]; \
    for (int i = 0; i < m*n; ++i) c_orig_float[i] = c_orig[i]; \
\
    /* cast the kernel buffers into the float buffers to ensure that the values match */ \
    for (int i = 0; i < m*k; ++i) a_float[i] = a[i]; \
    for (int i = 0; i < k*n; ++i) b_float[i] = b[i]; \
\
    /* init alpha and beta */ \
    alpha = 1; \
    beta  = 1; \
\
    /* run kernel to get result in c_ans */ \
    memcpy(c_ans, c_orig, m * n * sizeof(output_t)); \
    kernel( \
            BLIS_NO_TRANSPOSE, \
            BLIS_NO_TRANSPOSE, \
            m, \
            n, \
            k, \
            &alpha, \
            a, rsa, csa, \
            b, rsb, csb, \
            &beta, \
            c_ans, rsc, csc \
    ); \
\
    /* cast integer result into float buffer since float is our std format for correctness checking */ \
    for (int i = 0; i < m*n; ++i) c_ans_float[i] = c_ans[i]; \
\
    /* using the BLIS GEMM correctness check method, get the resid */ \
    correctness_checker( \
        m, n, k, \
        a_float, rsa, csa, \
        b_float, rsb, csb, \
        c_orig_float, rsc, csc, \
        c_ans_float, \
        (float) alpha, (float) beta \
    ); \
\
    free(a); \
    free(b); \
    free(c_ans); \
    free(c_orig); \
    free(a_float); \
    free(b_float); \
    free(c_ans_float); \
    free(c_orig_float); \
\
} 
