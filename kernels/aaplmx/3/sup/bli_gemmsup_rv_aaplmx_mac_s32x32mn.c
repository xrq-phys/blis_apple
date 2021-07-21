/*

   Kernel for the Apple matrix coprocessor.

*/

#include "blis.h"
#include "../amx.h"
#include "../amx_ext.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#ifndef MIN
#define MIN( a, b ) ( (a) > (b) ? (b) : (a) )
#endif

// Const memory for zeroizing.
extern const uint8_t amx_zeros[64];

// Prototype reference and supplementary microkernels.
GEMMSUP_KER_PROT( float, s, gemmsup_r_aaplmx_ref2 )
GEMMSUP_KER_PROT( float, s, gemmsup_ccr_aaplmx_mac_inner_32xn )


void bli_sgemmsup_rv_aaplmx_mac_32x32mn
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       float*     restrict alpha,
       float*     restrict a0, inc_t rs_a, inc_t cs_a,
       float*     restrict b0, inc_t rs_b, inc_t cs_b,
       float*     restrict beta,
       float*     restrict c0, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    void*  a_next = bli_auxinfo_next_a( data );
    void*  b_next = bli_auxinfo_next_b( data );

    inc_t  ps_a   = bli_auxinfo_ps_a( data );
    inc_t  ps_b   = bli_auxinfo_ps_b( data );

    dim_t  m      = m0;
    dim_t  n;
    dim_t  k;
    float* a_rows = a0;
    float* a;
    float* b_cols;
    float* b;
    float* c_rows = c0;
    float* c_cols;
    float* c;

    // Fix ignored panel strides for the short dims.
    if ( !ps_a || m0 <= 32 ) ps_a = 32;
    if ( !ps_b || n0 <= 32 ) ps_b = 32;

    // Isotropic kernel.
    if ( cs_c == 1 && rs_c != 1 )
    {
        auxinfo_t datat;
        bli_auxinfo_set_ps_a( ps_b, &datat );
        bli_auxinfo_set_ps_b( ps_a, &datat );
        bli_auxinfo_set_next_a( a_next, &datat );
        bli_auxinfo_set_next_b( b_next, &datat );

        return bli_sgemmsup_rv_aaplmx_mac_32x32mn
            (
              conjb, conja,
              n0, m0, k0, alpha,
              b0, cs_b, rs_b,
              a0, cs_a, rs_a, beta,
              c0, cs_c, rs_c,
              &datat, cntx
            );
    }

    // Edge-size microkernel. Safe or unsafe.
#ifdef GEMMSUP_ALLOW_UNSAFE_MEMORY_ACCESS
#warning "GEMMSUP is allowed to perform excess read (not write)."
    const sgemmsup_ker_ft edge_mker = bli_sgemmsup_ccr_aaplmx_mac_inner_32xn;
#else
    const sgemmsup_ker_ft edge_mker = bli_sgemmsup_r_aaplmx_ref2;
#endif

    // In-reg transpose is worse than millikernel transpose above.
    assert( rs_c == 1 );

    // ?CR case or ?RR case with A packed.
    assert( rs_a == 1 && cs_b == 1 );

    // As current the RE work has not discovered any
    //  broadcasting-load instruction yet, use this
    //  halfway solution for alpha & beta.
    static float alphac[16] = { 0 };
    static float beta_c[16] = { 0 };

    // Duplicate alpha & beta.
    if ( alphac[0] != *alpha )
        for ( int i = 0; i < 16; ++i )
            alphac[i] = *alpha;

    if ( beta_c[0] != *beta )
        for ( int i = 0; i < 16; ++i )
            beta_c[i] = *beta;

    // As the BLIS API has no self-finalization,
    //  AMX_START and AMX_STOP has to be included within
    //  kernels. They do not seem to take much time,
    //  either.
    AMX_START();

    // NOTE: Due to caller properties, only one of the (m, n)
    //  should be actually looping.
    for ( ; m >= 32; m -= 32 )
    {
        n = n0;
        c_cols = c_rows;
        b_cols = b0;

        for ( ; n >= 32; n -= 32 )
        {
            k = k0;
            a = a_rows;
            b = b_cols;
            c = c_cols;

            // Zeroize Z.
            AMX_MEM( LDY, amx_zeros, 0 );
            AMX_FMUL32_COMMON_REGALIGNED( 0, 0, 0 );
            AMX_FMUL32_COMMON_REGALIGNED( 0, 0, 1 );
            AMX_FMUL32_COMMON_REGALIGNED( 0, 0, 2 );
            AMX_FMUL32_COMMON_REGALIGNED( 0, 0, 3 );

            // Launch microkernel.
#pragma nounroll
            for ( ; k >= 4; k -= 4 )
            {
                AMX_MEM( LDX, a + cs_a * 0 + 0 , 0 ); // A column 0.
                AMX_MEM( LDX, a + cs_a * 0 + 16, 1 );
                AMX_MEM( LDX, a + cs_a * 1 + 0 , 2 ); // A column 1.
                AMX_MEM( LDX, a + cs_a * 1 + 16, 3 );
                AMX_MEM( LDX, a + cs_a * 2 + 0 , 4 ); // A column 2.
                AMX_MEM( LDX, a + cs_a * 2 + 16, 5 );
                AMX_MEM( LDX, a + cs_a * 3 + 0 , 6 ); // A column 3.
                AMX_MEM( LDX, a + cs_a * 3 + 16, 7 );

                AMX_MEM( LDY, b + rs_b * 0 + 0 , 0 ); // B row 0.
                AMX_MEM( LDY, b + rs_b * 0 + 16, 1 );
                AMX_MEM( LDY, b + rs_b * 1 + 0 , 2 ); // B row 1.
                AMX_MEM( LDY, b + rs_b * 1 + 16, 3 );
                AMX_MEM( LDY, b + rs_b * 2 + 0 , 4 ); // B row 2.
                AMX_MEM( LDY, b + rs_b * 2 + 16, 5 );
                AMX_MEM( LDY, b + rs_b * 3 + 0 , 6 ); // B row 3.
                AMX_MEM( LDY, b + rs_b * 3 + 16, 7 );

                AMX_FMA32_COMMON_REGALIGNED( 0, 0, 0 ); // Block (0, 0)
                AMX_FMA32_COMMON_REGALIGNED( 1, 0, 1 ); // Block (1, 0)
                AMX_FMA32_COMMON_REGALIGNED( 0, 1, 2 ); // Block (0, 1)
                AMX_FMA32_COMMON_REGALIGNED( 1, 1, 3 ); // Block (1, 1)

                AMX_FMA32_COMMON_REGALIGNED( 2, 2, 0 );
                AMX_FMA32_COMMON_REGALIGNED( 3, 2, 1 );
                AMX_FMA32_COMMON_REGALIGNED( 2, 3, 2 );
                AMX_FMA32_COMMON_REGALIGNED( 3, 3, 3 );

                AMX_FMA32_COMMON_REGALIGNED( 4, 4, 0 );
                AMX_FMA32_COMMON_REGALIGNED( 5, 4, 1 );
                AMX_FMA32_COMMON_REGALIGNED( 4, 5, 2 );
                AMX_FMA32_COMMON_REGALIGNED( 5, 5, 3 );

                AMX_FMA32_COMMON_REGALIGNED( 6, 6, 0 );
                AMX_FMA32_COMMON_REGALIGNED( 7, 6, 1 );
                AMX_FMA32_COMMON_REGALIGNED( 6, 7, 2 );
                AMX_FMA32_COMMON_REGALIGNED( 7, 7, 3 );

                // Address forward.
                a += 4 * cs_a;
                b += 4 * rs_b;
            }
#pragma nounroll
            for ( ; k >= 1; k -= 1 )
            {
                AMX_MEM( LDX, a + 0 , 0 );
                AMX_MEM( LDX, a + 16, 1 );

                AMX_MEM( LDY, b + 0 , 0 );
                AMX_MEM( LDY, b + 16, 1 );

                AMX_FMA32_COMMON_REGALIGNED( 0, 0, 0 );
                AMX_FMA32_COMMON_REGALIGNED( 1, 0, 1 );
                AMX_FMA32_COMMON_REGALIGNED( 0, 1, 2 );
                AMX_FMA32_COMMON_REGALIGNED( 1, 1, 3 );

                a += cs_a;
                b += rs_b;
            }
            // Load alpha & beta.
            AMX_MEM( LDY, alphac, 0 );
            AMX_MEM( LDY, beta_c, 1 );

            // Multiply by alpha.
            if ( *alpha != 1.0 )
                for ( int i = 0; i < 16; ++i ) {
                    AMX_EXTRX_REGALIGNED( i * 4 + 0, 4 );
                    AMX_EXTRX_REGALIGNED( i * 4 + 1, 5 );
                    AMX_EXTRX_REGALIGNED( i * 4 + 2, 6 );
                    AMX_EXTRX_REGALIGNED( i * 4 + 3, 7 );

                    AMX_FMUL32_SELCOL_REGALIGNED( i, 4, 0, 0 );
                    AMX_FMUL32_SELCOL_REGALIGNED( i, 5, 0, 1 );
                    AMX_FMUL32_SELCOL_REGALIGNED( i, 6, 0, 2 );
                    AMX_FMUL32_SELCOL_REGALIGNED( i, 7, 0, 3 );
                }

            // Load and multiply by beta.
            // Write into Z registers.
            if ( *beta != 0.0 )
            {
                float *c_ldr = c;
                if ( rs_c == 1 )
                {
                    // Load blocks (0, 0) and (1, 0).
                    for ( int i = 0; i < 16; ++i )
                    {
                        AMX_MEM( LDX, c_ldr + i * cs_c + 0 , 2 );
                        AMX_MEM( LDX, c_ldr + i * cs_c + 16, 3 );

                        AMX_FMA32_SELCOL_REGALIGNED( i, 2, 1, 0 );
                        AMX_FMA32_SELCOL_REGALIGNED( i, 3, 1, 1 );
                    }
                    c_ldr += 16 * cs_c;

                    // Load blocks (0, 1) and (1, 1).
                    for ( int i = 0; i < 16; ++i )
                    {
                        AMX_MEM( LDX, c_ldr + i * cs_c + 0 , 2 );
                        AMX_MEM( LDX, c_ldr + i * cs_c + 16, 3 );

                        AMX_FMA32_SELCOL_REGALIGNED( i, 2, 1, 2 );
                        AMX_FMA32_SELCOL_REGALIGNED( i, 3, 1, 3 );
                    }
                }
            }

            if ( rs_c == 1 )
            {
                // Store blocks (0, 0) and (1, 0).
                for ( int i = 0; i < 16; ++i )
                {
                    AMX_MEM( STZ, c + i * cs_c + 0 , i * 4 + 0 );
                    AMX_MEM( STZ, c + i * cs_c + 16, i * 4 + 1 );
                }
                c += 16 * cs_c;

                // Store blocks (0, 1) and (1, 1).
                for ( int i = 0; i < 16; ++i )
                {
                    AMX_MEM( STZ, c + i * cs_c + 0 , i * 4 + 2 );
                    AMX_MEM( STZ, c + i * cs_c + 16, i * 4 + 3 );
                }
            }

            b_cols += ps_b; // 32 * cs_b;
            c_cols += 32 * cs_c;
        }

        if ( n > 0 )
        {
            edge_mker
                (
                  conja, conjb,
                  32, n, k0, alpha,
                  a_rows, rs_a, cs_a,
                  b_cols, rs_b, cs_b, beta,
                  c_cols, rs_c, cs_c,
                  data, cntx
                );
        }

        a_rows += ps_a; // 32 * rs_a;
        c_rows += 32 * rs_c;
    }

    if ( m > 0 )
    {
        n = n0;
        b_cols = b0;
        c_cols = c_rows;

        for ( ; n >= 32; n -= 32 )
        {
            edge_mker
                (
                  conja, conjb,
                  m, 32, k0, alpha,
                  a_rows, rs_a, cs_a,
                  b_cols, rs_b, cs_b, beta,
                  c_cols, rs_c, cs_c,
                  data, cntx
                );

            b_cols += ps_b;
            c_cols += 32 * cs_c;
        }

        if ( n > 0 )
            edge_mker
                (
                  conja, conjb,
                  m, n, k0, alpha,
                  a_rows, rs_a, cs_a,
                  b_cols, rs_b, cs_b, beta,
                  c_cols, rs_c, cs_c,
                  data, cntx
                );
    }

    AMX_STOP();
}

void bli_sgemmsup_ccr_aaplmx_mac_inner_32xn
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m,
       dim_t               n,
       dim_t               k,
       float*     restrict alpha,
       float*     restrict a, inc_t rs_a, inc_t cs_a,
       float*     restrict b, inc_t rs_b, inc_t cs_b,
       float*     restrict beta,
       float*     restrict c, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    // This microkernel is supposed to be called between
    //  an AMX_START() and AMX_STOP() block.

    assert( rs_c == 1 && rs_a == 1 && cs_b == 1 );
    assert( m <= 32 && n <= 32 );

    // Scratchpad for safe storage.
    static float c_scr[256];

    static float alphac[16] = { 0 };
    static float beta_c[16] = { 0 };

    // Duplicate alpha & beta.
    if ( alphac[0] != *alpha )
        for ( int i = 0; i < 16; ++i )
            alphac[i] = *alpha;

    if ( beta_c[0] != *beta )
        for ( int i = 0; i < 16; ++i )
            beta_c[i] = *beta;

    // Zeroize Z.
    AMX_MEM( LDY, amx_zeros, 0 );
    AMX_FMUL32_COMMON_REGALIGNED( 0, 0, 0 );
    AMX_FMUL32_COMMON_REGALIGNED( 0, 0, 1 );
    AMX_FMUL32_COMMON_REGALIGNED( 0, 0, 2 );
    AMX_FMUL32_COMMON_REGALIGNED( 0, 0, 3 );

#pragma nounroll
    for ( ; k >= 4; k -= 4 )
    {
        AMX_MEM( LDX, a + cs_a * 0 + 0 , 0 ); // A column 0.
        AMX_MEM( LDX, a + cs_a * 1 + 0 , 2 ); // A column 1.
        AMX_MEM( LDX, a + cs_a * 2 + 0 , 4 ); // A column 2.
        AMX_MEM( LDX, a + cs_a * 3 + 0 , 6 ); // A column 3.
        if ( m > 16 )
        {
            AMX_MEM( LDX, a + cs_a * 0 + 16, 1 );
            AMX_MEM( LDX, a + cs_a * 1 + 16, 3 );
            AMX_MEM( LDX, a + cs_a * 2 + 16, 5 );
            AMX_MEM( LDX, a + cs_a * 3 + 16, 7 );
        }

        AMX_MEM( LDY, b + rs_b * 0 + 0 , 0 ); // B row 0.
        AMX_MEM( LDY, b + rs_b * 1 + 0 , 2 ); // B row 1.
        AMX_MEM( LDY, b + rs_b * 2 + 0 , 4 ); // B row 2.
        AMX_MEM( LDY, b + rs_b * 3 + 0 , 6 ); // B row 3.
        if ( n > 16 )
        {
            AMX_MEM( LDY, b + rs_b * 0 + 16, 1 );
            AMX_MEM( LDY, b + rs_b * 1 + 16, 3 );
            AMX_MEM( LDY, b + rs_b * 2 + 16, 5 );
            AMX_MEM( LDY, b + rs_b * 3 + 16, 7 );
        }

        AMX_FMA32_COMMON_REGALIGNED( 0, 0, 0 ); // Block (0, 0)
        AMX_FMA32_COMMON_REGALIGNED( 2, 2, 0 );
        AMX_FMA32_COMMON_REGALIGNED( 4, 4, 0 );
        AMX_FMA32_COMMON_REGALIGNED( 6, 6, 0 );
        if ( m > 16 )
        {
            AMX_FMA32_COMMON_REGALIGNED( 1, 0, 1 ); // Block (1, 0)
            AMX_FMA32_COMMON_REGALIGNED( 3, 2, 1 );
            AMX_FMA32_COMMON_REGALIGNED( 5, 4, 1 );
            AMX_FMA32_COMMON_REGALIGNED( 7, 6, 1 );
        }

        if ( n > 16 )
        {
            AMX_FMA32_COMMON_REGALIGNED( 0, 1, 2 ); // Block (0, 1)
            AMX_FMA32_COMMON_REGALIGNED( 2, 3, 2 );
            AMX_FMA32_COMMON_REGALIGNED( 4, 5, 2 );
            AMX_FMA32_COMMON_REGALIGNED( 6, 7, 2 );
            if ( m > 16 )
            {
                AMX_FMA32_COMMON_REGALIGNED( 1, 1, 3 ); // Block (1, 1)
                AMX_FMA32_COMMON_REGALIGNED( 3, 3, 3 );
                AMX_FMA32_COMMON_REGALIGNED( 5, 5, 3 );
                AMX_FMA32_COMMON_REGALIGNED( 7, 7, 3 );
            }
        }

        // Address forward.
        a += 4 * cs_a;
        b += 4 * rs_b;
    }
#pragma nounroll
    for ( ; k >= 1; k -= 1 )
    {
        AMX_MEM( LDX, a + 0 , 0 );
        if ( m > 16 )
            AMX_MEM( LDX, a + 16, 1 );

        AMX_MEM( LDY, b + 0 , 0 );
        if ( n > 16 )
            AMX_MEM( LDY, b + 16, 1 );

        AMX_FMA32_COMMON_REGALIGNED( 0, 0, 0 );
        if ( m > 16 )
            AMX_FMA32_COMMON_REGALIGNED( 1, 0, 1 );
        if ( n > 16 )
        {
            AMX_FMA32_COMMON_REGALIGNED( 0, 1, 2 );
            if ( m > 16 )
                AMX_FMA32_COMMON_REGALIGNED( 1, 1, 3 );
        }

        a += cs_a;
        b += rs_b;
    }
    // Load alpha & beta.
    AMX_MEM( LDY, alphac, 0 );
    AMX_MEM( LDY, beta_c, 1 );

    // Multiply by alpha.
    if ( *alpha != 1.0 )
        for ( int i = 0; i < 16; ++i ) {
            AMX_EXTRX_REGALIGNED( i * 4 + 0, 4 );
            AMX_EXTRX_REGALIGNED( i * 4 + 1, 5 );
            AMX_EXTRX_REGALIGNED( i * 4 + 2, 6 );
            AMX_EXTRX_REGALIGNED( i * 4 + 3, 7 );

            AMX_FMUL32_SELCOL_REGALIGNED( i, 4, 0, 0 );
            AMX_FMUL32_SELCOL_REGALIGNED( i, 5, 0, 1 );
            AMX_FMUL32_SELCOL_REGALIGNED( i, 6, 0, 2 );
            AMX_FMUL32_SELCOL_REGALIGNED( i, 7, 0, 3 );
        }

    // Load and multiply by beta.
    // Write into Z registers.
    if ( *beta != 0.0 )
    {
        float *c_ldr = c;
        if ( rs_c == 1 )
        {
            // Load blocks (0, 0) and (1, 0).
            for ( int i = 0; i < MIN( n, 16 ); ++i )
            {
                AMX_MEM( LDX, c_ldr + i * cs_c + 0 , 2 );
                AMX_MEM( LDX, c_ldr + i * cs_c + 16, 3 );

                AMX_FMA32_SELCOL_REGALIGNED( i, 2, 1, 0 );
                AMX_FMA32_SELCOL_REGALIGNED( i, 3, 1, 1 );
            }
            c_ldr += 16 * cs_c;

            // Load blocks (0, 1) and (1, 1).
            for ( int i = 0; i + 16 < n; ++i )
            {
                AMX_MEM( LDX, c_ldr + i * cs_c + 0 , 2 );
                AMX_MEM( LDX, c_ldr + i * cs_c + 16, 3 );

                AMX_FMA32_SELCOL_REGALIGNED( i, 2, 1, 2 );
                AMX_FMA32_SELCOL_REGALIGNED( i, 3, 1, 3 );
            }
        }
    }

    if ( rs_c == 1 )
    {
        // Store blocks (0, 0) and (1, 0).
        for ( int i = 0; i < MIN( n, 16 ); ++i )
        {
            if ( m >= 16 )
                AMX_MEM( STZ, c + i * cs_c + 0, i * 4 + 0 );
            else
            {
                AMX_MEM( STZ, c_scr + i * 16, i * 4 + 0 );
                memcpy( c + i * cs_c + 0, c_scr + i * 16, m * sizeof(float) );
                continue;
            }

            if ( m == 32 )
                AMX_MEM( STZ, c + i * cs_c + 16, i * 4 + 1 );
            else
            {
                AMX_MEM( STZ, c_scr + i * 16, i * 4 + 1 );
                memcpy( c + i * cs_c + 16, c_scr + i * 16, (m - 16) * sizeof(float) );
            }
        }
        c += 16 * cs_c;

        // Store blocks (0, 1) and (1, 1).
        for ( int i = 0; i + 16 < n; ++i )
        {
            if ( m >= 16 )
                AMX_MEM( STZ, c + i * cs_c + 0, i * 4 + 2 );
            else
            {
                AMX_MEM( STZ, c_scr + i * 16, i * 4 + 2 );
                memcpy( c + i * cs_c + 0, c_scr + i * 16, m * sizeof(float) );
                continue;
            }

            if ( m == 32 )
                AMX_MEM( STZ, c + i * cs_c + 16, i * 4 + 3 );
            else
            {
                AMX_MEM( STZ, c_scr + i * 16, i * 4 + 3 );
                memcpy( c + i * cs_c + 16, c_scr + i * 16, (m - 16) * sizeof(float) );
            }
        }
    }
}

