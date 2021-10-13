/* 

   Kernel for the Apple matrix coprocessor.

*/

#include "blis.h"
#include "amx.h"
#include "amx_ext.h"
#include <stdlib.h>
#include <assert.h>

// Const memory for zeroizing.
extern const uint8_t amx_zeros[64];


void bli_dgemm_aaplmx_mac_32x16
     (
       dim_t               k,
       double*    restrict alpha,
       double*    restrict a,
       double*    restrict b,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    void* a_next;
    void* b_next;
    if ( beta )
    {
        a_next = bli_auxinfo_next_a( data );
        b_next = bli_auxinfo_next_b( data );
    }

    // As current the RE work has not discovered any
    //  broadcasting-load instruction yet, use this
    //  halfway solution for alpha & beta.
    static double alphac[8] = { 0 };
    static double beta_c[8] = { 0 };

    // Transposed kernel.
    // Not implemented yet.
    /*
    if ( cs_c == 1 && rs_c != 1 )
        return bli_dgemm_aaplmx_mac_16x32
            (
              k, alpha, b, a,
              beta, c, cs_c, rs_c,
              data, cntx
            );
    */

    // TODO: Support generic strided storage.
    assert( rs_c == 1 || cs_c == 1 );

    // Duplicate alpha & beta.
    if ( alphac[0] != *alpha )
        for ( int i = 0; i < 8; ++i )
            alphac[i] = *alpha;

    if ( beta_c[0] != *beta )
        for ( int i = 0; i < 8; ++i )
            beta_c[i] = *beta;

    // As the BLIS API has no self-finalization,
    //  AMX_START and AMX_STOP has to be included within
    //  kernels. They do not seem to take much time,
    //  either.
    AMX_START();

    // Zeroize Z.
    AMX_MEM( LDX, amx_zeros, 0 );
    AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 0 );
    AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 1 );
    AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 2 );
    AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 3 );
    AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 4 );
    AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 5 );
    AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 6 );
    AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 7 );

#pragma nounroll
    for ( ; k >= 4; k -= 4 )
    {
        AMX_MEM( LDY, b + 16 * 0 + 0, 0 ); // B row 0.
        AMX_MEM( LDY, b + 16 * 0 + 8, 1 );
        AMX_MEM( LDY, b + 16 * 1 + 0, 2 ); // B row 1.
        AMX_MEM( LDY, b + 16 * 1 + 8, 3 );
        AMX_MEM( LDY, b + 16 * 2 + 0, 4 ); // B row 2.
        AMX_MEM( LDY, b + 16 * 2 + 8, 5 );
        AMX_MEM( LDY, b + 16 * 3 + 0, 6 ); // B row 3.
        AMX_MEM( LDY, b + 16 * 3 + 8, 7 );

        AMX_MEM( LDX, a + 32 * 0 + 0, 0 ); // A first half column 0.
        AMX_MEM( LDX, a + 32 * 0 + 8, 1 );
        AMX_MEM( LDX, a + 32 * 1 + 0, 2 ); // A first half column 1.
        AMX_MEM( LDX, a + 32 * 1 + 8, 3 );
        AMX_MEM( LDX, a + 32 * 2 + 0, 4 ); // A first half column 2.
        AMX_MEM( LDX, a + 32 * 2 + 8, 5 );
        AMX_MEM( LDX, a + 32 * 3 + 0, 6 ); // A first half column 3.
        AMX_MEM( LDX, a + 32 * 3 + 8, 7 );

        AMX_FMA64_COMMON_REGALIGNED( 0, 0, 0 ); // Block (0, 0)
        AMX_FMA64_COMMON_REGALIGNED( 1, 0, 1 ); // Block (1, 0)
        AMX_FMA64_COMMON_REGALIGNED( 0, 1, 4 ); // Block (0, 1)
        AMX_FMA64_COMMON_REGALIGNED( 1, 1, 5 ); // Block (1, 1)

        AMX_FMA64_COMMON_REGALIGNED( 2, 2, 0 );
        AMX_FMA64_COMMON_REGALIGNED( 3, 2, 1 );
        AMX_FMA64_COMMON_REGALIGNED( 2, 3, 4 );
        AMX_FMA64_COMMON_REGALIGNED( 3, 3, 5 );

        AMX_FMA64_COMMON_REGALIGNED( 4, 4, 0 );
        AMX_FMA64_COMMON_REGALIGNED( 5, 4, 1 );
        AMX_FMA64_COMMON_REGALIGNED( 4, 5, 4 );
        AMX_FMA64_COMMON_REGALIGNED( 5, 5, 5 );

        AMX_FMA64_COMMON_REGALIGNED( 6, 6, 0 );
        AMX_FMA64_COMMON_REGALIGNED( 7, 6, 1 );
        AMX_FMA64_COMMON_REGALIGNED( 6, 7, 4 );
        AMX_FMA64_COMMON_REGALIGNED( 7, 7, 5 );

        AMX_MEM( LDX, a + 32 * 0 + 16, 0 ); // A last half column 0.
        AMX_MEM( LDX, a + 32 * 0 + 24, 1 );
        AMX_MEM( LDX, a + 32 * 1 + 16, 2 ); // A last half column 1.
        AMX_MEM( LDX, a + 32 * 1 + 24, 3 );
        AMX_MEM( LDX, a + 32 * 2 + 16, 4 ); // A last half column 2.
        AMX_MEM( LDX, a + 32 * 2 + 24, 5 );
        AMX_MEM( LDX, a + 32 * 3 + 16, 6 ); // A last half column 3.
        AMX_MEM( LDX, a + 32 * 3 + 24, 7 );

        AMX_FMA64_COMMON_REGALIGNED( 0, 0, 2 ); // Block (2, 0)
        AMX_FMA64_COMMON_REGALIGNED( 1, 0, 3 ); // Block (3, 0)
        AMX_FMA64_COMMON_REGALIGNED( 0, 1, 6 ); // Block (2, 1)
        AMX_FMA64_COMMON_REGALIGNED( 1, 1, 7 ); // Block (3, 1)

        AMX_FMA64_COMMON_REGALIGNED( 2, 2, 2 );
        AMX_FMA64_COMMON_REGALIGNED( 3, 2, 3 );
        AMX_FMA64_COMMON_REGALIGNED( 2, 3, 6 );
        AMX_FMA64_COMMON_REGALIGNED( 3, 3, 7 );

        AMX_FMA64_COMMON_REGALIGNED( 4, 4, 2 );
        AMX_FMA64_COMMON_REGALIGNED( 5, 4, 3 );
        AMX_FMA64_COMMON_REGALIGNED( 4, 5, 6 );
        AMX_FMA64_COMMON_REGALIGNED( 5, 5, 7 );

        AMX_FMA64_COMMON_REGALIGNED( 6, 6, 2 );
        AMX_FMA64_COMMON_REGALIGNED( 7, 6, 3 );
        AMX_FMA64_COMMON_REGALIGNED( 6, 7, 6 );
        AMX_FMA64_COMMON_REGALIGNED( 7, 7, 7 );

        // Address forward.
        a += 4 * 32;
        b += 4 * 16;
    }
#pragma nounroll
    for ( ; k >= 1; k -= 1 )
    {
        AMX_MEM( LDY, b + 0, 0 );
        AMX_MEM( LDY, b + 8, 1 );

        AMX_MEM( LDX, a + 0 , 0 );
        AMX_MEM( LDX, a + 8 , 1 );
        AMX_MEM( LDX, a + 16, 2 );
        AMX_MEM( LDX, a + 24, 3 );

        AMX_FMA64_COMMON_REGALIGNED( 0, 0, 0 );
        AMX_FMA64_COMMON_REGALIGNED( 1, 0, 1 );
        AMX_FMA64_COMMON_REGALIGNED( 2, 0, 2 );
        AMX_FMA64_COMMON_REGALIGNED( 3, 0, 3 );
        AMX_FMA64_COMMON_REGALIGNED( 0, 1, 4 );
        AMX_FMA64_COMMON_REGALIGNED( 1, 1, 5 );
        AMX_FMA64_COMMON_REGALIGNED( 2, 1, 6 );
        AMX_FMA64_COMMON_REGALIGNED( 3, 1, 7 );

        a += 32;
        b += 16;
    }

    // Load alpha & beta.
    AMX_MEM( LDY, alphac, 0 );
    AMX_MEM( LDY, beta_c, 1 );

    // Multiply by alpha.
    if ( *alpha != 1.0 )
        for ( int i = 0; i < 8; ++i ) {
            AMX_EXTRX_REGALIGNED( i * 8 + 0, 4 );
            AMX_EXTRX_REGALIGNED( i * 8 + 1, 5 );
            AMX_EXTRX_REGALIGNED( i * 8 + 2, 6 );
            AMX_EXTRX_REGALIGNED( i * 8 + 3, 7 );

            AMX_FMUL64_SELCOL_REGALIGNED( i, 4, 0, 0 );
            AMX_FMUL64_SELCOL_REGALIGNED( i, 5, 0, 1 );
            AMX_FMUL64_SELCOL_REGALIGNED( i, 6, 0, 2 );
            AMX_FMUL64_SELCOL_REGALIGNED( i, 7, 0, 3 );

            AMX_EXTRX_REGALIGNED( i * 8 + 4, 4 );
            AMX_EXTRX_REGALIGNED( i * 8 + 5, 5 );
            AMX_EXTRX_REGALIGNED( i * 8 + 6, 6 );
            AMX_EXTRX_REGALIGNED( i * 8 + 7, 7 );

            AMX_FMUL64_SELCOL_REGALIGNED( i, 4, 0, 4 );
            AMX_FMUL64_SELCOL_REGALIGNED( i, 5, 0, 5 );
            AMX_FMUL64_SELCOL_REGALIGNED( i, 6, 0, 6 );
            AMX_FMUL64_SELCOL_REGALIGNED( i, 7, 0, 7 );
        }

    if ( data )
    {
        __asm__ volatile
        (
          "prfm PLDL2STRM, [%[a_next], 64*0] \n\t"
          "prfm PLDL2STRM, [%[a_next], 64*1] \n\t"
          "prfm PLDL2STRM, [%[a_next], 64*2] \n\t"
          "prfm PLDL2STRM, [%[a_next], 64*3] \n\t"
          "prfm PLDL2STRM, [%[b_next], 64*0] \n\t"
          "prfm PLDL2STRM, [%[b_next], 64*1] \n\t"
          "prfm PLDL2STRM, [%[b_next], 64*2] \n\t"
          "prfm PLDL2STRM, [%[b_next], 64*3] \n\t"
          :
          : [a_next] "r" (a_next),
            [b_next] "r" (b_next)
        );
    }

    // Load and multiply by beta.
    // Write into Z registers.
    if ( *beta != 0.0 )
    {
        double *c_ldr = c;
        if ( rs_c == 1 )
        {
            // Load blocks (0, 0), (1, 0), (2, 0) and (3, 0).
            for (int i = 0; i < 8; ++i) {
                AMX_MEM( LDX, c_ldr + i * cs_c + 0 , 4 );
                AMX_MEM( LDX, c_ldr + i * cs_c + 8 , 5 );
                AMX_MEM( LDX, c_ldr + i * cs_c + 16, 6 );
                AMX_MEM( LDX, c_ldr + i * cs_c + 24, 7 );

                AMX_FMA64_SELCOL_REGALIGNED( i, 4, 1, 0 );
                AMX_FMA64_SELCOL_REGALIGNED( i, 5, 1, 1 );
                AMX_FMA64_SELCOL_REGALIGNED( i, 6, 1, 2 );
                AMX_FMA64_SELCOL_REGALIGNED( i, 7, 1, 3 );
            }
            c_ldr += 8 * cs_c;

            // Load blocks (0, 1), (1, 1), (2, 1) and (3, 1).
            for (int i = 0; i < 8; ++i) {
                AMX_MEM( LDX, c_ldr + i * cs_c + 0 , 4 );
                AMX_MEM( LDX, c_ldr + i * cs_c + 8 , 5 );
                AMX_MEM( LDX, c_ldr + i * cs_c + 16, 6 );
                AMX_MEM( LDX, c_ldr + i * cs_c + 24, 7 );

                AMX_FMA64_SELCOL_REGALIGNED( i, 4, 1, 4 );
                AMX_FMA64_SELCOL_REGALIGNED( i, 5, 1, 5 );
                AMX_FMA64_SELCOL_REGALIGNED( i, 6, 1, 6 );
                AMX_FMA64_SELCOL_REGALIGNED( i, 7, 1, 7 );
            }
        }
        else if ( cs_c == 1 )
        {
            // Reload beta into X.
            AMX_MEM( LDX, beta_c, 1 );

            // Load blocks (0, 0) and (0, 1).
            for ( int i = 0; i < 8; ++i ) {
                AMX_MEM( LDY, c_ldr + i * rs_c + 0, 4 );
                AMX_MEM( LDY, c_ldr + i * rs_c + 8, 5 );

                AMX_FMA64_SELROW_REGALIGNED( i, 1, 4, 0 );
                AMX_FMA64_SELROW_REGALIGNED( i, 1, 5, 4 );
            }
            c_ldr += 8 * rs_c;

            // Load blocks (1, 0) and (1, 1).
            for ( int i = 0; i < 8; ++i ) {
                AMX_MEM( LDY, c_ldr + i * rs_c + 0, 4 );
                AMX_MEM( LDY, c_ldr + i * rs_c + 8, 5 );

                AMX_FMA64_SELROW_REGALIGNED( i, 1, 4, 1 );
                AMX_FMA64_SELROW_REGALIGNED( i, 1, 5, 5 );
            }
            c_ldr += 8 * rs_c;

            // Load blocks (2, 0) and (2, 1).
            for ( int i = 0; i < 8; ++i ) {
                AMX_MEM( LDY, c_ldr + i * rs_c + 0, 4 );
                AMX_MEM( LDY, c_ldr + i * rs_c + 8, 5 );

                AMX_FMA64_SELROW_REGALIGNED( i, 1, 4, 2 );
                AMX_FMA64_SELROW_REGALIGNED( i, 1, 5, 6 );
            }
            c_ldr += 8 * rs_c;

            // Load blocks (3, 0) and (3, 1).
            for ( int i = 0; i < 8; ++i ) {
                AMX_MEM( LDY, c_ldr + i * rs_c + 0, 4 );
                AMX_MEM( LDY, c_ldr + i * rs_c + 8, 5 );

                AMX_FMA64_SELROW_REGALIGNED( i, 1, 4, 3 );
                AMX_FMA64_SELROW_REGALIGNED( i, 1, 5, 7 );
            }
        }
    }

    if ( rs_c == 1 )
    {
        // Store blocks (0, 0), (1, 0), (2, 0) and (3, 0).
        for (int i = 0; i < 8; ++i) {
            AMX_MEM( STZ, c + i * cs_c + 0 , i * 8 + 0 );
            AMX_MEM( STZ, c + i * cs_c + 8 , i * 8 + 1 );
            AMX_MEM( STZ, c + i * cs_c + 16, i * 8 + 2 );
            AMX_MEM( STZ, c + i * cs_c + 24, i * 8 + 3 );
        }
        c += 8 * cs_c;

        // Store blocks (0, 1), (1, 1), (2, 1) and (3, 1).
        for (int i = 0; i < 8; ++i) {
            AMX_MEM( STZ, c + i * cs_c + 0 , i * 8 + 4 );
            AMX_MEM( STZ, c + i * cs_c + 8 , i * 8 + 5 );
            AMX_MEM( STZ, c + i * cs_c + 16, i * 8 + 6 );
            AMX_MEM( STZ, c + i * cs_c + 24, i * 8 + 7 );
        }
    }
    else if ( cs_c == 1 )
    {
        for ( int i = 0; i < 8; ++i ) {
            // (0, 0) and (0, 1).
            AMX_EXTRY64_REGALIGNED( i * 8 + 0, 0 );
            AMX_EXTRY64_REGALIGNED( i * 8 + 4, 1 );
            // (1, 0) and (1, 1).
            AMX_EXTRY64_REGALIGNED( i * 8 + 1, 2 );
            AMX_EXTRY64_REGALIGNED( i * 8 + 5, 3 );
            // (2, 0) and (2, 1).
            AMX_EXTRY64_REGALIGNED( i * 8 + 2, 4 );
            AMX_EXTRY64_REGALIGNED( i * 8 + 6, 5 );
            // (3, 0) and (3, 1).
            AMX_EXTRY64_REGALIGNED( i * 8 + 3, 6 );
            AMX_EXTRY64_REGALIGNED( i * 8 + 7, 7 );

            AMX_MEM( STY, c + (i + 0 ) * rs_c + 0 , 0 );
            AMX_MEM( STY, c + (i + 0 ) * rs_c + 8 , 1 );
            AMX_MEM( STY, c + (i + 8 ) * rs_c + 0 , 2 );
            AMX_MEM( STY, c + (i + 8 ) * rs_c + 8 , 3 );
            AMX_MEM( STY, c + (i + 16) * rs_c + 0 , 4 );
            AMX_MEM( STY, c + (i + 16) * rs_c + 8 , 5 );
            AMX_MEM( STY, c + (i + 24) * rs_c + 0 , 6 );
            AMX_MEM( STY, c + (i + 24) * rs_c + 8 , 7 );
        }
    }

    AMX_STOP();
}

