#if ( defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_ARCH_PPC) ) && defined(__APPLE__)

#include "blis.h"

#include "amx.h"
#include "amx_ext.h"
#include <assert.h>

// Const memory for zeroizing as
//  zeroizing instruction is not found yet.
const uint8_t amx_zeros[64] = { 0 };


BLIS_INLINE void bli_dgemmsup2_appleamx2_asm_16x32
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       double*    restrict alpha_vec,
       double*    restrict a, inc_t cs_a,
       double*    restrict b, inc_t rs_b,
       double*    restrict beta_vec,
       double*    restrict c, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx,
       double*    restrict a_p, int pack_a,
       double*    restrict b_p, int pack_b
     )
{
    const double *a_next = bli_auxinfo_next_a( data );
    const double *b_next = bli_auxinfo_next_b( data );

    if ( !k )
    {
        // Zeroize Z if no loop.
        AMX_MEM( LDX, amx_zeros, 0 );
        AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 0 );
        AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 1 );
        AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 2 );
        AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 3 );
        AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 4 );
        AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 5 );
        AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 6 );
        AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 7 );
    }
    else
    {
        // Do first loop w/ mul.
        AMX_MEM( LDX, a + 0, 0 );
        AMX_MEM( LDX, a + 8, 1 );

        AMX_MEM( LDY, b + 8*0, 0 );
        AMX_MEM( LDY, b + 8*1, 1 );
        AMX_MEM( LDY, b + 8*2, 2 );
        AMX_MEM( LDY, b + 8*3, 3 );

        AMX_FMUL64_COMMON_REGALIGNED( 0, 0, 0 );
        AMX_FMUL64_COMMON_REGALIGNED( 0, 1, 1 );
        AMX_FMUL64_COMMON_REGALIGNED( 0, 2, 2 );
        AMX_FMUL64_COMMON_REGALIGNED( 0, 3, 3 );

        if ( pack_a )
        {
            AMX_MEM( STX, a_p + 0, 0 );
            AMX_MEM( STX, a_p + 8, 1 );
        }
        if ( pack_b )
        {
            AMX_MEM( STY, b_p + 8*0, 0 );
            AMX_MEM( STY, b_p + 8*1, 1 );
            AMX_MEM( STY, b_p + 8*2, 2 );
            AMX_MEM( STY, b_p + 8*3, 3 );
        }

        AMX_FMUL64_COMMON_REGALIGNED( 1, 0, 4 );
        AMX_FMUL64_COMMON_REGALIGNED( 1, 1, 5 );
        AMX_FMUL64_COMMON_REGALIGNED( 1, 2, 6 );
        AMX_FMUL64_COMMON_REGALIGNED( 1, 3, 7 );

        a += cs_a;
        b += rs_b;
        a_p += 16;
        b_p += 32;
        k -= 1;
    }

    // TODO: Prefetch C. Or PRELOAD C??

#pragma nounroll
    for ( ; k >= 4; k -= 4 )
    {
        AMX_MEM( LDX, a + cs_a * 0 + 0, 0 ); // A column 0.
        AMX_MEM( LDX, a + cs_a * 0 + 8, 1 );
        AMX_MEM( LDX, a + cs_a * 1 + 0, 2 ); // A column 1.
        AMX_MEM( LDX, a + cs_a * 1 + 8, 3 );
        AMX_MEM( LDX, a + cs_a * 2 + 0, 4 ); // A column 2.
        AMX_MEM( LDX, a + cs_a * 2 + 8, 5 );
        AMX_MEM( LDX, a + cs_a * 3 + 0, 6 ); // A column 3.
        AMX_MEM( LDX, a + cs_a * 3 + 8, 7 );

        AMX_MEM( LDY, b + rs_b * 0 + 0, 0 ); // B row 0 first half.
        AMX_MEM( LDY, b + rs_b * 0 + 8, 1 );
        AMX_MEM( LDY, b + rs_b * 1 + 0, 2 ); // B row 1 first half.
        AMX_MEM( LDY, b + rs_b * 1 + 8, 3 );
        AMX_MEM( LDY, b + rs_b * 2 + 0, 4 ); // B row 2 first half.
        AMX_MEM( LDY, b + rs_b * 2 + 8, 5 );
        AMX_MEM( LDY, b + rs_b * 3 + 0, 6 ); // B row 3 first half.
        AMX_MEM( LDY, b + rs_b * 3 + 8, 7 );

        AMX_FMA64_COMMON_REGALIGNED( 0, 0, 0 ); // Block (0, 0)
        AMX_FMA64_COMMON_REGALIGNED( 0, 1, 1 ); // Block (0, 1)
        AMX_FMA64_COMMON_REGALIGNED( 1, 0, 4 ); // Block (1, 0)
        AMX_FMA64_COMMON_REGALIGNED( 1, 1, 5 ); // Block (1, 1)

        AMX_FMA64_COMMON_REGALIGNED( 2, 2, 0 );
        AMX_FMA64_COMMON_REGALIGNED( 2, 3, 1 );
        AMX_FMA64_COMMON_REGALIGNED( 3, 2, 4 );
        AMX_FMA64_COMMON_REGALIGNED( 3, 3, 5 );

        if ( pack_b )
        {
            AMX_MEM( STY, b_p + 32 * 0 + 0, 0 );
            AMX_MEM( STY, b_p + 32 * 0 + 8, 1 );
            AMX_MEM( STY, b_p + 32 * 1 + 0, 2 );
            AMX_MEM( STY, b_p + 32 * 1 + 8, 3 );
            AMX_MEM( STY, b_p + 32 * 2 + 0, 4 );
            AMX_MEM( STY, b_p + 32 * 2 + 8, 5 );
            AMX_MEM( STY, b_p + 32 * 3 + 0, 6 );
            AMX_MEM( STY, b_p + 32 * 3 + 8, 7 );
        }

        AMX_FMA64_COMMON_REGALIGNED( 4, 4, 0 );
        AMX_FMA64_COMMON_REGALIGNED( 4, 5, 1 );
        AMX_FMA64_COMMON_REGALIGNED( 5, 4, 4 );
        AMX_FMA64_COMMON_REGALIGNED( 5, 5, 5 );

        AMX_FMA64_COMMON_REGALIGNED( 6, 6, 0 );
        AMX_FMA64_COMMON_REGALIGNED( 6, 7, 1 );
        AMX_FMA64_COMMON_REGALIGNED( 7, 6, 4 );
        AMX_FMA64_COMMON_REGALIGNED( 7, 7, 5 );

        if ( pack_a )
        {
            AMX_MEM( STX, a_p + 16 * 0 + 0, 0 );
            AMX_MEM( STX, a_p + 16 * 0 + 8, 1 );
            AMX_MEM( STX, a_p + 16 * 1 + 0, 2 );
            AMX_MEM( STX, a_p + 16 * 1 + 8, 3 );
            AMX_MEM( STX, a_p + 16 * 2 + 0, 4 );
            AMX_MEM( STX, a_p + 16 * 2 + 8, 5 );
            AMX_MEM( STX, a_p + 16 * 3 + 0, 6 );
            AMX_MEM( STX, a_p + 16 * 3 + 8, 7 );
        }

        AMX_MEM( LDY, b + rs_b * 0 + 16, 0 ); // B row 0 last half.
        AMX_MEM( LDY, b + rs_b * 0 + 24, 1 );
        AMX_MEM( LDY, b + rs_b * 1 + 16, 2 ); // B row 1 last half.
        AMX_MEM( LDY, b + rs_b * 1 + 24, 3 );
        AMX_MEM( LDY, b + rs_b * 2 + 16, 4 ); // B row 2 last half.
        AMX_MEM( LDY, b + rs_b * 2 + 24, 5 );
        AMX_MEM( LDY, b + rs_b * 3 + 16, 6 ); // B row 3 last half.
        AMX_MEM( LDY, b + rs_b * 3 + 24, 7 );

        AMX_FMA64_COMMON_REGALIGNED( 0, 0, 2 ); // Block (0, 2)
        AMX_FMA64_COMMON_REGALIGNED( 0, 1, 3 ); // Block (0, 3)
        AMX_FMA64_COMMON_REGALIGNED( 1, 0, 6 ); // Block (1, 2)
        AMX_FMA64_COMMON_REGALIGNED( 1, 1, 7 ); // Block (1, 3)

        AMX_FMA64_COMMON_REGALIGNED( 2, 2, 2 );
        AMX_FMA64_COMMON_REGALIGNED( 2, 3, 3 );
        AMX_FMA64_COMMON_REGALIGNED( 3, 2, 6 );
        AMX_FMA64_COMMON_REGALIGNED( 3, 3, 7 );

        if ( pack_b )
        {
            AMX_MEM( STY, b_p + 32 * 0 + 16, 0 );
            AMX_MEM( STY, b_p + 32 * 0 + 24, 1 );
            AMX_MEM( STY, b_p + 32 * 1 + 16, 2 );
            AMX_MEM( STY, b_p + 32 * 1 + 24, 3 );
            AMX_MEM( STY, b_p + 32 * 2 + 16, 4 );
            AMX_MEM( STY, b_p + 32 * 2 + 24, 5 );
            AMX_MEM( STY, b_p + 32 * 3 + 16, 6 );
            AMX_MEM( STY, b_p + 32 * 3 + 24, 7 );
        }

        AMX_FMA64_COMMON_REGALIGNED( 4, 4, 2 );
        AMX_FMA64_COMMON_REGALIGNED( 4, 5, 3 );
        AMX_FMA64_COMMON_REGALIGNED( 5, 4, 6 );
        AMX_FMA64_COMMON_REGALIGNED( 5, 5, 7 );

        AMX_FMA64_COMMON_REGALIGNED( 6, 6, 2 );
        AMX_FMA64_COMMON_REGALIGNED( 6, 7, 3 );
        AMX_FMA64_COMMON_REGALIGNED( 7, 6, 6 );
        AMX_FMA64_COMMON_REGALIGNED( 7, 7, 7 );

        // Address forward.
        a += 4 * cs_a;
        b += 4 * rs_b;
        a_p += 4 * 16;
        b_p += 4 * 32;
    }

#pragma nounroll
    for ( ; k >= 1; k -= 1 )
    {
        AMX_MEM( LDX, a + 0, 0 );
        AMX_MEM( LDX, a + 8, 1 );

        AMX_MEM( LDY, b + 8*0, 0 );
        AMX_MEM( LDY, b + 8*1, 1 );
        AMX_MEM( LDY, b + 8*2, 2 );
        AMX_MEM( LDY, b + 8*3, 3 );

        AMX_FMA64_COMMON_REGALIGNED( 0, 0, 0 );
        AMX_FMA64_COMMON_REGALIGNED( 0, 1, 1 );
        AMX_FMA64_COMMON_REGALIGNED( 0, 2, 2 );
        AMX_FMA64_COMMON_REGALIGNED( 0, 3, 3 );

        if ( pack_a )
        {
            AMX_MEM( STX, a_p + 0, 0 );
            AMX_MEM( STX, a_p + 8, 1 );
        }
        if ( pack_b )
        {
            AMX_MEM( STY, b_p + 8*0, 0 );
            AMX_MEM( STY, b_p + 8*1, 1 );
            AMX_MEM( STY, b_p + 8*2, 2 );
            AMX_MEM( STY, b_p + 8*3, 3 );
        }

        AMX_FMA64_COMMON_REGALIGNED( 1, 0, 4 );
        AMX_FMA64_COMMON_REGALIGNED( 1, 1, 5 );
        AMX_FMA64_COMMON_REGALIGNED( 1, 2, 6 );
        AMX_FMA64_COMMON_REGALIGNED( 1, 3, 7 );

        a += cs_a;
        b += rs_b;
        a_p += 16;
        b_p += 32;
    }

    // Load alpha & beta.
    // Caller takes care duplicating.
    AMX_MEM( LDY, alpha_vec, 0 );
    AMX_MEM( LDY,  beta_vec, 1 );

    // Multiply by alpha.
    if ( alpha_vec[0] != 1.0 )
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

    bool tot_block = m == 16 && n == 32 ;

    // Load and multiply by beta.
    // Write into Z registers.
    if ( tot_block && beta_vec[0] != 0.0 )
    {
        double *c_ldr = c;
        if ( rs_c != 1 )
        {
            // Reload beta into X.
            AMX_MEM( LDX, beta_vec, 1 );

            // Load blocks (0, 0), (0, 1), (0, 2) and (0, 3).
            for (int i = 0; i < 8; ++i) {
                AMX_MEM( LDY, c_ldr + i * rs_c + 0 , 4 ); // TODO: Use 0-3?
                AMX_MEM( LDY, c_ldr + i * rs_c + 8 , 5 );
                AMX_MEM( LDY, c_ldr + i * rs_c + 16, 6 );
                AMX_MEM( LDY, c_ldr + i * rs_c + 24, 7 );

                AMX_FMA64_SELROW_REGALIGNED( i, 1, 4, 0 );
                AMX_FMA64_SELROW_REGALIGNED( i, 1, 5, 1 );
                AMX_FMA64_SELROW_REGALIGNED( i, 1, 6, 2 );
                AMX_FMA64_SELROW_REGALIGNED( i, 1, 7, 3 );
            }
            c_ldr += 8 * rs_c;

            // Load blocks (1, 0), (1, 1), (1, 2) and (1, 3).
            for (int i = 0; i < 8; ++i) {
                AMX_MEM( LDY, c_ldr + i * rs_c + 0 , 4 );
                AMX_MEM( LDY, c_ldr + i * rs_c + 8 , 5 );
                AMX_MEM( LDY, c_ldr + i * rs_c + 16, 6 );
                AMX_MEM( LDY, c_ldr + i * rs_c + 24, 7 );

                AMX_FMA64_SELROW_REGALIGNED( i, 1, 4, 4 );
                AMX_FMA64_SELROW_REGALIGNED( i, 1, 5, 5 );
                AMX_FMA64_SELROW_REGALIGNED( i, 1, 6, 6 );
                AMX_FMA64_SELROW_REGALIGNED( i, 1, 7, 7 );
            }
        }
        else
        {
            // Load blocks (0, 0) and (0, 1).
            for ( int i = 0; i < 8; ++i ) {
                AMX_MEM( LDX, c_ldr + i * cs_c + 0, 4 );
                AMX_MEM( LDX, c_ldr + i * cs_c + 8, 5 );

                AMX_FMA64_SELCOL_REGALIGNED( i, 4, 1, 0 );
                AMX_FMA64_SELCOL_REGALIGNED( i, 5, 1, 4 );
            }
            c_ldr += 8 * cs_c;

            // Load blocks (1, 0) and (1, 1).
            for ( int i = 0; i < 8; ++i ) {
                AMX_MEM( LDX, c_ldr + i * cs_c + 0, 4 );
                AMX_MEM( LDX, c_ldr + i * cs_c + 8, 5 );

                AMX_FMA64_SELCOL_REGALIGNED( i, 4, 1, 1 );
                AMX_FMA64_SELCOL_REGALIGNED( i, 5, 1, 5 );
            }
            c_ldr += 8 * cs_c;

            // Load blocks (2, 0) and (2, 1).
            for ( int i = 0; i < 8; ++i ) {
                AMX_MEM( LDX, c_ldr + i * cs_c + 0, 4 );
                AMX_MEM( LDX, c_ldr + i * cs_c + 8, 5 );

                AMX_FMA64_SELCOL_REGALIGNED( i, 4, 1, 2 );
                AMX_FMA64_SELCOL_REGALIGNED( i, 5, 1, 6 );
            }
            c_ldr += 8 * cs_c;

            // Load blocks (3, 0) and (3, 1).
            for ( int i = 0; i < 8; ++i ) {
                AMX_MEM( LDX, c_ldr + i * cs_c + 0, 4 );
                AMX_MEM( LDX, c_ldr + i * cs_c + 8, 5 );

                AMX_FMA64_SELCOL_REGALIGNED( i, 4, 1, 3 );
                AMX_FMA64_SELCOL_REGALIGNED( i, 5, 1, 7 );
            }
        }
    }

    static double c_t[ 16 * 32 ];
    double * c_s = tot_block ? c : c_t;
    inc_t rs_c_s = tot_block ? rs_c : 1;
    inc_t cs_c_s = tot_block ? cs_c : 16;

    if ( rs_c_s != 1 )
    {
        for ( int i = 0; i < 8; ++i ) {
            // (0, 0), (0, 1), (0, 2) and (0, 3).
            AMX_EXTRY64_REGALIGNED( i * 8 + 0, 0 );
            AMX_EXTRY64_REGALIGNED( i * 8 + 1, 1 );
            AMX_EXTRY64_REGALIGNED( i * 8 + 2, 2 );
            AMX_EXTRY64_REGALIGNED( i * 8 + 3, 3 );
            // (1, 0), (1, 1), (1, 2) and (1, 3).
            AMX_EXTRY64_REGALIGNED( i * 8 + 4, 4 );
            AMX_EXTRY64_REGALIGNED( i * 8 + 5, 5 );
            AMX_EXTRY64_REGALIGNED( i * 8 + 6, 6 );
            AMX_EXTRY64_REGALIGNED( i * 8 + 7, 7 );

            AMX_MEM( STY, c_s + (i + 0) * rs_c_s + 0 , 0 );
            AMX_MEM( STY, c_s + (i + 0) * rs_c_s + 8 , 1 );
            AMX_MEM( STY, c_s + (i + 0) * rs_c_s + 16, 2 );
            AMX_MEM( STY, c_s + (i + 0) * rs_c_s + 24, 3 );
            AMX_MEM( STY, c_s + (i + 8) * rs_c_s + 0 , 4 );
            AMX_MEM( STY, c_s + (i + 8) * rs_c_s + 8 , 5 );
            AMX_MEM( STY, c_s + (i + 8) * rs_c_s + 16, 6 );
            AMX_MEM( STY, c_s + (i + 8) * rs_c_s + 24, 7 );
        }
    }
    else
    {
        // Store blocks (0, 0) and (1, 0)
        for (int i = 0; i < 8; ++i) {
            AMX_MEM( STZ, c_s + i * cs_c_s + 0, i * 8 + 0 );
            AMX_MEM( STZ, c_s + i * cs_c_s + 8, i * 8 + 4 );
        }
        c_s += 8 * cs_c_s;

        // Store blocks (0, 1) and (1, 1)
        for (int i = 0; i < 8; ++i) {
            AMX_MEM( STZ, c_s + i * cs_c_s + 0, i * 8 + 1 );
            AMX_MEM( STZ, c_s + i * cs_c_s + 8, i * 8 + 5 );
        }
        c_s += 8 * cs_c_s;

        // Store blocks (0, 2) and (1, 2)
        for (int i = 0; i < 8; ++i) {
            AMX_MEM( STZ, c_s + i * cs_c_s + 0, i * 8 + 2 );
            AMX_MEM( STZ, c_s + i * cs_c_s + 8, i * 8 + 6 );
        }
        c_s += 8 * cs_c_s;

        // Store blocks (0, 3) and (1, 3)
        for (int i = 0; i < 8; ++i) {
            AMX_MEM( STZ, c_s + i * cs_c_s + 0, i * 8 + 3 );
            AMX_MEM( STZ, c_s + i * cs_c_s + 8, i * 8 + 7 );
        }
        c_s -= 24 * cs_c_s;
    }

    if ( c_s != c )
        for ( int i = 0; i < m; ++i )
            for ( int j = 0; j < n; ++j )
                c[ i * rs_c + j * cs_c ] = c[ i * rs_c + j * cs_c ] * beta_vec[0] +
                    c_t[ i + j * 16 ];
}


void bli_dgemmsup2_appleamx2_asm_16x32m
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a, inc_t cs_a,
       double*    restrict b, inc_t rs_b, inc_t cs_b,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx,
       double*    restrict a_p, int pack_a,
       double*    restrict b_p, int pack_b
     )
{
    const double *a_next = bli_auxinfo_next_a( data );
    const double *b_next = bli_auxinfo_next_b( data );
    inc_t ps_a_p    = bls_aux_ps_ext_p   ( data );
    inc_t ps_a      = bls_aux_ps_ext     ( data );
    inc_t cs_a_next = bls_aux_ls_ext_next( data );

    static double alpha_vec[8], beta_vec[8];
    for ( int i = 0; i < 8; ++i )
    {
        alpha_vec[i] = alpha[0];
        beta_vec[i] = beta[0];
    }
    
#ifdef DEBUG
    assert( rs_a == 1 && cs_b == 1 );
    assert( rs_c == 1 || cs_c == 1 );
#endif

    if ( pack_b && m >= 16 )
    {
        if ( m > 16 )
        {
            bli_auxinfo_set_next_a( a + ps_a, data );
            bli_auxinfo_set_next_b( b_p, data );
        }
        else
        {
            bli_auxinfo_set_next_a( a_next, data );
            bli_auxinfo_set_next_b( b_next, data );
        }

        bli_dgemmsup2_appleamx2_asm_16x32
            ( 16, n, k, alpha_vec,
              a, cs_a,
              b, rs_b, beta_vec,
              c, rs_c, cs_c,
              data, cntx,
              a_p, pack_a, b_p, 1 );

        m -= 16;
        a += ps_a;
        a_p += ps_a_p;
        c += 16 * rs_c;
        b = b_p;
        rs_b = 32;
    }

    for ( ; m > 0; m -= 16 )
    {
        dim_t m_loc = bli_min( m, 16 );

        if ( m > 16 )
        {
            bli_auxinfo_set_next_a( a + ps_a, data );
            bli_auxinfo_set_next_b( b_p, data );
        }
        else
        {
            bli_auxinfo_set_next_a( a_next, data );
            bli_auxinfo_set_next_b( b_next, data );
        }

        bli_dgemmsup2_appleamx2_asm_16x32
            ( m_loc, n, k, alpha_vec,
              a, cs_a,
              b, rs_b, beta_vec,
              c, rs_c, cs_c,
              data, cntx,
              a_p, pack_a, b_p, 0 );
        a += ps_a;
        a_p += ps_a_p;
        c += 16 * rs_c;
    }
}

#endif
