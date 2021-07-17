/* 

   Kernel for the Apple matrix coprocessor.

*/

#include "blis.h"
#include "amx.h"
#include "amx_ext.h"
#include <stdlib.h>
#include <assert.h>

// BLIs Single-precision GEMM for Apple Matrix Coprocessor,
//   implemented with MACros, of size 32 * 32.
void bli_sgemm_aaplmx_mac_32x32
     (
       dim_t               k,
       float*     restrict alpha,
       float*     restrict a,
       float*     restrict b,
       float*     restrict beta,
       float*     restrict c, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    void* a_next = bli_auxinfo_next_a( data );
    void* b_next = bli_auxinfo_next_b( data );

    // Isotropic kernel.
    if ( cs_c == 1 && rs_c != 1 )
        return bli_sgemm_aaplmx_mac_32x32
            (
              k, alpha, b, a,
              beta, c, cs_c, rs_c,
              data, cntx
            );

    // TODO: Support generic strided storage.
    assert( rs_c == 1 );

    // TODO: Support Alpha.
    assert( *alpha == 1.0 );

    // TODO: Move this to context init.
    AMX_START();

    // TODO: Other ways to zeroize Z.
    const static float zeros[16] = { 0 };
    if ( *beta == 0.0 )
        for ( int i = 0; i < 16 * 4; ++i )
            AMX_MEM( LDZ, zeros, i );
    else if ( *beta == 1.0 )
    {
        float *c_ldr = c;
        if ( rs_c == 1 )
        {
            // Load blocks (0, 0) and (1, 0).
            for (int i = 0; i < 16; ++i) {
                AMX_MEM( LDZ, c_ldr + i * cs_c + 0 , i * 4 + 0 );
                AMX_MEM( LDZ, c_ldr + i * cs_c + 16, i * 4 + 1 );
            }
            c_ldr += 16 * cs_c;

            // Load blocks (0, 1) and (1, 1).
            for (int i = 0; i < 16; ++i) {
                AMX_MEM( LDZ, c_ldr + i * cs_c + 0 , i * 4 + 2 );
                AMX_MEM( LDZ, c_ldr + i * cs_c + 16, i * 4 + 3 );
            }
        }
    }
    else
        // TODO: Support Beta.
        assert( false );

#pragma nounroll
    for ( ; k >= 4; k -= 4 )
    {
        AMX_MEM( LDX, a + 32 * 0 + 0 , 0 ); // A column 0.
        AMX_MEM( LDX, a + 32 * 0 + 16, 1 );
        AMX_MEM( LDX, a + 32 * 1 + 0 , 2 ); // A column 1.
        AMX_MEM( LDX, a + 32 * 1 + 16, 3 );
        AMX_MEM( LDX, a + 32 * 2 + 0 , 4 ); // A column 2.
        AMX_MEM( LDX, a + 32 * 2 + 16, 5 );
        AMX_MEM( LDX, a + 32 * 3 + 0 , 6 ); // A column 3.
        AMX_MEM( LDX, a + 32 * 3 + 16, 7 );

        AMX_MEM( LDY, b + 32 * 0 + 0 , 0 ); // B row 0.
        AMX_MEM( LDY, b + 32 * 0 + 16, 1 );
        AMX_MEM( LDY, b + 32 * 1 + 0 , 2 ); // B row 1.
        AMX_MEM( LDY, b + 32 * 1 + 16, 3 );
        AMX_MEM( LDY, b + 32 * 2 + 0 , 4 ); // B row 2.
        AMX_MEM( LDY, b + 32 * 2 + 16, 5 );
        AMX_MEM( LDY, b + 32 * 3 + 0 , 6 ); // B row 3.
        AMX_MEM( LDY, b + 32 * 3 + 16, 7 );

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
        a += 4 * 32;
        b += 4 * 32;
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

        a += 32;
        b += 32;
    }

    __asm__ volatile
    (
      "prfm PLDL1STRM, [%[a_next], 64*0] \n\t"
      "prfm PLDL1STRM, [%[a_next], 64*1] \n\t"
      // "prfm PLDL1STRM, [%[a_next], 64*2] \n\t"
      // "prfm PLDL1STRM, [%[a_next], 64*3] \n\t"
      "prfm PLDL1STRM, [%[b_next], 64*0] \n\t"
      "prfm PLDL1STRM, [%[b_next], 64*1] \n\t"
      // "prfm PLDL1STRM, [%[b_next], 64*2] \n\t"
      // "prfm PLDL1STRM, [%[b_next], 64*3] \n\t"
      :
      : [a_next] "r" (a_next),
        [b_next] "r" (b_next)
    );

    if ( rs_c == 1 )
    {
        // Store blocks (0, 0) and (1, 0).
        for (int i = 0; i < 16; ++i) {
            AMX_MEM( STZ, c + i * cs_c + 0 , i * 4 + 0 );
            AMX_MEM( STZ, c + i * cs_c + 16, i * 4 + 1 );
        }
        c += 16 * cs_c;

        // Store blocks (0, 1) and (1, 1).
        for (int i = 0; i < 16; ++i) {
            AMX_MEM( STZ, c + i * cs_c + 0 , i * 4 + 2 );
            AMX_MEM( STZ, c + i * cs_c + 16, i * 4 + 3 );
        }
    }

    AMX_STOP();
}


