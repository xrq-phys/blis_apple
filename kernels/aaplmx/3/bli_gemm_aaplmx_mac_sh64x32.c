/* 

   Kernel for the Apple matrix coprocessor.

*/

#include "blis.h"
#include "amx.h"
#include "amx_ext.h"
#include <arm_neon.h>
#include <stdlib.h>
#include <assert.h>

// Const memory for zeroizing.
extern const uint8_t amx_zeros[64];

#define KERNELNAME(Ch,FuncSuffix) bli_## Ch ##gemm_## FuncSuffix

#define PASTEPAC_16BIT(Ch,TypeName,OpMulAdd,OpMulAddSelCol,OpMul,OpMulSelCol) \
\
/* BLIs 16-bit(SH/I16) GEMM for Apple Matrix Coprocessor,
 *   implemented with MACros, of size 64 * 32. */ \
void KERNELNAME(Ch,aaplmx_mac_64x32) \
     ( \
       dim_t               k,     \
       TypeName*  restrict alpha, \
       TypeName*  restrict a,     \
       TypeName*  restrict b,     \
       TypeName*  restrict beta,  \
       TypeName*  restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data,  \
       cntx_t*    restrict cntx   \
     ) \
{ \
    void* a_next; \
    void* b_next; \
    if ( data ) \
    { \
        a_next = bli_auxinfo_next_a( data ); \
        b_next = bli_auxinfo_next_b( data ); \
    } \
\
    /* As current the RE work has not discovered any
     *  broadcasting-load instruction yet, use this
     *  halfway solution for alpha & beta. */ \
    static TypeName alphac[32] = { 0 }; \
    static TypeName beta_c[32] = { 0 }; \
\
    /* Call dual kernel.
     * if ( cs_c == 1 && rs_c != 1 )
     *     return bli_shgemm_aaplmx_mac_32x64
     *         (
     *           k, alpha, b, a,
     *           beta, c, cs_c, rs_c,
     *           data, cntx
     *         ); */ \
\
    /* TODO: Support generic strided storage. */ \
    assert( rs_c == 1 ); \
\
    /* Duplicate alpha & beta. */ \
    if ( alphac[0] != *alpha ) \
        for ( int i = 0; i < 32; ++i ) \
            alphac[i] = *alpha; \
\
    if ( beta_c[0] != *beta ) \
        for ( int i = 0; i < 32; ++i ) \
            beta_c[i] = *beta; \
\
    /* As the BLIS API has no self-finalization,
     *  AMX_START and AMX_STOP has to be included within
     *  kernels. They do not seem to take much time,
     *  either. */ \
    AMX_START(); \
\
    /* Zeroize Z.
     * AMX_START seems clearing already? */ \
    AMX_MEM( LDY, amx_zeros, 0 ); \
    OpMul( 0, 0, 0 ); \
    OpMul( 0, 0, 1 ); \
\
    _Pragma("nounroll") \
    for ( ; k >= 8; k -= 8 ) \
    { \
        AMX_MEM( LDX, a + 64 * 0 + 32 * 0, 0 ); /* A column 0. */ \
        AMX_MEM( LDX, a + 64 * 0 + 32 * 1, 1 ); \
        AMX_MEM( LDX, a + 64 * 1 + 32 * 0, 2 ); /* A column 1. */ \
        AMX_MEM( LDX, a + 64 * 1 + 32 * 1, 3 ); \
        AMX_MEM( LDX, a + 64 * 2 + 32 * 0, 4 ); /* A column 2. */ \
        AMX_MEM( LDX, a + 64 * 2 + 32 * 1, 5 ); \
        AMX_MEM( LDX, a + 64 * 3 + 32 * 0, 6 ); /* A column 3. */ \
        AMX_MEM( LDX, a + 64 * 3 + 32 * 1, 7 ); \
\
        AMX_MEM( LDY, b + 32 * 0, 0 ); /* B row 0. */ \
        AMX_MEM( LDY, b + 32 * 1, 1 ); /* B row 1. */ \
        AMX_MEM( LDY, b + 32 * 2, 2 ); /* B row 2. */ \
        AMX_MEM( LDY, b + 32 * 3, 3 ); /* B row 3. */ \
\
        OpMulAdd( 0, 0, 0 ); /* Block (0, 0) */ \
        OpMulAdd( 1, 0, 1 ); /* Block (1, 0) */ \
\
        OpMulAdd( 2, 1, 0 ); \
        OpMulAdd( 3, 1, 1 ); \
\
        OpMulAdd( 4, 2, 0 ); \
        OpMulAdd( 5, 2, 1 ); \
\
        OpMulAdd( 6, 3, 0 ); \
        OpMulAdd( 7, 3, 1 ); \
\
        AMX_MEM( LDX, a + 64 * 4 + 32 * 0, 0 ); \
        AMX_MEM( LDX, a + 64 * 4 + 32 * 1, 1 ); \
        AMX_MEM( LDX, a + 64 * 5 + 32 * 0, 2 ); \
        AMX_MEM( LDX, a + 64 * 5 + 32 * 1, 3 ); \
        AMX_MEM( LDX, a + 64 * 6 + 32 * 0, 4 ); \
        AMX_MEM( LDX, a + 64 * 6 + 32 * 1, 5 ); \
        AMX_MEM( LDX, a + 64 * 7 + 32 * 0, 6 ); \
        AMX_MEM( LDX, a + 64 * 7 + 32 * 1, 7 ); \
\
        AMX_MEM( LDY, b + 32 * 4, 4 ); \
        AMX_MEM( LDY, b + 32 * 5, 5 ); \
        AMX_MEM( LDY, b + 32 * 6, 6 ); \
        AMX_MEM( LDY, b + 32 * 7, 7 ); \
\
        OpMulAdd( 0, 4, 0 ); \
        OpMulAdd( 1, 4, 1 ); \
\
        OpMulAdd( 2, 5, 0 ); \
        OpMulAdd( 3, 5, 1 ); \
\
        OpMulAdd( 4, 6, 0 ); \
        OpMulAdd( 5, 6, 1 ); \
\
        OpMulAdd( 6, 7, 0 ); \
        OpMulAdd( 7, 7, 1 ); \
\
        /* Address forward. */ \
        a += 8 * 64; \
        b += 8 * 32; \
    } \
_Pragma("nounroll") \
    for ( ; k >= 1; k -= 1 ) \
    { \
        AMX_MEM( LDX, a + 0 , 0 ); \
        AMX_MEM( LDX, a + 32, 1 ); \
        AMX_MEM( LDY, b + 0 , 0 ); \
\
        OpMulAdd( 0, 0, 0 ); \
        OpMulAdd( 1, 0, 1 ); \
\
        a += 64; \
        b += 32; \
    } \
\
    /* Load alpha & beta. */ \
    AMX_MEM( LDY, alphac, 0 ); \
    AMX_MEM( LDY, beta_c, 1 ); \
\
    /* Multiply by alpha. */ \
    if ( *alpha != (TypeName) 1 ) \
        for ( int i = 0; i < 32; ++i ) { \
            AMX_EXTRX_REGALIGNED( i * 2 + 0, 2 ); \
            AMX_EXTRX_REGALIGNED( i * 2 + 1, 3 ); \
\
            OpMulSelCol( i, 2, 0, 0 ); \
            OpMulSelCol( i, 3, 0, 1 ); \
        } \
\
    if ( data ) \
    { \
        __asm__ volatile \
        ( \
          "prfm PLDL2STRM, [%[a_next], 64*0] \n\t" \
          "prfm PLDL2STRM, [%[a_next], 64*1] \n\t" \
          "prfm PLDL2STRM, [%[a_next], 64*2] \n\t" \
          "prfm PLDL2STRM, [%[a_next], 64*3] \n\t" \
          "prfm PLDL2STRM, [%[b_next], 64*0] \n\t" \
          "prfm PLDL2STRM, [%[b_next], 64*1] \n\t" \
          "prfm PLDL2STRM, [%[b_next], 64*2] \n\t" \
          "prfm PLDL2STRM, [%[b_next], 64*3] \n\t" \
          : \
          : [a_next] "r" (a_next), \
            [b_next] "r" (b_next)  \
        ); \
    } \
\
    /* Load and multiply by beta.
     * Write into Z registers. */ \
    if ( *beta != (TypeName) 0 ) \
    { \
        for (int i = 0; i < 32; ++i) { \
            AMX_MEM( LDX, c + i * cs_c + 0 , 2 ); \
            AMX_MEM( LDX, c + i * cs_c + 32, 3 ); \
\
            OpMulAddSelCol( i, 2, 1, 0 ); \
            OpMulAddSelCol( i, 3, 1, 1 ); \
        } \
    } \
\
    if ( rs_c == 1 ) \
    { \
        /* Store blocks (0, 0) and (1, 0). */ \
        for (int i = 0; i < 32; ++i) { \
            AMX_MEM( STZ, c + i * cs_c + 0 , i * 2 + 0 ); \
            AMX_MEM( STZ, c + i * cs_c + 32, i * 2 + 1 ); \
        } \
    } \
\
    AMX_STOP(); \
}

PASTEPAC_16BIT(
        sh,
        float16_t,
        AMX_FMA16_COMMON_REGALIGNED,
        AMX_FMA16_SELCOL_REGALIGNED,
        AMX_FMUL16_COMMON_REGALIGNED,
        AMX_FMUL16_SELCOL_REGALIGNED)

PASTEPAC_16BIT(
        i16,
        int16_t,
        AMX_MAC16_COMMON_REGALIGNED,
        AMX_MAC16_SELCOL_REGALIGNED,
        AMX_MUL16_COMMON_REGALIGNED,
        AMX_MUL16_SELCOL_REGALIGNED)

