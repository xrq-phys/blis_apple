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

#define PASTEPAC_16BIT_TO_32BIT(Ch,TypeIn,TypeOut,OpMulAdd,OpMulAddSelCol,OpMul,OpMulSelCol) \
\
/* BLIs 16-bit(SH/I16) to 32-bit(S/INT) GEMM for Apple Matrix Coprocessor,
 *   implemented with MACros, of size 32 * 32. */ \
void KERNELNAME(Ch,aaplmx_mac_32x32) \
     ( \
       dim_t               k,     \
       TypeOut*   restrict alpha, \
       TypeIn*    restrict a,     \
       TypeIn*    restrict b,     \
       TypeOut*   restrict beta,  \
       TypeOut*   restrict c, inc_t rs_c, inc_t cs_c, \
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
    static TypeOut alphac[16] = { 0 }; \
    static TypeOut beta_c[16] = { 0 }; \
\
    /* Isotropic kernel. */ \
    if ( cs_c == 1 && rs_c != 1 ) \
        return KERNELNAME(Ch,aaplmx_mac_32x32) \
            ( \
              k, alpha, b, a, \
              beta, c, cs_c, rs_c, \
              data, cntx \
            ); \
\
    /* TODO: Support generic strided storage. */ \
    assert( rs_c == 1 ); \
\
    /* Duplicate alpha & beta. */ \
    if ( alphac[0] != *alpha ) \
        for ( int i = 0; i < 16; ++i ) \
            alphac[i] = *alpha; \
\
    if ( beta_c[0] != *beta ) \
        for ( int i = 0; i < 16; ++i ) \
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
\
    _Pragma("nounroll") \
    for ( ; k >= 8; k -= 8 ) \
    { \
        AMX_MEM( LDX, a + 32 * 0, 0 ); /* A column 0. */ \
        AMX_MEM( LDX, a + 32 * 1, 1 ); /* A column 1. */ \
        AMX_MEM( LDX, a + 32 * 2, 2 ); /* A column 2. */ \
        AMX_MEM( LDX, a + 32 * 3, 3 ); /* A column 3. */ \
        AMX_MEM( LDX, a + 32 * 4, 4 ); /* A column 4. */ \
        AMX_MEM( LDX, a + 32 * 5, 5 ); /* A column 5. */ \
        AMX_MEM( LDX, a + 32 * 6, 6 ); /* A column 6. */ \
        AMX_MEM( LDX, a + 32 * 7, 7 ); /* A column 7. */ \
\
        AMX_MEM( LDY, b + 32 * 0, 0 ); /* B row 0. */ \
        AMX_MEM( LDY, b + 32 * 1, 1 ); /* B row 1. */ \
        AMX_MEM( LDY, b + 32 * 2, 2 ); /* B row 2. */ \
        AMX_MEM( LDY, b + 32 * 3, 3 ); /* B row 3. */ \
        AMX_MEM( LDY, b + 32 * 4, 4 ); /* B row 4. */ \
        AMX_MEM( LDY, b + 32 * 5, 5 ); /* B row 5. */ \
        AMX_MEM( LDY, b + 32 * 6, 6 ); /* B row 6. */ \
        AMX_MEM( LDY, b + 32 * 7, 7 ); /* B row 7. */ \
\
        OpMulAdd( 0, 0, 0 ); \
        OpMulAdd( 1, 1, 0 ); \
        OpMulAdd( 2, 2, 0 ); \
        OpMulAdd( 3, 3, 0 ); \
        OpMulAdd( 4, 4, 0 ); \
        OpMulAdd( 5, 5, 0 ); \
        OpMulAdd( 6, 6, 0 ); \
        OpMulAdd( 7, 7, 0 ); \
\
        /* Address forward. */ \
        a += 8 * 32; \
        b += 8 * 32; \
    } \
_Pragma("nounroll") \
    for ( ; k >= 1; k -= 1 ) \
    { \
        AMX_MEM( LDX, a + 0 , 0 ); \
        AMX_MEM( LDY, b + 0 , 0 ); \
\
        OpMulAdd( 0, 0, 0 ); \
\
        a += 32; \
        b += 32; \
    } \
\
    /* Load alpha & beta. */ \
    AMX_MEM( LDY, alphac, 0 ); \
    AMX_MEM( LDY, beta_c, 1 ); \
\
    /* Multiply by alpha. */ \
    if ( *alpha != 1.0 || *beta != 0.0 ) \
        for ( int i = 0; i < 16; ++i ) { \
            AMX_EXTRX_REGALIGNED( i * 4 + 0, 4 ); \
            AMX_EXTRX_REGALIGNED( i * 4 + 1, 5 ); \
            AMX_EXTRX_REGALIGNED( i * 4 + 2, 6 ); \
            AMX_EXTRX_REGALIGNED( i * 4 + 3, 7 ); \
\
            if ( *beta != 0.0 ) \
            { \
                AMX_MEM( LDZI, c + (i * 2 + 0) * cs_c + 0 , i * 4 + 0 ); \
                AMX_MEM( LDZI, c + (i * 2 + 0) * cs_c + 16, i * 4 + 1 ); \
                AMX_MEM( LDZI, c + (i * 2 + 1) * cs_c + 0 , i * 4 + 2 ); \
                AMX_MEM( LDZI, c + (i * 2 + 1) * cs_c + 16, i * 4 + 3 ); \
\
                if ( *beta != 1.0 ) \
                { /* Self-scale by beta. */ \
                    AMX_EXTRX_REGALIGNED( i * 4 + 0, 0 ); \
                    AMX_EXTRX_REGALIGNED( i * 4 + 1, 1 ); \
                    AMX_EXTRX_REGALIGNED( i * 4 + 2, 2 ); \
                    AMX_EXTRX_REGALIGNED( i * 4 + 3, 3 ); \
\
                    OpMulSelCol( i, 0, 1, 0 ); \
                    OpMulSelCol( i, 1, 1, 1 ); \
                    OpMulSelCol( i, 2, 1, 2 ); \
                    OpMulSelCol( i, 3, 1, 3 ); \
                } \
\
                OpMulAddSelCol( i, 4, 0, 0 ); \
                OpMulAddSelCol( i, 5, 0, 1 ); \
                OpMulAddSelCol( i, 6, 0, 2 ); \
                OpMulAddSelCol( i, 7, 0, 3 ); \
            } \
            else \
            { \
                OpMulSelCol( i, 4, 0, 0 ); \
                OpMulSelCol( i, 5, 0, 1 ); \
                OpMulSelCol( i, 6, 0, 2 ); \
                OpMulSelCol( i, 7, 0, 3 ); \
            } \
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
    if ( rs_c == 1 ) \
    { \
        /* Store blocks (0, 0) and (1, 0). */ \
        for (int i = 0; i < 32; ++i) { \
            AMX_MEM( STZI, c + i * cs_c + 0 , i * 2 + 0 ); \
            AMX_MEM( STZI, c + i * cs_c + 16, i * 2 + 1 ); \
        } \
    } \
\
    AMX_STOP(); \
}

PASTEPAC_16BIT_TO_32BIT(
        s_sh,
        float16_t,
        float32_t,
        AMX_FMA16_32_COMMON_REGALIGNED,
        AMX_FMA32_SELCOL_REGALIGNED,
        AMX_FMUL16_32_COMMON_REGALIGNED,
        AMX_FMUL32_SELCOL_REGALIGNED)

// Some int32-instructions are not supported by hardware.
// TODO: Find way for circumventing this.
#define AMX_MAC32_SELCOL_REGALIGNED(COLIDX, XREG, YREG, ZREGS) assert( 0 )
#define AMX_MUL32_SELCOL_REGALIGNED(COLIDX, XREG, YREG, ZREGS) assert( 0 )

PASTEPAC_16BIT_TO_32BIT(
        i32_i16,
        int16_t,
        int32_t,
        AMX_MAC16_32_COMMON_REGALIGNED,
        AMX_MAC32_SELCOL_REGALIGNED,
        AMX_MUL16_32_COMMON_REGALIGNED,
        AMX_MUL32_SELCOL_REGALIGNED)

