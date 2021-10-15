#include "bli_sandbox.h"
#include "../../kernels/aaplmx/3/amx.h"
#include "../../kernels/aaplmx/3/amx_ext.h"
#include <arm_neon.h>


#define BIT16_PACK_A(ch, DTYPE_IN, OpMulSelRow) \
\
void PACK_FUNC_NAME(ch, a) \
    ( \
        dim_t MR, \
        int m, int k, \
        DTYPE_IN* ap, int rs_a, int cs_a, \
        DTYPE_IN* apack \
    ) \
{ \
    int p; \
    int p_idx; \
    static const DTYPE_IN kappa[32] = \
        { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 }; \
    AMX_START(); \
\
    DTYPE_IN* adest = apack; \
    for (int i=0; i<m; i+=MR) \
    { \
        int ib = bli_min(MR, m-i); \
        if (ib == MR) /* Full size column height */ \
        { \
            p     = 0; \
            p_idx = 0; \
            if ( cs_a == 1 ) \
            { \
                AMX_MEM( LDX, kappa, 0 ); \
                for ( ; p+32<k; p+=32 ) \
                { \
                    _Pragma("unroll") \
                    for ( int ivec = 0; ivec < 4; ++ivec ) \
                        for ( int jvec = 0; jvec < 8; ++jvec ) \
                        { \
                            AMX_MEM( LDY, ap + (i + 8*ivec + jvec +  0)*rs_a + p_idx*cs_a, jvec ); \
                            OpMulSelRow( 8*ivec + jvec, 0, jvec, 0 ); \
                            AMX_MEM( LDY, ap + (i + 8*ivec + jvec + 32)*rs_a + p_idx*cs_a, jvec ); \
                            OpMulSelRow( 8*ivec + jvec, 0, jvec, 1 ); \
                        } \
                    _Pragma("unroll") \
                    for ( int ivec = 0; ivec < 32; ++ivec ) \
                    { \
                        AMX_MEM( STZ, adest + 64 * ivec +  0, ivec * 2 + 0 ); \
                        AMX_MEM( STZ, adest + 64 * ivec + 32, ivec * 2 + 1 ); \
                    } \
                    adest += 32 * 64; \
                    p_idx += 32; \
                } \
            } \
\
            for ( ; p<k; p++ ) \
            {  \
                if ( rs_a == 1 ) \
                { \
                    AMX_MEM( LDX, ap    + p_idx*cs_a, 0 ); \
                    AMX_MEM( LDX, ap+32 + p_idx*cs_a, 1 ); \
                    AMX_MEM( STX, adest   , 0 ); \
                    AMX_MEM( STX, adest+32, 1 ); \
                    adest += 64; \
                } \
                else \
                { \
                    *adest++ = ap[ (i+ 0+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 1+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 2+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 3+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 4+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 5+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 6+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 7+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 8+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 9+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+10+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+11+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+12+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+13+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+14+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+15+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+16+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+17+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+18+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+19+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+20+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+21+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+22+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+23+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+24+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+25+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+26+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+27+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+28+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+29+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+30+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+31+ 0)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 0+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 1+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 2+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 3+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 4+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 5+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 6+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 7+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 8+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+ 9+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+10+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+11+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+12+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+13+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+14+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+15+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+16+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+17+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+18+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+19+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+20+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+21+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+22+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+23+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+24+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+25+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+26+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+27+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+28+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+29+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+30+32)*rs_a + p_idx*cs_a ]; \
                    *adest++ = ap[ (i+31+32)*rs_a + p_idx*cs_a ]; \
                } \
                p_idx += 1; \
            } \
        } \
\
        else /* Not full size, pad with zeros */ \
        { \
            p_idx = 0; \
            for (int p=0; p<k; p++) \
            { \
                for (int ir=0; ir<ib; ir++) \
                { \
                    *adest++ = ap[ (i+ir)*rs_a + p_idx*cs_a ]; \
                } \
                for (int ir=ib; ir<MR; ir++) \
                { \
                    *adest++ = 0; \
                } \
                p_idx += 1; \
            } \
        } \
    } \
    AMX_STOP(); \
} 

#define BIT16_PACK_B(ch, DTYPE_IN, OpMulSelRow) \
\
void PACK_FUNC_NAME(ch, b) \
    ( \
        dim_t NR, \
        int k, int n, \
        DTYPE_IN* bp, int rs_b, int cs_b, \
        DTYPE_IN* bpack \
    ) \
{ \
    int p; \
    int p_idx; \
    static const DTYPE_IN kappa[32] = \
        { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 }; \
    AMX_START(); \
\
    DTYPE_IN* bdest = bpack; \
    for( int j=0; j<n; j += NR ) \
    { \
        int jb = bli_min(NR, n-j); \
        if ( jb == NR ) /* Full column width micro-panel.*/  \
        { \
            p     = 0; \
            p_idx = 0; \
            if ( rs_b == 1 ) \
            { \
                AMX_MEM( LDX, kappa, 0 ); \
                for ( ; p+32<k; p+=32 ) \
                { \
                    _Pragma("unroll") \
                    for ( int ivec = 0; ivec < 4; ++ivec ) \
                        for ( int jvec = 0; jvec < 8; ++jvec ) \
                        { \
                            AMX_MEM( LDY, bp + p_idx*rs_b + (j + 8*ivec + jvec)*cs_b, jvec ); \
                            OpMulSelRow( 8*ivec + jvec, 0, jvec, 0 ); \
                        } \
                    _Pragma("unroll") \
                    for ( int ivec = 0; ivec < 32; ++ivec ) \
                        AMX_MEM( STZ, bdest + 32 * ivec, ivec * 2 + 0 ); \
                    bdest += 32 * 32; \
                    p_idx += 32; \
                } \
            } \
\
            for ( ; p<k; p++ ) \
            { \
                if ( cs_b == 1 ) \
                { \
                    AMX_MEM( LDY, bp + p_idx*rs_b, 0 ); \
                    AMX_MEM( STY, bdest, 0 ); \
                    bdest += 32; \
                } \
                else \
                { \
                    *bdest++ = bp[ p_idx*rs_b + (j+ 0)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+ 1)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+ 2)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+ 3)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+ 4)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+ 5)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+ 6)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+ 7)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+ 8)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+ 9)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+10)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+11)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+12)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+13)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+14)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+15)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+16)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+17)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+18)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+19)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+20)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+21)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+22)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+23)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+24)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+25)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+26)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+27)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+28)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+29)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+30)*cs_b ]; \
                    *bdest++ = bp[ p_idx*rs_b + (j+31)*cs_b ]; \
                } \
                p_idx += 1; \
            } \
        } \
\
        else /* Not a full row size micro-panel.  We pad with zeroes. */ \
        { \
            p_idx = 0; \
            for ( int p=0; p<k; p++ )  \
            { \
                for ( int jr=0; jr<jb; jr++ ) \
                { \
                    *bdest++ = bp[ p_idx*rs_b + (j+jr)*cs_b ]; \
                } \
                for ( int jr=jb; jr<NR; jr++ ) \
                { \
                    *bdest++ = 0; \
                } \
                p_idx += 1; \
            } \
        } \
    } \
    AMX_STOP(); \
};


// 16 bit routines
BIT16_PACK_A( sh, float16_t, AMX_FMUL16_SELROW_REGALIGNED);
BIT16_PACK_B( sh, float16_t, AMX_FMUL16_SELROW_REGALIGNED);
BIT16_PACK_A(i16,   int16_t,  AMX_MUL16_SELROW_REGALIGNED);
BIT16_PACK_B(i16,   int16_t,  AMX_MUL16_SELROW_REGALIGNED);

