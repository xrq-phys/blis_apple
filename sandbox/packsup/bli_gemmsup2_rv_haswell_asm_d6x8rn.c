#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

#include "blis.h"
#include <assert.h>


#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"
// vpmovsxbq, vmaskmovpd was not defined.
#define vpmovsxbq(_0, _1) INSTR_(vpmovsxbq, _0, _1)
#define vmaskmovpd(_0, _1, _2) INSTR_(vmaskmovpd, _0, _1, _2)


#define LOAD_BROW_6(BADDR,B0_,B1_) \
    vmovupd(mem(BADDR    ), ymm(B0_)) \
    vmovupd(mem(BADDR, 32), xmm(B1_))

#define LOAD_BROW_5(BADDR,B0_,B1_) \
    vmovupd(mem(BADDR    ), ymm(B0_)) \
    vmovsd( mem(BADDR, 32), xmm(B1_))

#define LOAD_BROW_4(BADDR,B0_,B1_) vmovupd(mem(BADDR), ymm(B0_))
#define LOAD_BROW_3(BADDR,B0_,B1_) vmaskmovpd(mem(BADDR), ymm(B1_), ymm(B0_))
#define LOAD_BROW_2(BADDR,B0_,B1_) vmovupd(mem(BADDR), xmm(B0_))
#define LOAD_BROW_1(BADDR,B0_,B1_) vmovsd( mem(BADDR), xmm(B0_))

#define FMA_ROW_6(INST,A_,B0_,B1_,C0_,C1_) \
    INST (ymm(A_), ymm(B0_), ymm(C0_)) \
    INST (xmm(A_), xmm(B1_), xmm(C1_)) \

#define FMA_ROW_5(INST,A_,B0_,B1_,C0_,C1_) FMA_ROW_6(INST,A_,B0_,B1_,C0_,C1_)
#define FMA_ROW_4(INST,A_,B0_,B1_,C0_,C1_) INST (ymm(A_), ymm(B0_), ymm(C0_))
#define FMA_ROW_3(INST,A_,B0_,B1_,C0_,C1_) INST (ymm(A_), ymm(B0_), ymm(C0_))
#define FMA_ROW_2(INST,A_,B0_,B1_,C0_,C1_) INST (xmm(A_), xmm(B0_), xmm(C0_))
#define FMA_ROW_1(INST,A_,B0_,B1_,C0_,C1_) INST (xmm(A_), xmm(B0_), xmm(C0_))

#define DGEMM_6X8_NANOKER_6(N,INST,C00_,C01_,C10_,C11_,C20_,C21_,C30_,C31_,C40_,C41_,C50_,C51_,A0_,A1_,B0_,B1_,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB) \
    LOAD_BROW_## N (BADDR,B0_,B1_) \
    add(RSB, BADDR) \
    vbroadcastsd(mem(AADDR        ), ymm(A0_)) \
    vbroadcastsd(mem(AADDR, RSA, 1), ymm(A1_)) \
    FMA_ROW_## N (INST,A0_,B0_,B1_,C00_,C01_) \
    PACK_ ##PACKA (vunpcklpd(xmm(A1_), xmm(A0_), xmm(A0_))) \
    FMA_ROW_## N (INST,A1_,B0_,B1_,C10_,C11_) \
    PACK_ ##PACKA (vmovapd(xmm(A0_), mem(PAADDR))) \
    vbroadcastsd(mem(AADDR, RSA, 2), ymm(A0_)) \
    vbroadcastsd(mem(AADDR, RSA3,1), ymm(A1_)) \
    FMA_ROW_## N (INST,A0_,B0_,B1_,C20_,C21_) \
    PACK_ ##PACKA (vunpcklpd(xmm(A1_), xmm(A0_), xmm(A0_))) \
    FMA_ROW_## N (INST,A1_,B0_,B1_,C30_,C31_) \
    PACK_ ##PACKA (vmovapd(xmm(A0_), mem(PAADDR, 2*8))) \
    vbroadcastsd(mem(AADDR, RSA, 4), ymm(A0_)) \
    vbroadcastsd(mem(AADDR, RSA5,1), ymm(A1_)) \
    add(CSA, AADDR) \
    FMA_ROW_## N (INST,A0_,B0_,B1_,C40_,C41_) \
    PACK_ ##PACKA (vunpcklpd(xmm(A1_), xmm(A0_), xmm(A0_))) \
    FMA_ROW_## N (INST,A1_,B0_,B1_,C50_,C51_) \
    PACK_ ##PACKA (vmovapd(xmm(A0_), mem(PAADDR, 4*8))) \
    PACK_ ##PACKA (add(imm(6*8), PAADDR)) \
    PACK_ ##PACKB (vmovapd(ymm(B0_), mem(PBADDR    ))) /* TODO: Fixme: p_b n<=j<8 is undef. Zeroize it. */ \
    PACK_ ##PACKB (vmovapd(ymm(B1_), mem(PBADDR, 32))) \
    PACK_ ##PACKB (add(imm(8*8), PBADDR))

// This scheme supports 6, 5, 4, 3, 2, 1.
#define DGEMM_6X8_NANOKER_5(N,INST,C00_,C01_,C10_,C11_,C20_,C21_,C30_,C31_,C40_,C41_,C50_,C51_,A0_,A1_,B0_,B1_,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB) DGEMM_6X8_NANOKER_6(N,INST,C00_,C01_,C10_,C11_,C20_,C21_,C30_,C31_,C40_,C41_,C50_,C51_,A0_,A1_,B0_,B1_,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB)
#define DGEMM_6X8_NANOKER_4(N,INST,C00_,C01_,C10_,C11_,C20_,C21_,C30_,C31_,C40_,C41_,C50_,C51_,A0_,A1_,B0_,B1_,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB) DGEMM_6X8_NANOKER_6(N,INST,C00_,C01_,C10_,C11_,C20_,C21_,C30_,C31_,C40_,C41_,C50_,C51_,A0_,A1_,B0_,B1_,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB)
#define DGEMM_6X8_NANOKER_3(N,INST,C00_,C01_,C10_,C11_,C20_,C21_,C30_,C31_,C40_,C41_,C50_,C51_,A0_,A1_,B0_,B1_,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB) DGEMM_6X8_NANOKER_6(N,INST,C00_,C01_,C10_,C11_,C20_,C21_,C30_,C31_,C40_,C41_,C50_,C51_,A0_,A1_,B0_,B1_,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB)
#define DGEMM_6X8_NANOKER_2(N,INST,C00_,C01_,C10_,C11_,C20_,C21_,C30_,C31_,C40_,C41_,C50_,C51_,A0_,A1_,B0_,B1_,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB) DGEMM_6X8_NANOKER_6(N,INST,C00_,C01_,C10_,C11_,C20_,C21_,C30_,C31_,C40_,C41_,C50_,C51_,A0_,A1_,B0_,B1_,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB)
#define DGEMM_6X8_NANOKER_1(N,INST,C00_,C01_,C10_,C11_,C20_,C21_,C30_,C31_,C40_,C41_,C50_,C51_,A0_,A1_,B0_,B1_,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB) DGEMM_6X8_NANOKER_6(N,INST,C00_,C01_,C10_,C11_,C20_,C21_,C30_,C31_,C40_,C41_,C50_,C51_,A0_,A1_,B0_,B1_,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB)

#define DGEMM_6X8_NANOKER_7(N,INST,C00_,C01_,C10_,C11_,C20_,C21_,C30_,C31_,C40_,C41_,C50_,C51_,A0_,A1_,B0_,B1_,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB) \
    vmovupd(mem(BADDR), ymm(B0_)) \
    vmaskmovpd(mem(BADDR, 32), ymm(A1_), ymm(B1_)) \
    add(RSB, BADDR) \
    vbroadcastsd(mem(AADDR), ymm(A0_)) \
    INST (ymm(A0_), ymm(B0_), ymm(C00_)) \
    INST (ymm(A0_), ymm(B1_), ymm(C01_)) \
    PACK_ ##PACKA (vmovsd(xmm(A0_), mem(PAADDR))) \
    vbroadcastsd(mem(AADDR, RSA, 1), ymm(A0_)) \
    INST (ymm(A0_), ymm(B0_), ymm(C10_)) \
    INST (ymm(A0_), ymm(B1_), ymm(C11_)) \
    PACK_ ##PACKA (vmovsd(xmm(A0_), mem(PAADDR, 1*8))) \
    vbroadcastsd(mem(AADDR, RSA, 2), ymm(A0_)) \
    INST (ymm(A0_), ymm(B0_), ymm(C20_)) \
    INST (ymm(A0_), ymm(B1_), ymm(C21_)) \
    PACK_ ##PACKA (vmovsd(xmm(A0_), mem(PAADDR, 2*8))) \
    vbroadcastsd(mem(AADDR, RSA3,1), ymm(A0_)) \
    INST (ymm(A0_), ymm(B0_), ymm(C30_)) \
    INST (ymm(A0_), ymm(B1_), ymm(C31_)) \
    PACK_ ##PACKA (vmovsd(xmm(A0_), mem(PAADDR, 3*8))) \
    vbroadcastsd(mem(AADDR, RSA, 4), ymm(A0_)) \
    INST (ymm(A0_), ymm(B0_), ymm(C40_)) \
    INST (ymm(A0_), ymm(B1_), ymm(C41_)) \
    PACK_ ##PACKA (vmovsd(xmm(A0_), mem(PAADDR, 4*8))) \
    vbroadcastsd(mem(AADDR, RSA5,1), ymm(A0_)) \
    add(CSA, AADDR) \
    INST (ymm(A0_), ymm(B0_), ymm(C50_)) \
    INST (ymm(A0_), ymm(B1_), ymm(C51_)) \
    PACK_ ##PACKA (vmovsd(xmm(A0_), mem(PAADDR, 5*8))) \
    PACK_ ##PACKA (add(imm(6*8), PAADDR)) \
    PACK_ ##PACKB (vmovapd(ymm(B0_), mem(PBADDR    ))) \
    PACK_ ##PACKB (vmovapd(ymm(B1_), mem(PBADDR, 32))) \
    PACK_ ##PACKB (add(imm(8*8), PBADDR))

#define PACK_nopack(_1)
#define PACK_pack(_1) _1

#define DGENMASK_1(A1_,B1_,RTMP)
#define DGENMASK_2(A1_,B1_,RTMP)
#define DGENMASK_4(A1_,B1_,RTMP)
#define DGENMASK_5(A1_,B1_,RTMP)
#define DGENMASK_6(A1_,B1_,RTMP)

#define DGENMASK_3(A1_,B1_,RTMP) /* Use B1 */ \
    mov(imm(0b00000000111111111111111111111111), RTMP) \
    vmovd(RTMP, xmm(B1_)) \
    vpmovsxbq(xmm(B1_), ymm(B1_))

#define DGENMASK_7(A1_,B1_,RTMP) /* Use A1 */ \
    mov(imm(0b00000000111111111111111111111111), RTMP) \
    vmovd(RTMP, xmm(A1_)) \
    vpmovsxbq(xmm(A1_), ymm(A1_))

#define DMOVMASK_LOC_1
#define DMOVMASK_LOC_2
#define DMOVMASK_LOC_3
#define DMOVMASK_LOC_4
#define DMOVMASK_LOC_5
#define DMOVMASK_LOC_6
#define DMOVMASK_LOC_7 vmovapd(ymm1, ymm3)

#define DGEMM_6X8_NANOKER_LOC(N,INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB) \
    DGEMM_6X8_NANOKER_ ## N(N,INST,4,5,6,7,8,9,10,11,12,13,14,15,0,1,2,3,rax,RSA,RSA3,RSA5,CSA,rbx,RSB,rcx,rdx,PACKA,PACKB)

// No in-reg transpose support.
#define C1ROW_BETA_FWD_7(VFH_,VLH_,VBETA_,VMASK,CADDR,CSC) \
    vfmadd231pd(mem(CADDR), ymm(VBETA_), ymm(VFH_)) \
    vmovupd(ymm(VFH_), mem(CADDR)) \
    vmaskmovpd(mem(CADDR, 32), VMASK, ymm(VFH_)) \
    vfmadd231pd(ymm(VFH_), ymm(VBETA_), ymm(VLH_)) \
    vmaskmovpd(ymm(VLH_), VMASK, mem(CADDR, 32)) \
    add(CSC, CADDR)

#define C1ROW_BETA_FWD_6(VFH_,VLH_,VBETA_,VMASK,CADDR,CSC) \
    vfmadd231pd(mem(CADDR), ymm(VBETA_), ymm(VFH_)) \
    vmovupd(ymm(VFH_), mem(CADDR)) \
    vfmadd231pd(mem(CADDR, 32), xmm(VBETA_), xmm(VLH_)) \
    vmovupd(xmm(VLH_), mem(CADDR, 32)) \
    add(CSC, CADDR)

#define C1ROW_BETA_FWD_5(VFH_,VLH_,VBETA_,VMASK,CADDR,CSC) \
    vfmadd231pd(mem(CADDR), ymm(VBETA_), ymm(VFH_)) \
    vmovupd(ymm(VFH_), mem(CADDR)) \
    vmovsd(mem(CADDR, 32), xmm(VFH_)) \
    vfmadd231pd(xmm(VFH_), xmm(VBETA_), xmm(VLH_)) \
    vmovsd(xmm(VLH_), mem(CADDR, 32)) \
    add(CSC, CADDR)

#define C1ROW_BETA_FWD_4(VFH_,VLH_,VBETA_,VMASK,CADDR,CSC) \
    vfmadd231pd(mem(CADDR), ymm(VBETA_), ymm(VFH_)) \
    vmovupd(ymm(VFH_), mem(CADDR)) \
    add(CSC, CADDR)

#define C1ROW_BETA_FWD_3(VFH_,VLH_,VBETA_,VMASK,CADDR,CSC) \
    vmaskmovpd(mem(CADDR), VMASK, ymm(VLH_)) \
    vfmadd231pd(ymm(VLH_), ymm(VBETA_), ymm(VFH_)) \
    vmaskmovpd(ymm(VFH_), VMASK, mem(CADDR)) \
    add(CSC, CADDR)

#define C1ROW_BETA_FWD_2(VFH_,VLH_,VBETA_,VMASK,CADDR,CSC) \
    vfmadd231pd(mem(CADDR), xmm(VBETA_), xmm(VFH_)) \
    vmovupd(xmm(VFH_), mem(CADDR)) \
    add(CSC, CADDR)

#define C1ROW_BETA_FWD_1(VFH_,VLH_,VBETA_,VMASK,CADDR,CSC) \
    vmovsd(mem(CADDR), xmm(VLH_)) \
    vfmadd231pd(xmm(VLH_), xmm(VBETA_), xmm(VFH_)) \
    vmovsd(xmm(VFH_), mem(CADDR)) \
    add(CSC, CADDR)

#define C1ROW_FWD_7(VFH_,VLH_,VMASK,CADDR,CSC) \
    vmovupd(ymm(VFH_), mem(CADDR)) \
    vmaskmovpd(ymm(VLH_), VMASK, mem(CADDR, 32)) \
    add(CSC, CADDR)

#define C1ROW_FWD_6(VFH_,VLH_,VMASK,CADDR,CSC) \
    vmovupd(ymm(VFH_), mem(CADDR)) \
    vmovupd(xmm(VLH_), mem(CADDR, 32)) \
    add(CSC, CADDR)

#define C1ROW_FWD_5(VFH_,VLH_,VMASK,CADDR,CSC) \
    vmovupd(ymm(VFH_), mem(CADDR)) \
    vmovsd(xmm(VLH_), mem(CADDR, 32)) \
    add(CSC, CADDR)

#define C1ROW_FWD_4(VFH_,VLH_,VMASK,CADDR,CSC) \
    vmovupd(ymm(VFH_), mem(CADDR)) \
    add(CSC, CADDR)

#define C1ROW_FWD_3(VFH_,VLH_,VMASK,CADDR,CSC) \
    vmaskmovpd(ymm(VFH_), VMASK, mem(CADDR)) \
    add(CSC, CADDR)

#define C1ROW_FWD_2(VFH_,VLH_,VMASK,CADDR,CSC) \
    vmovupd(xmm(VFH_), mem(CADDR)) \
    add(CSC, CADDR)

#define C1ROW_FWD_1(VFH_,VLH_,VMASK,CADDR,CSC) \
    vmovsd(xmm(VFH_), mem(CADDR)) \
    add(CSC, CADDR)


#define GENDECL(N,PACKA,PACKB) \
void bli_dgemmsup2_rv_haswell_asm_6x## N ##r_ ## PACKA ## _ ## PACKB \
    ( \
     dim_t            m, \
     dim_t            n, \
     dim_t            k, \
     double *restrict alpha, \
     double *restrict a, inc_t rs_a0, inc_t cs_a0, \
     double *restrict b, inc_t rs_b0, inc_t cs_b0, \
     double *restrict beta, \
     double *restrict c, inc_t rs_c0, inc_t cs_c0, \
     auxinfo_t       *data, \
     cntx_t          *cntx, \
     double *restrict a_p, \
     double *restrict b_p \
    )

#define GENDEF(N,PACKA,PACKB) \
    GENDECL(N,PACKA,PACKB) \
{ \
    const void* a_next = bli_auxinfo_next_a( data ); \
    const void* b_next = bli_auxinfo_next_b( data ); \
    uint64_t cs_a_next = bli_auxinfo_ps_a( data ); /* Borrow the space. */ \
\
    /* Typecast local copies of integers in case dim_t and inc_t are a
     * different size than is expected by load instructions. */ \
    uint64_t k_mker = k / 4; \
    uint64_t k_left = k % 4; \
    uint64_t rs_a   = rs_a0; \
    uint64_t cs_a   = cs_a0; \
    uint64_t rs_b   = rs_b0; \
    uint64_t rs_c   = rs_c0; \
    uint64_t cs_c   = cs_c0; \
\
    begin_asm() \
\
    DGENMASK_## N (1,3,r8) \
\
    mov(var(k_mker), rsi) \
    mov(var(k_left), r14) \
    mov(var(a), rax) \
    mov(var(b), rbx) \
    mov(var(a_p), rcx) \
    mov(var(b_p), rdx) \
    mov(var(rs_a), rdi) \
    lea(mem(, rdi, 8), rdi) /* rs_a *= sizeof(double) */ \
    mov(var(cs_a), r10) \
    lea(mem(, r10, 8), r10) /* cs_a *= sizeof(double) */ \
    mov(var(rs_b), r11) \
    lea(mem(, r11, 8), r11) /* rs_b *= sizeof(double) */ \
    lea(mem(rdi, rdi, 2), r13) /* r13 = 3*rs_a; */ \
    lea(mem(r13, rdi, 2), r15) /* r15 = 5*rs_a; */ \
    mov(var(a_next), r8) \
    mov(var(cs_a2), r9) /* r9 = cs_a_next; */ \
    lea(mem(, r9, 8), r9) /* cs_a_next *= sizeof(double) */ \
    lea(mem(r9, r9, 2), r12) /* r12 = 3*cs_a_next; */ \
\
    /* TODO: Prefetch C. */ \
\
    test(rsi, rsi) \
    je(.DEMPTYALL) \
\
    label(.DK_4LOOP_INIT) \
        prefetch(0, mem(r8, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(N,vmulpd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r9, 1, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r9, 2, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r12, 1, 5*8)) \
        lea(mem(r8, r9, 4), r8) \
        DGEMM_6X8_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        dec(rsi) \
    je(.DK_LEFT_LOOP_PREP) \
\
    label(.DK_4LOOP) \
        prefetch(0, mem(r8, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r9, 1, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r9, 2, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r12, 1, 5*8)) \
        lea(mem(r8, r9, 4), r8) \
        DGEMM_6X8_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        dec(rsi) \
    jne(.DK_4LOOP) \
\
    label(.DK_LEFT_LOOP_PREP) \
    test(r14, r14) \
    je(.DWRITEMEM_PREP) \
    jmp(.DK_LEFT_LOOP) \
\
    label(.DEMPTYALL) \
        vxorpd(ymm4,  ymm4,  ymm4) \
        vxorpd(ymm5,  ymm5,  ymm5) \
        vxorpd(ymm6,  ymm6,  ymm6) \
        vxorpd(ymm7,  ymm7,  ymm7) \
        vxorpd(ymm8,  ymm8,  ymm8) \
        vxorpd(ymm9,  ymm9,  ymm9) \
        vxorpd(ymm10, ymm10, ymm10) \
        vxorpd(ymm11, ymm11, ymm11) \
        vxorpd(ymm12, ymm12, ymm12) \
        vxorpd(ymm13, ymm13, ymm13) \
        vxorpd(ymm14, ymm14, ymm14) \
        vxorpd(ymm15, ymm15, ymm15) \
\
    label(.DK_LEFT_LOOP) \
        prefetch(0, mem(r8, 5*8)) \
        add(r9, r8) \
        DGEMM_6X8_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        dec(r14) \
    jne(.DK_LEFT_LOOP) \
\
    label(.DWRITEMEM_PREP) \
\
    mov(var(alpha), rax) \
    mov(var(beta), rbx) \
    /* mov(var(a_next), rcx) */ \
    mov(var(b_next), rdx) \
    vbroadcastsd(mem(rax), ymm0) \
    vbroadcastsd(mem(rbx), ymm2) \
\
        vmulpd(ymm0, ymm4, ymm4) \
        vmulpd(ymm0, ymm5, ymm5) \
        vmulpd(ymm0, ymm6, ymm6) \
        vmulpd(ymm0, ymm7, ymm7) \
        vmulpd(ymm0, ymm8, ymm8) \
        vmulpd(ymm0, ymm9, ymm9) \
        vmulpd(ymm0, ymm10, ymm10) \
        vmulpd(ymm0, ymm11, ymm11) \
        vmulpd(ymm0, ymm12, ymm12) \
        vmulpd(ymm0, ymm13, ymm13) \
        vmulpd(ymm0, ymm14, ymm14) \
        vmulpd(ymm0, ymm15, ymm15) \
\
        /* prefetch(0, mem(rcx)) */ \
        /* prefetch(0, mem(rcx, 1*64)) \
         * prefetch(0, mem(rcx, 2*64)) */ \
        prefetch(0, mem(rdx)) \
        /* prefetch(0, mem(rdx, 1*64)) \
         * prefetch(0, mem(rdx, 2*64)) \
         * prefetch(0, mem(rdx, 3*64)) */ \
\
    mov(var(c), rcx) /* load c */ \
    mov(var(rs_c), rdi) /* load rs_c */ \
    lea(mem(, rdi, 8), rdi) /* rdi = rs_c * sizeof(double) */ \
\
    /* now avoid loading C if beta == 0 */ \
\
    DMOVMASK_LOC_## N /* ymm3 is mask when n == 3 or n == 7. */ \
    vxorpd(ymm0, ymm0, ymm0) /* set ymm0 to zero. */ \
    vucomisd(xmm0, xmm2) /* set ZF if beta == 0. */ \
    je(.DBETAZERO) /* if ZF = 1, jump to beta == 0 case */ \
\
        C1ROW_BETA_FWD_ ## N(4, 5, 2, ymm3, rcx, rdi) \
        C1ROW_BETA_FWD_ ## N(6, 7, 2, ymm3, rcx, rdi) \
        C1ROW_BETA_FWD_ ## N(8, 9, 2, ymm3, rcx, rdi) \
        C1ROW_BETA_FWD_ ## N(10, 11, 2, ymm3, rcx, rdi) \
        C1ROW_BETA_FWD_ ## N(12, 13, 2, ymm3, rcx, rdi) \
        C1ROW_BETA_FWD_ ## N(14, 15, 2, ymm3, rcx, rdi) \
\
        jmp(.DDONE) /* jump to end. */ \
\
    label(.DBETAZERO) \
\
        C1ROW_FWD_ ## N(4, 5, ymm3, rcx, rdi) \
        C1ROW_FWD_ ## N(6, 7, ymm3, rcx, rdi) \
        C1ROW_FWD_ ## N(8, 9, ymm3, rcx, rdi) \
        C1ROW_FWD_ ## N(10, 11, ymm3, rcx, rdi) \
        C1ROW_FWD_ ## N(12, 13, ymm3, rcx, rdi) \
        C1ROW_FWD_ ## N(14, 15, ymm3, rcx, rdi) \
\
    label(.DDONE) \
\
    end_asm( \
    :: \
      [a]      "m" (a), \
      [rs_a]   "m" (rs_a), \
      [cs_a]   "m" (cs_a), \
      [cs_a2]  "m" (cs_a_next), \
      [b]      "m" (b), \
      [rs_b]   "m" (rs_b), \
      [c]      "m" (c), \
      [rs_c]   "m" (rs_c), \
      [cs_c]   "m" (cs_c), \
      [k_mker] "m" (k_mker), \
      [k_left] "m" (k_left), \
      [alpha]  "m" (alpha), \
      [beta]   "m" (beta), \
      [a_next] "m" (a_next), \
      [b_next] "m" (b_next), \
      [a_p]    "m" (a_p), \
      [b_p]    "m" (b_p) \
    : "rax", "rbx", "rcx", "rdx", "rsi", "rdi", \
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", \
      "xmm0", "xmm1", "xmm2", "xmm3", \
      "xmm4", "xmm5", "xmm6", "xmm7", \
      "xmm8", "xmm9", "xmm10", "xmm11", \
      "xmm12", "xmm13", "xmm14", "xmm15", \
      "memory" \
    ) \
\
}

// GENDECL(8,pack,pack);
// GENDECL(8,pack,nopack);

GENDEF(1,nopack,pack)
GENDEF(2,nopack,pack)
GENDEF(3,nopack,pack)
GENDEF(4,nopack,pack)
GENDEF(5,nopack,pack)
GENDEF(6,nopack,pack)
GENDEF(7,nopack,pack)
// GENDECL(8,nopack,pack);

GENDEF(1,nopack,nopack)
GENDEF(2,nopack,nopack)
GENDEF(3,nopack,nopack)
GENDEF(4,nopack,nopack)
GENDEF(5,nopack,nopack)
GENDEF(6,nopack,nopack)
GENDEF(7,nopack,nopack)
// GENDECL(8,nopack,nopack);
#undef GENDEF

void bli_dgemmsup2_rv_haswell_asm_6x8rn
    (
     dim_t            m,
     dim_t            n,
     dim_t            k,
     double *restrict alpha,
     double *restrict a, inc_t rs_a, inc_t cs_a,
     double *restrict b, inc_t rs_b, inc_t cs_b,
     double *restrict beta0,
     double *restrict c0, inc_t rs_c0, inc_t cs_c0,
     auxinfo_t       *data,
     cntx_t          *cntx,
     double *restrict a_p, int pack_a,
     double *restrict b_p, int pack_b
    )
{
#if DEBUG
    assert( m == 6 );
    assert( cs_b == 1 );
    assert( rs_c0 == 1 ||
            cs_c0 == 1 );
#endif
    // In-register transpose is still unsupported.
    static double ct[6 * 8];
    double zero = 0.0;
    double *c; inc_t rs_c, cs_c;
    double *beta;
    if ( cs_c0 != 1 )
    {
        c = ct;
        rs_c = 8;
        cs_c = 1;
        beta = &zero;
    }
    else
    {
        c = c0;
        rs_c = rs_c0,
        cs_c = cs_c0;
        beta = beta0;
    }

    switch ( !!pack_a << 9 | !!pack_b << 8 | n ) {
#define EXPAND_CASE_BASE( N, PA, PB, PACKA, PACKB ) \
    case ( PA << 9 | PB << 8 | N ): \
        bli_dgemmsup2_rv_haswell_asm_6x## N ##r_## PACKA ##_## PACKB \
            ( m, n, k, \
              alpha, \
              a, rs_a, cs_a, \
              b, rs_b, cs_b, \
              beta, \
              c, rs_c, cs_c, \
              data, cntx, a_p, b_p \
            ); break;
    // Final col-block. A tiles'll never be reused.
#define EXPAND_CASE2( N ) \
    case ( 1 << 9 | 1 << 8 | N ): \
      EXPAND_CASE_BASE( N, 0, 1, nopack, pack ) \
    case ( 1 << 9 | 0 << 8 | N ): \
      EXPAND_CASE_BASE( N, 0, 0, nopack, nopack )
    EXPAND_CASE2(7)
    EXPAND_CASE2(6)
    EXPAND_CASE2(5)
    EXPAND_CASE2(4)
    EXPAND_CASE2(3)
    EXPAND_CASE2(2)
    EXPAND_CASE2(1)

    default:
#ifdef DEBUG
        assert( 0 );
#endif
        break;
    }

    if ( c == ct )
    {
        for ( dim_t j = 0; j < n; ++j )
            for ( dim_t i = 0; i < m; ++i )
                c0[i * rs_c0 + j * cs_c0] =
                    c0[i * rs_c0 + j * cs_c0] * *beta0 + ct[i * 8 + j];
    }
}

#endif
