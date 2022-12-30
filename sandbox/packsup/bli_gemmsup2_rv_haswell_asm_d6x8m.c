#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

#include "blis.h"
#include <assert.h>


#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"


#define DGEMM_6X8_NANOKER(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB) \
    vmovupd(mem(BADDR    ), B0) \
    vmovupd(mem(BADDR, 32), B1) \
    add(RSB, BADDR) \
    vbroadcastsd(mem(AADDR        ), ymm(A0_)) \
    vbroadcastsd(mem(AADDR, RSA, 1), ymm(A1_)) \
    INST (ymm(A0_), B0, C00) \
    INST (ymm(A0_), B1, C01) \
    PACK_ ##PACKA (vunpcklpd(xmm(A1_), xmm(A0_), xmm(A0_))) \
    INST (ymm(A1_), B0, C10) \
    INST (ymm(A1_), B1, C11) \
    PACK_ ##PACKA (vmovapd(xmm(A0_), mem(PAADDR))) \
    vbroadcastsd(mem(AADDR, RSA, 2), ymm(A0_)) \
    vbroadcastsd(mem(AADDR, RSA3,1), ymm(A1_)) \
    INST (ymm(A0_), B0, C20) \
    INST (ymm(A0_), B1, C21) \
    PACK_ ##PACKA (vunpcklpd(xmm(A1_), xmm(A0_), xmm(A0_))) \
    INST (ymm(A1_), B0, C30) \
    INST (ymm(A1_), B1, C31) \
    PACK_ ##PACKA (vmovapd(xmm(A0_), mem(PAADDR, 2*8))) \
    vbroadcastsd(mem(AADDR, RSA, 4), ymm(A0_)) \
    vbroadcastsd(mem(AADDR, RSA5,1), ymm(A1_)) \
    add(CSA, AADDR) \
    INST (ymm(A0_), B0, C40) \
    INST (ymm(A0_), B1, C41) \
    PACK_ ##PACKA (vunpcklpd(xmm(A1_), xmm(A0_), xmm(A0_))) \
    INST (ymm(A1_), B0, C50) \
    INST (ymm(A1_), B1, C51) \
    PACK_ ##PACKA (vmovapd(xmm(A0_), mem(PAADDR, 4*8))) \
    PACK_ ##PACKA (add(imm(6*8), PAADDR)) \
    PACK_ ##PACKB (vmovapd(B0, mem(PBADDR    ))) \
    PACK_ ##PACKB (vmovapd(B1, mem(PBADDR, 32))) \
    PACK_ ##PACKB (add(imm(8*8), PBADDR))

#define PACK_nopack(_1)
#define PACK_pack(_1) _1

#define DGEMM_6X8_NANOKER_LOC(INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB) \
    DGEMM_6X8_NANOKER(INST,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm14,ymm15,0,1,ymm2,ymm3,rax,RSA,RSA3,RSA5,CSA,rbx,RSB,rcx,rdx,PACKA,PACKB)

#define C1ROW_BETA_FWD(VFH,VLH,VBETA,CADDR,CSC) \
    vfmadd231pd(mem(CADDR), VBETA, VFH) \
    vmovupd(VFH, mem(CADDR)) \
    vfmadd231pd(mem(CADDR, 32), VBETA, VLH) \
    vmovupd(VLH, mem(CADDR, 32)) \
    add(CSC, CADDR)

#define C1ROW_FWD(VFH,VLH,CADDR,CSC) \
    vmovupd(VFH, mem(CADDR)) \
    vmovupd(VLH, mem(CADDR, 32)) \
    add(CSC, CADDR)

#define DTRANSPOSE_4X4(C0,C1,C2,C3,V0_,V1_,V2_,V3_) /* C0-3: In row, out col; V0-3: Scratch. */ \
    vunpcklpd(C1, C0, ymm(V0_)) /* Unpack bientries into V0: { C0[0], C1[0], C0[2], C1[2] } */ \
    vunpckhpd(C1, C0, ymm(V1_)) /* Unpack bientries into V1: { C0[1], C1[1], C0[3], C1[3] } */ \
    vunpcklpd(C3, C2, ymm(V2_)) /* Unpack bientries into V2: { C2[0], C3[0], C2[2], C3[2] } */ \
    vunpckhpd(C3, C2, ymm(V3_)) /* Unpack bientries into V3: { C2[1], C3[1], C2[3], C3[3] } */ \
    vinsertf128(imm(0x1), xmm(V2_), ymm(V0_), C0) /* Insert first half of V2 to last half of { C0 <- V0 } */ \
    vinsertf128(imm(0x1), xmm(V3_), ymm(V1_), C1) /* Insert first half of V3 to last half of { C1 <- V1 } */ \
    vperm2f128(imm(0x31), ymm(V2_), ymm(V0_), C2) /* Shuffle last half of V0 to first half of { C2 <- V2 } */ \
    vperm2f128(imm(0x31), ymm(V3_), ymm(V1_), C3) /* Shuffle last half of V1 to first half of { C3 <- V3 } */

#define DTRANSPOSE_2X4(D0_,D1_,D2_,D3_,S0,S1) /* 2X4 row-stored (2 ymm) into col-stored (4 xmm). */ \
    vunpcklpd(S1, S0, ymm(D0_)) /* Unpack bientries into D0: { S0[0], S1[0], S0[2], S1[2] } */ \
    vunpckhpd(S1, S0, ymm(D1_)) /* Unpack bientries into D1: { S0[1], S1[1], S0[3], S1[3] } */ \
    vextractf128(imm(0x1), ymm(D0_), xmm(D2_)) /* Extract last half of D0 into D2 */ \
    vextractf128(imm(0x1), ymm(D1_), xmm(D3_)) /* Extract last half of D1 into D3 */

#define CCOL_4V_BETA(C0,C1,C2,C3,VBETA,CADDR,CSC,CSC3) \
    vfmadd231pd(mem(CADDR), VBETA, C0) /* Column 0, 0:4 */ \
    vfmadd231pd(mem(CADDR, CSC, 1), VBETA, C1) /* Column 1, 0:4 */ \
    vfmadd231pd(mem(CADDR, CSC, 2), VBETA, C2) /* Column 2, 0:4 */ \
    vfmadd231pd(mem(CADDR, CSC3, 1), VBETA, C3) /* Column 3, 0:4 */ \
    vmovupd(C0, mem(CADDR)) \
    vmovupd(C1, mem(CADDR, CSC, 1)) \
    vmovupd(C2, mem(CADDR, CSC, 2)) \
    vmovupd(C3, mem(CADDR, CSC3, 1))

#define CCOL_4V(C0,C1,C2,C3,CADDR,CSC,CSC3) \
    vmovupd(C0, mem(CADDR)) \
    vmovupd(C1, mem(CADDR, CSC, 1)) \
    vmovupd(C2, mem(CADDR, CSC, 2)) \
    vmovupd(C3, mem(CADDR, CSC3, 1))

// Define microkernel here.
// It will be instantiated multiple times by the millikernel assembly.
#define DGEMM_6X8M_UKER_LOC(PACKA,PACKB,LABEL_SUFFIX) \
    /* The microkernel code does not take care of loading a-related address.
     * The millikernel asm takes care of forwarding a to the location required,
     * *** while C addresses aught to be forwarded & stored *** .
     * mov(var(a), rax)
     * mov(var(a_p), rcx)
     * mov(var(a_next), r8)
     * mov(var(cs_a2), r9) ** r9 = cs_a_next */ \
    mov(var(b), rbx) \
    mov(var(b_p), rdx) \
    mov(var(rs_a), rdi) \
    mov(var(cs_a), r10) \
    mov(var(rs_b), r11) \
    lea(mem(rdi, rdi, 2), r13) /* r13 = 3*rs_a; */ \
    lea(mem(r13, rdi, 2), r15) /* r15 = 5*rs_a; */ \
    lea(mem(r9, r9, 2), r12) /* r12 = 3*cs_a_next; */ \
    mov(var(k_iter), rsi) \
    mov(var(k_left), r14) \
\
    /* TODO: Prefetch C. */ \
\
    test(rsi, rsi) \
    je(.DEMPTYALL_ ## LABEL_SUFFIX) \
\
    label(.DK_4LOOP_INIT_ ## LABEL_SUFFIX) \
        prefetch(0, mem(r8, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(vmulpd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r9, 1, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r9, 2, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r12, 1, 5*8)) \
        lea(mem(r8, r9, 4), r8) \
        DGEMM_6X8_NANOKER_LOC(vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        dec(rsi) \
    je(.DK_LEFT_LOOP_PREP_ ## LABEL_SUFFIX) \
\
    label(.DK_4LOOP_ ## LABEL_SUFFIX) \
        prefetch(0, mem(r8, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r9, 1, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r9, 2, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r12, 1, 5*8)) \
        lea(mem(r8, r9, 4), r8) \
        DGEMM_6X8_NANOKER_LOC(vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        dec(rsi) \
    jne(.DK_4LOOP_ ## LABEL_SUFFIX) \
\
    label(.DK_LEFT_LOOP_PREP_ ## LABEL_SUFFIX) \
    test(r14, r14) \
    je(.DWRITEMEM_PREP_ ## LABEL_SUFFIX) \
    jmp(.DK_LEFT_LOOP_ ## LABEL_SUFFIX) \
\
    label(.DEMPTYALL_ ## LABEL_SUFFIX) \
        vzeroall() \
\
    label(.DK_LEFT_LOOP_ ## LABEL_SUFFIX) \
        prefetch(0, mem(r8, 5*8)) \
        add(r9, r8) \
        DGEMM_6X8_NANOKER_LOC(vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        dec(r14) \
    jne(.DK_LEFT_LOOP_ ## LABEL_SUFFIX) \
\
    label(.DWRITEMEM_PREP_ ## LABEL_SUFFIX) \
\
    /* For millikernels:
     * New B is packed. Use it. */ \
    PACK_ ## PACKB( mov(var(b_p), rcx) ) \
    PACK_ ## PACKB( movq(rcx, var(b)) ) \
    PACK_ ## PACKB( movq(imm(8*8), var(rs_b)) ) \
\
    mov(var(alpha), rax) \
    mov(var(beta), rbx) \
    mov(var(b_next), rdx) \
    vbroadcastsd(mem(rax), ymm0) \
    vbroadcastsd(mem(rbx), ymm3) \
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
        prefetch(0, mem(rdx)) \
        /* prefetch(0, mem(rdx, 1*64)) \
         * prefetch(0, mem(rdx, 2*64)) \
         * prefetch(0, mem(rdx, 3*64)) */ \
\
    mov(var(c), rcx) /* load c */ \
    mov(var(rs_c), rdi) /* load rs_c */ \
    mov(var(cs_c), rsi) /* load cs_c */ \
\
    lea(mem(rcx, rdi, 4), r14) /* load address of c + 4*rs_c; */ \
    lea(mem(rsi, rsi, 2), r13) /* r13 = 3*cs_c; */ \
\
    vxorpd(ymm0, ymm0, ymm0) /* set ymm0 to zero. */ \
    vucomisd(xmm0, xmm3) /* set ZF if beta == 0. */ \
    je(.DBETAZERO_ ## LABEL_SUFFIX) /* if ZF = 1, jump to beta == 0 case */ \
\
        cmp(imm(8*8), rdi) /* set ZF if (8*rs_c) == 8. */ \
        jz(.DCOLSTORED_ ## LABEL_SUFFIX) /* jump to column storage case */ \
\
            C1ROW_BETA_FWD(ymm4, ymm5, ymm3, rcx, rdi) \
            C1ROW_BETA_FWD(ymm6, ymm7, ymm3, rcx, rdi) \
            C1ROW_BETA_FWD(ymm8, ymm9, ymm3, rcx, rdi) \
            C1ROW_BETA_FWD(ymm10, ymm11, ymm3, rcx, rdi) \
            C1ROW_BETA_FWD(ymm12, ymm13, ymm3, rcx, rdi) \
            C1ROW_BETA_FWD(ymm14, ymm15, ymm3, rcx, rdi) \
\
            jmp(.DDONE_ ## LABEL_SUFFIX) \
\
        label(.DCOLSTORED_ ## LABEL_SUFFIX) \
\
            DTRANSPOSE_4X4(ymm4, ymm6, ymm8, ymm10, 0, 1, 2, 3) \
            vbroadcastsd(mem(rbx), ymm3) /* Reload beta */ \
            CCOL_4V_BETA(ymm4, ymm6, ymm8, ymm10, ymm3, rcx, rsi, r13) /* mem * beta into regs */ \
            lea(mem(rcx, rsi, 4), rcx) /* rcx forward 4 cols to the next 4x4 block */ \
\
            DTRANSPOSE_2X4(0, 1, 2, 4, ymm12, ymm14) /* ymm3 holds beta. xmm4 spare now */ \
            CCOL_4V_BETA(xmm0, xmm1, xmm2, xmm4, xmm3, r14, rsi, r13) /* scale & store 2x4 block */ \
            lea(mem(r14, rsi, 4), r14) /* r14 forward 4 cols to the next 2x4 block */ \
\
            DTRANSPOSE_4X4(ymm5, ymm7, ymm9, ymm11, 0, 1, 2, 8) /* ymm3 holds beta. Use ymm4/8 */ \
            CCOL_4V_BETA(ymm5, ymm7, ymm9, ymm11, ymm3, rcx, rsi, r13) \
\
            DTRANSPOSE_2X4(0, 1, 2, 4, ymm13, ymm15) \
            CCOL_4V_BETA(xmm0, xmm1, xmm2, xmm4, xmm3, r14, rsi, r13) \
\
            jmp(.DDONE_ ## LABEL_SUFFIX) \
\
    label(.DBETAZERO_ ## LABEL_SUFFIX) \
\
        cmp(imm(8*8), rdi) /* set ZF if (8*rs_c) == 8. */ \
        jz(.DCOLSTORBZ_ ## LABEL_SUFFIX) /* jump to column storage case */ \
\
            C1ROW_FWD(ymm4, ymm5, rcx, rdi) \
            C1ROW_FWD(ymm6, ymm7, rcx, rdi) \
            C1ROW_FWD(ymm8, ymm9, rcx, rdi) \
            C1ROW_FWD(ymm10, ymm11, rcx, rdi) \
            C1ROW_FWD(ymm12, ymm13, rcx, rdi) \
            C1ROW_FWD(ymm14, ymm15, rcx, rdi) \
\
            jmp(.DDONE_ ## LABEL_SUFFIX) \
\
        label(.DCOLSTORBZ_ ## LABEL_SUFFIX) \
\
            DTRANSPOSE_4X4(ymm4, ymm6, ymm8, ymm10, 0, 1, 2, 3) \
            CCOL_4V(ymm4, ymm6, ymm8, ymm10, rcx, rsi, r13) \
            lea(mem(rcx, rsi, 4), rcx) \
\
            DTRANSPOSE_2X4(0, 1, 2, 4, ymm12, ymm14) \
            CCOL_4V(xmm0, xmm1, xmm2, xmm4, r14, rsi, r13) \
            lea(mem(r14, rsi, 4), r14) \
\
            DTRANSPOSE_4X4(ymm5, ymm7, ymm9, ymm11, 0, 1, 2, 8) \
            CCOL_4V(ymm5, ymm7, ymm9, ymm11, rcx, rsi, r13) \
\
            DTRANSPOSE_2X4(0, 1, 2, 4, ymm13, ymm15) \
            CCOL_4V(xmm0, xmm1, xmm2, xmm4, r14, rsi, r13) \
\
    label(.DDONE_ ## LABEL_SUFFIX)

// Start defining the millikernel.
// This is the asm entry point for x86.
#define GENDEF(PACKA) \
BLIS_INLINE void bli_dgemmsup2_rv_haswell_asm_6x8m_ ## PACKA \
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
    ) \
{ \
    const void* a_next = bli_auxinfo_next_a( data ); \
    const void* b_next = bli_auxinfo_next_b( data ); \
    uint64_t ps_a      = bli_auxinfo_ps_a( data ) << 3; \
    uint64_t ps_a_p    = bli_auxinfo_is_a( data ) << 3; /* Borrow the space. */ \
    uint64_t cs_a_next = bli_auxinfo_ps_b( data ) << 3; /* Borrow the space. */ \
\
    /* Typecast local copies of integers in case dim_t and inc_t are a
     * different size than is expected by load instructions. */ \
    uint64_t m_iter = m / 6; \
    uint64_t m_left = m % 6; \
    uint64_t k_iter = k / 4; \
    uint64_t k_left = k % 4; \
    uint64_t rs_a   = rs_a0 << 3; \
    uint64_t cs_a   = cs_a0 << 3; \
    uint64_t rs_b   = rs_b0 << 3; \
    uint64_t rs_c   = rs_c0 << 3; \
    uint64_t cs_c   = cs_c0 << 3; \
\
    /* Spare the first(?) m_iter for doing B-packing */ \
    if ( !m_left ) \
    { \
      m_iter--; \
      m_left = 8; \
    } \
\
    begin_asm() \
\
    /* First (B-packing) microkernel */ \
    mov(var(a), rax) \
    mov(var(a_p), rcx) \
    mov(rax, r8) \
    add(var(ps_a), r8) \
    mov(var(cs_a), r9) \
    mov(var(m_iter), rsi) \
    test(rsi, rsi) \
    je(.DM_LEFT) \
    DGEMM_6X8M_UKER_LOC(PACKA,pack,init) \
    mov(var(m_iter), rsi) \
    mov(var(a), rax) /*********** Prepare a for next uker */ \
    mov(var(ps_a), rdi) /******** Prepare a for next uker */ \
    lea(mem(rax, rdi, 1), rax) /* Prepare a for next uker */ \
    lea(mem(rax, rdi, 1), r8) /* Prepare a_next for next uker */ \
    mov(var(cs_a), r9) /* Prepare cs_a_next for next uker */ \
    mov(var(a_p), rcx) /*** Prepare a_p for next uker */ \
    add(var(ps_a_p), rcx) /* Prepare a_p for next uker */ \
    mov(var(c),    rdx) /******** Prepare c for next uker */ \
    mov(var(rs_c), rdi) /******** Prepare c for next uker */ \
    lea(mem(rdi, rdi, 2), rdi) /* Prepare c for next uker */ \
    lea(mem(rdx, rdi, 2), rdx) /* Prepare c for next uker */ \
    movq(rdx, var(c)) /********** Prepare c for next uker */ \
    movq(rax, var(a)) \
    movq(rcx, var(a_p)) \
    dec(rsi) \
    je(.DM_LEFT) \
    mov(rsi, var(m_iter)) \
\
    /* Microkernels in between */ \
    label(.DM_ITER) \
    DGEMM_6X8M_UKER_LOC(PACKA,nopack,iter) \
    mov(var(m_iter), rsi) \
    mov(var(a), rax) /*********** Prepare a for next uker */ \
    mov(var(ps_a), rdi) /******** Prepare a for next uker */ \
    lea(mem(rax, rdi, 1), rax) /* Prepare a for next uker */ \
    lea(mem(rax, rdi, 1), r8) /* Prepare a_next for next uker */ \
    mov(var(cs_a), r9) /* Prepare cs_a_next for next uker */ \
    mov(var(a_p), rcx) /*** Prepare a_p for next uker */ \
    add(var(ps_a_p), rcx) /* Prepare a_p for next uker */ \
    mov(var(c),    rdx) /******** Prepare c for next uker */ \
    mov(var(rs_c), rdi) /******** Prepare c for next uker */ \
    lea(mem(rdi, rdi, 2), rdi) /* Prepare c for next uker */ \
    lea(mem(rdx, rdi, 2), rdx) /* Prepare c for next uker */ \
    movq(rdx, var(c)) /********** Prepare c for next uker */ \
    movq(rax, var(a)) \
    movq(rcx, var(a_p)) \
    dec(rsi) \
    je(.DM_LEFT) \
    mov(rsi, var(m_iter)) \
    jmp(.DM_ITER) \
\
    /* Final microkernel */ \
    label(.DM_LEFT) \
    mov(var(a_next), r8) /* Override a_next w/ next millikernel. */ \
    mov(var(cs_a2), r9) /* Override cs_a_next w/ next millikernel. */ \
    /* TODO: Support for edge cases. */ \
    DGEMM_6X8M_UKER_LOC(PACKA,nopack,final) \
\
    end_asm( \
    : [a]     "+m" (a), \
      [b]     "+m" (b), \
      [rs_b]  "+m" (rs_b), \
      [c]     "+m" (c), \
      [a_p]   "+m" (a_p), \
      [m_iter]"+m" (m_iter) \
    : [m_left] "m" (m_left), \
      [rs_a]   "m" (rs_a), \
      [cs_a]   "m" (cs_a), \
      [cs_a2]  "m" (cs_a_next), \
      [ps_a]   "m" (ps_a), \
      [ps_a_p] "m" (ps_a_p), \
      [rs_c]   "m" (rs_c), \
      [cs_c]   "m" (cs_c), \
      [k_iter] "m" (k_iter), \
      [k_left] "m" (k_left), \
      [alpha]  "m" (alpha), \
      [beta]   "m" (beta), \
      [b_p]    "m" (b_p), \
      [a_next] "m" (a_next), \
      [b_next] "m" (b_next) \
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

GENDEF(pack)
GENDEF(nopack)

#if 1
void bli_dgemmsup2_rv_haswell_asm_6x8m
    (
     dim_t            m,
     dim_t            n,
     dim_t            k,
     double *restrict alpha,
     double *restrict a, inc_t rs_a0, inc_t cs_a0,
     double *restrict b, inc_t rs_b0, inc_t cs_b0,
     double *restrict beta,
     double *restrict c, inc_t rs_c0, inc_t cs_c0,
     auxinfo_t       *data,
     cntx_t          *cntx,
     double *restrict a_p, int pack_a,
     double *restrict b_p, int pack_b
    )
{
    assert( n == 8 );
    assert( m % 6 == 0 );
    assert( cs_b0 == 1 );
    assert( rs_c0 == 1 ||
            cs_c0 == 1 );

    if ( pack_a )
        bli_dgemmsup2_rv_haswell_asm_6x8m_pack
            ( m, n, k, alpha,
              a, rs_a0, cs_a0,
              b, rs_b0, cs_b0, beta,
              c, rs_c0, cs_c0,
              data, cntx, a_p, b_p );
    else
        bli_dgemmsup2_rv_haswell_asm_6x8m_nopack
            ( m, n, k, alpha,
              a, rs_a0, cs_a0,
              b, rs_b0, cs_b0, beta,
              c, rs_c0, cs_c0,
              data, cntx, a_p, b_p );
}
#endif

#endif
