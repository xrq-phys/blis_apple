#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

#include "blis.h"
#include <assert.h>


#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"
// vpmovsxbq, vmaskmovpd was not defined.
#define vpmovsxbq(_0, _1) INSTR_(vpmovsxbq, _0, _1)
#define vmaskmovpd(_0, _1, _2) INSTR_(vmaskmovpd, _0, _1, _2)

#define min_(A, B) ((A) < (B) ? (A) : (B))

#define PACK_nopack(_1)
#define PACK_pack(_1) _1

#define VMOV_align vmovapd
#define VMOV_noalign vmovupd

#define BRANCH_noj(_1)
#define BRANCH_j(_1) _1

#define LOAD_BROW_8(BADDR,B0_,B1_,BAlign) \
    VMOV_ ## BAlign (mem(BADDR    ), ymm(B0_)) \
    VMOV_ ## BAlign (mem(BADDR, 32), ymm(B1_))

#define LOAD_BROW_6(BADDR,B0_,B1_,BAlign) \
    VMOV_ ## BAlign (mem(BADDR    ), ymm(B0_)) \
    VMOV_ ## BAlign (mem(BADDR, 32), xmm(B1_))

#define LOAD_BROW_5(BADDR,B0_,B1_,BAlign) \
    VMOV_ ## BAlign (mem(BADDR    ), ymm(B0_)) \
    vmovsd( mem(BADDR, 32), xmm(B1_))

#define LOAD_BROW_7(BADDR,B0_,B1_,BAlign) LOAD_BROW_8(BADDR,B0_,B1_,BAlign)
#define LOAD_BROW_4(BADDR,B0_,B1_,BAlign) VMOV_ ## BAlign (mem(BADDR), ymm(B0_))
#define LOAD_BROW_3(BADDR,B0_,B1_,BAlign) VMOV_ ## BAlign (mem(BADDR), ymm(B0_))
#define LOAD_BROW_2(BADDR,B0_,B1_,BAlign) VMOV_ ## BAlign (mem(BADDR), xmm(B0_))
#define LOAD_BROW_1(BADDR,B0_,B1_,BAlign) vmovsd( mem(BADDR), xmm(B0_))

#define FMA_ROW_8(INST,A_,B0_,B1_,C0_,C1_) \
    INST (ymm(A_), ymm(B0_), ymm(C0_)) \
    INST (ymm(A_), ymm(B1_), ymm(C1_))

#define FMA_ROW_6(INST,A_,B0_,B1_,C0_,C1_) \
    INST (ymm(A_), ymm(B0_), ymm(C0_)) \
    INST (xmm(A_), xmm(B1_), xmm(C1_))

#define FMA_ROW_7(INST,A_,B0_,B1_,C0_,C1_) FMA_ROW_8(INST,A_,B0_,B1_,C0_,C1_)
#define FMA_ROW_5(INST,A_,B0_,B1_,C0_,C1_) FMA_ROW_6(INST,A_,B0_,B1_,C0_,C1_)
#define FMA_ROW_4(INST,A_,B0_,B1_,C0_,C1_) INST (ymm(A_), ymm(B0_), ymm(C0_))
#define FMA_ROW_3(INST,A_,B0_,B1_,C0_,C1_) INST (ymm(A_), ymm(B0_), ymm(C0_))
#define FMA_ROW_2(INST,A_,B0_,B1_,C0_,C1_) INST (xmm(A_), xmm(B0_), xmm(C0_))
#define FMA_ROW_1(INST,A_,B0_,B1_,C0_,C1_) INST (xmm(A_), xmm(B0_), xmm(C0_))

#define DGEMM_6X8_NANOKER(N,INST,C00_,C01_,C10_,C11_,C20_,C21_,C30_,C31_,C40_,C41_,C50_,C51_,A0_,A1_,B0_,B1_,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch) \
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
    PACK_ ##PACKB (add(imm(8*8), PBADDR)) \
\
    BRANCH_ ##Branch ( dec(Counter) ) \
    BRANCH_ ##Branch ( je(.D ## ELabel ##_## ELSuffix) ) \
    LOAD_BROW_## N (BADDR,B0_,B1_,BAlign) \
    add(RSB, BADDR)

#define DGEMM_NANOKER_LOC(N,INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch) \
    DGEMM_6X8_NANOKER(N,INST,4,5,6,7,8,9,10,11,12,13,14,15,0,1,2,3,rax,RSA,RSA3,RSA5,CSA,rbx,RSB,rcx,rdx,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch)

#define BETA_nz(_1) _1
#define BETA_z(_1)

#define C1ROW_FWD_8(BETA,VFH_,VLH_,VBETA_,VMASK,CADDR,CSC) \
    BETA_ ## BETA( vfmadd231pd(mem(CADDR), ymm(VBETA_), ymm(VFH_)) ) \
    BETA_ ## BETA( vfmadd231pd(mem(CADDR, 32), ymm(VBETA_), ymm(VLH_)) ) \
    vmovupd(ymm(VFH_), mem(CADDR)) \
    vmovupd(ymm(VLH_), mem(CADDR, 32)) \
    add(CSC, CADDR)

#define C1ROW_FWD_7(BETA,VFH_,VLH_,VBETA_,VMASK,CADDR,CSC) \
    BETA_ ## BETA( vfmadd231pd(mem(CADDR), ymm(VBETA_), ymm(VFH_)) ) \
    BETA_ ## BETA( vfmadd231pd(mem(CADDR, 32), ymm(VBETA_), ymm(VLH_)) ) \
    vmovupd(ymm(VFH_), mem(CADDR)) \
    vmaskmovpd(ymm(VLH_), VMASK, mem(CADDR, 32)) \
    add(CSC, CADDR)

#define C1ROW_FWD_6(BETA,VFH_,VLH_,VBETA_,VMASK,CADDR,CSC) \
    BETA_ ## BETA( vfmadd231pd(mem(CADDR), ymm(VBETA_), ymm(VFH_)) ) \
    vmovupd(ymm(VFH_), mem(CADDR)) \
    BETA_ ## BETA( vfmadd231pd(mem(CADDR, 32), xmm(VBETA_), xmm(VLH_)) ) \
    vmovupd(xmm(VLH_), mem(CADDR, 32)) \
    add(CSC, CADDR)

#define C1ROW_FWD_5(BETA,VFH_,VLH_,VBETA_,VMASK,CADDR,CSC) \
    BETA_ ## BETA( vfmadd231pd(mem(CADDR), ymm(VBETA_), ymm(VFH_)) ) \
    vmovupd(ymm(VFH_), mem(CADDR)) \
    BETA_ ## BETA( vmovsd(mem(CADDR, 32), xmm(VFH_)) ) \
    BETA_ ## BETA( vfmadd231pd(xmm(VFH_), xmm(VBETA_), xmm(VLH_)) ) \
    vmovsd(xmm(VLH_), mem(CADDR, 32)) \
    add(CSC, CADDR)

#define C1ROW_FWD_4(BETA,VFH_,VLH_,VBETA_,VMASK,CADDR,CSC) \
    BETA_ ## BETA( vfmadd231pd(mem(CADDR), ymm(VBETA_), ymm(VFH_)) ) \
    vmovupd(ymm(VFH_), mem(CADDR)) \
    add(CSC, CADDR)

#define C1ROW_FWD_3(BETA,VFH_,VLH_,VBETA_,VMASK,CADDR,CSC) \
    BETA_ ## BETA( vfmadd231pd(mem(CADDR), ymm(VBETA_), ymm(VFH_)) ) \
    vmaskmovpd(ymm(VFH_), VMASK, mem(CADDR)) \
    add(CSC, CADDR)

#define C1ROW_FWD_2(BETA,VFH_,VLH_,VBETA_,VMASK,CADDR,CSC) \
    BETA_ ## BETA( vfmadd231pd(mem(CADDR), xmm(VBETA_), xmm(VFH_)) ) \
    vmovupd(xmm(VFH_), mem(CADDR)) \
    add(CSC, CADDR)

#define C1ROW_FWD_1(BETA,VFH_,VLH_,VBETA_,VMASK,CADDR,CSC) \
    BETA_ ## BETA( vmovsd(mem(CADDR), xmm(VLH_)) ) \
    BETA_ ## BETA( vfmadd231pd(xmm(VLH_), xmm(VBETA_), xmm(VFH_)) ) \
    vmovsd(xmm(VFH_), mem(CADDR)) \
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

// Using scratch ymm0, ymm1, ymm2, ymm3. Beta to be reloaded.
#define C1COL_STORE_1(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    DTRANSPOSE_4X4(ymm(C00_),C10,C20,C30,0,1,2,3) \
    BETA_ ## BETA ( vbroadcastsd(mem(BETAADDR), ymm3) ) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR), ymm3, ymm(C00_)) ) \
    vmovupd(ymm(C00_), mem(CADDR)) \
    DTRANSPOSE_2X4(0,1,2,C00_,C40,C50) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR4), xmm3, xmm0) ) \
    vmovupd(xmm0, mem(CADDR4))

#define C1COL_STORE_2(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    C1COL_STORE_1(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR, CSC, 1), ymm3, C10) ) \
    vmovupd(C10, mem(CADDR, CSC, 1)) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR4, CSC, 1), xmm3, xmm1) ) \
    vmovupd(xmm1, mem(CADDR4, CSC, 1))

#define C1COL_STORE_3(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    C1COL_STORE_2(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR, CSC, 2), ymm3, C20) ) \
    vmovupd(C20, mem(CADDR, CSC, 2)) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR4, CSC, 2), xmm3, xmm2) ) \
    vmovupd(xmm2, mem(CADDR4, CSC, 2))

#define C1COL_STORE_4(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    C1COL_STORE_3(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR, CSC3, 1), ymm3, C30) ) \
    vmovupd(C30, mem(CADDR, CSC3, 1)) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR4, CSC3, 1), xmm3, xmm(C00_)) ) \
    vmovupd(xmm(C00_), mem(CADDR4, CSC3, 1))

#define C1COL_STORE_5(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    C1COL_STORE_4(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    lea(mem(CADDR,  CSC, 4), CADDR) \
    lea(mem(CADDR4, CSC, 4), CADDR4) \
    DTRANSPOSE_4X4(C01,C11,C21,C31,0,1,2,C00_) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR), ymm3, C01) ) \
    vmovupd(C01, mem(CADDR)) \
    DTRANSPOSE_2X4(0,1,2,C00_,C41,C51) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR4), xmm3, xmm0) ) \
    vmovupd(xmm0, mem(CADDR4))

#define C1COL_STORE_6(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    C1COL_STORE_5(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR, CSC, 1), ymm3, C11) ) \
    vmovupd(C11, mem(CADDR, CSC, 1)) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR4, CSC, 1), xmm3, xmm1) ) \
    vmovupd(xmm1, mem(CADDR4, CSC, 1))

#define C1COL_STORE_7(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    C1COL_STORE_6(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR, CSC, 2), ymm3, C21) ) \
    vmovupd(C21, mem(CADDR, CSC, 2)) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR4, CSC, 2), xmm3, xmm2) ) \
    vmovupd(xmm2, mem(CADDR4, CSC, 2))

#define C1COL_STORE_8(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    C1COL_STORE_7(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR, CSC3, 1), ymm3, C31) ) \
    vmovupd(C31, mem(CADDR, CSC3, 1)) \
    BETA_ ## BETA ( vfmadd231pd(mem(CADDR4, CSC3, 1), xmm3, xmm(C00_)) ) \
    vmovupd(xmm(C00_), mem(CADDR4, CSC3, 1))

#define DGENMASK_1(V_,RTMP)
#define DGENMASK_2(V_,RTMP)
#define DGENMASK_4(V_,RTMP)
#define DGENMASK_5(V_,RTMP)
#define DGENMASK_6(V_,RTMP)
#define DGENMASK_7(V_,RTMP) DGENMASK_3(V_,RTMP)
#define DGENMASK_8(V_,RTMP)

#define DGENMASK_3(V_,RTMP) \
    mov(imm(0b00000000111111111111111111111111), RTMP) \
    vmovd(RTMP, xmm(V_)) \
    vpmovsxbq(xmm(V_), ymm(V_))

// Define microkernel here.
// It will be instantiated multiple times by the millikernel assembly.
#define DGEMM_6X8M_UKER_LOC(N,PACKA,PACKB,BAlign,LABEL_SUFFIX) \
    /* The microkernel code does not take care of loading a-related address.
     * The millikernel asm takes care of forwarding A & C to the location required,
     * + C addresses aught to be stored as well.
     * mov(var(a), rax)
     * mov(var(c), rdx)
     * mov(var(a_p), rcx)
     * mov(var(a_next), r8)
     * mov(var(cs_a2), r9) ** r9 = cs_a_next */ \
\
    mov(var(rs_c), rdi) \
    mov(var(cs_c), rsi) \
	cmp(imm(8), rdi) \
	jz(.DCOLPREFETCH_ ## LABEL_SUFFIX) \
		lea(mem(rdi, rdi, 2), r13) /* r13 = 3*rs_c; */ \
		lea(mem(rdx, r13, 1), r14) /* r14 = c + 3*rs_c; */ \
		prefetch(0, mem(rdx, 7*8)) /* prefetch c + 0*rs_c */ \
		prefetch(0, mem(rdx, rdi, 1, 7*8)) /* prefetch c + 1*rs_c */ \
		prefetch(0, mem(rdx, rdi, 2, 7*8)) /* prefetch c + 2*rs_c */ \
		prefetch(0, mem(r14, 7*8)) /* prefetch c + 3*rs_c */ \
		prefetch(0, mem(r14, rdi, 1, 7*8)) /* prefetch c + 4*rs_c */ \
		prefetch(0, mem(r14, rdi, 2, 7*8)) /* prefetch c + 5*rs_c */ \
		jmp(.DPREFETCHDONE_ ## LABEL_SUFFIX) \
	label(.DCOLPREFETCH_ ## LABEL_SUFFIX) \
		lea(mem(rsi, rsi, 2), r13) /* r13 = 3*cs_c; */ \
		lea(mem(rdx, r13, 1), r14) /* r14 = c + 3*cs_c; */ \
		prefetch(0, mem(rdx, 5*8)) /* prefetch c + 0*cs_c */ \
		prefetch(0, mem(rdx, rsi, 1, 5*8)) /* prefetch c + 1*cs_c */ \
		prefetch(0, mem(rdx, rsi, 2, 5*8)) /* prefetch c + 2*cs_c */ \
		prefetch(0, mem(r14, 5*8)) /* prefetch c + 3*cs_c */ \
		prefetch(0, mem(r14, rsi, 1, 5*8)) /* prefetch c + 4*cs_c */ \
		prefetch(0, mem(r14, rsi, 2, 5*8)) /* prefetch c + 5*cs_c */ \
		prefetch(0, mem(r14, r13, 1, 5*8)) /* prefetch c + 6*cs_c */ \
		prefetch(0, mem(r14, rsi, 4, 5*8)) /* prefetch c + 7*cs_c */ \
	label(.DPREFETCHDONE_ ## LABEL_SUFFIX) \
\
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
    test(rsi, rsi) \
    je(.DEMPTYALL_ ## LABEL_SUFFIX) \
    LOAD_BROW_ ## N(rbx, 2, 3, BAlign) \
    add(r11, rbx) \
\
    label(.DK_4LOOP_INIT_ ## LABEL_SUFFIX) \
        prefetch(0, mem(r8, 5*8)) \
        DGEMM_NANOKER_LOC(N,vmulpd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,_,_,_,noj) \
        prefetch(0, mem(r8, r9, 1, 5*8)) \
        DGEMM_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,_,_,_,noj) \
        prefetch(0, mem(r8, r9, 2, 5*8)) \
        DGEMM_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,_,_,_,noj) \
        prefetch(0, mem(r8, r12, 1, 5*8)) \
        lea(mem(r8, r9, 4), r8) \
        DGEMM_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,rsi,K_LEFT_LOOP_PREP,LABEL_SUFFIX,j) \
\
    label(.DK_4LOOP_ ## LABEL_SUFFIX) \
        prefetch(0, mem(r8, 5*8)) \
        DGEMM_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,_,_,_,noj) \
        prefetch(0, mem(r8, r9, 1, 5*8)) \
        DGEMM_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,_,_,_,noj) \
        prefetch(0, mem(r8, r9, 2, 5*8)) \
        DGEMM_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,_,_,_,noj) \
        prefetch(0, mem(r8, r12, 1, 5*8)) \
        lea(mem(r8, r9, 4), r8) \
        DGEMM_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,rsi,K_LEFT_LOOP_PREP,LABEL_SUFFIX,j) \
    jmp(.DK_4LOOP_ ## LABEL_SUFFIX) \
\
    label(.DEMPTYALL_ ## LABEL_SUFFIX) \
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
    label(.DK_LEFT_LOOP_PREP_ ## LABEL_SUFFIX) \
    test(r14, r14) \
    je(.DWRITEMEM_PREP_ ## LABEL_SUFFIX) \
    LOAD_BROW_ ## N(rbx, 2, 3, BAlign) \
    add(r11, rbx) \
\
    label(.DK_LEFT_LOOP_ ## LABEL_SUFFIX) \
        prefetch(0, mem(r8, 5*8)) \
        add(r9, r8) \
        DGEMM_NANOKER_LOC(N,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,r14,WRITEMEM_PREP,LABEL_SUFFIX,j) \
    jmp(.DK_LEFT_LOOP_ ## LABEL_SUFFIX) \
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
    DGENMASK_## N (3,r15) /* ymm3 is mask when n == 3 or n == 7. */ \
    vxorpd(ymm0, ymm0, ymm0) /* set ymm0 to zero. */ \
    vucomisd(xmm0, xmm2) /* set ZF if beta == 0. */ \
    je(.DBETAZERO_ ## LABEL_SUFFIX) /* if ZF = 1, jump to beta == 0 case */ \
\
        cmp(imm(8), rdi) /* set ZF if (8*rs_c) == 8. */ \
        jz(.DCOLSTORED_ ## LABEL_SUFFIX) /* jump to column storage case */ \
\
            C1ROW_FWD_ ## N(nz, 4, 5, 2, ymm3, rcx, rdi) \
            C1ROW_FWD_ ## N(nz, 6, 7, 2, ymm3, rcx, rdi) \
            C1ROW_FWD_ ## N(nz, 8, 9, 2, ymm3, rcx, rdi) \
            C1ROW_FWD_ ## N(nz, 10, 11, 2, ymm3, rcx, rdi) \
            C1ROW_FWD_ ## N(nz, 12, 13, 2, ymm3, rcx, rdi) \
            C1ROW_FWD_ ## N(nz, 14, 15, 2, ymm3, rcx, rdi) \
            jmp(.DDONE_ ## LABEL_SUFFIX) \
\
        label(.DCOLSTORED_ ## LABEL_SUFFIX) \
\
            C1COL_STORE_ ## N (nz,4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm14,ymm15,rbx,rcx,r14,rsi,r13) \
            jmp(.DDONE_ ## LABEL_SUFFIX) \
\
    label(.DBETAZERO_ ## LABEL_SUFFIX) \
\
        cmp(imm(8), rdi) /* set ZF if (8*rs_c) == 8. */ \
        jz(.DCOLSTORBZ_ ## LABEL_SUFFIX) /* jump to column storage case */ \
\
            C1ROW_FWD_ ## N(z, 4, 5, 2, ymm3, rcx, rdi) \
            C1ROW_FWD_ ## N(z, 6, 7, 2, ymm3, rcx, rdi) \
            C1ROW_FWD_ ## N(z, 8, 9, 2, ymm3, rcx, rdi) \
            C1ROW_FWD_ ## N(z, 10, 11, 2, ymm3, rcx, rdi) \
            C1ROW_FWD_ ## N(z, 12, 13, 2, ymm3, rcx, rdi) \
            C1ROW_FWD_ ## N(z, 14, 15, 2, ymm3, rcx, rdi) \
            jmp(.DDONE_ ## LABEL_SUFFIX) \
\
        label(.DCOLSTORBZ_ ## LABEL_SUFFIX) \
\
            C1COL_STORE_ ## N (z,4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm14,ymm15,rbx,rcx,r14,rsi,r13) \
\
    label(.DDONE_ ## LABEL_SUFFIX)

// Start defining the millikernel.
// This is the asm entry point for x86.
#define GENDEF(N,PACKA,BAlign_,BAlign__) \
BLIS_INLINE void bli_dgemmsup2_rv_haswell_asm_6x ## N ## m_ ## PACKA ## _ ## BAlign_ \
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
     double *restrict b_p, uint64_t pack_b \
    ) \
{ \
    const void* a_next = bli_auxinfo_next_a( data ); \
    const void* b_next = bli_auxinfo_next_b( data ); \
    uint64_t ps_a      = bli_auxinfo_ps_a( data ) << 3; \
    uint64_t ps_a_p    = bli_auxinfo_is_a( data ) << 3; /* Borrow the space. */ \
    uint64_t cs_a_next = bli_auxinfo_ps_b( data ) << 3; /* Borrow the space. */ \
    uint64_t ps_a_prfm = min_(ps_a, 64*8); /* Packed A: Avoid prefetching too far away. */ \
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
    begin_asm() \
\
    /* First (B-packing) microkernel */ \
    mov(var(a), rax) \
    mov(var(c), rdx) \
    mov(var(a_p), rcx) \
    mov(var(pack_b), rdi) \
    mov(var(m_iter), rsi) \
    cmp(imm(1), rsi) \
    je(.DM_BEFORE_INIT_NEXT_IS_FINAL) \
        mov(rax, r8) \
        add(var(ps_a2), r8) \
        mov(var(cs_a), r9) \
        jmp(.DM_BEFORE_INIT_CONTINUE) \
    label(.DM_BEFORE_INIT_NEXT_IS_FINAL) \
        mov(var(a_next), r8) \
        mov(var(cs_a2), r9) \
    label(.DM_BEFORE_INIT_CONTINUE) \
    test(rdi, rdi) \
    je(.DM_ITER) \
    DGEMM_6X8M_UKER_LOC(N,PACKA,pack,noalign,init) \
    mov(var(m_iter), rsi) \
    mov(var(a), rax) /*********** Prepare a for next uker */ \
    mov(var(ps_a), rdi) /******** Prepare a for next uker */ \
    lea(mem(rax, rdi, 1), rax) /* Prepare a for next uker */ \
    mov(var(a_p), rcx) /**** Prepare a_p for next uker */ \
    add(var(ps_a_p), rcx) /* Prepare a_p for next uker */ \
    mov(var(c),    rdx) /******** Prepare c for next uker */ \
    mov(var(rs_c), rdi) /******** Prepare c for next uker */ \
    lea(mem(rdi, rdi, 2), rdi) /* Prepare c for next uker */ \
    lea(mem(rdx, rdi, 2), rdx) /* Prepare c for next uker */ \
    cmp(imm(2), rsi) \
    je(.DM_INIT_NEXT_IS_FINAL) \
        mov(rax, r8) /******* Prepare a_next for next uker */ \
        add(var(ps_a2), r8)/* Prepare a_next for next uker */ \
        mov(var(cs_a), r9) /* Prepare cs_a_next for next uker */ \
        jmp(.DM_INIT_CONTINUE) \
    label(.DM_INIT_NEXT_IS_FINAL) \
        mov(var(a_next), r8)/* Prepare a_next for final uker. */ \
        mov(var(cs_a2), r9) /* Prepare cs_a_next for final uker. */ \
    label(.DM_INIT_CONTINUE) \
    movq(rdx, var(c)) \
    movq(rax, var(a)) \
    movq(rcx, var(a_p)) \
    dec(rsi) \
    je(.DMILLIKER_END) \
    mov(rsi, var(m_iter)) \
\
    /* Microkernels in between */ \
    label(.DM_ITER) \
    DGEMM_6X8M_UKER_LOC(N,PACKA,nopack,BAlign_,iter) \
    mov(var(m_iter), rsi) \
    mov(var(a), rax) /*********** Prepare a for next uker */ \
    mov(var(ps_a), rdi) /******** Prepare a for next uker */ \
    lea(mem(rax, rdi, 1), rax) /* Prepare a for next uker */ \
    mov(var(a_p), rcx) /**** Prepare a_p for next uker */ \
    add(var(ps_a_p), rcx) /* Prepare a_p for next uker */ \
    mov(var(c),    rdx) /******** Prepare c for next uker */ \
    mov(var(rs_c), rdi) /******** Prepare c for next uker */ \
    lea(mem(rdi, rdi, 2), rdi) /* Prepare c for next uker */ \
    lea(mem(rdx, rdi, 2), rdx) /* Prepare c for next uker */ \
    cmp(imm(2), rsi) \
    je(.DM_ITER_NEXT_IS_FINAL) \
        mov(rax, r8) /******* Prepare a_next for next uker */ \
        add(var(ps_a2), r8)/* Prepare a_next for next uker */ \
        mov(var(cs_a), r9) /* Prepare cs_a_next for next uker */ \
        jmp(.DM_ITER_CONTINUE) \
    label(.DM_ITER_NEXT_IS_FINAL) \
        mov(var(a_next), r8)/* Prepare a_next for final uker. */ \
        mov(var(cs_a2), r9) /* Prepare cs_a_next for final uker. */ \
    label(.DM_ITER_CONTINUE) \
    movq(rdx, var(c)) \
    movq(rax, var(a)) \
    movq(rcx, var(a_p)) \
    dec(rsi) \
    je(.DMILLIKER_END) \
    mov(rsi, var(m_iter)) \
    jmp(.DM_ITER) \
\
    label(.DMILLIKER_END) \
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
      [ps_a2]  "m" (ps_a_prfm), \
      [rs_c]   "m" (rs_c), \
      [cs_c]   "m" (cs_c), \
      [k_iter] "m" (k_iter), \
      [k_left] "m" (k_left), \
      [alpha]  "m" (alpha), \
      [beta]   "m" (beta), \
      [b_p]    "m" (b_p), \
      [pack_b] "m" (pack_b), \
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
    if ( m_left ) \
    { assert( 0 ); } \
\
}

// Packing milliker can have edge case unaligned? TODO: Caller exclusion.
GENDEF(8,pack,align,noalign)

GENDEF(8,nopack,align,align)
GENDEF(7,nopack,align,align)
GENDEF(6,nopack,align,align)
GENDEF(5,nopack,align,align)
GENDEF(4,nopack,align,align)
GENDEF(3,nopack,align,align)
GENDEF(2,nopack,align,align)
GENDEF(1,nopack,align,align)

GENDEF(8,nopack,noalign,noalign)
GENDEF(7,nopack,noalign,noalign)
GENDEF(6,nopack,noalign,noalign)
GENDEF(5,nopack,noalign,noalign)
GENDEF(4,nopack,noalign,noalign)
GENDEF(3,nopack,noalign,noalign)
GENDEF(2,nopack,noalign,noalign)
GENDEF(1,nopack,noalign,noalign)

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
#ifdef DEBUG
    assert( n <= 8 );
    assert( cs_b0 == 1 );
    assert( rs_c0 == 1 ||
            cs_c0 == 1 );
#endif
    const int b_align = (((uint64_t)b % (4*8)) + rs_b0 % 4) == 0;

    switch ( pack_a << 13 | b_align << 12 | n ) {
#define EXPAND_CASE(N,PACKA_C,BAlign_C,PACKA,BAlign) \
        case ( PACKA_C << 13 | BAlign_C << 12 | N ): \
            /* Fully packed. */ \
            bli_dgemmsup2_rv_haswell_asm_6x ## N ## m_## PACKA ##_## BAlign \
                ( m, n, k, alpha, \
                  a, rs_a0, cs_a0, \
                  b, rs_b0, cs_b0, beta, \
                  c, rs_c0, cs_c0, \
                  data, cntx, a_p, b_p, pack_b ); \
            break;
        case ( 1 << 13 | 1 << 12 | 8 ):
        EXPAND_CASE(8, 1, 0, pack, align) // Use aligned kernel regardless b_align.
        EXPAND_CASE(8, 0, 1, nopack, align)
        EXPAND_CASE(8, 0, 0, nopack, noalign)
        EXPAND_CASE(7, 0, 1, nopack, align) // Edge kernels require no packing.
        EXPAND_CASE(7, 0, 0, nopack, noalign)
        EXPAND_CASE(6, 0, 1, nopack, align)
        EXPAND_CASE(6, 0, 0, nopack, noalign)
        EXPAND_CASE(5, 0, 1, nopack, align)
        EXPAND_CASE(5, 0, 0, nopack, noalign)
        EXPAND_CASE(4, 0, 1, nopack, align)
        EXPAND_CASE(4, 0, 0, nopack, noalign)
        EXPAND_CASE(3, 0, 1, nopack, align)
        EXPAND_CASE(3, 0, 0, nopack, noalign)
        EXPAND_CASE(2, 0, 1, nopack, align)
        EXPAND_CASE(2, 0, 0, nopack, noalign)
        EXPAND_CASE(1, 0, 1, nopack, align)
        EXPAND_CASE(1, 0, 0, nopack, noalign)
        default:
#ifdef DEBUG
            fprintf( stderr, "error: n=%ld,pa=%d,ab=%d\n", n, pack_a, b_align );
            assert( 0 );
#endif
            break;
    }
}

#endif
