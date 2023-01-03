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

#define DGEMM_6X8_NANOKER(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch) \
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
    PACK_ ##PACKB (add(imm(8*8), PBADDR)) \
\
    BRANCH_ ##Branch ( dec(Counter) ) \
    BRANCH_ ##Branch ( je(.D ## ELabel ##_## ELSuffix) ) \
    VMOV_ ## BAlign (mem(BADDR    ), B0) \
    VMOV_ ## BAlign (mem(BADDR, 32), B1) \
    add(RSB, BADDR)

#define DGEMM_MX8_NANOKER_ALINE_1(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    vbroadcastsd(mem(AADDR), ymm(A0_)) \
    INST (ymm(A0_), B0, C00) \
    INST (ymm(A0_), B1, C01) \
    PACK_ ##PACKA (vmovsd(xmm(A0_), mem(PAADDR)))

#define DGEMM_MX8_NANOKER_ALINE_2(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    vbroadcastsd(mem(AADDR        ), ymm(A0_)) \
    vbroadcastsd(mem(AADDR, RSA, 1), ymm(A1_)) \
    INST (ymm(A0_), B0, C00) \
    INST (ymm(A0_), B1, C01) \
    PACK_ ##PACKA (vunpcklpd(xmm(A1_), xmm(A0_), xmm(A0_))) \
    INST (ymm(A1_), B0, C10) \
    INST (ymm(A1_), B1, C11) \
    PACK_ ##PACKA (vmovapd(xmm(A0_), mem(PAADDR)))

#define DGEMM_MX8_NANOKER_ALINE_3(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    DGEMM_MX8_NANOKER_ALINE_2(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    vbroadcastsd(mem(AADDR, RSA, 2), ymm(A0_)) \
    INST (ymm(A0_), B0, C20) \
    INST (ymm(A0_), B1, C21) \
    PACK_ ##PACKA (vmovsd(xmm(A0_), mem(PAADDR, 2*8)))

#define DGEMM_MX8_NANOKER_ALINE_4(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    DGEMM_MX8_NANOKER_ALINE_2(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    vbroadcastsd(mem(AADDR, RSA, 2), ymm(A0_)) \
    vbroadcastsd(mem(AADDR, RSA3,1), ymm(A1_)) \
    INST (ymm(A0_), B0, C20) \
    INST (ymm(A0_), B1, C21) \
    PACK_ ##PACKA (vunpcklpd(xmm(A1_), xmm(A0_), xmm(A0_))) \
    INST (ymm(A1_), B0, C30) \
    INST (ymm(A1_), B1, C31) \
    PACK_ ##PACKA (vmovapd(xmm(A0_), mem(PAADDR, 2*8)))

#define DGEMM_MX8_NANOKER_ALINE_5(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    DGEMM_MX8_NANOKER_ALINE_4(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    vbroadcastsd(mem(AADDR, RSA, 4), ymm(A0_)) \
    INST (ymm(A0_), B0, C40) \
    INST (ymm(A0_), B1, C41) \
    PACK_ ##PACKA (vmovsd(xmm(A0_), mem(PAADDR, 4*8)))

#define DGEMM_MX8_NANOKER(M,INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch) \
    DGEMM_MX8_NANOKER_ALINE_## M (INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    add(CSA, AADDR) \
    PACK_ ##PACKA (add(imm(6*8), PAADDR)) \
    PACK_ ##PACKB (vmovapd(B0, mem(PBADDR    ))) \
    PACK_ ##PACKB (vmovapd(B1, mem(PBADDR, 32))) \
    PACK_ ##PACKB (add(imm(8*8), PBADDR)) \
\
    BRANCH_ ##Branch ( dec(Counter) ) \
    BRANCH_ ##Branch ( je(.D ## ELabel ##_## ELSuffix) ) \
    VMOV_ ## BAlign (mem(BADDR    ), B0) \
    VMOV_ ## BAlign (mem(BADDR, 32), B1) \
    add(RSB, BADDR)

#define DGEMM_6X8_NANOKER_LOC(INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch) \
    DGEMM_6X8_NANOKER(INST,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm14,ymm15,0,1,ymm2,ymm3,rax,RSA,RSA3,RSA5,CSA,rbx,RSB,rcx,rdx,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch)

#define DGEMM_MX8_NANOKER_LOC(M,INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch) \
    DGEMM_MX8_NANOKER(M,INST,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm14,ymm15,0,1,ymm2,ymm3,rax,RSA,RSA3,RSA5,CSA,rbx,RSB,rcx,rdx,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch)

#define DGEMM_MX8_NANOKER_LOC_1(INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch) DGEMM_MX8_NANOKER_LOC(1,INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch)
#define DGEMM_MX8_NANOKER_LOC_2(INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch) DGEMM_MX8_NANOKER_LOC(2,INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch)
#define DGEMM_MX8_NANOKER_LOC_3(INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch) DGEMM_MX8_NANOKER_LOC(3,INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch)
#define DGEMM_MX8_NANOKER_LOC_4(INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch) DGEMM_MX8_NANOKER_LOC(4,INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch)
#define DGEMM_MX8_NANOKER_LOC_5(INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch) DGEMM_MX8_NANOKER_LOC(5,INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch)
#define DGEMM_MX8_NANOKER_LOC_6(INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch) DGEMM_6X8_NANOKER_LOC(INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch)
#define DGEMM_NANOKER_LOC(M,INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch) DGEMM_MX8_NANOKER_LOC_## M(INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB,BAlign,Counter,ELabel,ELSuffix,Branch)

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

#define CSTORE_LOC_1(CADDR,RSC) \
    C1ROW_BETA_FWD(ymm4, ymm5, ymm3, CADDR, RSC)
#define CSTORE_LOC_2(CADDR,RSC) \
    CSTORE_LOC_1(CADDR,RSC) \
    C1ROW_BETA_FWD(ymm6, ymm7, ymm3, CADDR, RSC)
#define CSTORE_LOC_3(CADDR,RSC) \
    CSTORE_LOC_2(CADDR,RSC) \
    C1ROW_BETA_FWD(ymm8, ymm9, ymm3, CADDR, RSC)
#define CSTORE_LOC_4(CADDR,RSC) \
    CSTORE_LOC_3(CADDR,RSC) \
    C1ROW_BETA_FWD(ymm10, ymm11, ymm3, CADDR, RSC)
#define CSTORE_LOC_5(CADDR,RSC) \
    CSTORE_LOC_4(CADDR,RSC) \
    C1ROW_BETA_FWD(ymm12, ymm13, ymm3, CADDR, RSC)
#define CSTORE_LOC_6(CADDR,RSC) \
    CSTORE_LOC_5(CADDR,RSC) \
    C1ROW_BETA_FWD(ymm14, ymm15, ymm3, CADDR, RSC)

#define CSTOREZB_LOC_1(CADDR,RSC) \
    C1ROW_FWD(ymm4, ymm5, CADDR, RSC)
#define CSTOREZB_LOC_2(CADDR,RSC) \
    CSTOREZB_LOC_1(CADDR,RSC) \
    C1ROW_FWD(ymm6, ymm7, CADDR, RSC)
#define CSTOREZB_LOC_3(CADDR,RSC) \
    CSTOREZB_LOC_2(CADDR,RSC) \
    C1ROW_FWD(ymm8, ymm9, CADDR, RSC)
#define CSTOREZB_LOC_4(CADDR,RSC) \
    CSTOREZB_LOC_3(CADDR,RSC) \
    C1ROW_FWD(ymm10, ymm11, CADDR, RSC)
#define CSTOREZB_LOC_5(CADDR,RSC) \
    CSTOREZB_LOC_4(CADDR,RSC) \
    C1ROW_FWD(ymm12, ymm13, CADDR, RSC)
#define CSTOREZB_LOC_6(CADDR,RSC) \
    CSTOREZB_LOC_5(CADDR,RSC) \
    C1ROW_FWD(ymm14, ymm15, CADDR, RSC)

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

#define CCOL_4V_nz(C0,C1,C2,C3,VBETA,CADDR,CSC,CSC3) \
    vfmadd231pd(mem(CADDR), VBETA, C0) /* Column 0, 0:4 */ \
    vfmadd231pd(mem(CADDR, CSC, 1), VBETA, C1) /* Column 1, 0:4 */ \
    vfmadd231pd(mem(CADDR, CSC, 2), VBETA, C2) /* Column 2, 0:4 */ \
    vfmadd231pd(mem(CADDR, CSC3, 1), VBETA, C3) /* Column 3, 0:4 */ \
    vmovupd(C0, mem(CADDR)) \
    vmovupd(C1, mem(CADDR, CSC, 1)) \
    vmovupd(C2, mem(CADDR, CSC, 2)) \
    vmovupd(C3, mem(CADDR, CSC3, 1))

#define CCOL_4V_z(C0,C1,C2,C3,VBETA,CADDR,CSC,CSC3) \
    vmovupd(C0, mem(CADDR)) \
    vmovupd(C1, mem(CADDR, CSC, 1)) \
    vmovupd(C2, mem(CADDR, CSC, 2)) \
    vmovupd(C3, mem(CADDR, CSC3, 1))

#define BETA_nz(_1) _1
#define BETA_z(_1)

// In-reg transpose needs implementing case-by-case.
#define CSTORECOL_LOC_6(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40_,C41_,C50_,C51_,VBETA_,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    DTRANSPOSE_4X4(ymm(C00_), C10, C20, C30, 0, 1, 2, VBETA_) \
    BETA_ ## BETA ( vbroadcastsd(mem(BETAADDR), ymm(VBETA_)) ) /* Reload beta */ \
    CCOL_4V_ ## BETA (ymm(C00_), C10, C20, C30, ymm(VBETA_), CADDR, CSC, CSC3) \
    lea(mem(CADDR, CSC, 4), CADDR) /* CADDR forward 4 cols to the next 4x4 block */ \
    DTRANSPOSE_2X4(0, 1, 2, C00_, ymm(C40_), ymm(C50_)) /* VBETA holds beta. Use spare */ \
    CCOL_4V_ ## BETA (xmm0, xmm1, xmm2, xmm(C00_), xmm(VBETA_), CADDR4, CSC, CSC3) \
    lea(mem(CADDR4, CSC, 4), CADDR4) /* CADDR4 forward 4 cols to the next 2x4 block */ \
    DTRANSPOSE_4X4(C01, C11, C21, C31, 0, 1, 2, C00_) \
    CCOL_4V_ ## BETA (C01, C11, C21, C31, ymm(VBETA_), CADDR, CSC, CSC3) \
    DTRANSPOSE_2X4(0, 1, 2, C00_, ymm(C41_), ymm(C51_)) \
    CCOL_4V_ ## BETA (xmm0, xmm1, xmm2, xmm(C00_), xmm(VBETA_), CADDR4, CSC, CSC3)

#define CSTORECOL_LOC_5(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40_,C41_,C50_,C51_,VBETA_,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    /* Here: ymm0-2 used additionally as scratch. Don't occupy them outside. */ \
    /* Transpose bulk part. */ \
    DTRANSPOSE_4X4(ymm(C00_),C10,C20,C30,0,1,2,C50_) \
    DTRANSPOSE_4X4(    C01,  C11,C21,C31,0,1,2,C50_) \
    /* Store block 1. */ \
    CCOL_4V_ ## BETA (ymm(C00_),C10,C20,C30,ymm(VBETA_),CADDR,CSC,CSC3) \
    lea(mem(CADDR, CSC, 4), CADDR) \
    vpermpd(imm(0b01010101), ymm(C40_), ymm0) \
    vpermpd(imm(0b10101010), ymm(C40_), ymm1) \
    vpermpd(imm(0b11111111), ymm(C40_), ymm2) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4), xmm(C50_)) ) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4, CSC, 1), xmm(C51_)) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C50_), xmm(VBETA_), xmm(C40_)) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C51_), xmm(VBETA_), xmm0) ) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4, CSC, 2), xmm(C50_)) ) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4, CSC3, 1), xmm(C51_)) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C50_), xmm(VBETA_), xmm1) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C51_), xmm(VBETA_), xmm2) ) \
    vmovsd(xmm(C40_), mem(CADDR4)) \
    vmovsd(xmm0, mem(CADDR4, CSC, 1)) \
    vmovsd(xmm1, mem(CADDR4, CSC, 2)) \
    vmovsd(xmm2, mem(CADDR4, CSC3, 1)) \
    lea(mem(CADDR4, CSC, 4), CADDR4) \
    /* Store block 2. */ \
    CCOL_4V_ ## BETA (C01,C11,C21,C31,ymm(VBETA_),CADDR,CSC,CSC3) \
    vpermpd(imm(0b01010101), ymm(C41_), ymm0) \
    vpermpd(imm(0b10101010), ymm(C41_), ymm1) \
    vpermpd(imm(0b11111111), ymm(C41_), ymm2) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4), xmm(C50_)) ) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4, CSC, 1), xmm(C51_)) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C50_), xmm(VBETA_), xmm(C41_)) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C51_), xmm(VBETA_), xmm0) ) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4, CSC, 2), xmm(C50_)) ) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4, CSC3, 1), xmm(C51_)) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C50_), xmm(VBETA_), xmm1) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C51_), xmm(VBETA_), xmm2) ) \
    vmovsd(xmm(C41_), mem(CADDR4)) \
    vmovsd(xmm0, mem(CADDR4, CSC, 1)) \
    vmovsd(xmm1, mem(CADDR4, CSC, 2)) \
    vmovsd(xmm2, mem(CADDR4, CSC3, 1))

#define CSTORECOL_LOC_4(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40_,C41_,C50_,C51_,VBETA_,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    /* Transpose bulk part. */ \
    DTRANSPOSE_4X4(ymm(C00_),C10,C20,C30,0,1,2,C50_) \
    DTRANSPOSE_4X4(    C01,  C11,C21,C31,0,1,2,C50_) \
    CCOL_4V_ ## BETA (ymm(C00_),C10,C20,C30,ymm(VBETA_),CADDR,CSC,CSC3) \
    lea(mem(CADDR, CSC, 4), CADDR) \
    CCOL_4V_ ## BETA (C01,C11,C21,C31,ymm(VBETA_),CADDR,CSC,CSC3)

#define CSTORECOL_LOC_3(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40_,C41_,C50_,C51_,VBETA_,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    /* Transpose bulk part. */ \
    DTRANSPOSE_4X4(ymm(C00_),C10,C20,C30,0,1,2,C50_) \
    DTRANSPOSE_4X4(    C01,  C11,C21,C31,0,1,2,C50_) \
    /* Prepare mask */ \
    mov(imm(0b00000000111111111111111111111111), CADDR4) \
    vmovd(CADDR4, xmm(C50_)) \
    vpmovsxbq(xmm(C50_), ymm(C50_)) \
    /* Store block 1. */ \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR), ymm(C50_), ymm0) ) \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR, CSC, 1), ymm(C50_), ymm1) ) \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR, CSC, 2), ymm(C50_), ymm2) ) \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR, CSC3, 1), ymm(C50_), ymm(C51_)) ) \
    BETA_ ## BETA( vfmadd231pd(ymm0, ymm(VBETA_), ymm(C00_)) ) \
    BETA_ ## BETA( vfmadd231pd(ymm1, ymm(VBETA_),     C10) ) \
    BETA_ ## BETA( vfmadd231pd(ymm2, ymm(VBETA_),     C20) ) \
    BETA_ ## BETA( vfmadd231pd(ymm(C51_), ymm(VBETA_), C30) ) \
    vmaskmovpd(ymm(C00_), ymm(C50_), mem(CADDR)) \
    vmaskmovpd(    C10,   ymm(C50_), mem(CADDR, CSC, 1)) \
    vmaskmovpd(    C20,   ymm(C50_), mem(CADDR, CSC, 2)) \
    vmaskmovpd(    C30,   ymm(C50_), mem(CADDR, CSC3, 1)) \
    lea(mem(CADDR, CSC, 4), CADDR) \
    /* Store block 2. */ \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR), ymm(C50_), ymm0) ) \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR, CSC, 1), ymm(C50_), ymm1) ) \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR, CSC, 2), ymm(C50_), ymm2) ) \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR, CSC3, 1), ymm(C50_), ymm(C51_)) ) \
    BETA_ ## BETA( vfmadd231pd(ymm0, ymm(VBETA_), C01) ) \
    BETA_ ## BETA( vfmadd231pd(ymm1, ymm(VBETA_), C11) ) \
    BETA_ ## BETA( vfmadd231pd(ymm2, ymm(VBETA_), C21) ) \
    BETA_ ## BETA( vfmadd231pd(ymm(C51_), ymm(VBETA_), C31) ) \
    vmaskmovpd(C01, ymm(C50_), mem(CADDR)) \
    vmaskmovpd(C11, ymm(C50_), mem(CADDR, CSC, 1)) \
    vmaskmovpd(C21, ymm(C50_), mem(CADDR, CSC, 2)) \
    vmaskmovpd(C31, ymm(C50_), mem(CADDR, CSC3, 1))

#define CSTORECOL_LOC_2(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40_,C41_,C50_,C51_,VBETA_,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    /* Transpose bulk part. */ \
    DTRANSPOSE_2X4(0,1,2,C50_,ymm(C00_),C10) \
    CCOL_4V_ ## BETA (xmm0,xmm1,xmm2,xmm(C50_),xmm(VBETA_),CADDR,CSC,CSC3) \
    lea(mem(CADDR, CSC, 4), CADDR) \
    DTRANSPOSE_2X4(0,1,2,C50_,C01,C11) \
    CCOL_4V_ ## BETA (xmm0,xmm1,xmm2,xmm(C50_),xmm(VBETA_),CADDR,CSC,CSC3)

#define CSTORECOL_LOC_1(BETA,C00_,C01,C10,C11,C20,C21,C30,C31,C40_,C41_,C50_,C51_,VBETA_,BETAADDR,CADDR,CADDR4,CSC,CSC3) \
    /* Block 1. */ \
    vpermpd(imm(0b01010101), ymm(C00_), ymm0) \
    vpermpd(imm(0b10101010), ymm(C00_), ymm1) \
    vpermpd(imm(0b11111111), ymm(C00_), ymm2) \
    BETA_ ## BETA( vmovsd(mem(CADDR), xmm(C40_)) ) \
    BETA_ ## BETA( vmovsd(mem(CADDR, CSC, 1), xmm(C41_)) ) \
    BETA_ ## BETA( vmovsd(mem(CADDR, CSC, 2), xmm(C50_)) ) \
    BETA_ ## BETA( vmovsd(mem(CADDR, CSC3, 1), xmm(C51_)) ) \
    BETA_ ## BETA( vfmadd231pd(xmm(C41_), xmm(VBETA_), xmm0) ) \
    BETA_ ## BETA( vfmadd231pd(xmm(C50_), xmm(VBETA_), xmm1) ) \
    BETA_ ## BETA( vfmadd231pd(xmm(C51_), xmm(VBETA_), xmm2) ) \
    vmovsd( xmm0, mem(CADDR, CSC, 1) ) \
    vmovsd( xmm1, mem(CADDR, CSC, 2) ) \
    vmovsd( xmm2, mem(CADDR, CSC3, 1) ) \
    vmovapd(ymm(C00_), ymm0) \
    BETA_ ## BETA( vfmadd231pd(xmm(C40_), xmm(VBETA_), xmm0) ) \
    vmovsd( xmm0, mem(CADDR) ) \
    lea(mem(CADDR, CSC, 4), CADDR) \
    /* Block 2. */ \
    vpermpd(imm(0b01010101), C01, ymm0) \
    vpermpd(imm(0b10101010), C01, ymm1) \
    vpermpd(imm(0b11111111), C01, ymm2) \
    BETA_ ## BETA( vmovsd(mem(CADDR), xmm(C40_)) ) \
    BETA_ ## BETA( vmovsd(mem(CADDR, CSC, 1), xmm(C41_)) ) \
    BETA_ ## BETA( vmovsd(mem(CADDR, CSC, 2), xmm(C50_)) ) \
    BETA_ ## BETA( vmovsd(mem(CADDR, CSC3, 1), xmm(C51_)) ) \
    BETA_ ## BETA( vfmadd231pd(xmm(C41_), xmm(VBETA_), xmm0) ) \
    BETA_ ## BETA( vfmadd231pd(xmm(C50_), xmm(VBETA_), xmm1) ) \
    BETA_ ## BETA( vfmadd231pd(xmm(C51_), xmm(VBETA_), xmm2) ) \
    vmovsd( xmm0, mem(CADDR, CSC, 1) ) \
    vmovsd( xmm1, mem(CADDR, CSC, 2) ) \
    vmovsd( xmm2, mem(CADDR, CSC3, 1) ) \
    vmovapd(C01, ymm0) \
    BETA_ ## BETA( vfmadd231pd(xmm(C40_), xmm(VBETA_), xmm0) ) \
    vmovsd( xmm0, mem(CADDR) )

// Define microkernel here.
// It will be instantiated multiple times by the millikernel assembly.
#define DGEMM_6X8N_UKER_LOC(M,PACKA,PACKB,BAlign,LABEL_SUFFIX) \
    /* The microkernel code does not take care of loading B-related address.
     * The millikernel asm takes care of forwarding B & C to the location required,
     * + C addresses aught to be stored as well.
     * mov(var(b), rbx)
     * mov(var(c), rcx)
     * mov(var(b_p), rdx)
     * mov(var(b_next), r8)
     * mov(var(rs_b2), r9) ** r9 = rs_b_next */ \
\
    mov(var(rs_c), rdi) \
    mov(var(cs_c), rsi) \
	cmp(imm(8), rdi) \
	jz(.DCOLPREFETCH_ ## LABEL_SUFFIX) \
		lea(mem(rdi, rdi, 2), r13) /* r13 = 3*rs_c; */ \
		lea(mem(rcx, r13, 1), r14) /* r14 = c + 3*rs_c; */ \
		prefetch(0, mem(rcx, 7*8)) /* prefetch c + 0*rs_c */ \
		prefetch(0, mem(rcx, rdi, 1, 7*8)) /* prefetch c + 1*rs_c */ \
		prefetch(0, mem(rcx, rdi, 2, 7*8)) /* prefetch c + 2*rs_c */ \
		prefetch(0, mem(r14, 7*8)) /* prefetch c + 3*rs_c */ \
		prefetch(0, mem(r14, rdi, 1, 7*8)) /* prefetch c + 4*rs_c */ \
		prefetch(0, mem(r14, rdi, 2, 7*8)) /* prefetch c + 5*rs_c */ \
		jmp(.DPREFETCHDONE_ ## LABEL_SUFFIX) \
	label(.DCOLPREFETCH_ ## LABEL_SUFFIX) \
		lea(mem(rsi, rsi, 2), r13) /* r13 = 3*cs_c; */ \
		lea(mem(rcx, r13, 1), r14) /* r14 = c + 3*cs_c; */ \
		prefetch(0, mem(rcx, 5*8)) /* prefetch c + 0*cs_c */ \
		prefetch(0, mem(rcx, rsi, 1, 5*8)) /* prefetch c + 1*cs_c */ \
		prefetch(0, mem(rcx, rsi, 2, 5*8)) /* prefetch c + 2*cs_c */ \
		prefetch(0, mem(r14, 5*8)) /* prefetch c + 3*cs_c */ \
		prefetch(0, mem(r14, rsi, 1, 5*8)) /* prefetch c + 4*cs_c */ \
		prefetch(0, mem(r14, rsi, 2, 5*8)) /* prefetch c + 5*cs_c */ \
		prefetch(0, mem(r14, r13, 1, 5*8)) /* prefetch c + 6*cs_c */ \
		prefetch(0, mem(r14, rsi, 4, 5*8)) /* prefetch c + 7*cs_c */ \
	label(.DPREFETCHDONE_ ## LABEL_SUFFIX) \
\
    mov(var(a), rax) \
    mov(var(a_p), rcx) \
    mov(var(rs_a), rdi) \
    mov(var(cs_a), r10) \
    mov(var(rs_b), r11) \
    lea(mem(rdi, rdi, 2), r13) /* r13 = 3*rs_a; */ \
    lea(mem(r13, rdi, 2), r15) /* r15 = 5*rs_a; */ \
    lea(mem(r9, r9, 2), r12) /* r12 = 3*rs_b_next; */ \
    mov(var(k_iter), rsi) \
    mov(var(k_left), r14) \
\
    test(rsi, rsi) \
    je(.DEMPTYALL_ ## LABEL_SUFFIX) \
    VMOV_ ## BAlign (mem(rbx), ymm2) \
    VMOV_ ## BAlign (mem(rbx, 32), ymm3) \
    add(r11, rbx) \
\
    label(.DK_4LOOP_INIT_ ## LABEL_SUFFIX) \
        prefetch(0, mem(r8, 7*8)) \
        DGEMM_NANOKER_LOC(M,vmulpd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,_,_,_,noj) \
        prefetch(0, mem(r8, r9, 1, 7*8)) \
        DGEMM_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,_,_,_,noj) \
        prefetch(0, mem(r8, r9, 2, 7*8)) \
        DGEMM_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,_,_,_,noj) \
        prefetch(0, mem(r8, r12, 1, 7*8)) \
        lea(mem(r8, r9, 4), r8) \
        DGEMM_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,rsi,K_LEFT_LOOP_PREP,LABEL_SUFFIX,j) \
\
    label(.DK_4LOOP_ ## LABEL_SUFFIX) \
        prefetch(0, mem(r8, 7*8)) \
        DGEMM_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,_,_,_,noj) \
        prefetch(0, mem(r8, r9, 1, 7*8)) \
        DGEMM_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,_,_,_,noj) \
        prefetch(0, mem(r8, r9, 2, 7*8)) \
        DGEMM_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,_,_,_,noj) \
        prefetch(0, mem(r8, r12, 1, 7*8)) \
        lea(mem(r8, r9, 4), r8) \
        DGEMM_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,rsi,K_LEFT_LOOP_PREP,LABEL_SUFFIX,j) \
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
    VMOV_ ## BAlign (mem(rbx), ymm2) \
    VMOV_ ## BAlign (mem(rbx, 32), ymm3) \
    add(r11, rbx) \
\
    label(.DK_LEFT_LOOP_ ## LABEL_SUFFIX) \
        prefetch(0, mem(r8, 7*8)) \
        add(r9, r8) \
        DGEMM_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB,BAlign,r14,WRITEMEM_PREP,LABEL_SUFFIX,j) \
    jmp(.DK_LEFT_LOOP_ ## LABEL_SUFFIX) \
\
    label(.DWRITEMEM_PREP_ ## LABEL_SUFFIX) \
\
    /* For millikernels:
     * New A is packed. Use it. */ \
    PACK_ ## PACKA( mov(var(a_p), rcx) ) \
    PACK_ ## PACKA( movq(rcx, var(a)) ) \
    PACK_ ## PACKA( movq(imm(8), var(rs_a)) ) \
    PACK_ ## PACKA( movq(imm(6*8), var(cs_a)) ) \
\
    mov(var(alpha), rax) \
    mov(var(beta), rbx) \
    vbroadcastsd(mem(rax), ymm0) \
    vbroadcastsd(mem(rbx), ymm3) \
    mov(var(a_next), rax) \
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
        prefetch(0, mem(rax)) \
        /* prefetch(0, mem(rax, 1*64)) \
         * prefetch(0, mem(rax, 2*64)) \
         * prefetch(0, mem(rax, 3*64)) */ \
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
        cmp(imm(8), rdi) /* set ZF if (8*rs_c) == 8. */ \
        jz(.DCOLSTORED_ ## LABEL_SUFFIX) /* jump to column storage case */ \
\
            CSTORE_LOC_## M (rcx, rdi) \
            jmp(.DDONE_ ## LABEL_SUFFIX) \
\
        label(.DCOLSTORED_ ## LABEL_SUFFIX) \
\
            CSTORECOL_LOC_## M (nz,4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,12,13,14,15,3,rbx,rcx,r14,rsi,r13) \
            jmp(.DDONE_ ## LABEL_SUFFIX) \
\
    label(.DBETAZERO_ ## LABEL_SUFFIX) \
\
        cmp(imm(8), rdi) /* set ZF if (8*rs_c) == 8. */ \
        jz(.DCOLSTORBZ_ ## LABEL_SUFFIX) /* jump to column storage case */ \
\
            CSTOREZB_LOC_## M (rcx, rdi) \
            jmp(.DDONE_ ## LABEL_SUFFIX) \
\
        label(.DCOLSTORBZ_ ## LABEL_SUFFIX) \
\
            CSTORECOL_LOC_## M (z,4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,12,13,14,15,3,rbx,rcx,r14,rsi,r13) \
\
    label(.DDONE_ ## LABEL_SUFFIX)

// Start defining the millikernel.
// This is the asm entry point for x86.
#define GENDEF(M,PACKB,BAlign) \
BLIS_INLINE void bli_dgemmsup2_rv_haswell_asm_## M ##x8n_ ## PACKB ## _ ## BAlign \
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
     double *restrict a_p, uint64_t pack_a, \
     double *restrict b_p \
    ) \
{ \
    const void* a_next = bli_auxinfo_next_b( data ); \
    const void* b_next = bli_auxinfo_next_a( data ); /* Caller specifies A as "extending". */ \
    uint64_t ps_b      = bli_auxinfo_ps_a( data ) << 3; /* Panel stride for extending dim. */ \
    uint64_t ps_b_p    = bli_auxinfo_is_a( data ) << 3; /* Panel stride for packed extending. */ \
    uint64_t rs_b_next = bli_auxinfo_ps_b( data ) << 3; /* Borrow the space. */ \
    uint64_t ps_b_prfm = min_(ps_b, 64*8); /* Packed A: Avoid prefetching too far away. */ \
\
    /* Typecast local copies of integers in case dim_t and inc_t are a
     * different size than is expected by load instructions. */ \
    uint64_t n_iter = n / 8; \
    uint64_t n_left = n % 8; \
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
    /* First (A-packing) microkernel */ \
    mov(var(b), rbx) \
    mov(var(c), rcx) \
    mov(var(b_p), rdx) \
    mov(var(pack_a), rdi) \
    mov(var(n_iter), rsi) \
    cmp(imm(1), rsi) \
    je(.DN_BEFORE_INIT_NEXT_IS_FINAL) \
        mov(rbx, r8) \
        add(var(ps_b2), r8) \
        mov(var(rs_b), r9) \
        jmp(.DN_BEFORE_INIT_CONTINUE) \
    label(.DN_BEFORE_INIT_NEXT_IS_FINAL) \
        mov(var(b_next), r8) \
        mov(var(rs_b2), r9) \
    label(.DN_BEFORE_INIT_CONTINUE) \
    test(rdi, rdi) \
    je(.DN_ITER) \
    DGEMM_6X8N_UKER_LOC(M,pack,PACKB,BAlign,init) \
    mov(var(n_iter), rsi) \
    mov(var(b), rbx) /*********** Prepare b for next uker */ \
    mov(var(ps_b), rdi) /******** Prepare b for next uker */ \
    lea(mem(rbx, rdi, 1), rbx) /* Prepare b for next uker */ \
    mov(var(b_p), rdx) /**** Prepare b_p for next uker */ \
    add(var(ps_b_p), rdx) /* Prepare b_p for next uker */ \
    mov(var(c), rcx) /*********** Prepare c for next uker */ \
    mov(var(cs_c), r13) /******** Prepare c for next uker */ \
    lea(mem(rcx, r13, 8), rcx) /* Prepare c for next uker */ \
    cmp(imm(2), rsi) \
    je(.DN_INIT_NEXT_IS_FINAL) \
        mov(rbx, r8) \
        add(var(ps_b2), r8) \
        mov(var(rs_b), r9) \
        jmp(.DN_INIT_CONTINUE) \
    label(.DN_INIT_NEXT_IS_FINAL) \
        mov(var(b_next), r8) \
        mov(var(rs_b2), r9) \
    label(.DN_INIT_CONTINUE) \
    movq(rcx, var(c)) \
    movq(rbx, var(b)) \
    movq(rdx, var(b_p)) \
    dec(rsi) \
    je(.DMILLIKER_END) \
    mov(rsi, var(n_iter)) \
\
    /* Microkernels in between */ \
    label(.DN_ITER) \
    DGEMM_6X8N_UKER_LOC(M,nopack,PACKB,BAlign,iter) \
    mov(var(n_iter), rsi) \
    mov(var(b), rbx) /*********** Prepare b for next uker */ \
    mov(var(ps_b), rdi) /******** Prepare b for next uker */ \
    lea(mem(rbx, rdi, 1), rbx) /* Prepare b for next uker */ \
    mov(var(b_p), rdx) /**** Prepare b_p for next uker */ \
    add(var(ps_b_p), rdx) /* Prepare b_p for next uker */ \
    mov(var(c), rcx) /*********** Prepare c for next uker */ \
    mov(var(cs_c), r13) /******** Prepare c for next uker */ \
    lea(mem(rcx, r13, 8), rcx) /* Prepare c for next uker */ \
    cmp(imm(2), rsi) \
    je(.DN_ITER_NEXT_IS_FINAL) \
        mov(rbx, r8) \
        add(var(ps_b2), r8) \
        mov(var(rs_b), r9) \
        jmp(.DN_ITER_CONTINUE) \
    label(.DN_ITER_NEXT_IS_FINAL) \
        mov(var(b_next), r8) \
        mov(var(rs_b2), r9) \
    label(.DN_ITER_CONTINUE) \
    movq(rcx, var(c)) \
    movq(rbx, var(b)) \
    movq(rdx, var(b_p)) \
    dec(rsi) \
    je(.DMILLIKER_END) \
    mov(rsi, var(n_iter)) \
    jmp(.DN_ITER) \
\
    label(.DMILLIKER_END) \
\
    end_asm( \
    : [a]     "+m" (a), \
      [b]     "+m" (b), \
      [rs_a]  "+m" (rs_a), \
      [cs_a]  "+m" (cs_a), \
      [c]     "+m" (c), \
      [b_p]   "+m" (b_p), \
      [n_iter]"+m" (n_iter) \
    : [n_left] "m" (n_left), \
      [rs_b]   "m" (rs_b), \
      [rs_b2]  "m" (rs_b_next), \
      [ps_b]   "m" (ps_b), \
      [ps_b_p] "m" (ps_b_p), \
      [ps_b2]  "m" (ps_b_prfm), \
      [rs_c]   "m" (rs_c), \
      [cs_c]   "m" (cs_c), \
      [k_iter] "m" (k_iter), \
      [k_left] "m" (k_left), \
      [alpha]  "m" (alpha), \
      [beta]   "m" (beta), \
      [a_p]    "m" (a_p), \
      [pack_a] "m" (pack_a), \
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
    if ( n_left ) \
    { assert( 0 ); } \
\
}

GENDEF(6,pack,noalign)

GENDEF(6,nopack,align)
GENDEF(5,nopack,align)
GENDEF(4,nopack,align)
GENDEF(3,nopack,align)
GENDEF(2,nopack,align)
GENDEF(1,nopack,align)

GENDEF(6,nopack,noalign)
GENDEF(5,nopack,noalign)
GENDEF(4,nopack,noalign)
GENDEF(3,nopack,noalign)
GENDEF(2,nopack,noalign)
GENDEF(1,nopack,noalign)

void bli_dgemmsup2_rv_haswell_asm_6x8n
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
    const inc_t ps_b0 = bli_auxinfo_ps_a( data ); // Panel stride for extending dim.
    const int b_align = (((uint64_t)b % (4*8)) + rs_b0 % 4 + ps_b0 % 4) == 0;
#ifdef DEBUG
    assert( m <= 6 );
    assert( cs_b0 == 1 );
    assert( rs_c0 == 1 ||
            cs_c0 == 1 );
#endif

    switch ( pack_b << 13 | b_align << 12 | m ) {
#define EXPAND_CASE(M,PACKB_C,BAlign_C,PACKB,BAlign) \
        case ( PACKB_C << 13 | BAlign_C << 12 | M ): \
            /* Fully packed. */ \
            bli_dgemmsup2_rv_haswell_asm_## M ##x8n_## PACKB ##_## BAlign \
                ( m, n, k, alpha, \
                  a, rs_a0, cs_a0, \
                  b, rs_b0, cs_b0, beta, \
                  c, rs_c0, cs_c0, \
                  data, cntx, a_p, pack_a, b_p ); \
            break;
        case ( 1 << 13 | 1 << 12 | 6 ):
        EXPAND_CASE(6, 1, 0, pack, noalign) // Use unaligned kernel regardless b_align.
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
            fprintf( stderr, "error: m=%ld,pb=%d,ab=%d\n", m, pack_b, b_align );
            assert( 0 );
#endif
            break;
    }
}

void bli_dgemmsup2_cv_haswell_asm_8x6m
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
    bli_dgemmsup2_rv_haswell_asm_6x8n
        ( n, m, k, alpha,
          b, cs_b0, rs_b0,
          a, cs_a0, rs_b0, beta,
          c, cs_c0, rs_c0,
          data, cntx,
          b_p, pack_b,
          a_p, pack_a );
}

#endif
