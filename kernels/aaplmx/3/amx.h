#pragma once

#include <stdint.h>

// Encode AMX' operand registers.
__asm__ (
  ".equ enc_x0,  0   \n\t"
  ".equ enc_x1,  1   \n\t"
  ".equ enc_x2,  2   \n\t"
  ".equ enc_x3,  3   \n\t"
  ".equ enc_x4,  4   \n\t"
  ".equ enc_x5,  5   \n\t"
  ".equ enc_x6,  6   \n\t"
  ".equ enc_x7,  7   \n\t"
  ".equ enc_x8,  8   \n\t"
  ".equ enc_x9,  9   \n\t"
  ".equ enc_x10, 10  \n\t"
  ".equ enc_x11, 11  \n\t"
  ".equ enc_x12, 12  \n\t"
  ".equ enc_x13, 13  \n\t"
  ".equ enc_x14, 14  \n\t"
  ".equ enc_x15, 15  \n\t"
  ".equ enc_x16, 16  \n\t"
  ".equ enc_x17, 17  \n\t"
  ".equ enc_x18, 18  \n\t"
  ".equ enc_x19, 19  \n\t"
  ".equ enc_x20, 20  \n\t"
  ".equ enc_x21, 21  \n\t"
  ".equ enc_x22, 22  \n\t"
  ".equ enc_x23, 23  \n\t"
  ".equ enc_x24, 24  \n\t"
  ".equ enc_x25, 25  \n\t"
  ".equ enc_x26, 26  \n\t"
  ".equ enc_x27, 27  \n\t"
  ".equ enc_x28, 28  \n\t"
  ".equ enc_x29, 29  \n\t"
  ".equ enc_x30, 30  \n\t"
  ".macro amx opcode reg  \n\t" // Here is the instruction encoding.
  ".word ( \\opcode | enc_\\reg ) \n\t"
  ".endm \n\t"
);

// TODO: do I need memory as an input?
#define AMX_LDX(V)                                                             \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (0 << 5)), %0" ::"r"((uint64_t)V)     \
      : "x0", "memory")
#define AMX_LDY(V)                                                             \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (1 << 5)), %0" ::"r"((uint64_t)V)     \
      : "x0", "memory")
#define AMX_STX(V)                                                             \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (2 << 5)), %0" ::"r"((uint64_t)V)     \
      : "x0", "memory")
#define AMX_STY(V)                                                             \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (3 << 5)), %0" ::"r"((uint64_t)V)     \
      : "x0", "memory")
#define AMX_LDZ(V)                                                             \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (4 << 5)), %0" ::"r"((uint64_t)V)     \
      : "x0", "memory")
#define AMX_STZ(V)                                                             \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (5 << 5)), %0" ::"r"((uint64_t)V)     \
      : "x0", "memory")

#define AMX_LDZI(V)                                                            \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (6 << 5)), %0" ::"r"((uint64_t)V)     \
      : "x0", "memory")
#define AMX_STZI(V)                                                            \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (7 << 5)), %0" ::"r"((uint64_t)V)     \
      : "x0", "memory")

// TODO: probably shouldn't say these clobber memory?
#define AMX_EXTRX(V)                                                           \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (8 << 5)), %0" ::"r"((uint64_t)V)     \
      : "x0", "memory")
#define AMX_EXTRY(V)                                                           \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (9 << 5)), %0" ::"r"((uint64_t)V)     \
      : "x0", "memory")

#define AMX_FMA64(V)                                                           \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (10 << 5)), %0" ::"r"((uint64_t)V)    \
      : "x0", "memory")
#define AMX_FMS64(V)                                                           \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (11 << 5)), %0" ::"r"((uint64_t)V)    \
      : "x0", "memory")

#define AMX_FMA32(V)                                                           \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (12 << 5)), %0" ::"r"((uint64_t)V)    \
      : "x0", "memory")
#define AMX_FMS32(V)                                                           \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (13 << 5)), %0" ::"r"((uint64_t)V)    \
      : "x0", "memory")

#define AMX_MAC16(V)                                                           \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (14 << 5)), %0" ::"r"((uint64_t)V)    \
      : "x0", "memory")
#define AMX_FMA16(V)                                                           \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (15 << 5)), %0" ::"r"((uint64_t)V)    \
      : "x0", "memory")
#define AMX_FMS16(V)                                                           \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (16 << 5)), %0" ::"r"((uint64_t)V)    \
      : "x0", "memory")

#define AMX_START()                                                            \
  __asm__ volatile(                                                            \
      "nop \r\n nop \r\n nop \r\n .word (0x201000 | (17 << 5) | 0)" ::         \
          : "memory")
#define AMX_STOP()                                                             \
  __asm__ volatile(                                                            \
      "nop \r\n nop \r\n nop \r\n .word (0x201000 | (17 << 5) | 1)" ::         \
          : "memory")

// horizontal multiply uint16_ts? (doesn't mac16 have a flag for this?)
// z0[i] += x0[i] + y0[i]
#define AMX_VECINT(V)                                                          \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (18 << 5)), %0" ::"r"((uint64_t)V)    \
      : "x0", "memory")

// horizontal multiply float16_ts? (doesn't fma16 have a flag for this?)
// z0[i] += x0[i] + y0[i]
#define AMX_VECFP(V)                                                           \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (19 << 5)), %0" ::"r"((uint64_t)V)    \
      : "x0", "memory")

// uint16_t matrix multiply? (doesn't mac16 do this?)
#define AMX_MATINT(V)                                                          \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (20 << 5)), %0" ::"r"((uint64_t)V)    \
      : "x0", "memory")

// float16_t matrix multiply? (doesn't fma16 do this?)
#define AMX_MATFP(V)                                                           \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (21 << 5)), %0" ::"r"((uint64_t)V)    \
      : "x0", "memory")

// looks only at z0, clears it, and generates a 64-bit value in x0[0]:
// [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -> 0xffffffffffffffff
// [0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1] -> 0xf0
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] -> 0xfedcba9876543210
// [0x0, 0x10000000, 0x20000000, 0x30000000, 0x40000000, 0x50000000, 0x60000000,
// 0x70000000, 0x80000000, 0x90000000, 0xA0000000, 0xB0000000, 0xC0000000,
// 0xD0000000, 0xE0000000, 0xF0000000] -> fffffff0f6543210
#define AMX_GENLUT(V)                                                          \
  __asm__ volatile(                                                            \
      "amx (0x201000 | (22 << 5)), %0" ::"r"((uint64_t)V)    \
      : "x0", "memory")

typedef _Float16 float16;

union amx_row {
  // not all supported types, just useful ones
  uint8_t u8[64];
  uint16_t u16[32];
  uint32_t u32[16];
  uint64_t u64[8];
  float16 f16[32];
  float f32[16];
  double f64[8];
};

struct amx_state {
  union amx_row x[8];
  union amx_row y[8];
  union amx_row z[64];
};

BLIS_INLINE void store_amx_state(struct amx_state *state) {
  memset(state, 0xAA, sizeof *state);
  for (uint64_t i = 0; i < 8; i++) {
    AMX_STX((i << 56) | (uint64_t)&state->x[i]);
  }
  for (uint64_t i = 0; i < 8; i++) {
    AMX_STY((i << 56) | (uint64_t)&state->y[i]);
  }
  for (uint64_t i = 0; i < 64; i++) {
    AMX_STZ((i << 56) | (uint64_t)&state->z[i]);
  }
}

BLIS_INLINE void load_amx_state(struct amx_state *state) {
  for (uint64_t i = 0; i < 8; i++) {
    AMX_LDX((i << 56) | (uint64_t)&state->x[i]);
  }
  for (uint64_t i = 0; i < 8; i++) {
    AMX_LDY((i << 56) | (uint64_t)&state->y[i]);
  }
  for (uint64_t i = 0; i < 64; i++) {
    AMX_LDZ((i << 56) | (uint64_t)&state->z[i]);
  }
}
