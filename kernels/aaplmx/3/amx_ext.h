/* 

   Additional macros for AMX instructions

*/
#define AMX_MEM(OP, ADDR, REG) \
  AMX_## OP( (uint64_t)(ADDR) | ((uint64_t)(REG) << 56) )

#define AMX_FMA32_COMMON(PADROWS, PADCOLS, ZREGS) \
  AMX_FMA32( (uint64_t)(PADCOLS) | ((uint64_t)(PADROWS) << 10) | ((uint64_t)(ZREGS)<< 20) )

#define AMX_FMA32_COMMON_REGALIGNED(XREG, YREG, ZREGS) \
  AMX_FMA32_COMMON( ((uint64_t)(XREG) << 6), ((uint64_t)(YREG) << 6), (ZREGS) )


