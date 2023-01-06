*This sandbox was originally named* `packsup`.

*Currently only* `dgemm` *is supported*.

# GEMMHOT: The "hot-plugging" solution to GEMM calls.

This sandbox is a `GEMM` solution to cover up packing overheads of regular the `GEMM` codepath. By integrating packing instructions to the `GEMMSUP` millikernel on the unpacked memory, `bls_?gemm` here tries to make a hot-plugging solution to kick calculation off without the need to wait for `memcpy`.
