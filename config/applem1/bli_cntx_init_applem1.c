/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

void bli_cntx_init_applem1( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];
	blksz_t thresh[ BLIS_NUM_THRESH ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_applem1_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels and
	//  their storage preferences.
	// Secret Apple matrix coprocessor instructions are used here.
	bli_cntx_set_l3_nat_ukrs
	(
	  2,
	  BLIS_GEMM_UKR, BLIS_FLOAT,    bli_sgemm_aaplmx_mac_32x32, FALSE,
	  BLIS_GEMM_UKR, BLIS_DOUBLE,   bli_dgemm_aaplmx_mac_32x16, FALSE,
	  cntx
	);

	// Set packing routines.
	bli_cntx_set_packm_kers
	(
	  3,
	  BLIS_PACKM_32XK_KER, BLIS_FLOAT,  bli_spackm_aaplmx_mac_32xk,
	  BLIS_PACKM_16XK_KER, BLIS_DOUBLE, bli_dpackm_aaplmx_mac_16xk,
	  BLIS_PACKM_32XK_KER, BLIS_DOUBLE, bli_dpackm_aaplmx_mac_32xk,
	  cntx
	);

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],    32,    32,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    32,    16,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   384,   256,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],  2016,  2048,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ], 16384,  8192,    -1,    -1 );

	// Update the context with the current architecture's register and cache
	// blocksizes (and multiples) for native execution.
	bli_cntx_set_blkszs
	(
	  BLIS_NAT, 5,
	  BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
	  BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
	  BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
	  BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
	  BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,
	  cntx
	);

	// -------------------------------------------------------------------------

	// Initialize sup thresholds with architecture-appropriate values.
	//                                          s     d     c     z
	bli_blksz_init_easy( &thresh[ BLIS_MT ],  601,  601,   -1,   -1 );
	bli_blksz_init_easy( &thresh[ BLIS_NT ],  601,  601,   -1,   -1 );
	bli_blksz_init_easy( &thresh[ BLIS_KT ],  601,  601,   -1,   -1 );

	// Initialize the context with the sup thresholds.
	bli_cntx_set_l3_sup_thresh
	(
	  3,
	  BLIS_MT, &thresh[ BLIS_MT ],
	  BLIS_NT, &thresh[ BLIS_NT ],
	  BLIS_KT, &thresh[ BLIS_KT ],
	  cntx
	);

	// Update the context with optimized small/unpacked gemm kernels.
	bli_cntx_set_l3_sup_kers
	(
	  12,
	  BLIS_RRR, BLIS_FLOAT, bli_sgemmsup_rv_aaplmx_mac_32x32mn, TRUE,
	  BLIS_RCR, BLIS_FLOAT, bli_sgemmsup_rv_aaplmx_mac_32x32mn, TRUE,
	  BLIS_RCC, BLIS_FLOAT, bli_sgemmsup_rv_aaplmx_mac_32x32mn, TRUE,
	  BLIS_CRR, BLIS_FLOAT, bli_sgemmsup_rv_aaplmx_mac_32x32mn, TRUE,
	  BLIS_CCR, BLIS_FLOAT, bli_sgemmsup_rv_aaplmx_mac_32x32mn, TRUE,
	  BLIS_CCC, BLIS_FLOAT, bli_sgemmsup_rv_aaplmx_mac_32x32mn, TRUE,
	  // BLIS_RRC, BLIS_DOUBLE, bli_dgemmsup_rd_armv8a_asm_6x8m, TRUE,
	  // BLIS_CRC, BLIS_DOUBLE, bli_dgemmsup_rd_armv8a_asm_6x8n, TRUE,
	  BLIS_RRR, BLIS_DOUBLE, bli_dgemmsup_rv_aaplmx_mac_16x32mn, TRUE,
	  BLIS_RCR, BLIS_DOUBLE, bli_dgemmsup_rv_aaplmx_mac_16x32mn, TRUE,
	  BLIS_RCC, BLIS_DOUBLE, bli_dgemmsup_rv_aaplmx_mac_16x32mn, TRUE,
	  BLIS_CRR, BLIS_DOUBLE, bli_dgemmsup_rv_aaplmx_mac_16x32mn, TRUE,
	  BLIS_CCR, BLIS_DOUBLE, bli_dgemmsup_rv_aaplmx_mac_16x32mn, TRUE,
	  BLIS_CCC, BLIS_DOUBLE, bli_dgemmsup_rv_aaplmx_mac_16x32mn, TRUE,
	  cntx
	);

	// Initialize level-3 sup blocksize objects with architecture-specific
	// values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],    32,    16,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    32,    32,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],  1024,   640,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],  2016,  2016,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  4096,  4096,    -1,    -1 );

	// Update the context with the current architecture's register and cache
	// blocksizes for small/unpacked level-3 problems.
	bli_cntx_set_l3_sup_blkszs
	(
	  5,
	  BLIS_NC, &blkszs[ BLIS_NC ],
	  BLIS_KC, &blkszs[ BLIS_KC ],
	  BLIS_MC, &blkszs[ BLIS_MC ],
	  BLIS_NR, &blkszs[ BLIS_NR ],
	  BLIS_MR, &blkszs[ BLIS_MR ],
	  cntx
	);
}

