#include "blis.h"
#include <assert.h>

void bli_gemm_ex
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// If C has a zero dimension, return early.
	if ( bli_obj_has_zero_dim( c ) ) return;

	// If alpha is zero, or if A or B has a zero dimension, scale C by beta
	// and return early.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) ||
	     bli_obj_has_zero_dim( a ) ||
	     bli_obj_has_zero_dim( b ) )
	{
		bli_scalm( beta, c );
		return;
	}

	// When the datatype is elidgible // & k is not too small,
	// invoke the sandbox method where dgemmsup and dpack interleaves
	// each other.
	if ( // ( k > m / 2 || k > n / 2 ) &&
		bli_obj_dt( a ) == BLIS_DOUBLE &&
		bli_obj_dt( b ) == BLIS_DOUBLE &&
		bli_obj_dt( c ) == BLIS_DOUBLE &&
		bli_obj_dt( alpha ) == BLIS_DOUBLE &&
		bli_obj_dt( beta  ) == BLIS_DOUBLE )
	{
		/* Detect small-m and transpose into small-n.
		 * TODO: For x86 only? It seems that arm64 does not need this. */
		if ( bli_obj_has_notrans( c ) ?
			( bli_obj_dim( BLIS_M, c ) < bli_min( bli_obj_dim( BLIS_N, c ), 50 ) ) :
			( bli_obj_dim( BLIS_N, c ) < bli_min( bli_obj_dim( BLIS_M, c ), 50 ) ) )
		{
			// Call C' += B'A' <=> C += A B.
			obj_t at, bt, ct;
			bli_obj_alias_to( a, &at );
			bli_obj_alias_to( b, &bt );
			bli_obj_alias_to( c, &ct );
			bli_obj_toggle_trans( &at );
			bli_obj_toggle_trans( &bt );
			bli_obj_toggle_trans( &ct );
			bli_gemm_ex ( alpha, &bt, &at, beta, &ct, cntx, rntm );
			return ;
		}

		dim_t m, n, k;
		inc_t rs_a, cs_a, rs_b, cs_b, rs_c, cs_c;
		if ( bli_obj_has_notrans( a ) )
		{
			k = bli_obj_dim( BLIS_N, a );
			rs_a = bli_obj_row_stride( a );
			cs_a = bli_obj_col_stride( a );
		}
		else
		{
			k = bli_obj_dim( BLIS_M, a );
			rs_a = bli_obj_col_stride( a );
			cs_a = bli_obj_row_stride( a );
		}
		if ( bli_obj_has_notrans( b ) )
		{
			rs_b = bli_obj_row_stride( b );
			cs_b = bli_obj_col_stride( b );
		}
		else
		{
			rs_b = bli_obj_col_stride( b );
			cs_b = bli_obj_row_stride( b );
		}
		if ( bli_obj_has_notrans( c ) )
		{
			m = bli_obj_dim( BLIS_M, c );
			n = bli_obj_dim( BLIS_N, c );
			rs_c = bli_obj_row_stride( c );
			cs_c = bli_obj_col_stride( c );
		}
		else
		{
			m = bli_obj_dim( BLIS_N, c );
			n = bli_obj_dim( BLIS_M, c );
			rs_c = bli_obj_col_stride( c );
			cs_c = bli_obj_row_stride( c );
		}

		dim_t mr, nr;
		ukr_dgemm_sup_t milliker;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
		if ( cs_b == 1 ) { mr = 6; nr = 8; milliker = bli_dgemmsup2_rv_haswell_asm_6x8m; }
		else { mr = 8; nr = 6; milliker = bli_dgemmsup2_cv_haswell_asm_8x6m; }
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_ARCH_PPC)
		milliker = rs_a == 1 ? bli_dgemmsup2_cv_armv8a_asm_8x6m :
			bli_dgemmsup2_rv_armv8a_asm_8x6m; mr = 8; nr = 6;
#else
		// TODO: Use reference kernels.
#error "This architecture is not supported yet."
#endif

		rntm_t rntm_l;
		if ( rs_a == 1 || cs_b == 1 )
		{
			// Query the context for block size & packing kernels.
			if ( cntx == NULL ) cntx = bli_gks_query_cntx();

			// Check the operands.
			if ( bli_error_checking_is_enabled() )
				bli_gemm_check( alpha, a, b, beta, c, cntx );

			bls_dgemm
			(
				m, n, k,
				bli_obj_buffer( alpha ),
				bli_obj_buffer( a ), rs_a, cs_a,
				bli_obj_buffer( b ), rs_b, cs_b,
				bli_obj_buffer( beta ),
				bli_obj_buffer( c ), rs_c, cs_c,
				cntx, &rntm_l,
				milliker, mr, nr
			);
			return ;
		}

		// Otherwise, the program would pop back to the original path of exec.
	}

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); }
	else                { rntm_l = *rntm;                       }

	// Default to using native execution.
	num_t dt = bli_obj_dt( c );
	ind_t im = BLIS_NAT;

	// If each matrix operand has a complex storage datatype, try to get an
	// induced method (if one is available and enabled). NOTE: Allowing
	// precisions to vary while using 1m, which is what we do here, is unique
	// to gemm; other level-3 operations use 1m only if all storage datatypes
	// are equal (and they ignore the computation precision).
	if ( bli_obj_is_complex( c ) &&
	     bli_obj_is_complex( a ) &&
	     bli_obj_is_complex( b ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_gemmind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im );

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_gemm_check( alpha, a, b, beta, c, cntx );

	// Invoke the operation's front-end and request the default control tree.
	bli_gemm_front( alpha, a, b, beta, c, cntx, &rntm_l );
}

