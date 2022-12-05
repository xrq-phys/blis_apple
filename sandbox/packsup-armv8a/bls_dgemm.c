#include "blis.h"
#include <stdlib.h>
#include <assert.h>


BLIS_INLINE void abort_(const char *msg)
{ fprintf(stderr, "%s\n", msg); abort(); }

BLIS_INLINE void assert_(const bool cond, const char *msg)
{ if (!cond) abort_(msg); }

#define min_(a, b) ( (a) < (b) ? (a) : (b) )

void bls_dgemm
    (
     dim_t m0,
     dim_t n0,
     dim_t k,
     double *restrict alpha,
     double *restrict a, inc_t rs_a, inc_t cs_a,
     double *restrict b, inc_t rs_b, inc_t cs_b,
     double *restrict beta,
     double *restrict c, inc_t rs_c, inc_t cs_c,
     cntx_t *cntx,
     rntm_t *rntm,
     ukr_dgemm_bulk_t ukr_bulk,
     ukr_dgemm_sup_t  ukr_sup
    )
{
    const dim_t mc = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MC, cntx );
    const dim_t nc = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_NC, cntx );
    const dim_t mr = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MR, cntx );
    const dim_t nr = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_NR, cntx );
    assert_( !(mc % mr), "MC not multiple of MR." );
    assert_( !(nc % nr), "NC not multiple of NR." );
    dim_t num_ir = mc / mr;
    dim_t num_jr = nc / nr;

    // Only has *r* kers.
    // assert( cs_b == 1 );

    auxinfo_t data;
    bli_auxinfo_set_next_a( 0, &data );
    bli_auxinfo_set_next_b( 0, &data );

    // Initialize packing env vars.
    bli_pack_init_rntm_from_env( rntm );

    // For querying BLIS' memory pool.
    pba_t *pba = bli_pba_query();
    mem_t  mem_a, mem_b;

    int b_size = nr * num_jr * k * sizeof( double );
    int a_size = mr * num_ir * k * sizeof( double );

    // Query the pool for packing space.
    bli_pba_acquire_m( pba, b_size, BLIS_BUFFER_FOR_B_PANEL, &mem_b );
    bli_pba_acquire_m( pba, a_size, BLIS_BUFFER_FOR_A_BLOCK, &mem_a );

    double *b_panels = bli_mem_buffer( &mem_b );
    double *a_panels = bli_mem_buffer( &mem_a );

    for ( dim_t jc_offset = 0; jc_offset < n0; jc_offset += nc ) {
        double *b_l3 = b + jc_offset * cs_b;
        double *c_l3 = c + jc_offset * cs_c;

        // double *b_panels = calloc(nr * k * num_jr, sizeof(double));

        for ( dim_t ic_offset = 0; ic_offset < m0; ic_offset += mc ) {
            double *a_l2 = a    + ic_offset * rs_a;
            double *c_l2 = c_l3 + ic_offset * rs_c;

            double *a_uker, *b_uker;
            inc_t rs_a_uker, cs_a_uker;
            inc_t rs_b_uker, cs_b_uker;

            // double *a_panels = calloc(mr * k * num_ir, sizeof(double));
            if ( bli_rntm_pack_b( rntm ) && ic_offset == 0 )
            {
                // Ahead-of-time packing case.
                //
                double one = 1.0;
                l1mukr_t dpackm = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_PACKM_NRXK_KER, cntx );
                for ( dim_t jr = 0; jr < num_jr && n0 - jc_offset - jr * nr > 0; ++jr )
                {
                    double *b_l1 = b_l3 + jr * nr * cs_b;;
                    double *b_p = b_panels + nr * k * jr;
                    dim_t n_uker = min_(n0 - jc_offset - jr * nr, nr);
                    dpackm
                        (
                         BLIS_NO_CONJUGATE, BLIS_PACKED_ROWS,
                         n_uker, k, k,
                         &one,
                         b_l1, cs_b, rs_b,
                         b_p, nr,
                         cntx
                        );
                }
            }

            for ( dim_t jr = 0; jr < num_jr && n0 - jc_offset - jr * nr > 0; ++jr ) {
                double *c_l1 = c_l2 + jr * nr * cs_c;
                double *b_l1 = b_l3 + jr * nr * cs_b;;
                double *b_p = b_panels + nr * k * jr;
                dim_t n_uker = min_(n0 - jc_offset - jr * nr, nr);

                if ( bli_rntm_pack_b( rntm ) || ic_offset > 0 ) {
                    // Reuse packed b.
                    b_uker = b_p;
                    rs_b_uker = nr;
                    cs_b_uker = 1;
                } else {
                    b_uker = b_l1;
                    rs_b_uker = rs_b;
                    cs_b_uker = cs_b;
                }

                if ( bli_rntm_pack_a( rntm ) && jr == 0 )
                {
                    // Ahead-of-time packing case.
                    //
                    double one = 1.0;
                    l1mukr_t dpackm = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_PACKM_MRXK_KER, cntx );
                    for ( dim_t ir = 0; ir < num_ir && m0 - ic_offset - ir * mr > 0; ++ir )
                    {
                        double *a_l1 = a_l2 + ir * mr * rs_a;
                        double *a_p = a_panels + mr * k * ir;
                        dim_t m_uker = min_(m0 - ic_offset - ir * mr, mr);
                        dpackm
                            (
                             BLIS_NO_CONJUGATE, BLIS_PACKED_COLUMNS,
                             m_uker, k, k,
                             &one,
                             a_l1, rs_a, cs_a,
                             a_p, mr,
                             cntx
                            );
                    }
                }

                for ( dim_t ir = 0; ir < num_ir && m0 - ic_offset - ir * mr > 0; ++ir )
                {
                    double *c_r  = c_l1 + ir * mr * rs_c;
                    double *a_l1 = a_l2 + ir * mr * rs_a;
                    double *a_p = a_panels + mr * k * ir;
                    dim_t m_uker = min_(m0 - ic_offset - ir * mr, mr);

                    if ( ir > 0 ) {
                        b_uker = b_p;
                        rs_b_uker = nr;
                        cs_b_uker = 1;
                    }

                    if ( bli_rntm_pack_a( rntm ) || jr > 0 ) {
                        a_uker = a_p;
                        rs_a_uker = 1;
                        cs_a_uker = mr;
                    } else {
                        a_uker = a_l1;
                        rs_a_uker = rs_a;
                        cs_a_uker = cs_a;
                    }

                    // Set next_a.
                    if ( ir + 1 < num_ir ) {
                        if ( jr > 0 ) // Previous jr packed a.
                            bli_auxinfo_set_next_a( a_p + mr * k, &data );
                        else
                            bli_auxinfo_set_next_a( a_l1 + mr * rs_a, &data );
                    } else if ( jr + 1 < num_jr )
                        // Still using the already-packed a panels.
                        bli_auxinfo_set_next_a( a_panels, &data );
                    else if ( ic_offset + mc < m0 )
                        bli_auxinfo_set_next_a( a_l2 + mc * rs_a, &data );
                    else
                        bli_auxinfo_set_next_a( a, &data );

                    // Set next_b
                    if ( ir + 1 < num_ir )
                        // Next b must be next same jr & packed by this / some previous uker call.
                        bli_auxinfo_set_next_b( b_p, &data );
                    else if ( jr + 1 < num_jr ) {
                        if ( ic_offset > 0 ) // Previous ic has packed b for the next jr.
                            bli_auxinfo_set_next_b( b_p + nr * k, &data );
                        else
                            bli_auxinfo_set_next_b( b_l1 + nr * cs_b, &data );
                    } else if ( ic_offset + mc < m0 )
                        // Return jr.
                        bli_auxinfo_set_next_b( b_panels, &data );
                    else
                        bli_auxinfo_set_next_b( b_l3 + nc * cs_b, &data );

                    if ( a_uker == a_p && b_uker == b_p && m_uker + n_uker > mr + nr - 3 )
                        ukr_bulk
                            (
                             m_uker, n_uker, k,
                             alpha,
                             a_uker, b_uker,
                             beta,
                             c_r, rs_c, cs_c,
                             &data, cntx
                            );
                    else
                        ukr_sup
                            (
                             m_uker, n_uker, k,
                             alpha,
                             a_uker, rs_a_uker, cs_a_uker,
                             b_uker, rs_b_uker, cs_b_uker,
                             beta,
                             c_r, rs_c, cs_c,
                             &data, cntx,
                             a_p, a_uker != a_p,
                             b_p, b_uker != b_p
                            );
                }
            }

            // free( a_panels );
        }

        // free( b_panels );
    }

    bli_pba_release( pba, &mem_b );
    bli_pba_release( pba, &mem_a );
}

