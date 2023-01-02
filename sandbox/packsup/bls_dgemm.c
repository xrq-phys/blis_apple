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
     dim_t k0,
     double *restrict alpha0,
     double *restrict a, inc_t rs_a, inc_t cs_a,
     double *restrict b, inc_t rs_b, inc_t cs_b,
     double *restrict beta0,
     double *restrict c, inc_t rs_c, inc_t cs_c,
     cntx_t *cntx,
     rntm_t *rntm,
     ukr_dgemm_sup_t ukr_sup,
     dim_t mr, dim_t nr
    )
{
    const dim_t mc_= bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MC, cntx );
    const dim_t nc_= bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_NC, cntx );
    const dim_t kc = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_KC, cntx );
    const dim_t kc_max = kc + 47;
#if 0
    const dim_t mc = mc_;
    const dim_t nc = nc_;
    assert_( !(mc % mr), "MC not multiple of MR." );
    assert_( !(nc % nr), "NC not multiple of NR." );
#else
    const dim_t mc = ( mc_ + mr - 1 ) / mr * mr;
    const dim_t nc = ( nc_ + nr - 1 ) / nr * nr;
#endif
    const dim_t num_ir = mc / mr;
    const dim_t num_jr = nc / nr;
    const bool has_pack_a =
        n0 >= bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_NT, cntx ) &&
        k0 >= bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_KT, cntx );
    const bool has_pack_b =
        m0 >= bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MT, cntx ) &&
        k0 >= bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_KT, cntx );

    auxinfo_t data;
    bli_auxinfo_set_next_a( 0, &data );
    bli_auxinfo_set_next_b( 0, &data );

    // Initialize packing env vars.
    bli_pack_init_rntm_from_env( rntm );

    // For querying BLIS' memory pool.
    pba_t *pba = bli_pba_query();
    mem_t  mem_a, mem_b;

    int b_size = nr * num_jr * kc_max * sizeof( double );
    int a_size = mr * num_ir * kc_max * sizeof( double );

    // Query the pool for packing space.
    bli_pba_acquire_m( pba, b_size, BLIS_BUFFER_FOR_B_PANEL, &mem_b );
    bli_pba_acquire_m( pba, a_size, BLIS_BUFFER_FOR_A_BLOCK, &mem_a );

    double *b_panels = bli_mem_buffer( &mem_b );
    double *a_panels = bli_mem_buffer( &mem_a );

    // Constants.
    double one = 1.0;

    for ( dim_t jc_offset = 0; jc_offset < n0; jc_offset += nc ) {
        double *b_l4 = b + jc_offset * cs_b;
        double *c_l4 = c + jc_offset * cs_c;
        double *alpha = alpha0;
        double *beta  = beta0;

        for ( dim_t lc_offset = 0; lc_offset < k0; /* lc_offset += k_uker. */ ) {
            double *a_l3 = a    + lc_offset * cs_a;
            double *b_l3 = b_l4 + lc_offset * rs_b;
            dim_t k_uker = k0 - lc_offset < kc_max ? k0 - lc_offset : kc;
            // Determine whether to use k_uker * ?r or kc * ?r as the packing stride.
            // On CPU basically k_uker * ?r is better since it ensures equential HW prefetching.
            dim_t k_ps = k_uker;

            for ( dim_t ic_offset = 0; ic_offset < m0; ic_offset += mc ) {
                double *a_l2 = a_l3 + ic_offset * rs_a;
                double *c_l2 = c_l4 + ic_offset * rs_c;

                double *a_uker, *b_uker;
                inc_t rs_a_uker, cs_a_uker;
                inc_t rs_b_uker, cs_b_uker;

                if ( bli_rntm_pack_b( rntm ) && ic_offset == 0 )
                {
                    // Ahead-of-time packing case.
                    //
                    l1mukr_t dpackm = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_PACKM_NRXK_KER, cntx );
                    for ( dim_t jr = 0; jr < num_jr && n0 - jc_offset - jr * nr > 0; ++jr )
                    {
                        double *b_l1 = b_l3 + jr * nr * cs_b;
                        double *b_p = b_panels + nr * k_ps * jr;
                        dim_t n_uker = min_(n0 - jc_offset - jr * nr, nr);
                        dpackm
                            (
                             BLIS_NO_CONJUGATE, BLIS_PACKED_ROWS,
                             n_uker, k_uker, k_uker,
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
                    double *b_p = b_panels + nr * k_ps * jr;
                    dim_t jr_offset = jc_offset + jr * nr;
                    dim_t n_uker = min_(n0 - jr_offset, nr);

                    if ( bli_rntm_pack_b( rntm ) || ( ic_offset > 0 && has_pack_b ) ) {
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
                            double *a_p = a_panels + mr * k_ps * ir;
                            dim_t m_uker = min_(m0 - ic_offset - ir * mr, mr);
                            dpackm
                                (
                                 BLIS_NO_CONJUGATE, BLIS_PACKED_COLUMNS,
                                 m_uker, k_uker, k_uker,
                                 &one,
                                 a_l1, rs_a, cs_a,
                                 a_p, mr,
                                 cntx
                                );
                        }
                    }

                    // Set next_a.
                    if ( jr + 1 < num_jr && jr_offset + n_uker < n0 ) {
                        if ( has_pack_a ) {
                            // Still using the already-packed a panels.
                            bli_auxinfo_set_next_a( a_panels, &data );
                            bli_auxinfo_set_ps_b( mr, &data ); // cs_a_next.
                        } else {
                            bli_auxinfo_set_next_a( a_l2, &data );
                            bli_auxinfo_set_ps_b( cs_a, &data ); // cs_a_next.
                        }
                    } else {
                        bli_auxinfo_set_ps_b( cs_a, &data ); // cs_a_next.
                        if ( ic_offset + mc < m0 )
                            bli_auxinfo_set_next_a( a_l2 + mc * rs_a, &data );
                        else
                            if ( lc_offset + kc < k0 )
                                bli_auxinfo_set_next_a( a_l3 + kc * cs_a, &data );
                            else
                                bli_auxinfo_set_next_a( a, &data );
                    }

                    // Set next_b
                    if ( jr + 1 < num_jr && jr_offset + n_uker < n0 ) {
                        if ( ic_offset > 0 && has_pack_b ) // Previous ic has packed b for the next jr.
                            bli_auxinfo_set_next_b( b_p + nr * k_ps, &data );
                        else
                            bli_auxinfo_set_next_b( b_l1 + nr * cs_b, &data );
                    } else if ( ic_offset + mc < m0 )
                        // Return jr.
                        bli_auxinfo_set_next_b( b_panels, &data );
                    else
                        if ( lc_offset + kc < k0 )
                            bli_auxinfo_set_next_b( b_l3 + kc * rs_b, &data );
                        else
                            bli_auxinfo_set_next_b( b_l3 + nc * cs_b, &data );

                    dim_t m_mker = min_( m0 - ic_offset, mc );
                    bli_auxinfo_set_is_a( mr * k_ps, &data ); // ps_a_p.

                    if ( bli_rntm_pack_a( rntm ) || ( jr > 0 && has_pack_a ) ) {
                        a_uker = a_panels;
                        rs_a_uker = 1;
                        cs_a_uker = mr;
                        bli_auxinfo_set_ps_a( bli_auxinfo_is_a( &data ), &data );
                    } else {
                        a_uker = a_l2;
                        rs_a_uker = rs_a;
                        cs_a_uker = cs_a;
                        bli_auxinfo_set_ps_a( rs_a * 6, &data );
                    }

                    ukr_sup
                        (
                         m_mker, n_uker, k_uker,
                         alpha,
                         a_uker, rs_a_uker, cs_a_uker,
                         b_uker, rs_b_uker, cs_b_uker,
                         beta,
                         c_l1, rs_c, cs_c,
                         &data, cntx,
                         a_panels, a_uker != a_panels && jr_offset + n_uker < n0 && has_pack_a,
                         b_p,      b_uker != b_p                                 && has_pack_b
                        );

                }
            }
            beta = &one;
            lc_offset += k_uker;
        }
    }

    bli_pba_release( pba, &mem_b );
    bli_pba_release( pba, &mem_a );
}

