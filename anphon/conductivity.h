/*
 conductivity.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "pointers.h"
#include "kpoint.h"
#include <vector>
#include <set>
#include <complex>

namespace PHON_NS {
class Conductivity : protected Pointers {
public:
    Conductivity(class PHON *);

    ~Conductivity();

    void setup_kappa();

    void prepare_restart();

    void setup_kappa_mc();

    void calc_anharmonic_imagself();

    void compute_kappa();

    int calc_kappa_spec;
    unsigned int ntemp;
    double **damping3;
    double ***kappa;
    double ***kappa_spec;
    double ***kappa_coherent;
    double *temperature;
    int calc_coherent;

    //store relative error of gamma
    double **rel_err;

    //MC for kappa
    unsigned int seed;
    int calc_kappa_mc;  //0: full mode, 1: MC
    double nsample_kappa_density;
    int nsample_kappa_ini;
    unsigned int *nsample_kappa;
    double coef_b;
    std::vector<std::vector<std::vector<unsigned int>>> map_sample_to_mode;  //[temperature][dim1*3+dim2][sample_id]
    unsigned int *gamma_calculated;
    double ***weighting_factor_mc;  //[temperature][dim1*3+dim2][ik*ns+is]
    double ***weighting_factor_map;  //[temperature][dim1*3+dim2][ik*ns+is], accumulation of weighting_factor_mc
    std::string vv_dim;  //"full","diagonal","diagonal_sum","xx"-"zz"

private:
    void set_default_variables();

    void deallocate_variables();

    double ***vel;
    std::complex<double> ****velmat;
    unsigned int nk, ns;
    int nshift_restart;
    std::vector<int> vks_l, vks_done;
    std::set<int> vks_job;
    std::string file_coherent_elems;

    void write_result_gamma(unsigned int,
                            unsigned int,
                            double ***,
                            double **) const;

    void write_result_err(unsigned int,
                            unsigned int,
                            double ***,
                            double **) const;

    void write_result_gamma_each(unsigned int,
                            double ***,
                            double **) const;

    void write_result_err_each(unsigned int,
                            double ***,
                            double **) const;
                            
    void generate_nsample_kappa();

    void generate_kappa_mc_map(const KpointMeshUniform *kmesh_in, const double *const *eval_in);

    void generate_kappa_mc_sample(const unsigned int *nshift, const unsigned int *nsample);
    
    void setup_vks_for_mc();

    void average_self_energy_at_degenerate_point(const int n,
                                                 const int m,
                                                 const KpointMeshUniform *kmesh_in,
                                                 const double *const *eval_in,
                                                 double **damping) const;

    void compute_frequency_resolved_kappa(const int ntemp,
                                          const int smearing_method,
                                          const KpointMeshUniform *kmesh_in,
                                          const double *const *eval_in,
                                          const double *const *const *const *kappa_mode,
                                          double ***kappa_spec_out) const;

    void compute_kappa_intraband(const KpointMeshUniform *kmesh_in,
                                 const double *const *eval_in,
                                 const double *const *lifetime,
                                 double ***kappa_intra,
                                 double ***kappa_spec_out) const;

    void compute_kappa_intraband_with_mc(const KpointMeshUniform *kmesh_in,
                                 const double *const *eval_in,
                                 const double *const *lifetime,
                                 const int nshift_sample,
                                 double ***kappa_sample,
                                 double ***sample_error_tau,
                                 double ***sample_error_mc,
                                 double ***kappa_intra,
                                 double ***kappa_intra_error_tau_out,
                                 double ***kappa_intra_error_mc_out,
                                 double ***kappa_spec_out) const;

    void compute_kappa_coherent(const KpointMeshUniform *kmesh_in,
                                const double *const *eval_in,
                                const double *const *gamma_total,
                                double ***kappa_coherent_out) const;
};
}
