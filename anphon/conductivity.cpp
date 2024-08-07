/*
 conductivity.cpp

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "mpi_common.h"
#include "conductivity.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "integration.h"
#include "parsephon.h"
#include "isotope.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "memory.h"
#include "phonon_dos.h"
#include "thermodynamics.h"
#include "phonon_velocity.h"
#include "anharmonic_core.h"
#include "system.h"
#include "write_phonons.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>

using namespace PHON_NS;

Conductivity::Conductivity(PHON *phon) : Pointers(phon)
{
    set_default_variables();
}

Conductivity::~Conductivity()
{
    deallocate_variables();
}

void Conductivity::set_default_variables()
{
    calc_kappa_spec = 0;
    ntemp = 0;
    damping3 = nullptr;
    kappa = nullptr;
    kappa_spec = nullptr;
    kappa_coherent = nullptr;
    temperature = nullptr;
    vel = nullptr;
    velmat = nullptr;
    calc_coherent = 0;
    file_coherent_elems = "";

    //calc_kappa_mc=0;
    //nsample_kappa_density=0.1;
    //coef_b=1.5;
    //vv_dim = "sum";  //these value were initially set in parsephon.cpp
    std::random_device rnd;
    seed=rnd();
}

void Conductivity::deallocate_variables()
{
    if (damping3) {
        deallocate(damping3);
    }
    if (rel_err) {
        deallocate(rel_err);
    }
    if (kappa) {
        deallocate(kappa);
    }
    if (kappa_spec) {
        deallocate(kappa_spec);
    }
    if (kappa_coherent) {
        deallocate(kappa_coherent);
    }
    if (temperature) {
        deallocate(temperature);
    }
    if (vel) {
        deallocate(vel);
    }
    if (velmat) {
        deallocate(velmat);
    }
}

void Conductivity::setup_kappa()
{
    MPI_Bcast(&calc_coherent, 1, MPI_INT, 0, MPI_COMM_WORLD);

    unsigned int i, j, k;

    nk = dos->kmesh_dos->nk;
    ns = dynamical->neval;

    ntemp = static_cast<unsigned int>((system->Tmax - system->Tmin) / system->dT) + 1;
    allocate(temperature, ntemp);

    for (i = 0; i < ntemp; ++i) {
        temperature[i] = system->Tmin + static_cast<double>(i) * system->dT;
    }

    const auto nks_total = dos->kmesh_dos->nk_irred * ns;
    const auto nks_each_thread = nks_total / mympi->nprocs;
    const auto nrem = nks_total - nks_each_thread * mympi->nprocs;

    if (nrem > 0) {
        allocate(damping3, (nks_each_thread + 1) * mympi->nprocs, ntemp);
        allocate(rel_err, (nks_each_thread + 1) * mympi->nprocs, ntemp);
    } else {
        allocate(damping3, nks_total, ntemp);
        allocate(rel_err, nks_total, ntemp);
    }

    const auto factor = Bohr_in_Angstrom * 1.0e-10 / time_ry;

    if (mympi->my_rank == 0) {
        allocate(vel, nk, ns, 3);
        if (calc_coherent) {
            allocate(velmat, nk, ns, ns, 3);
        }
    } else {
        allocate(vel, 1, 1, 1);
        if (calc_coherent) {
            allocate(velmat, 1, 1, 1, 3);
        }
    }

    phonon_velocity->get_phonon_group_velocity_mesh_mpi(*dos->kmesh_dos,
                                                        system->lavec_p,
                                                        vel);
    if (mympi->my_rank == 0) {
        for (i = 0; i < nk; ++i) {
            for (j = 0; j < ns; ++j) {
                for (k = 0; k < 3; ++k) vel[i][j][k] *= factor;
            }
        }
    }

    if (calc_coherent) {
        phonon_velocity->calc_phonon_velmat_mesh(velmat);
        if (calc_coherent == 2) {
            file_coherent_elems = input->job_title + ".kc_elem";
        }
    }

    vks_job.clear();

    for (i = 0; i < dos->kmesh_dos->nk_irred; ++i) {
        for (j = 0; j < ns; ++j) {
            vks_job.insert(i * ns + j);
        }
    }
}

void Conductivity::prepare_restart()
{
    // Write phonon frequency to result file

    int i;
    std::string line_tmp;
    unsigned int nk_tmp, ns_tmp;
    unsigned int multiplicity;
    int nks_done, *arr_done;

    double vel_dummy[3];

    nshift_restart = 0;

    vks_done.clear();

    if (mympi->my_rank == 0) {

        if (!phon->restart_flag) {

            writes->fs_result << "##Phonon Frequency" << std::endl;
            writes->fs_result << "#K-point (irreducible), Branch, Omega (cm^-1)" << std::endl;

            for (i = 0; i < dos->kmesh_dos->nk_irred; ++i) {
                const auto ik = dos->kmesh_dos->kpoint_irred_all[i][0].knum;
                for (auto is = 0; is < dynamical->neval; ++is) {
                    writes->fs_result << std::setw(6) << i + 1 << std::setw(6) << is + 1;
                    writes->fs_result << std::setw(15) << writes->in_kayser(dos->dymat_dos->get_eigenvalues()[ik][is])
                                      << std::
                                      endl;
                }
            }

            writes->fs_result << "##END Phonon Frequency" << std::endl << std::endl;
            writes->fs_result << "##Phonon Relaxation Time" << std::endl;

        } else {

            while (writes->fs_result >> line_tmp) {

                if (line_tmp == "#GAMMA_EACH") {

                    writes->fs_result >> nk_tmp >> ns_tmp;
                    writes->fs_result >> multiplicity;

                    const auto nks_tmp = (nk_tmp - 1) * ns + ns_tmp - 1;

                    for (i = 0; i < multiplicity; ++i) {
                        writes->fs_result >> vel_dummy[0] >> vel_dummy[1] >> vel_dummy[2];
                    }

                    for (i = 0; i < ntemp; ++i) {
                        writes->fs_result >> damping3[nks_tmp][i];
                        damping3[nks_tmp][i] *= time_ry / Hz_to_kayser;
                    }
                    vks_done.push_back(nks_tmp);
                }
            }

            try {
                double* err_calc_tmp;
                allocate(err_calc_tmp,ntemp);
                while (!writes->fs_err.eof()) {
                    for (i = 0; i < ntemp; ++i) {
                        writes->fs_err >> err_calc_tmp[i];  //read calculated error for each T
                    }
                    writes->fs_err >> line_tmp;  //dummy, #GAMMA_EACH
                    writes->fs_err >> nk_tmp >> ns_tmp;
                    const auto nks_tmp = (nk_tmp - 1) * ns + ns_tmp - 1;
                    //for debug
                    //std::cout << nk_tmp << " " << ns_tmp;
                    for (i = 0; i < ntemp; ++i) {
                        rel_err[nks_tmp][i]=err_calc_tmp[i];
                        //for debug
                        //std::cout << " " << rel_err[nks_tmp][i];
                    }
                    //for debug
                    //std::cout << std::endl;
                }
                writes->fs_err.clear();
                deallocate(err_calc_tmp);
            }catch (...){
                std::cerr << "error is detected in MC_err.log" << std::endl;
                std::exit(1);
            }
        }

        writes->fs_result.close();
        writes->fs_result.open(writes->file_result.c_str(), std::ios::app | std::ios::out);
    }

    // Add vks_done list here

    if (mympi->my_rank == 0) {
        nks_done = vks_done.size();
    }
    MPI_Bcast(&nks_done, 1, MPI_INT, 0, MPI_COMM_WORLD);
    nshift_restart = nks_done;

    if (nks_done > 0) {
        allocate(arr_done, nks_done);

        if (mympi->my_rank == 0) {
            for (i = 0; i < nks_done; ++i) {
                arr_done[i] = vks_done[i];
            }
        }
        MPI_Bcast(&arr_done[0], nks_done, MPI_INT, 0, MPI_COMM_WORLD);

        // Remove vks_done elements from vks_job

        for (i = 0; i < nks_done; ++i) {

            const auto it_set = vks_job.find(arr_done[i]);

            if (it_set == vks_job.end()) {
                std::cout << " rank = " << mympi->my_rank
                          << " arr_done = " << arr_done[i] << std::endl;
                exit("prepare_restart", "This cannot happen");
            } else {
                vks_job.erase(it_set);
            }
        }
        deallocate(arr_done);
    }
    //vks_done.clear();
}

void PHON_NS::Conductivity::setup_kappa_mc()
{
    //broadcast vars
    //MPI_Bcast(&calc_kappa_mc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //MPI_Bcast(&nsample_kappa_density, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //setup nsample_kappa
    generate_nsample_kappa();
    generate_kappa_mc_map(dos->kmesh_dos, dos->dymat_dos->get_eigenvalues());
    unsigned int *nshift_tmp;
    allocate(nshift_tmp,ntemp);
    for(int id=0;id<ntemp;id++)nshift_tmp[id]=0;
    //resize map_sample_to_mode
    map_sample_to_mode.assign(ntemp,std::vector<std::vector<unsigned int>>(9));
    generate_kappa_mc_sample(nshift_tmp,nsample_kappa);
    setup_vks_for_mc();
}

void Conductivity::generate_nsample_kappa(){
    const auto nk_irred = dos->kmesh_dos->nk_irred;
    if(nsample_kappa_density <= 0 || nsample_kappa_density > 1){
        for (int i = 0; i < ntemp; ++i) {
            nsample_kappa[i]=static_cast<unsigned int>(nsample_kappa_ini);
        }
    }else{
        for (int i = 0; i < ntemp; ++i) {
            nsample_kappa[i]=static_cast<unsigned int>(nk_irred*ns*nsample_kappa_density);
        }
    }
}

void Conductivity::generate_kappa_mc_map(const KpointMeshUniform *kmesh_in, const double *const *eval_in){
    const auto nk_irred = kmesh_in->nk_irred;
    allocate(weighting_factor_mc,ntemp,9,nk_irred*ns);
    allocate(weighting_factor_map,ntemp,9,nk_irred*ns);

    //generate map
    for (int i = 0; i < ntemp; ++i) {

        if (temperature[i] < eps) {
        } else {
            for (int is = 0; is < ns; ++is) {
                for (int ik = 0; ik < nk_irred; ++ik) {
                    const auto knum = kmesh_in->kpoint_irred_all[ik][0].knum;
                    const auto omega = eval_in[knum][is];
                    const auto nk_equiv = kmesh_in->kpoint_irred_all[ik].size();
                    const double omega_b=pow(omega,coef_b);
                    for (unsigned int j = 0; j < 3; ++j) {
                        for (unsigned int k = 0; k < 3; ++k) {
                            if(vv_dim == "diagonal" || vv_dim == "diagonal_sum"){
                                //if(j != k)continue;
                            }
                            auto vv_tmp = 0.0;

                            // Accumulate group velocity (diad product) for the reducible k points
                            for (auto ieq = 0; ieq < nk_equiv; ++ieq) {
                                const auto ktmp = kmesh_in->kpoint_irred_all[ik][ieq].knum;
                                vv_tmp += vel[ktmp][is][j] * vel[ktmp][is][k];
                            }

                            if (thermodynamics->classical) {
                                weighting_factor_mc[i][3 * j + k][ik * ns + is]
                                        = thermodynamics->Cv_classical(omega, temperature[i])
                                          * vv_tmp * omega_b;
                            } else {
                                weighting_factor_mc[i][3 * j + k][ik * ns + is]
                                        = thermodynamics->Cv(omega, temperature[i])
                                          * vv_tmp * omega_b;
                            }
                            if(is==0 && ik==0){
                                weighting_factor_map[i][3 * j + k][ik * ns + is]=weighting_factor_mc[i][3 * j + k][ik * ns + is];
                            }else{
                                weighting_factor_map[i][3 * j + k][ik * ns + is]=weighting_factor_mc[i][3 * j + k][ik * ns + is-1]
                                                                                    +weighting_factor_mc[i][3 * j + k][ik * ns + is];
                            }
                        }
                    }

                    if(vv_dim == "full"){
                    }else if(vv_dim == "diagonal"){
                        //xy=xz = xx, yx=yz = yy, zx=zy = zz
                        for(int j=0;j<3;j++){
                            for(int k=0;k<3;k++){
                                if(j==k)continue;
                                weighting_factor_mc[i][j*3+k][ik * ns + is] = weighting_factor_mc[i][j*3+j][ik * ns + is];
                                weighting_factor_map[i][j*3+k][ik * ns + is] = weighting_factor_map[i][j*3+j][ik * ns + is];
                            }
                        }
                    }else if(vv_dim == "diagonal_sum"){
                        weighting_factor_mc[i][0][ik * ns + is]=weighting_factor_mc[i][0][ik * ns + is]
                                                                    +weighting_factor_mc[i][4][ik * ns + is]
                                                                    +weighting_factor_mc[i][8][ik * ns + is];
                        weighting_factor_map[i][0][ik * ns + is]=weighting_factor_map[i][0][ik * ns + is]
                                                                    +weighting_factor_map[i][4][ik * ns + is]
                                                                    +weighting_factor_map[i][8][ik * ns + is];
                        for(int j=1;j<9;j++){
                            weighting_factor_mc[i][j][ik * ns + is]=weighting_factor_mc[i][0][ik * ns + is];
                            weighting_factor_map[i][j][ik * ns + is]=weighting_factor_map[i][0][ik * ns + is];
                        }
                    }
                }
            }
        }
    }
}

void PHON_NS::Conductivity::generate_kappa_mc_sample(const unsigned int *nshift, const unsigned int *nsample)
{
    std::mt19937_64 mt(seed);
    const auto nk_irred = dos->kmesh_dos->nk_irred;
    for (int i = 0; i < ntemp; ++i) {
        if (temperature[i] < eps) {
        } else {
            if(vv_dim == "full" || vv_dim == "diagonal"){
                for(int j=0;j<3;j++){
                    for(int k=0;k<3;k++){
                        if(vv_dim == "diagonal" && j != k)continue;

                        if(map_sample_to_mode[i][j*3+k].size() < nshift[i]+nsample[i]){
                            map_sample_to_mode[i][j*3+k].reserve(nshift[i]+nsample[i]);
                        }
                        
                        //generate random values
                        double* rand_val;
                        allocate(rand_val, nsample[i]);
                        std::uniform_real_distribution<> rand_gen(0, 1);
                        
                        double rand_max=weighting_factor_map[i][j*3+k][nk_irred*ns-1];  //upper limit of rand_val
                        for(unsigned int mcid=0;mcid<nsample[i];mcid++){
                            rand_val[mcid]=rand_gen(mt)*rand_max;
                        }
                        std::sort(rand_val,rand_val+nsample[i]);

                        unsigned int tmp=0;
                        for(unsigned int mcid=0;mcid<nsample[i];mcid++){
                            for(int ib=tmp;ib<nk_irred*ns;ib++){
                                if(rand_val[mcid] < weighting_factor_map[i][j*3+k][ib] || ib == nk_irred*ns-1){
                                    tmp=ib;
                                    break;
                                }
                            }
                            map_sample_to_mode[i][j*3+k].push_back(tmp);
                        }
                    }
                }
                //copy to non-diagonal sample set
                if(vv_dim == "diagonal"){
                    for(int j=0;j<3;j++){
                        for(int k=0;k<3;k++){
                            if(j == k)continue;
                            if(map_sample_to_mode[i][j*3+k].size() != nshift[i]){
                                std::cerr << "Error in generating map_sample_to_mode" << std::endl;
                                std::exit(1);
                            }
                            if(map_sample_to_mode[i][j*3+k].size() < nshift[i]+nsample[i]){
                                map_sample_to_mode[i][j*3+k].reserve(nshift[i]+nsample[i]);
                            }
                            for(unsigned int mcid=0;mcid<nsample[i];mcid++){
                                map_sample_to_mode[i][j*3+k].push_back(map_sample_to_mode[i][j*3+j][nshift[i]+mcid]);
                            }
                        }
                    }
                }
            }else{
                if(map_sample_to_mode[i][0].size() < nshift[i]+nsample[i]){
                    map_sample_to_mode[i][0].reserve(nshift[i]+nsample[i]);
                }
                
                //generate random values
                double* rand_val;
                allocate(rand_val, nsample[i]);
                std::uniform_real_distribution<> rand_gen(0, 1);
                
                double rand_max=weighting_factor_map[i][0][nk_irred*ns-1];  //upper limit of rand_val
                for(unsigned int mcid=0;mcid<nsample[i];mcid++){
                    rand_val[mcid]=rand_gen(mt)*rand_max;
                }
                std::sort(rand_val,rand_val+nsample[i]);

                unsigned int tmp=0;
                for(unsigned int mcid=0;mcid<nsample[i];mcid++){
                    for(int ib=tmp;ib<nk_irred*ns;ib++){
                        if(rand_val[mcid] < weighting_factor_map[i][0][ib] || ib == nk_irred*ns-1){
                            tmp=ib;
                            break;
                        }
                    }
                    map_sample_to_mode[i][0].push_back(tmp);
                }
                //copy to other sample set
                for(int j=1;j<9;j++){
                    if(map_sample_to_mode[i][j].size() != nshift[i]){
                        std::cerr << "Error in generating map_sample_to_mode" << std::endl;
                        std::exit(1);
                    }
                    if(map_sample_to_mode[i][j].size() < nshift[i]+nsample[i]){
                        map_sample_to_mode[i][j].reserve(nshift[i]+nsample[i]);
                    }
                    for(unsigned int mcid=0;mcid<nsample[i];mcid++){
                        map_sample_to_mode[i][j].push_back(map_sample_to_mode[i][0][nshift[i]+mcid]);
                    }
                }

            }
        }
    }
}

void PHON_NS::Conductivity::setup_vks_for_mc()
{
    const auto nk_irred = dos->kmesh_dos->nk_irred;
    allocate(gamma_calculated,nk_irred*ns);
    int* gamma_calculated_tmp;
    allocate(gamma_calculated_tmp,nk_irred*ns);
    for(int iks=0;iks<nk_irred*ns;iks++){
        gamma_calculated[iks]=0;
        gamma_calculated_tmp[iks]=0;
    }
    
    //setup calculated map
    for(int i=0;vks_done.size();i++){
        auto iks_tmp = vks_done[i];
        gamma_calculated[iks_tmp]=1;
        gamma_calculated_tmp[iks_tmp]=1;
    }

    //reset vks_job
    vks_job.clear();
    for(int i=0;i<ntemp;i++){
        for(int j=0;j<9;j++){
            for(int mcid=0;mcid<nsample_kappa[i];mcid++){
                auto iks_tmp=map_sample_to_mode[i][j][mcid];
                if(gamma_calculated_tmp[iks_tmp]>0)continue;
                gamma_calculated_tmp[iks_tmp]=1;
                vks_job.insert(iks_tmp);
            }
        }
    }
    deallocate(gamma_calculated_tmp);
}

void Conductivity::calc_anharmonic_imagself()
{
    unsigned int i;
    unsigned int *nks_thread = nullptr;
    double *damping3_loc = nullptr;
    double *std_ret = nullptr;

    //initialize time elapsed
    anharmonic_core->elapsed_com=0;
    anharmonic_core->elapsed_SPS=0;
    anharmonic_core->elapsed_sample=0;
    anharmonic_core->elapsed_V3=0;
    anharmonic_core->elapsed_other=0;

    //setup kappa_mc
    std::vector<int> vks_vec;
    if(calc_kappa_mc>0){
        int nks_g_tmp;
        if (mympi->my_rank == 0) {
            setup_kappa_mc();
            vks_vec.reserve(vks_job.size());
            for (const auto &it: vks_job) {
                vks_vec.push_back(it);
            }
            nks_g_tmp=vks_vec.size();
        }
        MPI_Bcast(&nks_g_tmp, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (mympi->my_rank != 0) {
            vks_vec.resize(nks_g_tmp);
        }
        MPI_Bcast(&vks_vec[0], nks_g_tmp, MPI_INT, 0, MPI_COMM_WORLD);
        if (mympi->my_rank != 0) {
            vks_job.clear();
            for(i=0;i<nks_g_tmp;i++){
                vks_job.insert(vks_vec[i]);
            }
        }
    }

    // Distribute (k,s) to individual MPI threads

    const auto nks_g = vks_job.size();
    vks_l.clear();

    unsigned int icount = 0;

    for (const auto &it: vks_job) {
        if (icount % mympi->nprocs == mympi->my_rank) {
            vks_l.push_back(it);
        }
        ++icount;
    }

    if (mympi->my_rank == 0) {
        allocate(nks_thread, mympi->nprocs);
    }

    auto nks_tmp = vks_l.size();
    MPI_Gather(&nks_tmp, 1, MPI_UNSIGNED, &nks_thread[mympi->my_rank],
               1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (mympi->my_rank == 0) {
        std::cout << std::endl;
        std::cout << " Start calculating anharmonic phonon self-energies ... " << std::endl;
        std::cout << " Total Number of phonon modes to be calculated : " << nks_g << std::endl;
        std::cout << " All modes are distributed to MPI threads as the following :" << std::endl;
        for (i = 0; i < mympi->nprocs; ++i) {
            std::cout << " RANK: " << std::setw(5) << i + 1;
            std::cout << std::setw(8) << "MODES: " << std::setw(5) << nks_thread[i] << std::endl;
        }
        if(anharmonic_core->integration_method>0){
            std::cout << std::endl;
            std::cout << " Monte-Carlo integration is active : " << std::endl;
            std::cout << "  Monte-Carlo integration mode is " << anharmonic_core->integration_method
             << std::endl;
            std::cout << "  Sample type is ";
            if(anharmonic_core->use_sample_density){
                std::cout << "density : " << anharmonic_core->sample_density*100
                 << "%" << std::endl;
            }else{
                std::cout << "number : " << anharmonic_core->nsample_input << std::endl;
            }
        }
        std::cout << std::endl << std::flush;

        deallocate(nks_thread);
    }

    unsigned int nk_tmp;

    if (nks_g % mympi->nprocs != 0) {
        nk_tmp = nks_g / mympi->nprocs + 1;
    } else {
        nk_tmp = nks_g / mympi->nprocs;
    }

    if (vks_l.size() < nk_tmp) {
        vks_l.push_back(-1);
    }

    allocate(damping3_loc, ntemp);
    allocate(std_ret,ntemp);

    for (i = 0; i < nk_tmp; ++i) {

        const auto iks = vks_l[i];

        if (iks == -1) {

            for (unsigned int j = 0; j < ntemp; ++j) damping3_loc[j] = eps; // do nothing

        } else {

            const auto knum = dos->kmesh_dos->kpoint_irred_all[iks / ns][0].knum;
            const auto snum = iks % ns;

            const auto omega = dos->dymat_dos->get_eigenvalues()[knum][snum];
            
            if (integration->ismear == 0 || integration->ismear == 1) {
                
                if(anharmonic_core->integration_method<=0){
                    anharmonic_core->calc_damping_smearing(ntemp,
                                                       temperature,
                                                       omega,
                                                       iks / ns,
                                                       snum,
                                                       dos->kmesh_dos,
                                                       dos->dymat_dos->get_eigenvalues(),
                                                       dos->dymat_dos->get_eigenvectors(),
                                                       damping3_loc);
                }else{
                    anharmonic_core->calc_damping_smearing_MC(ntemp,
                                                       temperature,
                                                       omega,
                                                       iks / ns,
                                                       snum,
                                                       dos->kmesh_dos,
                                                       dos->dymat_dos->get_eigenvalues(),
                                                       dos->dymat_dos->get_eigenvectors(),
                                                       damping3_loc,
                                                       std_ret);
                }
            } else if (integration->ismear == -1) {
                if(anharmonic_core->integration_method<=0){
                    anharmonic_core->calc_damping_tetrahedron(ntemp,
                                                          temperature,
                                                          omega,
                                                          iks / ns,
                                                          snum,
                                                          dos->kmesh_dos,
                                                          dos->dymat_dos->get_eigenvalues(),
                                                          dos->dymat_dos->get_eigenvectors(),
                                                          damping3_loc);
                }else{
                    anharmonic_core->calc_damping_tetrahedron_MC(ntemp,
                                                          temperature,
                                                          omega,
                                                          iks / ns,
                                                          snum,
                                                          dos->kmesh_dos,
                                                          dos->dymat_dos->get_eigenvalues(),
                                                          dos->dymat_dos->get_eigenvectors(),
                                                          damping3_loc,
                                                          std_ret);
                }
            }
        }
        std::chrono::system_clock::time_point  start, now;
        if (mympi->my_rank == 0) {
            start = std::chrono::system_clock::now();
        }

        if(calc_kappa_mc>0){
            double *damping3_tmp;
            double *rel_err_tmp;
            if (mympi->my_rank == 0) {
                allocate(damping3_tmp, ntemp * mympi->nprocs);
                allocate(rel_err_tmp, ntemp * mympi->nprocs);
            }
            MPI_Gather(&damping3_loc[0], ntemp, MPI_DOUBLE,
                    &damping3_tmp[0], ntemp,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
            
            MPI_Gather(&std_ret[0], ntemp, MPI_DOUBLE,
                    &rel_err_tmp[0], ntemp,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (mympi->my_rank == 0) {
                for(int id=0;id<mympi->nprocs;id++){
                    if(i*mympi->nprocs+id > vks_job.size())break;
                    auto iks_mc=vks_vec[i*mympi->nprocs+id];
                    gamma_calculated[iks_mc]=1;
                    for(int id_temp=0;id_temp<ntemp;id_temp++){
                        damping3[iks_mc][id_temp]=damping3_tmp[id*ntemp+id_temp];
                        rel_err[iks_mc][id_temp]=rel_err_tmp[id*ntemp+id_temp];
                    }
                }
            }
        }else{
            MPI_Gather(&damping3_loc[0], ntemp, MPI_DOUBLE,
                    damping3[nshift_restart + i * mympi->nprocs], ntemp,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
            
            MPI_Gather(&std_ret[0], ntemp, MPI_DOUBLE,
                    rel_err[nshift_restart + i * mympi->nprocs], ntemp,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        if (mympi->my_rank == 0) {
            now = std::chrono::system_clock::now();
            anharmonic_core->elapsed_com += std::chrono::duration_cast<std::chrono::milliseconds>(now-start).count();
            start = std::chrono::system_clock::now();
        }

        if (mympi->my_rank == 0) {
            //if MC_integration is active, 
            //note that some message is already shown in function calc_damping_smearing_MC
            if(calc_kappa_mc>0){
                //write_result_gamma_each(iks_mc, vel, damping3);
            }else{
                write_result_gamma(i, nshift_restart, vel, damping3);
            }
            if(anharmonic_core->integration_method<=0){
                std::cout << " MODE " << std::setw(5) << i + 1 << " done." << std::endl << std::flush;
            }else{
                write_result_err(i, nshift_restart, vel, rel_err);
                //std::cout << anharmonic_core->std_ret[0];
                std::cout << " MODE " << std::setw(5) << i + 1 << ", SPS:"
                 << anharmonic_core->elapsed_SPS  << ", sample:" 
                 << anharmonic_core->elapsed_sample  << ", V3:" 
                 << anharmonic_core->elapsed_V3  << ", other:"
                 << anharmonic_core->elapsed_other  << ", com:"  
                 << anharmonic_core->elapsed_com << std::endl << std::flush;
            }
        }
    }
    deallocate(damping3_loc);
    deallocate(std_ret);
}

void Conductivity::write_result_gamma(const unsigned int ik,
                                      const unsigned int nshift,
                                      double ***vel_in,
                                      double **damp_in) const
{
    const unsigned int np = mympi->nprocs;
    unsigned int k;

    for (unsigned int j = 0; j < np; ++j) {

        const auto iks_g = ik * np + j + nshift;

        if (iks_g >= dos->kmesh_dos->nk_irred * ns) break;

        writes->fs_result << "#GAMMA_EACH" << std::endl;
        writes->fs_result << iks_g / ns + 1 << " " << iks_g % ns + 1 << std::endl;

        const auto nk_equiv = dos->kmesh_dos->kpoint_irred_all[iks_g / ns].size();

        writes->fs_result << nk_equiv << std::endl;
        for (k = 0; k < nk_equiv; ++k) {
            const auto ktmp = dos->kmesh_dos->kpoint_irred_all[iks_g / ns][k].knum;
            writes->fs_result << std::setw(15) << vel_in[ktmp][iks_g % ns][0];
            writes->fs_result << std::setw(15) << vel_in[ktmp][iks_g % ns][1];
            writes->fs_result << std::setw(15) << vel_in[ktmp][iks_g % ns][2] << std::endl;
        }

        for (k = 0; k < ntemp; ++k) {
            writes->fs_result << std::setw(15)
                              << damp_in[iks_g][k] * Hz_to_kayser / time_ry << std::endl;
        }
        writes->fs_result << "#END GAMMA_EACH" << std::endl;
    }
}

void Conductivity::write_result_err(const unsigned int ik,
                                      const unsigned int nshift,
                                      double ***vel_in,
                                      double **ret_err) const
{
    const unsigned int np = mympi->nprocs;
    unsigned int k;

    for (unsigned int j = 0; j < np; ++j) {

        const auto iks_g = ik * np + j + nshift;

        if (iks_g >= dos->kmesh_dos->nk_irred * ns) break;

        for (k = 0; k < ntemp; ++k) {
            writes->fs_err << std::setw(15) << std::scientific
                              << ret_err[iks_g][k];
        }

        writes->fs_err << " #" << "GAMMA_EACH  ";
        writes->fs_err << iks_g / ns + 1 << " " << iks_g % ns + 1 << std::endl;
    }
}

void Conductivity::compute_kappa()
{
    unsigned int i;
    unsigned int iks;

    if (mympi->my_rank == 0) {

        std::string file_kl;
        std::ofstream ofs_kl;
        double damp_tmp;

        double **lifetime;
        double **gamma_total;

        allocate(lifetime, dos->kmesh_dos->nk_irred * ns, ntemp);
        allocate(gamma_total, dos->kmesh_dos->nk_irred * ns, ntemp);

        average_self_energy_at_degenerate_point(dos->kmesh_dos->nk_irred * ns,
                                                ntemp,
                                                dos->kmesh_dos,
                                                dos->dymat_dos->get_eigenvalues(),
                                                damping3);

        if (isotope->include_isotope) {
            for (iks = 0; iks < dos->kmesh_dos->nk_irred * ns; ++iks) {
                const auto snum = iks % ns;
                if (dynamical->is_imaginary[iks / ns][snum]) {
                    for (i = 0; i < ntemp; ++i) {
                        lifetime[iks][i] = 0.0;
                        gamma_total[iks][i] = 1.0e+100; // very big number
                    }
                } else {
                    for (i = 0; i < ntemp; ++i) {
                        damp_tmp = damping3[iks][i] + isotope->gamma_isotope[iks / ns][snum];
                        gamma_total[iks][i] = damp_tmp;
                        if (damp_tmp > 1.0e-100) {
                            lifetime[iks][i] = 1.0e+12 * time_ry * 0.5 / damp_tmp;
                        } else {
                            lifetime[iks][i] = 0.0;
                        }
                    }
                }
            }
        } else {
            for (iks = 0; iks < dos->kmesh_dos->nk_irred * ns; ++iks) {

                if (dynamical->is_imaginary[iks / ns][iks % ns]) {
                    for (i = 0; i < ntemp; ++i) {
                        lifetime[iks][i] = 0.0;
                        gamma_total[iks][i] = 1.0e+100; // very big number
                    }
                } else {
                    for (i = 0; i < ntemp; ++i) {
                        damp_tmp = damping3[iks][i];
                        gamma_total[iks][i] = damp_tmp;
                        if (damp_tmp > 1.0e-100) {
                            lifetime[iks][i] = 1.0e+12 * time_ry * 0.5 / damp_tmp;
                        } else {
                            lifetime[iks][i] = 0.0;
                            gamma_total[iks][i] = 1.0e+100;
                        }
                    }
                }
            }
        }

        allocate(kappa, ntemp, 3, 3);

        if (calc_kappa_spec) {
            allocate(kappa_spec, dos->n_energy, ntemp, 3);
        }

        compute_kappa_intraband(dos->kmesh_dos,
                                dos->dymat_dos->get_eigenvalues(),
                                lifetime,
                                kappa,
                                kappa_spec);
        deallocate(lifetime);

        if (calc_coherent) {
            allocate(kappa_coherent, ntemp, 3, 3);
            compute_kappa_coherent(dos->kmesh_dos,
                                   dos->dymat_dos->get_eigenvalues(),
                                   gamma_total,
                                   kappa_coherent);
        }

        deallocate(gamma_total);
    }
}

void Conductivity::average_self_energy_at_degenerate_point(const int n,
                                                           const int m,
                                                           const KpointMeshUniform *kmesh_in,
                                                           const double *const *eval_in,
                                                           double **damping) const
{
    int j, k, l;
    const auto nkr = kmesh_in->nk_irred;

    double *eval_tmp;
    const auto tol_omega = 1.0e-7; // Approximately equal to 0.01 cm^{-1}

    std::vector<int> degeneracy_at_k;

    allocate(eval_tmp, ns);

    double *damping_sum;

    allocate(damping_sum, m);

    for (auto i = 0; i < nkr; ++i) {
        const auto ik = kmesh_in->kpoint_irred_all[i][0].knum;

        for (j = 0; j < ns; ++j) eval_tmp[j] = eval_in[ik][j];

        degeneracy_at_k.clear();

        auto omega_prev = eval_tmp[0];
        auto ideg = 1;

        for (j = 1; j < ns; ++j) {
            const auto omega_now = eval_tmp[j];

            if (std::abs(omega_now - omega_prev) < tol_omega) {
                ++ideg;
            } else {
                degeneracy_at_k.push_back(ideg);
                ideg = 1;
                omega_prev = omega_now;
            }
        }
        degeneracy_at_k.push_back(ideg);

        int is = 0;
        for (j = 0; j < degeneracy_at_k.size(); ++j) {
            ideg = degeneracy_at_k[j];

            if (ideg > 1) {

                for (l = 0; l < m; ++l) damping_sum[l] = 0.0;

                for (k = is; k < is + ideg; ++k) {
                    for (l = 0; l < m; ++l) {
                        damping_sum[l] += damping[ns * i + k][l];
                    }
                }

                for (k = is; k < is + ideg; ++k) {
                    for (l = 0; l < m; ++l) {
                        damping[ns * i + k][l] = damping_sum[l] / static_cast<double>(ideg);
                    }
                }
            }

            is += ideg;
        }
    }
    deallocate(damping_sum);
}

void Conductivity::compute_kappa_intraband(const KpointMeshUniform *kmesh_in,
                                           const double *const *eval_in,
                                           const double *const *lifetime,
                                           double ***kappa_intra,
                                           double ***kappa_spec_out) const
{
    int i, is, ik;
    double ****kappa_mode;
    const auto factor_toSI = 1.0e+18 / (std::pow(Bohr_in_Angstrom, 3) * system->volume_p);

    const auto nk_irred = kmesh_in->nk_irred;
    allocate(kappa_mode, ntemp, 9, ns, nk_irred);

    double ***kappa_err;
    if(anharmonic_core->integration_method>0){
        allocate(kappa_err,ntemp,3,3);
    }

    for (i = 0; i < ntemp; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            for (unsigned int k = 0; k < 3; ++k) {

                if (temperature[i] < eps) {
                    // Set kappa as zero when T = 0.
                    for (is = 0; is < ns; ++is) {
                        for (ik = 0; ik < nk_irred; ++ik) {
                            kappa_mode[i][3 * j + k][is][ik] = 0.0;
                        }
                    }
                } else {
                    for (is = 0; is < ns; ++is) {
                        for (ik = 0; ik < nk_irred; ++ik) {
                            const auto knum = kmesh_in->kpoint_irred_all[ik][0].knum;
                            const auto omega = eval_in[knum][is];
                            auto vv_tmp = 0.0;
                            const auto nk_equiv = kmesh_in->kpoint_irred_all[ik].size();

                            // Accumulate group velocity (diad product) for the reducible k points
                            for (auto ieq = 0; ieq < nk_equiv; ++ieq) {
                                const auto ktmp = kmesh_in->kpoint_irred_all[ik][ieq].knum;
                                vv_tmp += vel[ktmp][is][j] * vel[ktmp][is][k];
                            }

                            if (thermodynamics->classical) {
                                kappa_mode[i][3 * j + k][is][ik]
                                        = thermodynamics->Cv_classical(omega, temperature[i])
                                          * vv_tmp * lifetime[ns * ik + is][i];
                            } else {
                                kappa_mode[i][3 * j + k][is][ik]
                                        = thermodynamics->Cv(omega, temperature[i])
                                          * vv_tmp * lifetime[ns * ik + is][i];
                            }

                            // Convert to SI unit
                            kappa_mode[i][3 * j + k][is][ik] *= factor_toSI;

                        }
                    }
                }

                kappa_intra[i][j][k] = 0.0;
                if(anharmonic_core->integration_method>0){
                    kappa_err[i][j][k] = 0.0;
                }

                for (is = 0; is < ns; ++is) {
                    for (ik = 0; ik < nk_irred; ++ik) {
                        kappa_intra[i][j][k] += kappa_mode[i][3 * j + k][is][ik];
                        if(anharmonic_core->integration_method>0){
                            /*for debug*/
                            //std::cout << "rel_err[" << ns * ik + is << "][" << i << "] = " << rel_err[ns * ik + is][i] << " ";
                            //std::cout << "kappa_mode[" << is << "][" << ik << "] = " << kappa_mode[i][3 * j + k][is][ik] << std::endl;
                            kappa_err[i][j][k] += rel_err[ns * ik + is][i]*rel_err[ns * ik + is][i]
                                                    *kappa_mode[i][3 * j + k][is][ik]*kappa_mode[i][3 * j + k][is][ik];
                                                    //variance of kappa_mode

                            //kappa_err[i][j][k] += rel_err[ns * ik + is][i]*std::abs(kappa_mode[i][3 * j + k][is][ik]);
                            // calculate simple summation of standard error 
                            // when there is a bias, this estimation is more reasonable
                        }
                    }
                }
                if(anharmonic_core->integration_method>0){
                    kappa_err[i][j][k]=sqrt(kappa_err[i][j][k])/kappa_intra[i][j][k];
                    // calculate relative deviation of kappa

                    // kappa_err[i][j][k] = kappa_err[i][j][k]/kappa_intra[i][j][k];
                    // calculate error due to bias
                }
                kappa_intra[i][j][k] /= static_cast<double>(nk);
            }
        }
    }

    if (calc_kappa_spec) {
        //allocate(kappa_spec_out, dos->n_energy, ntemp, 3);
        compute_frequency_resolved_kappa(ntemp,
                                         integration->ismear,
                                         dos->kmesh_dos,
                                         dos->dymat_dos->get_eigenvalues(),
                                         kappa_mode,
                                         kappa_spec_out);
    }

    deallocate(kappa_mode);
    if(anharmonic_core->integration_method>0){
        std::cout << std::endl << "Thermal conductivity calculation is finished" << std::endl;
        std::cout << " Estimated error sigma_err : " << std::endl;
        std::cout << " T [K] |kxx       |kxy       |kxz       |kyx       |kyy       |kyz       |kzx       |kzy       |kzz [%]    " << std::endl;
        for (i = 0; i < ntemp; ++i) {
            std::cout << "  " << std::setw(6) << std::fixed << static_cast<int>(temperature[i]);
            for (unsigned int j = 0; j < 3; ++j) {
                for (unsigned int k = 0; k < 3; ++k) {
                    std::cout << std::fixed << std::setw(11) << kappa_err[i][j][k]*100;
                }
            }
            std::cout << std::endl;
        }
        deallocate(kappa_err);
    }
}

void PHON_NS::Conductivity::compute_kappa_intraband_with_mc(const KpointMeshUniform *kmesh_in,
                                                            const double *const *eval_in,
                                                            const double *const *lifetime,
                                                            const int nshift_sample,
                                                            double ***kappa_sample,
                                                            double ***sample_error_tau,
                                                            double ***sample_error_mc,
                                                            double ***kappa_intra,
                                                            double ***kappa_intra_error_tau_out,
                                                            double ***kappa_intra_error_mc_out,
                                                            double ***kappa_spec_out) const
{
    int i, is, ik;
    //double ***kappa_sample;
    const auto factor_toSI = 1.0e+18 / (std::pow(Bohr_in_Angstrom, 3) * system->volume_p);

    const auto nk_irred = kmesh_in->nk_irred;
    //allocate(kappa_sample, ntemp, 9, nsample_kappa);

    for (i = 0; i < ntemp; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            for (unsigned int k = 0; k < 3; ++k) {

                if (temperature[i] < eps) {
                    // Set kappa as zero when T = 0.
                    for (int id_mc = nshift_sample; id_mc < nsample_kappa[i]; ++id_mc) {
                        kappa_sample[i][3 * j + k][id_mc] = 0.0;
                    }
                } else {
                    for (int id_mc = nshift_sample; id_mc < nsample_kappa[i]; ++id_mc) {
                        
                        if(map_sample_to_mode[i][3 * j + k][id_mc]<0)break;  //already sampling is finished
                        ik = map_sample_to_mode[i][3 * j + k][id_mc]/ns;
                        is = map_sample_to_mode[i][3 * j + k][id_mc]%ns;
                        const auto knum = kmesh_in->kpoint_irred_all[ik][0].knum;
                        const auto omega = eval_in[knum][is];
                        auto vv_tmp = 0.0;
                        const auto nk_equiv = kmesh_in->kpoint_irred_all[ik].size();

                        // Accumulate group velocity (diad product) for the reducible k points
                        for (auto ieq = 0; ieq < nk_equiv; ++ieq) {
                            const auto ktmp = kmesh_in->kpoint_irred_all[ik][ieq].knum;
                            vv_tmp += vel[ktmp][is][j] * vel[ktmp][is][k];
                        }

                        if (thermodynamics->classical) {
                            kappa_sample[i][3 * j + k][id_mc]
                                    = thermodynamics->Cv_classical(omega, temperature[i])
                                        * vv_tmp * lifetime[ns * ik + is][i]
                                        / weighting_factor_mc[i][3 * j + k][ns * ik + is];
                        } else {
                            kappa_sample[i][3 * j + k][id_mc]
                                    = thermodynamics->Cv(omega, temperature[i])
                                        * vv_tmp * lifetime[ns * ik + is][i]
                                        / weighting_factor_mc[i][3 * j + k][ns * ik + is];
                        }

                        // Convert to SI unit
                        kappa_sample[i][3 * j + k][id_mc] *= factor_toSI;

                        //
                        sample_error_mc[i][3 * j + k][id_mc] = kappa_sample[i][3 * j + k][id_mc]*kappa_sample[i][3 * j + k][id_mc];
                        if(anharmonic_core->integration_method>0){
                            sample_error_tau[i][3 * j + k][id_mc] = rel_err[ns * ik + is][i]*rel_err[ns * ik + is][i]
                                                    *kappa_sample[i][3 * j + k][id_mc]*kappa_sample[i][3 * j + k][id_mc];
                                                    //variance of kappa_sample

                            //kappa_err[i][j][k] += rel_err[ns * ik + is][i]*std::abs(kappa_sample[i][3 * j + k][id_mc]);
                            // calculate simple summation of standard error 
                            // when there is a bias, this estimation is more reasonable
                        }
                    }
                }

                kappa_intra[i][j][k] = 0.0;
                kappa_intra_error_mc_out[i][j][k]=0;
                kappa_intra_error_tau_out[i][j][k]=0.0;

                for (int id_mc = 0; id_mc < nsample_kappa[i]; ++id_mc) {
                    if(map_sample_to_mode[i][3 * j + k][id_mc]<0)break;  //already sampling is finished
                    ik = map_sample_to_mode[i][3 * j + k][id_mc]/ns;
                    is = map_sample_to_mode[i][3 * j + k][id_mc]%ns;
                    kappa_intra[i][j][k] += kappa_sample[i][3 * j + k][id_mc];
                    kappa_intra_error_mc_out[i][j][k] += sample_error_mc[i][3 * j + k][id_mc];
                    if(anharmonic_core->integration_method>0){
                        kappa_intra_error_tau_out[i][j][k]+=sample_error_tau[i][3 * j + k][id_mc];
                    }
                }
                if(kappa_intra[i][j][k] < 1.0e-100){
                    kappa_intra_error_tau_out[i][j][k]=0;
                    if(anharmonic_core->integration_method>0){
                        kappa_intra_error_tau_out[i][j][k]=0;
                    }
                }else{
                    kappa_intra_error_tau_out[i][j][k]=sqrt(kappa_intra_error_tau_out[i][j][k]/nsample_kappa[i]
                                                         - pow(kappa_intra[i][j][k]/nsample_kappa[i],2))
                                                         /(nsample_kappa[i]-1)
                                                         /(kappa_intra[i][j][k]/nsample_kappa[i]);
                    if(anharmonic_core->integration_method>0){
                        kappa_intra_error_tau_out[i][j][k]=sqrt(kappa_intra_error_tau_out[i][j][k])/kappa_intra[i][j][k];
                        // calculate relative deviation of kappa

                        // kappa_err[i][j][k] = kappa_err[i][j][k]/kappa_intra[i][j][k];
                        // calculate error due to bias
                    }
                }
                kappa_intra[i][j][k] *= weighting_factor_map[i][3 * j + k][nk*ns-1];
                kappa_intra[i][j][k] /= nsample_kappa[i];
                kappa_intra[i][j][k] /= static_cast<double>(nk);
            }
        }
    }

    if (calc_kappa_spec) {/*
        //allocate(kappa_spec_out, dos->n_energy, ntemp, 3);
        compute_frequency_resolved_kappa_mc(ntemp,
                                         integration->ismear,
                                         dos->kmesh_dos,
                                         dos->dymat_dos->get_eigenvalues(),
                                         kappa_sample,
                                         kappa_spec_out);*/
    }

    //deallocate(kappa_sample);
    //std::cout << std::endl << "Thermal conductivity calculation is finished" << std::endl;
    std::cout << " Calculated kappa : " << std::endl;
    std::cout << " T [K] |kxx       |kxy       |kxz       |kyx       |kyy       |kyz       |kzx       |kzy       |kzz [%]    " << std::endl;
    for (i = 0; i < ntemp; ++i) {
        std::cout << "  " << std::setw(6) << std::fixed << static_cast<int>(temperature[i]);
        for (unsigned int j = 0; j < 3; ++j) {
            for (unsigned int k = 0; k < 3; ++k) {
                std::cout << std::fixed << std::setw(10) << kappa_intra[i][j][k] << " ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << " Estimated error sigma_err : " << std::endl;
    std::cout << " T [K] |kxx       |kxy       |kxz       |kyx       |kyy       |kyz       |kzx       |kzy       |kzz [%]    " << std::endl;
    for (i = 0; i < ntemp; ++i) {
        std::cout << "  " << std::setw(6) << std::fixed << static_cast<int>(temperature[i]);
        for (unsigned int j = 0; j < 3; ++j) {
            for (unsigned int k = 0; k < 3; ++k) {
                std::cout << std::fixed << std::setw(10) << (kappa_intra_error_mc_out[i][j][k]+kappa_intra_error_tau_out[i][j][k])*100 << " ";
            }
        }
        std::cout << std::endl;
    }
}

void Conductivity::compute_kappa_coherent(const KpointMeshUniform *kmesh_in,
                                          const double *const *eval_in,
                                          const double *const *gamma_total,
                                          double ***kappa_coherent_out) const
{
    // Compute the coherent part of thermal conductivity
    // based on the Michelle's paper.
    int ib;
    const auto factor_toSI = 1.0e+18 / (std::pow(Bohr_in_Angstrom, 3) * system->volume_p);
    const auto common_factor = factor_toSI * 1.0e+12 * time_ry / static_cast<double>(nk);
    const auto common_factor_output = factor_toSI * 1.0e+12 * time_ry;
    const int ns2 = ns * ns;
    const auto czero = std::complex<double>(0.0, 0.0);
    std::vector<std::complex<double>> kappa_tmp(ns2, czero);
    std::complex<double> **kappa_save = nullptr;

    const auto nk_irred = kmesh_in->nk_irred;

    std::ofstream ofs;
    if (calc_coherent == 2) {
        ofs.open(file_coherent_elems.c_str(), std::ios::out);
        if (!ofs) exit("compute_kappa_coherent", "cannot open file_kc");
        ofs << "# Temperature [K], 1st and 2nd xyz components, ibranch, jbranch, ik_irred, "
               "omega1 [cm^-1], omega2 [cm^-1], kappa_elems real, kappa_elems imag" << std::endl;
        allocate(kappa_save, ns2, nk_irred);
    }

    for (auto i = 0; i < ntemp; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            for (unsigned int k = 0; k < 3; ++k) {

                kappa_coherent_out[i][j][k] = 0.0;

                if (temperature[i] > eps) {
#pragma omp parallel for
                    for (ib = 0; ib < ns2; ++ib) {
                        kappa_tmp[ib] = czero;
                        const int is = ib / ns;
                        const int js = ib % ns;

                        if (js == is) continue; // skip the diagonal component

                        for (auto ik = 0; ik < nk_irred; ++ik) {
                            const auto knum = kmesh_in->kpoint_irred_all[ik][0].knum;
                            const auto omega1 = eval_in[knum][is];
                            const auto omega2 = eval_in[knum][js];

                            if (omega1 < eps8 || omega2 < eps8) continue;
                            auto vv_tmp = czero;
                            const auto nk_equiv = kmesh_in->kpoint_irred_all[ik].size();

                            // Accumulate group velocity (diad product) for the reducible k points
                            for (auto ieq = 0; ieq < nk_equiv; ++ieq) {
                                const auto ktmp = kmesh_in->kpoint_irred_all[ik][ieq].knum;
                                vv_tmp += velmat[ktmp][is][js][j] * velmat[ktmp][js][is][k];
                            }
                            auto kcelem_tmp = 2.0 * (omega1 * omega2) / (omega1 + omega2)
                                              * (thermodynamics->Cv(omega1, temperature[i]) / omega1
                                                 + thermodynamics->Cv(omega2, temperature[i]) / omega2)
                                              * 2.0 * (gamma_total[ik * ns + is][i] + gamma_total[ik * ns + js][i])
                                              / (4.0 * std::pow(omega1 - omega2, 2.0)
                                                 + 4.0 * std::pow(gamma_total[ik * ns + is][i]
                                                                  + gamma_total[ik * ns + js][i], 2.0))
                                              * vv_tmp;
                            kappa_tmp[ib] += kcelem_tmp;

                            if (calc_coherent == 2 && j == k) {
                                kappa_save[ib][ik] = kcelem_tmp * common_factor_output;
                            }
                        }
                    } // end OpenMP parallelization over ib

                    for (ib = 0; ib < ns2; ++ib) {
                        if (std::abs(kappa_tmp[ib].imag()) > eps10) {
                            warn("compute_kappa_coherent",
                                 "The kappa_coherent_out has imaginary component.");
                        }
                        kappa_coherent_out[i][j][k] += kappa_tmp[ib].real();
                    }

                    if (calc_coherent == 2 && j == k) {
                        for (ib = 0; ib < ns2; ++ib) {

                            const int is = ib / ns;
                            const int js = ib % ns;

                            for (auto ik = 0; ik < nk_irred; ++ik) {
                                if (is == js) kappa_save[ib][ik] = czero;

                                ofs << std::setw(5) << temperature[i];
                                ofs << std::setw(3) << j + 1 << std::setw(3) << k + 1;
                                ofs << std::setw(4) << is + 1;
                                ofs << std::setw(4) << js + 1;
                                ofs << std::setw(6) << ik + 1;
                                const auto knum = kmesh_in->kpoint_irred_all[ik][0].knum;
                                const auto omega1 = eval_in[knum][is];
                                const auto omega2 = eval_in[knum][js];
                                ofs << std::setw(15) << writes->in_kayser(omega1);
                                ofs << std::setw(15) << writes->in_kayser(omega2);
                                ofs << std::setw(15) << kappa_save[ib][ik].real();
                                ofs << std::setw(15) << kappa_save[ib][ik].imag();
                                ofs << '\n';
                            }
                        }
                        ofs << '\n';
                    }
                }
                kappa_coherent_out[i][j][k] *= common_factor;
            }
        }
    }

    if (calc_coherent == 2) {
        ofs.close();
        deallocate(kappa_save);
    }
}

void Conductivity::compute_frequency_resolved_kappa(const int ntemp,
                                                    const int smearing_method,
                                                    const KpointMeshUniform *kmesh_in,
                                                    const double *const *eval_in,
                                                    const double *const *const *const *kappa_mode,
                                                    double ***kappa_spec_out) const
{
    int i, j;
    unsigned int *kmap_identity;
    double **eval;

    std::cout << std::endl;
    std::cout << " KAPPA_SPEC = 1 : Calculating thermal conductivity spectra ... ";

    allocate(kmap_identity, nk);
    allocate(eval, ns, nk);

    for (i = 0; i < nk; ++i) kmap_identity[i] = i;

    for (i = 0; i < nk; ++i) {
        for (j = 0; j < ns; ++j) {
            eval[j][i] = writes->in_kayser(eval_in[i][j]);
        }
    }

#ifdef _OPENMP
#pragma omp parallel private (j)
#endif
    {
        int k;
        int knum;
        double *weight;
        allocate(weight, nk);

#ifdef _OPENMP
#pragma omp for
#endif
        for (i = 0; i < dos->n_energy; ++i) {

            for (j = 0; j < ntemp; ++j) {
                for (k = 0; k < 3; ++k) {
                    kappa_spec_out[i][j][k] = 0.0;
                }
            }

            for (int is = 0; is < ns; ++is) {
                if (smearing_method == -1) {
                    integration->calc_weight_tetrahedron(nk, kmap_identity,
                                                         eval[is], dos->energy_dos[i],
                                                         dos->tetra_nodes_dos->get_ntetra(),
                                                         dos->tetra_nodes_dos->get_tetras(),
                                                         weight);
                } else {
                    integration->calc_weight_smearing(nk, nk, kmap_identity,
                                                      eval[is], dos->energy_dos[i],
                                                      smearing_method, weight);
                }

                for (j = 0; j < ntemp; ++j) {
                    for (k = 0; k < 3; ++k) {
                        for (int ik = 0; ik < kmesh_in->nk_irred; ++ik) {
                            knum = kmesh_in->kpoint_irred_all[ik][0].knum;
                            kappa_spec_out[i][j][k] += kappa_mode[j][3 * k + k][is][ik] * weight[knum];
                        }
                    }
                }
            }
        }
        deallocate(weight);
    }

    deallocate(kmap_identity);
    deallocate(eval);

    std::cout << " done!" << std::endl;
}
