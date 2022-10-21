import subprocess


def run_test(team_size=-1, vector_length=-1, nsolves_team=1, ortho_strategy=0, temp_data_strategy=0, shared_level=0, extra_options=''):
    default_options = '-dim 2 -dm_landau_amr_levels_max 2,2,2,2,2 -dm_landau_amr_post_refine 0 -dm_landau_batch_size 1 '
    default_options += '-dm_landau_batch_view_idx 0 -dm_landau_device_type kokkos -dm_landau_ion_charges 1,2,3,4 '
    default_options += '-dm_landau_ion_masses 2,32,64,128 -dm_landau_n 1.0000009,1,1e-7,1e-7,1e-7 -dm_landau_num_species_grid 1,1,1,1,1 '
    default_options += '-dm_landau_thermal_temps 2,1,1,1,1 -dm_landau_type p4est -dm_landau_verbose 2 '
    default_options += '-dm_mat_type aijkokkos -dm_preallocate_only false -dm_vec_type kokkos -ex2_grid_view_idx 0 '
    default_options += '-ksp_type preonly -pc_bjkokkos_ksp_converged_reason -pc_bjkokkos_ksp_max_it 150 '
    default_options += '-pc_bjkokkos_ksp_rtol 1e-12 -pc_bjkokkos_ksp_type gmres -pc_bjkokkos_pc_type jacobi '
    default_options += '-pc_type bjkokkos -petscspace_degree 3 -snes_converged_reason -snes_max_it 40 '
    default_options += '-snes_monitor -snes_rtol 1e-14 -snes_stol 1e-14 -ts_adapt_monitor -ts_dt .5 '
    default_options += '-ts_exact_final_time stepover -ts_max_snes_failures -1 -ts_max_steps 0 -ts_monitor -ts_type beuler '

    exe = './ex2 '
    exe += default_options
    exe += '-dm_landau_batch_size 100 -ts_max_steps 1  -log_view ' 
    exe += '-pc_bjkokkos_ksp_batch_nsolves_team ' + str(nsolves_team) + ' '
    exe += '-pc_bjkokkos_ksp_batch_team_size ' + str(team_size) + ' '
    exe += '-pc_bjkokkos_ksp_batch_vector_length ' + str(vector_length) + ' '
    exe += '-pc_bjkokkos_ksp_batch_ortho_strategy ' + str(ortho_strategy) + ' '
    exe += '-pc_bjkokkos_ksp_batch_temp_data_strategy ' + str(temp_data_strategy) + ' '
    exe += '-pc_bjkokkos_ksp_batch_shared_level ' + str(shared_level) + ' '
    exe += extra_options
    exe += '| tee out'
    print('exe = ' + exe)
    subprocess.call(exe, shell=True)

    process = subprocess.Popen(['grep', 'KSPSolve', 'out'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    out_array = out.split()
    if len(out_array) > 30:
        return float(out_array[30])
    return -1


if __name__ == "__main__":
    print(run_test())
