#############################################################
#  Script for processing a run                              #
#  All required parameters can be specified in              #
#  the first part of this script                            #
#############################################################

#
# per run parameters
# ==================
#

# predefined patterns
# more can be added if needed
base_patterns = {1: ['image', 'dark'],
                 2: ['pumped', 'dark1', 'unpumped', 'dark2']
                }

run_number = 50
dark_run = 49  # only required for 'run_normalised'
xgm_min = 0.1  # only required for 'run_normalised'
xgm_max = 0.5  # only required for 'run_normalised'
pattern = 1  # see available patterns above
total_frames = 199  # total number of frames per train


# select which normalisation to perform
# multiple normalisations can be selected
# _Note_ run_normalised requires an existing dark run

run_is_darkrun = False  # process 'run_number' as dark run
run_std = True  # process 'run_number' as standard run
run_normalised = False  # process 'run_number' as normalised run

run_xgm = False # not yet implemented
run_pump_probe = False  # not yet implemented

#
# additional parameters
# =====================
#

proposal = 2530
output_dir = 'processed_runs_xgm'
slurm_log_dir = 'slurm_log'
log_level = 'info'  # can be 'info', 'warning' or 'error'


#############################################################
#                                                           #
#  Excecution part                                          #
#  processing is done as specified above                    #
#                                                           #
#  No user interaction is required in the remaining part    #
#  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    #
#                                                           #
#############################################################


import os
import processing as pr
import subprocess as sp
import time


def init_log(log_level):
    if log_level == 'info':
        log_levels = ['info', 'warning', 'error']
    elif log_level == 'warning':
        log_levels = ['warning', 'error']
    else:
        log_levels = ['error']
    
    def wrapped(msg, level):
        if level in log_levels:
            print(str.upper(level), '\t', msg)
    return wrapped


def submit_jobs(pr_func, module_range=range(16), **kwargs):
    for module in module_range:
        process_sh = ('#!/bin/bash\n'
                      'source /usr/share/Modules/init/bash\n'
                      'module load exfel\n'
                      'module load exfel_anaconda3/1.1\n'
                      'python -c "import processing; '
                      f'processing.{pr_func}(**{kwargs}, module={module})"'
                     )
        file_name = f'process{time.time()}.sh'  # to avoid overriding files
        with open(file_name, 'w') as f:
            f.write(process_sh)
        command = ['sbatch', '-p', 'upex', '-t', '100', '-o',
                   f'{slurm_log_dir}/slurm-%A.out', file_name]
        res = sp.run(command, stdout=sp.PIPE)
        print_log(res.stdout.decode('utf-8', 'replace'), 'info')
        os.remove(file_name)


if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(slurm_log_dir):
        os.makedirs(slurm_log_dir)

    print_log = init_log(log_level)
        
    print_log('Starting the processing', 'info')
    print_log(f'Processing run {run_number}', 'info')
    print_log(f'Darkrun is run {dark_run}', 'info')
    
    
    if run_is_darkrun:
        if dark_run != 0:
            print_log((f'Run {run_number} is interpreted as darkrun; '
                       f'ignoring darkrun {dark_run} in the calculation'),
                       'warning')
        if run_std or run_normalised or run_xgm:
            print_log(f'Running darkrun. All other run requests are ignored.',
                      'info')
        
    if not run_is_darkrun and not run_std and not run_normalised and not run_xgm:
        print_log('No processing specified!', 'error')
    
    base_pattern = base_patterns[pattern]
    full_pattern = (base_pattern * (total_frames // len(base_pattern))
                    + ['end_image'] * (total_frames % len(base_pattern)))
    
    if run_is_darkrun:
        print_log('Submitting jobs for the dark run', 'info')
        submit_jobs('standard_run',
                    proposal=proposal,
                    run=run_number,
                    pattern=full_pattern,
                    dirname=output_dir)
    else:
        if run_std:
            print_log('Submitting jobs for the standard run', 'info')
            if  dark_run != 0:
                print_log((f'ignoring darkrun {dark_run}; no darkrun'
                           ' substraction is performed in standard processing'),
                          'info')

            submit_jobs('standard_run',
                        proposal=proposal,
                        run=run_number,
                        pattern=full_pattern,
                        dirname=output_dir)
        if run_normalised and xgm_min is not None and xgm_max is not None:
            print_log('Submitting jobs for the normalised run', 'info')
            submit_jobs('normalised_run',
                        proposal=proposal,
                        run=run_number,
                        pattern=full_pattern,
                        dark_run=dark_run,
                        xgm_threshold=(xgm_min, xgm_max),
                        dirname=output_dir)
        elif run_normalised:
            print_log(("xgm_min and xgm_max must not be 'None'"
                       " for the normalised run"),
                      'error')
        if run_pump_probe:
            print_log('Submitting jobs for the pump-probe run', 'info')
            submit_jobs('pump_probe_run',
                        proposal=proposal,
                        run=run_number,
                        pattern=full_pattern,
                        dirname=output_dir)
        if run_xgm:
            print_log('Submitting jobs for the xgm-run', 'info')
            submit_jobs('xgm_run',
                        proposal=proposal,
                        run=run_number,
                        pattern=full_pattern,
                        dirname=output_dir)
