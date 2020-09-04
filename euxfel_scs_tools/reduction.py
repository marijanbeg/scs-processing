import os
import time
import numpy as np
import subprocess as sp


base_script = ('import os, sys\n'
               'sys.path.append("'
               f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}'
               '")\nimport euxfel_scs_tools as scs\n'
               'module = scs.Module(proposal={proposal}, run={run}, '
               'module=$MODULE$, pattern={pattern})\n'
               )


def reduction_sum(proposal, run, pattern,
                  frame_type, trains=None, njobs=40, dirname=None):
    script = (base_script.format(proposal=proposal, run=run, pattern=pattern) +
              f'module.reduce_sum(frame_type="{frame_type}", trains={trains},'
              f' njobs={njobs}, dirname="{os.path.abspath(dirname)}")\n')
    _submit_jobs(script)


def reduction_sum_norm(proposal, run, pattern,
                       dark_run,
                       frames={'image': 'image',
                               'dark': 'dark'},
                       dark_run_frames={'image': 'image',
                                        'dark': 'dark'},
                       trains=None,
                       njobs=40,
                       dirname=None):
    script = (base_script.format(proposal=proposal, run=run, pattern=pattern) +
              f'module.reduce_sum_norm(dark_run={dark_run}, frames={frames}, '
              f'dark_run_frames={dark_run_frames}, trains={trains}, '
              f'njobs={njobs}, dirname="{os.path.abspath(dirname)}")\n')
    _submit_jobs(script)


def reduction_std(proposal, run, pattern,
                  frame_types=None, trains=None, njobs=40, dirname=None):
    script = (base_script.format(proposal=proposal, run=run, pattern=pattern) +
              f'module.reduce_std(frame_types={frame_types}, trains={trains}, '
              f'njobs={njobs}, dirname="{os.path.abspath(dirname)}")\n')
    _submit_jobs(script)


def reduction_norm(proposal, run, pattern,
                   dark_run,
                   frames={'image': 'image',
                           'dark': 'dark'},
                   dark_run_frames={'image': 'image',
                                    'dark': 'dark'},
                   trains=None, xgm_threshold=(1e-5, np.inf),
                   njobs=40,
                   dirname=None):
    script = (base_script.format(proposal=proposal, run=run, pattern=pattern) +
              f'module.reduce_norm(dark_run={dark_run}, '
              f'frames={frames}, dark_run_frames={dark_run_frames}, '
              f'trains={trains}, xgm_threshold={xgm_threshold}, '
              f'njobs={njobs}, dirname="{os.path.abspath(dirname)}")')
    # Replaces inf because "np." is missing.
    _submit_jobs(script.replace('inf),', 'np.inf),'))


def _submit_jobs(py_script, slurm_dir='slurm_log',
                 script_dir='autogenerated_scripts', modules=range(16)):
    if not os.path.exists(script_dir):
        os.makedirs(script_dir)

    if not os.path.exists(slurm_dir):
        os.makedirs(slurm_dir)

    for module in modules:
        filebasename = f'run_{time.time()}_module{module}'

        # Python file.
        with open(os.path.join(f'{script_dir}',
                               f'{filebasename}.py'), 'w') as f:
            f.write(py_script.replace('$MODULE$', str(module)))

        # Bash file.
        process_sh = ('#!/bin/bash\n'
                      'source /usr/share/Modules/init/bash\n'
                      'module load exfel\n'
                      'module load exfel_anaconda3/1.1\n'
                      f'python3 {filebasename}.py')

        with open(os.path.join(f'{script_dir}',
                               f'{filebasename}.sh'), 'w') as f:
            f.write(process_sh)

        # Submit job to the queue.
        command = ['sbatch', '-p', 'upex', '-t', '100',
                   '--chdir', f'{script_dir}', '-o',
                   f'{os.path.abspath(slurm_dir)}/slurm-%A-{filebasename}.out',
                   f'{filebasename}.sh']
        sp.run(command, stdout=sp.PIPE)

    print(f'Submitted {len(modules)} slurm jobs to the queue. '
          'Please wait for jobs to complete.')
