import os
import argparse
import processing as pr


proposal = 2530
pattern = ['image', 'dark'] * 99 + ['end_image']
dirname = 'processed_runs_xgm'

if not os.path.exists(dirname):
    os.makedirs(dirname)


def process_module(proposal, run, module, pattern, dark_run, xgm_threshold):
    module = pr.Module(proposal=proposal, run=run,
                       module=module, pattern=pattern)

    if dark_run == 0:
        module.process_std(dirname=dirname)
    else:
        module.process_normalised(dark_run=dark_run,
                                  xgm_threshold=xgm_threshold,
                                  dirname=dirname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-number', metavar='S',
                        action='store',
                        help='the run to be processed')
    parser.add_argument('--module', metavar='S',
                        action='store',
                        help='module to be processed')
    parser.add_argument('--dark-run', metavar='S',
                        action='store',
                        help='dark run number')
    parser.add_argument('--xgm-lower', metavar='S',
                        action='store',
                        help='lower XGM threshold')
    parser.add_argument('--xgm-upper', metavar='S',
                        action='store',
                        help='upper XGM threshold')
    args = parser.parse_args()

    process_module(proposal, run=int(args.run_number),
                   module=int(args.module), pattern=pattern,
                   dark_run=int(args.dark_run),
                   xgm_threshold=(float(args.xgm_lower),
                                  float(args.xgm_upper)))
