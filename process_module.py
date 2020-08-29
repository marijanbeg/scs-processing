import argparse
import processing as pr


proposal = 2530
pattern = ['image', 'dark'] * 99 + ['end_image']


def process_module(proposal, run, module, pattern, dark_run):
    module = pr.Module(proposal=proposal, run=run, module=module, pattern=pattern)
    
    if dark_run == 0:
        # Dark run
        module.process_std(dirname='../../Shared/processed_runs_xgm')
    else:
        module.process_normalised(dark_run=dark_run, dirname='../../Shared/processed_runs_xgm')


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
    args = parser.parse_args()

    process_module(proposal, int(args.run_number), int(args.module), pattern, int(args.dark_run))