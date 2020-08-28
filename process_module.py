import argparse
import processing as pr


proposal = 2530
pattern = ['image', 'dark'] * 99 + ['end_image']


def process_module(proposal, run, module, pattern):
    module = pr.Module(proposal=proposal, run=run, module=module, pattern=pattern)
    
    # Remove train_indices!!! Here just for testing.
    #module.process_std(train_indices=range(10), dirname='processed_runs_xgm')
    
    module.process_normalised(dark_run=49, train_indices=range(10), dirname='processed_runs_xgm')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-number', metavar='S',
                        action='store',
                        help='the run to be processed')
    parser.add_argument('--module', metavar='S',
                        action='store',
                        help='module to be processed')
    args = parser.parse_args()

    process_module(proposal, int(args.run_number), int(args.module), pattern)