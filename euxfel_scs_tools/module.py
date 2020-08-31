import os
import h5py
import joblib
import numpy as np
import extra_data as ed
from .xgm import XGM
from .train import Train
from .util import save_h5, job_chunks


class Module:
    def __init__(self, proposal, run, module, pattern):
        """Class used for processing individual modules.

        Parameters
        ----------
        proposal : int

            Proposal number.

        run : int

            Run number.

        module : int

            Module number.

        pattern : list

            Pattern is a list of strings marking the type of each frame in the
            train. For instance, `['image', 'dark', 'image', 'dark', ...,
            'end_image']`. By convention, the last frame (image frame) in the
            train is called `'end_image'` and it is nto processed because it
            does not have its corresponding dark frame.

        """
        self.proposal = proposal
        self.run = run
        self.module = module
        self.pattern = pattern

        # Run object.
        self.orun = ed.open_run(proposal=proposal,
                                run=run).select(self.selector, 'image.data')

        # XGM object
        self.xgm = XGM(proposal=proposal, run=run, module=module,
                       pattern=pattern)

    @property
    def selector(self):
        """Module's device name.

        Returns
        -------
        str

            Module's device name.

        """
        return f'SCS_DET_DSSC1M-1/DET/{self.module}CH0:xtdf'

    @property
    def ntrains(self):
        """Number of trains in the run.

        Returns
        -------
        int

            Number of trains in the run.

        """
        return len(self.orun.train_ids)

    def nframes(self, frame_type):
        return np.count_nonzero(np.array(self.pattern) == frame_type)

    def train(self, index):
        """Extracts Train object for the train with `index`.

        XGM data is propagated to the Train object.

        Parameters
        ----------
        index : int

            Index of the train (not its ID). For instance, 0, 1, 2, 3,...

        Returns
        -------
        Train

            Train object.

        """
        train_id, train_data = self.orun.train_from_index(index)
        return Train(train_id=train_id, train_data=train_data,
                     pattern=self.pattern)

    def sum_frame(self, frame_type, trains, njobs=40):
        """Sums all individual frame values."""
        # If train indices are not specified, all trains are processed.
        if trains is None:
            trains = range(self.ntrains)

        if 'dark' in frame_type:
            msg = f'XGM value cannot be extracted for {frame_type}.'
            raise ValueError(msg)

        def thread_func(job_trains):
            s = []
            for i in job_trains:
                train = self.train(i)

                if train.valid and train.train_id in self.xgm:
                    frame_sum = np.sum(train[frame_type].data, axis=(1, 2, 3),
                                       dtype='int64')
                    xgm_values = self.xgm.frame_data(train.train_id,
                                                     frame_type)

                s += list(zip(frame_sum, xgm_values.values))

            return s

        # Run jobs in parallel. Default backend is used at the moment.
        ranges = job_chunks(njobs, trains)  # distribute trains

        res = joblib.Parallel(n_jobs=njobs)(
            joblib.delayed(thread_func)(i) for i in ranges)

        # Sum results from individual jobs and return.
        repacked_tuple = tuple(zip(*[j for i in res for j in i]))

        return np.array(repacked_tuple[0]), np.array(repacked_tuple[1])

    def reduce_sum(self, frame_types=None, trains=None, njobs=40,
                   dirname=None):
        """Standard processing.

        Parameters
        ----------
        frame_types : array_like

            A list of unique frame types to be processed. If `None` all frame
            types present in the pattern will be processed. Deafults to `None`.

        trains : list

            List of trains to be processed. Defaults to `None`, in which case
            all trains in the run are processed.

        njobs : int

            The number of jobs in parallel processing. Defaults to 10.

        dirname : str

            Directory name in which resulting HDF5 file is saved. Defaults to
            `None` and result is returned instead of saved.

        """
        if frame_types is None:
            frame_types = [i for i in self.pattern if 'dark' not in i]

        summed_frames = {}
        for frame_type in frame_types:
            key = f'{frame_type}_sum'
            summed_frames[key] = self.sum_frame(frame_type,
                                                trains=trains,
                                                njobs=njobs)

        # Save data if dirname is specified.
        if dirname is not None:
            dirname += f'/run_{self.run}/'
            filename = f'module_{self.module}_sum.h5'
            save_h5(summed_frames, dirname, filename)

        return summed_frames

    def average_frame(self, frame_type, trains=None, njobs=40):
        """This method computes the average of all frames through trains.

        Parameters
        ----------
        frame_type : str

            It can be any of the frame types passed in pattern.

        trains : array_like

            Indices of trains to be processed. If `None`, all trains in the run
            are processed. Defaults to 'None'.

        njobs : int

            The number of jobs in parallel processing. Defaults to 10.

        Returns
        -------
        numpy.ndarray

            Shape: (number of frame_type frames in the train, xdim, ydim)

        """
        # If train indices are not specified, all trains are processed.
        if trains is None:
            trains = range(self.ntrains)

        # Function ran on an single thread.
        def thread_func(job_trains):
            accumulator = np.zeros((self.nframes(frame_type), 1, 128, 512),
                                   dtype='float64')
            counter = 0

            # We iterate through all trains.
            for i in job_trains:
                train = self.train(i)  # extract the train object
                if train.valid:  # Train is valid if it contains image.data.
                    accumulator += train[frame_type].data
                    counter += 1

            return accumulator, counter

        # Run jobs in parallel. Default backend is used at the moment.
        ranges = job_chunks(njobs, trains)  # distribute trains
        res = joblib.Parallel(n_jobs=njobs)(
            joblib.delayed(thread_func)(i) for i in ranges)

        # Extract and sum results from individual jobs.
        total_sum = sum(list(zip(*res))[0])
        total_number = sum(list(zip(*res))[1])

        # Compute average and "squeeze" to remove empty dimension.
        return np.squeeze(total_sum / total_number)

    def reduce_std(self, frame_types=None, trains=None, njobs=40,
                   dirname=None):
        """Standard processing.

        Parameters
        ----------
        frame_types : array_like

            A list of unique frame types to be processed. If `None` all frame
            types present in the pattern will be processed. Deafults to `None`.

        trains : list

            List of trains to be processed. Defaults to `None`, in which case
            all trains in the run are processed.

        njobs : int

            The number of jobs in parallel processing. Defaults to 10.

        dirname : str

            Directory name in which resulting HDF5 file is saved. Defaults to
            `None` and result is returned instead of saved.

        """
        if frame_types is None:
            frame_types = set(self.pattern)

        averaged_frames = {}
        for frame_type in frame_types:
            key = f'{frame_type}_average'
            averaged_frames[key] = self.average_frame(frame_type,
                                                      trains=trains,
                                                      njobs=njobs)

        # Save data if dirname is specified.
        if dirname is not None:
            dirname += f'/run_{self.run}/'
            filename = f'module_{self.module}_std.h5'
            save_h5(averaged_frames, dirname, filename)

        return averaged_frames

    def reduce_norm(self,
                    dark_run,
                    frames={'image': 'image',
                            'dark': 'dark'},
                    dark_run_frames={'image': 'image',
                                     'dark': 'dark'},
                    trains=None,
                    xgm_threshold=(1e-5, np.inf),
                    njobs=40,
                    dirname=None):
        """Normalisation processing.

        This processing does the following:

            1. It loads the results of an already processed dark run (RD). This
            run was processed using `process_std`.

            2. It computes the difference between image and dark averages ().


         computes the average of all dark and image frames
        across all trains. If `dirname` is provided, two numbpy arrays are
        saved: `dark_average` and `image_average` to an HDF5. If
        `train_indices` is `None`, all trains are computed.

        Parameters
        ----------
        train_indices : list

            List of trains to be processed. Defaults to `None`, in which case
            all trains in the run are processed.

        dirname : str

            Directory name in which resulting HDF5 file is saved. Defaults to
            `None` and result is returned instead of saved.

        """
        if trains is None:
            trains = range(self.ntrains)

        # First, we compute the average of darks.
        dark_average = self.average_frame(frame_type=frames['dark'],
                                          trains=trains, njobs=njobs)

        # Second, we load image_average and dark_average for the dark run and
        # compute their difference.
        filename = os.path.join(dirname, f'run_{dark_run}',
                                f'module_{self.module}_std.h5')
        with h5py.File(filename, 'r') as f:
            dark_run_diff = (f[f'{dark_run_frames["image"]}_average'][:] -
                             f[f'{dark_run_frames["dark"]}_average'][:])

        # This is the value we subtract from each image frame before we
        # normalise it by XGM value.
        sval = dark_average + dark_run_diff

        def thread_func(job_trains):
            accumulator = np.zeros((self.nframes(frames['image']), 128, 512),
                                   dtype='float64')
            counter = 0

            for i in job_trains:
                train = self.train(i)

                if train.valid and train.train_id in self.xgm:
                    images = train[frames['image']]
                    xgm_vals = self.xgm.frame_data(train_id=train.train_id,
                                                   frame_type=frames['image'])

                    s = np.zeros((images.n, 128, 512), dtype='float64')
                    for j in range(images.n):
                        if xgm_threshold[0] < xgm_values[j] < xgm_threshold[1]:
                            s[j, ...] = (np.squeeze(images.data[j, ...]) -
                                         sval[j, ...]) / xgm_vals[j].values

                    accumulator += s
                    counter += 1

            return accumulator, counter

        ranges = job_chunks(njobs, trains)
        res = joblib.Parallel(n_jobs=njobs)(
            joblib.delayed(thread_func)(i) for i in ranges)

        # Extract and sum results from individual jobs.
        total_sum = sum(list(zip(*res))[0])
        total_number = sum(list(zip(*res))[1])

        # Compute the average of frames.
        average = total_sum / total_number

        # Save data if dirname is specified.
        if dirname is not None:
            dirname += f'/run_{self.run}/'
            filename = f'module_{self.module}_norm.h5'
            save_h5({f'{frames["image"]}_average': average}, dirname, filename)

        return average
