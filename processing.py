import os
import h5py
import joblib
import numpy as np
import extra_data as ed
import xarray as xr


def save_h5(data, dirname, filename):
    """Saves data in HDF5 file.

    The file is saved as dirname/filename.

    data (numpy.ndarray) - numpy array to be saved
    dirname (str) - directory (will be created if it does not exist)
    filename (str) - should have extension .h5

    """

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    h5file = os.path.join(f'{dirname}', f'{filename}')
    with h5py.File(h5file, 'w') as f:
        f.create_dataset('data', data=data)


def job_chunks(n_jobs, ntrains):
    """Splitting ntrains to n_jobs.

    n_jobs (int) - how many cores are employed.
    ntrains (int) - the total number of trains to be processed

    """
    # Number of trains per process.
    chunk = int(np.ceil(ntrains / n_jobs))

    total_range = range(ntrains)
    ranges = [total_range[step:step + chunk]
              for step in np.arange(ntrains, step=chunk)]

    return ranges


class Train:
    """Class for processing individual trains."""
    def __init__(self, data, pattern, xgm):
        self.pattern = np.array(pattern)
        self.xgm = xgm
        
        # image.data might be missing
        try:
            self.data = data[list(data.keys())[0]]['image.data']
        except:
            self.data = None

    @property
    def valid(self):
        """If there is no data in train, False is returned. Otherwise, True."""
        if self.data is not None:
            return True
        else:
            return False

    def __getitem__(self, frame_type):
        return Container(self.data[self.pattern==frame_type, ...],
                         xgm=self.xgm)


class Container:
    def __init__(self, data, xgm):
        self.data = data
        self.xgm = xgm

    @property
    def n(self):
        """The number of frames in the container."""
        return self.data.shape[0]

    def __add__(self, other):
        return self.__class__(data=self.data+other.data, xgm=None)

    def process(self, normalised=True, subtraction_value=0):
        """Sums all frames in the container and counts how many trains were added.

        Returns a tuple: (sum of frames, number of summed frames)

        normalised (bool) - If True, from each image subtraction_value is subtracted.
                            The result is then divided by XGM value.

        """
        if normalised:
            frames_sum = 0
            frames_num = 0
            for frame in range(self.n):
                # The value of XGM can be zero.
                # This means that division would result in error or it can be very large, so that the final sum is corrupted.
                # Marijan: I arbitrarily chose this small value (1e-5)
                if self.xgm[frame] > 1e-5:  # discard low reading of XGM
                    frame_value = self.data[frame, ...]  # frame values
                    xgm_value = self.xgm[frame].values  # value of XGM for an image frame

                    frames_sum += (frame_value - subtraction_value) / xgm_value

                    # One frame processed -> counter incremented.
                    frames_num += 1

            return frames_sum, frames_num
        else:
            # If no normalisation is required, simple sum is computed.
            return np.sum(self.data, axis=0), self.n


class XGM:
    def __init__(self, proposal, run, module, pattern):
        """Class used for extraction and using XGM data.

        proposal (int)- proposal number
        run (int) - run number
        module (int) - module number (0-15)
        pattern (list) - for instance:
            'image', 'dark', 'image', 'dark', ... , 'end_image'

        Length of pattern list should be the same as the number of frames per
        train.

        """
        self.proposal = proposal
        self.run = run
        self.module = module
        self.pattern = pattern

        # Run object.
        str1 = 'SCS_BLU_XGM/XGM/DOOCS:output'
        str2 = 'data.intensitySa3TD'
        orun = ed.open_run(proposal=self.proposal,
                           run=self.run).select(str1, str2)
        
        # Read data
        self.data = orun.get_array(str1, str2)

    @property
    def n(self):
        """The number of XGM values.

        It is the same as the number of images in pattern.

        """
        return np.count_nonzero(np.array(self.pattern) == 'image')

    def train(self, index):
        return self.data[index, 0:self.n]


class Module:
    def __init__(self, proposal, run, module, pattern):
        """Class used for processing individual modules.

        proposal (int)- proposal number
        run (int) - run number
        module (int) - module number (0-15)
        pattern (list) - for instance:
            'image', 'dark', 'image', 'dark', ... , 'end_image'

        Length of pattern list should be the same as the number of frames per
        train.

        """
        self.proposal = proposal
        self.run = run
        self.module = module
        self.pattern = pattern

        # Run object.
        self.orun = ed.open_run(proposal=proposal,
                                run=run).select(self.selector, 'image.data')

        # XGM object
        self.xgm = XGM(proposal, run, module, pattern)

    @property
    def selector(self):
        """Module's device name."""
        return f'SCS_DET_DSSC1M-1/DET/{self.module}CH0:xtdf'

    @property
    def ntrains(self):
        """Number of trains in the run."""
        return len(self.orun.train_ids)

    @property
    def fpt(self):
        """Number of frames per train."""
        return self.orun.detector_info(self.selector)['frames_per_train']

    def train(self, index):
        """Returns train object.

        index (int) - 0, 1, 2, 3,....

        """
        _, data = self.orun.train_from_index(index)
        return Train(data=data, pattern=self.pattern,
                     xgm=self.xgm.train(index))

    def process_frames(self, frame_type, train_indices=None, dirname=None):
        """Sums frames of frame_type. Result is an average of frames.

        No normalisation is done here and it is used for processing dark runs.

        frame_type - 'image' or 'dark'
        train_indices - list of train indices to be summed.
                        If None, all trains are processed.
        dirname - if specified, data is saved as an HDF5 file in dirname.

        """
        # If train indices are not specified, all trains are processed.
        if train_indices is None:
            train_indices = range(self.ntrains)

        # Function for parallel processing
        def parallel_function(trains):
            trains_sum = 0  # sum of frames are added to it
            trains_num = 0  # number of summed frames are added to it

            # We iterate through all trains.
            for i in trains:
                train = self.train(i)  # get train object

                # Train is valid if it contains image.data.
                if train.valid:
                    trains_sum += train[frame_type].data
                    trains_num += 1

            return trains_sum, trains_num

        n_jobs = 10  # number of cores - can be exposed later
        ranges = job_chunks(n_jobs, len(train_indices))

        res = joblib.Parallel(n_jobs=10)(joblib.delayed(parallel_function)(i) for i in ranges)

        trains_sum = sum(list(zip(*res))[0])
        trains_num = sum(list(zip(*res))[1])

        # Compute average.
        trains_average = trains_sum / trains_num

        # Save data if dirname is specified.
        if dirname is not None:
            dirname += f'/run_{self.run}/'
            filename = f'module_{self.module}_{frame_type}.h5'
            save_h5(trains_average, dirname, filename)
        else:
            return trains_average

    def process_std(self, train_indices=None, dirname=None):
        """Standard processing.

        Result is: average(images) - average(darks).

        """
        darks_average = self.process_frames(frame_type='darks', train_indices=train_indices)
        images_average = self.process_frames(frame_type='images', train_indices=train_indices)

        diff = images_average - darks_average

        # Save data if dirname is specified.
        if dirname is not None:
            dirname += f'/run_{self.run}/'
            filename = f'module_{self.module}_diff.h5'
            save_h5(diff, dirname, filename)
        else:
            return diff

    def process_normalised(self, dark_run, train_indices=None, dirname=None):
        """Processing with normalisation.

        dark_run - dark run number for which the data has already been processed and saved.
        train_indices - indices of trains to be processed. If not specified, all trains in the run are summed.

        """
        # If train indices are not specified, all trains are processed.
        if train_indices is None:
            train_indices = range(self.ntrains)

        # First, we compute the average of darks (intradarks). No normalisation.
        darks_average = self.process_frames(frame_type='darks', train_indices=train_indices)

        # Second, we load average(images) - average(darks) for the dark run.
        # We assume the data has been saved already.
        filename = os.path.join('processed_runs_xgm', f'run_{dark_run}', f'module_{self.module}_diff.h5')
        with h5py.File(filename, 'r') as f:
            dark_run_data = f['data'][:]

        # This is the value we subtract from each image before we normalise it with XGM value.
        subtraction_value = darks_average + dark_run_data

        def parallel_function(trains):
            # For details of the following code, please refer to the previous function.
            frames_sum = 0
            frames_num = 0

            for i in trains:
                train = self.train(i)

                if train.valid:
                    # Please note we pass subtraction value here.
                    s, n = train.images.process(normalised=True, subtraction_value=subtraction_value)
                    frames_sum += s
                    frames_num += n

            return frames_sum, frames_num

        n_jobs = 10  # number of cores - can be exposed later
        ranges = job_chunks(n_jobs, len(train_indices))

        res = joblib.Parallel(n_jobs=10)(joblib.delayed(parallel_function)(i) for i in ranges)

        frames_sum = sum(list(zip(*res))[0])
        frames_num = sum(list(zip(*res))[1])

        # Compute the average of frames.
        frames_average = frames_sum / frames_num

        # Save data if dirname is specified.
        if dirname is not None:
            dirname += f'/run_{self.run}/'
            filename = f'module_{self.module}_normalised.h5'
            save_h5(frames_average, dirname, filename)
        else:
            return frames_average


def concat_module_images(dirname, run, image_type='normalised'):
    """Concatenate corrected module images to get one xarray for the whole detector
    
    This function expects normalised images for all modules in a subdirectory of the
    given basedirectory. 
    
    dirname (str) - the base directory in which processed runs are stored
    run (int) - the run number
    image_type (str) - the type of images to use. Can be one of
        'normalised', 'diff', 'image', 'dark'
    """
    data = np.empty((16,128,512), dtype=np.float64)
    for module_number in range(16):
        filename = os.path.join(dirname, f'run_{run}', f'module_{module_number}_{image_type}.h5')
        with h5py.File(filename, 'r') as f:
            module_data = f['data'][:]
        data[module_number, ...] = np.squeeze(module_data)
    return xr.DataArray(data, dims=('module', 'x', 'y'),
                        coords={'module': range(16), 'x': range(1,129), 'y': range(1,513)})
