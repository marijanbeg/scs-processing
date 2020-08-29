import os
import h5py
import joblib
import numpy as np
import xarray as xr
import extra_data as ed


def save_h5(data, dirname, filename):
    """Saves data in HDF5 file.

    `data` is a dictionary whose keys are the names of datasets in HDF5 file
    and values are numpy arrays. For multiple items in dictionary, multiple
    datasets are saved. The file is saved as `dirname/filename` and filename
    should have `.h5` extension. If the directory does not exist, it is going
    to be created.

    Parameters
    ----------
    data : dict

        Dictionary whose keys are the names of datasets in HDF5 file
        and values are numpy arrays.

    dirname : str

        Directory name.

    filename : str

        Name of the file which includes `.h5` extension.

    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    h5file = os.path.join(f'{dirname}', f'{filename}')
    with h5py.File(h5file, 'w') as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)


def job_chunks(njobs, ntrains):
    """Function for splitting an array of trains into chunks for parallel
    processing.

    Parameters
    ----------
    n_jobs : int

        The number of jobs/threads.

    ntrains : int

        The number of trains to be processed.

    """
    # The number of trains per process. The first njobs-1 will process chunk
    # number of trains, whereas the last job is going to process the rest.
    chunk = int(np.ceil(ntrains / njobs))

    total_range = range(ntrains)
    return [total_range[step:step + chunk]
            for step in np.arange(ntrains, step=chunk)]


class Train:
    def __init__(self, data, pattern, xgm):
        """Class representing a single train in the run.

        Parameters
        ----------
        data : train_data

            Data obtained by using extra_data.

        pattern : list

            Pattern is a list of strings marking the type of each frame in the
            train. For instance, `['image', 'dark', 'image', 'dark', ...,
            'end_image']`. By convention, the last frame (image frame) in the
            train is called `'end_image'` and it is nto processed because it
            does not have its corresponding dark frame.

        xgm : numpy.ndarray

            Numpy array of length which is the same as the number of images in
            the train. It contains XGM values, which can then be used for
            normalisation.

        """
        self.pattern = np.array(pattern)
        self.xgm = xgm

        # image.data might be missing. In that case None is assigned and later
        # used for train validation.
        try:
            self.data = data[list(data.keys())[0]]['image.data']
        except:
            self.data = None

    @property
    def valid(self):
        """Check whether train contains image.data.

        Returns
        -------
        bool

            `True` if train contains image.data, `False` otherwise.

        """
        if self.data is not None:
            return True
        else:
            return False

    def __getitem__(self, frame_type):
        """Extracting frames of the same type as a `Container`.

        Parameters
        ----------
        frame_type : str

            It can be `'image'`, `'dark'`, or `'end_image'`.

        Returns
        -------
        Container

            Container containing only data for the frames of the same type.

        """
        return Container(self.data[self.pattern==frame_type, ...])


class Container:
    def __init__(self, data):
        """Class for performing operations on frames of the same type.

        Parameters
        ----------
        data : np.ndarray

            Numpy array whose first dimension are individual frames.

        """
        self.data = data

    @property
    def n(self):
        """The number of frames in the container.

        Returns
        -------
        int

            The number of frames in the container.

        """
        return self.data.shape[0]

    def __add__(self, other):
        """Binary `+` operator.

        Parameters
        ----------
        other : Container

            Second operand.

        """
        return self.__class__(data=self.data+other.data)


class XGM:
    def __init__(self, proposal, run, module, pattern):
        """Class used for extraction and manipulation of XGM data.

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

        # Strings for getting XGM values from the data.
        str1 = 'SCS_BLU_XGM/XGM/DOOCS:output'
        str2 = 'data.intensitySa3TD'

        # Run object.
        orun = ed.open_run(proposal=self.proposal,
                           run=self.run).select(str1, str2)

        # Read data
        self.data = orun.get_array(str1, str2)

    @property
    def n(self):
        """The number of meaningful XGM values.

        Returns
        -------
        int

            This is the number of XGM values which correspond to particular
            image frames. It is the same as the number of images in pattern.

        """
        return np.count_nonzero(np.array(self.pattern) == 'image')

    def train(self, index):
        """Extract XGM data for an individual train.

        Parameters
        ----------
        index

            Index of the train (not its ID). For instance, 0, 1, 2, 3,...

        Returns
        -------
        numpy.ndarray

            An array of XGM values which contains the same number of elements
            as the number of image frames.

        """
        return self.data[index, 0:self.n]


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
        self.xgm = XGM(proposal, run, module, pattern)

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

    @property
    def fpt(self):
        """Number of Frames Per Train (FPT).

        Returns
        -------
        int

            Number of frames in each train.

        """
        return self.orun.detector_info(self.selector)['frames_per_train']

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
        _, data = self.orun.train_from_index(index)
        return Train(data=data, pattern=self.pattern,
                     xgm=self.xgm.train(index))

    def process_frames(self, frame_type, train_indices=None):
        """This method computes the average of all frames trgough trains.

        More precisely, it computes the average frame 1 across all trains,
        frame 2, frame 3, etc. No XGM normalisation is performed in this
        method. It is used for processing dark frames, as well as the image
        frames where normalisation is not important.

        Parameters
        ----------
        frame_type : str

            It can be `'image'`, `'dark'`, or `'end_image'`.

        train_indices : list

            If `None`, all trains in the run are processed.

        Returns
        -------
        numpy.ndarray

            Shape: (number of frame_type frames in the train, xdim, ydim)

        """
        # If train indices are not specified, all trains are processed.
        if train_indices is None:
            train_indices = range(self.ntrains)

        # Function for parallel processing
        def parallel_function(trains):
            ft_data = self.train(trains[0])[frame_type].data
            trains_sum = np.zeros_like(ft_data, dtype='float64')  # accumulator for the sum of trains
            trains_num = 0  # number of summed trains counter

            # We iterate through all trains.
            for i in trains:
                train = self.train(i)  # extract the train object

                if train.valid:  # Train is valid if it contains image.data.
                    trains_sum += train[frame_type].data
                    trains_num += 1

            return trains_sum, trains_num

        n_jobs = 10  # number of cores - should be exposed later to the user
        ranges = job_chunks(n_jobs, len(train_indices))  # distribute trains

        # Run jobs in parallel. Default backend is used.
        res = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(parallel_function)(i) for i in ranges)

        # Extract and sum results from individual jobs.
        trains_sum = sum(list(zip(*res))[0])
        trains_num = sum(list(zip(*res))[1])

        # Compute average and "squeeze" to remove empty dimension.
        trains_average = np.squeeze(trains_sum / trains_num)

        return trains_average

    def process_std(self, train_indices=None, dirname=None):
        """Standard processing.

        This processing computes the average of all dark and image frames
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
        dark_average = self.process_frames(frame_type='dark',
                                           train_indices=train_indices)
        image_average = self.process_frames(frame_type='image',
                                            train_indices=train_indices)

        # Save data if dirname is specified.
        if dirname is not None:
            dirname += f'/run_{self.run}/'
            filename = f'module_{self.module}_std.h5'
            data = {'dark_average': dark_average,
                    'image_average': image_average}
            save_h5(data, dirname, filename)
        else:
            return dark_average, image_average

    def process_normalised(self, dark_run, train_indices=None,
                           xgm_threshold=(1e-5, np.inf), dirname=None):
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
        # If train indices are not specified, all trains are processed.
        if train_indices is None:
            train_indices = range(self.ntrains)

        # First, we compute the average of darks (intradarks).
        dark_average = self.process_frames(frame_type='dark',
                                           train_indices=train_indices)

        # Second, we load image_average and dark_average for the dark run.
        filename = os.path.join('processed_runs_xgm', f'run_{dark_run}',
                                f'module_{self.module}_std.h5')
        with h5py.File(filename, 'r') as f:
            dr_image_average = f['image_average'][:]
            dr_dark_average = f['dark_average'][:]

        # Now, we compute the difference for the dark run:
        dr_diff = dr_image_average - dr_dark_average

        # This is the value we subtract from each image before we normalise it
        # with XGM value.
        sval = dark_average + dr_diff

        def parallel_function(trains):
            # For details of the following code, please refer to the previous
            # function.
            ft_data = np.squeeze(self.train(trains[0])['image'].data)
            trains_sum = np.zeros_like(ft_data, dtype='float64')  # accumulator for the sum of trains
            trains_num = 0

            for i in trains:
                train = self.train(i)

                if train.valid:
                    images = train['image']
                    s = np.zeros((images.n, images.data.shape[2],
                                  images.data.shape[3]))
                    for i in range(images.n):
                        xgm = train.xgm[i].values
                        if xgm_threshold[0] < xgm < xgm_threshold[1]:
                            s[i, ...] = (images.data[i, ...] -
                                         sval[i, ...]) / xgm

                    trains_sum += s
                    trains_num += 1

            return trains_sum, trains_num

        n_jobs = 10  # number of cores - can be exposed later
        ranges = job_chunks(n_jobs, len(train_indices))

        res = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(parallel_function)(i) for i in ranges)

        trains_sum = sum(list(zip(*res))[0])
        trains_num = sum(list(zip(*res))[1])

        # Compute the average of frames.
        trains_average = trains_sum / trains_num

        # Save data if dirname is specified.
        if dirname is not None:
            dirname += f'/run_{self.run}/'
            filename = f'module_{self.module}_normalised.h5'
            data = {'normalised_average': trains_average}
            save_h5(data, dirname, filename)
        else:
            return trains_average


def concat_module_images(dirname, run, run_type='normalised',
                         image_type='normalised_average'):
    """Concatenate corrected module images to get one xarray for the whole
    detector

    This function expects normalised images for all modules in a subdirectory
    of the given basedirectory.

    dirname (str) - the base directory in which processed runs are stored
    run (int) - the run number
    run_type (str) - the type of the run. Can be 'normalised' or 'diff'
    image_type (str) - the type of images. Can be 'normalised_average' if run_type='normalised'
        or 'image_agerage'/'dark_average' if run_type='diff'
    """
    data = None
    dark_data = None
    for module_number in range(16):
        filename = os.path.join(dirname, f'run_{run}', f'module_{module_number}_{run_type}.h5')
        with h5py.File(filename, 'r') as f:
            module_data = f[image_type][:]
        frames, x, y = module_data.shape
        if data is None:
                data = np.zeros((frames, 16, x, y), dtype=np.float64)
        data[:, module_number, ...] = module_data
    return xr.DataArray(data, dims=('frame', 'module', 'x', 'y'),
                        coords={'frame': range(frames), 'module': range(16),
                                'x': range(1, x + 1), 'y': range(1, y + 1)})
