import os
import h5py
import joblib
import numpy as np
import xarray as xr
import extra_data as ed
import subprocess as sp
import time

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
        self._data = orun.get_array(str1, str2)

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
        return self._data[index, 0:self.n]

    @property
    def data(self):
        """Returns all XGM per frame values for all trains.

        Returns
        -------
        numpy.ndarray

            XGM values across all trains.

        """
        return np.concatenate([self.train(i)
                               for i in range(self._data.shape[0])], axis=0)

# class PhaseShifter:
#     def __init__(self, proposal, run, module, pattern):
#         self.proposal = proposal
#         self.run = run
#         self.module = module
#         self.pattern = pattern
#
#         # Strings for getting XGM values from the data.
#         str1 = 'SCS_BLU_XGM/XGM/DOOCS:output'
#         str2 = 'data.intensitySa3TD'
#
#         # Run object.
#         orun = ed.open_run(proposal=self.proposal,
#                            run=self.run).select(str1, str2)
#
#         # Read data
#         self._data = orun.get_array(str1, str2)

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

    def nframes(self, frame_type):
        return np.count_nonzero(np.array(self.pattern)==frame_type)

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
        ranges = job_chunks(njobs, len(trains))  # distribute trains
        res = joblib.Parallel(n_jobs=njobs)(
            joblib.delayed(thread_func)(i) for i in ranges)

        # Extract and sum results from individual jobs.
        total_sum = sum(list(zip(*res))[0])
        total_number = sum(list(zip(*res))[1])

        # Compute average and "squeeze" to remove empty dimension.
        return np.squeeze(total_sum / total_number)

    def process_std(self, frame_types=None, trains=None, njobs=40,
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

    def process_norm(self,
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
        # If train indices are not specified, all trains are processed.
        if trains is None:
            trains = range(self.ntrains)

        # First, we compute the average of darks (intradarks).
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
            # For details of the following code, please refer to the previous
            # function.
            accumulator = np.zeros((self.nframes('image'), 128, 512),
                                   dtype='float64')
            counter = 0

            for i in job_trains:
                train = self.train(i)
                if train.valid:
                    images = train[frames['image']]
                    s = np.zeros((images.n, 128, 512), dtype='float64')
                    for i in range(images.n):
                        xgm = train.xgm[i].values
                        if xgm_threshold[0] < xgm < xgm_threshold[1]:
                            s[i, ...] = (images.data[i, ...] -
                                         sval[i, ...]) / xgm

                    accumulator += s
                    counter += 1

            return accumulator, counter

        ranges = job_chunks(njobs, len(trains))
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


def reduction_std(proposal, run, pattern, dirname=None,
                  frame_types=None, trains=None, njobs=40):
    script = ('import os, sys\n'
              'sys.path.append(os.path.dirname('
              'os.path.dirname(os.path.abspath(__file__))))\n'
              'import processing as pr\n'
              f'module = pr.Module(proposal={proposal}, run={run},'
              f' module=MODULE, pattern={pattern})\n'
              f'module.process_std(frame_types={frame_types}, trains={trains},'
              f' njobs={njobs}, dirname="../{dirname}")\n')
    _submit_jobs(script)

    
def reduction_norm(proposal, run, pattern, dark_run, dirname=None,
                   frames={'image': 'image',
                           'dark': 'dark'},
                   dark_run_frames={'image': 'image',
                                    'dark': 'dark'},
                   trains=None, xgm_threshold=(1e-5, np.inf), njobs=40):
    script = ('import os, sys\n'
              'import numpy as np\n'
              'sys.path.append(os.path.dirname('
              'os.path.dirname(os.path.abspath(__file__))))\n'
              'import processing as pr\n'
              f'module = pr.Module(proposal={proposal}, run={run},'
              f' module=MODULE, pattern={pattern})\n'
              f'module.process_norm(dark_run={dark_run}, '
              f'frames={frames}, dark_run_frames={dark_run_frames}, '
              f'trains={trains}, xgm_threshold={xgm_threshold}, '
              f'njobs={njobs}, dirname="../{dirname}")')
    _submit_jobs(script.replace('inf),', 'np.inf),'))


def _submit_jobs(py_script, slurm_dir='slurm_log', module_range=range(16)):
    script_dir = 'autogenerated_scripts'
    if not os.path.exists(script_dir):
        os.makedirs(script_dir)
    if not os.path.exists(slurm_dir):
        os.makedirs(slurm_dir)
    for module in module_range:
        file_name = f'run_{time.time()}_module{module}'
        process_sh = ('#!/bin/bash\n'
                      'source /usr/share/Modules/init/bash\n'
                      'module load exfel\n'
                      'module load exfel_anaconda3/1.1\n'
                      f'python3 {file_name}.py')
        
        with open(f'{script_dir}/{file_name}.py', 'w') as f:
            f.write(py_script.replace('MODULE', str(module)))

        with open(f'{script_dir}/{file_name}.sh', 'w') as f:
            f.write(process_sh)
        
        command = ['sbatch', '-p', 'upex', '-t', '100',
                   '--chdir', f'{script_dir}',
                   '-o', f'../{slurm_dir}/slurm-%A.out', f'{file_name}.sh']
        
        sp.run(command)
