import os
import h5py
import numpy as np


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


def job_chunks(njobs, trains, frames=None):
    """Function for splitting an array of trains into chunks for parallel
    processing.

    Parameters
    ----------
    n_jobs : int

        The number of jobs/threads.

    trains : list

        List of trains to be processed.

    """
    # The number of trains per process. The first njobs-1 will process chunk
    # number of trains, whereas the last job is going to process the rest.
    chunk = int(np.ceil(len(trains) / njobs))

    if frames is None:
        res = [trains[step:step + chunk]
               for step in np.arange(len(trains), step=chunk)]
    else:
        res = [[trains[step:step + chunk], frames[step:step + chunk]]
               for step in np.arange(len(trains), step=chunk)]
    
    return res
