import os
import h5py
import numpy as np
import xarray as xr


def read(dirname, run, run_type='normalised', image_type='normalised_average'):
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
