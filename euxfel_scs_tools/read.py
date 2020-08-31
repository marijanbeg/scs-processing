import os
import h5py
import numpy as np
import xarray as xr


def read(dirname, run, reduction_type, data):
    data = None
    dark_data = None

    if reduction_type in ['std', 'norm']:
        data = np.zeros((nframes, 16, x, y), dtype=np.float64)
    else:
        data = np.zeros((16, nframes), dtype=np.float64)

    for module in range(16):
        filename = os.path.join(dirname, f'run_{run}',
                                f'module_{module}_{reduction_type}.h5')

        with h5py.File(filename, 'r') as f:
            module_data = f[data][:]

        nframes, x, y = module_data.shape

        if reduction_type in ['std', 'norm']:
            data[:, module, ...] = module_data
        else:
            data[module, ...] = module_data

    if reduction_type in ['std', 'norm']:
        return xr.DataArray(data,
                            dims=('frame', 'module', 'x', 'y'),
                            coords={'frame': range(nframes),
                                    'module': range(16),
                                    'x': range(1, x + 1),
                                    'y': range(1, y + 1)})
    else:
        return xr.DataArray(data,
                            dims=('module', 'frame'),
                            coords={'module': range(16),
                                    'frame': range(nframes)}
