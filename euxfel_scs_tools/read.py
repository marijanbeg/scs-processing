import os
import h5py
import numpy as np
import xarray as xr


def read(dirname, run, reduction_type, dataset):
    filename = os.path.join(dirname, f'run_{run}',
                            'module_{module}_'
                            f'{reduction_type}.h5')

    if reduction_type in ['std', 'norm']:
        return _read_std_normal(filename, dataset)
    else:
        return _read_xgm(filename, dataset)


def _read_std_normal(filename, dataset):
    # first dataset to get dimensions
    with h5py.File(filename.format(module=0), 'r') as f:
        module_data = f[dataset][:]

    nframes, x, y = module_data.shape

    data = np.zeros((nframes, 16, x, y), dtype=np.float64)
    data[:, 0, ...] = module_data

    for module in range(1, 16):
        with h5py.File(filename.format(module=module), 'r') as f:
            module_data = f[dataset][:]
        data[:, module, ...] = module_data

    return xr.DataArray(data,
                        dims=('frame', 'module', 'x', 'y'),
                        coords={'frame': range(nframes),
                                'module': range(16),
                                'x': range(1, x + 1),
                                'y': range(1, y + 1)})


def _read_xgm(filename, dataset):
    # first dataset to get dimensions
    with h5py.File(filename.format(module=0), 'r') as f:
        module_data = f[dataset][:]
        xgm_data = f['xgm'][:]

    nframes, *rest = module_data.shape

    data = np.zeros((16, nframes), dtype=np.float64)
    xgm = np.zeros((16, nframes), dtype=np.float64)
    data[0, ...] = module_data
    xgm[0, ...] = xgm_data

    for module in range(1, 16):
        with h5py.File(filename.format(module=module), 'r') as f:
            module_data = f[dataset][:]
            xgm_data = f['xgm'][:]
        data[module, ...] = module_data
        xgm[module, ...] = xgm_data

    return (xr.DataArray(data,
                         dims=('module', 'frame'),
                         coords={'module': range(16),
                                 'frame': range(nframes)}),
            xr.DataArray(np.mean(xgm, axis=0),
                         dims=('frame'),
                         coords={'frame': range(nframes)}))
