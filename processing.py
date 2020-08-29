import os
import h5py
import joblib
import numpy as np
import extra_data as ed


def save_h5(data, dirname, filename):
    """Saves data in HDF5 file. The file is saved as dirname/filename."""
    
    # Make diretory if it does not exist.
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
    # Save file.
    h5file = os.path.join(f'{dirname}', f'{filename}')
    with h5py.File(h5file, 'w') as f:
        f.create_dataset('data', data=data)

        
def job_chunks(n_jobs, ntrains):
    chunk = int(np.ceil(ntrains / n_jobs))  # jobs per process
    
    train_range = range(ntrains)
    ranges = [train_range[step:step + chunk] for step in np.arange(ntrains, step=chunk)]

    return ranges


class Module:
    """Class used for processing individual modules."""
    def __init__(self, proposal, run, module, pattern):
        """
        proposal - proposal number
        run - run number
        module - module number (0-15)
        pattern - for instance: 'image', 'dark', 'image', 'dark', ... , 'end_image'
        
        Length of pattern list should be the same as the number of frames per train.
        
        """
        self.proposal = proposal
        self.run = run
        self.module = module
        self.pattern = pattern
        
        # Create run object.
        self.orun = ed.open_run(proposal=proposal, run=run).select(self.selector, 'image.data')
        
    @property
    def selector(self):
        """Device name for the analysed module"""
        return f'SCS_DET_DSSC1M-1/DET/{self.module}CH0:xtdf'
    
    @property
    def train_ids(self):
        """List of train_ids."""
        return self.orun.train_ids
    
    @property
    def ntrains(self):
        """Number of trains in the run."""
        return len(self.train_ids)
    
    @property
    def fpt(self):
        """Number of frames per train"""
        return self.orun.detector_info(self.selector)['frames_per_train']
    
    @property
    def xgm(self):
        """XGM data for the whole run."""
        tmprun = ed.open_run(proposal=self.proposal,
                             run=self.run).select('SCS_BLU_XGM/XGM/DOOCS:output',
                                                  'data.intensitySa3TD')
        return tmprun.get_array('SCS_BLU_XGM/XGM/DOOCS:output', 'data.intensitySa3TD')
    
    def train(self, index):
        """Returns train object.
        
        index (int) - 0, 1, 2, 3,....
        
        """
        _, data = self.orun.train_from_index(index)
        return Train(data=data, pattern=self.pattern, xgm=self.xgm[index, :])
    
    def process_frames(self, frame_type, train_indices=None, dirname=None):
        """Sums frames of frame_type. Result is an average of frames.
        
        No normalisation is done here and it is used for processing dark runs.
        
        frame_type - 'image' or 'dark'
        train_indices - list of train indices to be summed. If None, all trains are processed.
        dirname - if specified, data is saved as an HDF5 file in dirname.
        
        """
        # If train indices are not specified, all trains are processed.
        if train_indices is None:
            train_indices = range(self.ntrains)
        
        # Function for parallel processing
        def parallel_function(trains):
            frames_sum = 0  # sum of frames are added to it
            frames_num = 0  # number of summed frames are added to it
            
            # We iterate through all trains.
            for i in trains:
                train = self.train(i)  # get train object

                # Train is valid if it contains image.data.
                if train.valid:
                    # Compute sum of frames and the number of summed frames.
                    s, n = getattr(train, frame_type).process(normalised=False)
                    
                    frames_sum += s
                    frames_num += n
            
            return frames_sum, frames_num

        n_jobs = 10  # number of cores - can be exposed later
        ranges = job_chunks(n_jobs, len(train_indices))
        
        res = joblib.Parallel(n_jobs=10)(joblib.delayed(parallel_function)(i) for i in ranges)
        
        frames_sum = sum(list(zip(*res))[0])
        frames_num = sum(list(zip(*res))[1])
                
        # Compute average.
        frames_average = frames_sum / frames_num
        
        # Save data if dirname is specified.
        if dirname is not None:
            dirname += f'/run_{self.run}/'
            filename = f'module_{self.module}_{frame_type}.h5'
            save_h5(frames_average, dirname, filename)
        else:
            return frames_average
        
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


class Train:
    """Class for processing individual trains."""
    def __init__(self, data, pattern, xgm):
        self.data = data
        self.pattern = np.array(pattern)
        self.xgm = xgm
       
    @property
    def valid(self):
        """If there is no data in train, False is returned. Otherwise, True."""
        if 'image.data' in self.data[self.selector].keys():
            return True
        else:
            return False

    @property
    def selector(self):
        """Data selector."""
        return list(self.data.keys())[0]

    @property
    def frames(self):
        """Numpy array of all frames in the train."""
        return self.data[self.selector]['image.data']
    
    @property
    def images(self):
        """Returns container with image frames only."""
        return Container(self.frames[self.pattern=='image', ...], xgm=self.xgm)
    
    @property
    def darks(self):
        """Returns container with dark frames only."""
        return Container(self.frames[self.pattern=='dark', ...], xgm=self.xgm)


class Container:
    def __init__(self, data, xgm):
        self.data = data
        self.xgm = xgm
        
    @property
    def n(self):
        """The number of frames in the container."""
        return self.data.shape[0]
    
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