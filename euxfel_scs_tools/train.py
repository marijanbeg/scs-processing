import numpy as np
from .container import Container


class Train:
    def __init__(self, train_id, train_data, pattern):
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
        self.train_id = train_id
        self.pattern = np.array(pattern)

        # image.data might be missing. In that case None is assigned and later
        # used for train validation.
        try:
            self.data = train_data[list(train_data.keys())[0]]['image.data']
        except:  # what would be the right exception here?
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
