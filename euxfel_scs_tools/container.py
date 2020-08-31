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
