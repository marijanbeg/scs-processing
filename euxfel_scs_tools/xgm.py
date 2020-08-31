import extra_data as ed


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
