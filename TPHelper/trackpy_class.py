from copy import deepcopy
import numpy as np
import trackpy as tp

import TPHelper.input_output as tpio
import TPHelper.track_manager as tpmng
import TPHelper.trackpy_man as tpman

##-\-\-\-\-\-\-\-\-\
## PRIVATE FUNCTIONS
##-/-/-/-/-/-/-/-/-/

# -------------------------
# Round numbers to odd ones
def _nbr2odd(number):

    # Add 1 to even numbers
    if number % 2 == 0:
        number += 1

    return number

# ---------------------------------------
# Check if the array have multiple frames
def _check_multiple_frames(array=None, positions=None):

    # Check the shape of the array
    if array is not None:
        if array.shape[0] <= 1:
            raise Exception("The function cannot be called as it requires a sequence of frame.")

    # Check the content of the dataframe
    if positions is not None:
        if 'frame' in positions:
            if len(np.unique(positions['frame'])) <= 1:
                raise Exception("The function cannot be called as it requires multiple frames to be saved in the dataframe.")
        else:
            raise Exception("The function cannot be called as it requires multiple frames to be saved in the dataframe.")

##-\-\-\-\-\-\-\
## TRACKPY CLASS
##-/-/-/-/-/-/-/

class TrackingSession:
    def __init__(self, diameter=41, dark_spots=False, search_range=None, load_file=None, array=None):

        # Keep the array in memory
        self.array = array
        self.raw_trajectory = None
        self.trajectory = None

        # Initialize the default parameters for tp.locate and tp.batch
        self.diameter = _nbr2odd(diameter)
        self.minmass = None
        self.maxsize = None
        self.separation = None
        self.noise_size = 1
        self.smoothing_size = None
        self.threshold = None
        self.invert = dark_spots
        self.percentile = 64
        self.topn = None
        self.preprocess = True
        self.max_iterations = 10
        self.filter_before = None
        self.filter_after = None
        self.characterize = True
        self.engine = 'auto'

        # Initialize the default parameters for tp.link and trajectory filtering
        if search_range is None:
            search_range = diameter
        self.search_range = search_range
        self.memory = 0
        self.adaptive_stop = None
        self.adaptive_step = 0.95
        self.neighbor_strategy = None
        self.link_strategy = None
        self.filter_stubs = None

        if load_file is not None:
            self.load(load_file)

    ##-\-\-\-\-\-\-\-\-\-\-\
    ## CALL TRACKPY FUNCTIONS
    ##-/-/-/-/-/-/-/-/-/-/-/

    # ----------------------------------------
    # Preview the parameters on a single frame
    def locate(self, array=None, store=True, frame=0):

        # Retrieve the array
        if array is None:
            array = self.array

        # Deal with multiple frame arrays
        if len(array.shape) == 3:
            array = array[frame]

        # Check odd numbers
        self.diameter = _nbr2odd(self.diameter)

        # Run TrackPy
        dataframe = tp.locate(array,
            self.diameter,
            minmass = self.minmass,
            maxsize = self.maxsize,
            separation = self.separation,
            noise_size = self.noise_size,
            smoothing_size = self.smoothing_size,
            threshold = self.threshold,
            invert = self.invert,
            percentile = self.percentile,
            topn = self.topn,
            preprocess = self.preprocess,
            max_iterations = self.max_iterations,
            characterize = self.characterize,
            engine = self.engine
        )

        # Store in the instance
        if store:
            self.raw_trajectory = deepcopy(dataframe)
            self.trajectory = deepcopy(dataframe)

        return dataframe

    # -------------------------------------
    # Batch process all frames of the stack
    def batch(self, array=None, filter=True, store=True):

        # Retrieve the array
        if array is None:
            array = self.array

        # Check if there are multiple frames to read
        _check_multiple_frames(array=array)

        # Check odd numbers
        self.diameter = _nbr2odd(self.diameter)

        # Run TrackPy
        dataframe = tp.batch(array,
            self.diameter,
            minmass = self.minmass,
            maxsize = self.maxsize,
            separation = self.separation,
            noise_size = self.noise_size,
            smoothing_size = self.smoothing_size,
            threshold = self.threshold,
            invert = self.invert,
            percentile = self.percentile,
            topn = self.topn,
            preprocess = self.preprocess,
            max_iterations = self.max_iterations,
            characterize = self.characterize,
            engine = self.engine
        )

        # Store in the instance
        if store:
            self.raw_trajectory = deepcopy(dataframe)
            self.trajectory = deepcopy(dataframe)

        # Filter the trajectory
        if filter:
            dataframe = self.filter(dataframe, store=store)

        return dataframe

    # -----------------------------------------------------------
    # Filter the collected collection of points into a trajectory
    def filter(self, dataframe=None, store=True):

        # Retrieve the dataframe
        if dataframe is None:
            dataframe = self.raw_trajectory

        # Connect positions together
        dataframe = tp.link(dataframe,
            self.search_range,
            memory = self.memory,
            adaptive_stop = self.adaptive_stop,
            adaptive_step = self.adaptive_step,
            neighbor_strategy = self.neighbor_strategy,
            link_strategy = self.link_strategy,
        )

        # Remove spurious trajectory
        if self.filter_stubs is not None:
            dataframe = tp.filtering.filter_stubs(dataframe, threshold=self.filter_stubs)

        # Regenerate the index
        dataframe = dataframe.reset_index(drop=True)

        # Store in the instance
        if store:
            self.trajectory = deepcopy(dataframe)

        return dataframe

    ##-\-\-\-\-\-\-\-\-\
    ## DISPLAY FUNCTIONS
    ##-/-/-/-/-/-/-/-/-/

    # ----------------------------------------------------
    # Display the list and current value of all parameters
    def showParameters(self, group_by=True):
        tpman.showValues(self, group_by=group_by)

    # -------------------------------------------
    # Show the description of the given parameter
    def showDescription(self, name):
        tpman.showDescription(name, getattr(self, name))

    # ---------------------------------------
    # Display the position on the given frame
    def preview(self, array=None, dataframe=None, frame=None, show_frame=0):

        # Get the array
        if array is None:
            array = self.array

        # Get the dataframe
        if dataframe is None:
            dataframe = self.trajectory

        # Select the type of display to apply
        tpmng.displayDataframe(dataframe, array, frame=frame, show_frame=show_frame)

    ##-\-\-\-\-\-\-\-\-\-\-\
    ## LOAD AND SAVE SETTINGS
    ##-/-/-/-/-/-/-/-/-/-/-/

    # ---------------------------
    # Save the settings in a file
    def save(self, file_name=None):
        tpio.saveSettings(self, name=file_name)

    # -----------------------------
    # Load the settings from a file
    def load(self, file_name):

        # Load the file into a dictonary
        setting_dict = tpio.loadSettings(file_name)

        # Assign all the values
        for setting in setting_dict.keys():
            setattr(self, setting, setting_dict[setting])

##-\-\-\-\-\-\-\-\
## TRAJECTORY CLASS
##-/-/-/-/-/-/-/-/

class TrackManager:
    def __init__(self, positions, array=None):

        # Initialize the object
        self.trajectory = positions
        self.array = array

    ##-\-\-\-\-\-\-\
    ## PATH SELECTION
    ##-/-/-/-/-/-/-/

    # ---------------------------------
    # List all the tracks in the object
    def listTracks(self):
        return np.copy( self.trajectory['particle'].unique() )

    # -------------------
    # Re-index the tracks
    def resetID(self):
        self.trajectory = tpmng.renumberList(self.trajectory)

    # --------------------------------------
    # Return a copy of the current dataframe
    def duplicate(self):
        return deepcopy(self)

    ##-\-\-\-\-\-\-\-\-\
    ## PATH MODIFICATION
    ##-/-/-/-/-/-/-/-/-/

    # -------------------------
    # Merge two tracks into one
    def merge(self, track_id1, track_id2):

        _check_multiple_frames(positions=self.trajectory)
        self.trajectory = tpmng.mergeTracks(self.trajectory, track_id1, track_id2)

    # ----------------------
    # Split a track into two
    def split(self, track_id, split_after):

        _check_multiple_frames(positions=self.trajectory)
        self.trajectory = tpmng.splitTrack(self.trajectory, track_id, split_after)

    # -------------------------
    # Remove the selected track
    def remove(self, track_id):
        self.trajectory = tpmng.deleteTrack(self.trajectory, track_id)

    ##-\-\-\-\-\-\
    ## OUTPUT DATA
    ##-/-/-/-/-/-/

    # ------------------------------------
    # Display a specific path and/or frame
    def show(self, frame=None, track_ids=None, array=None, show_frame=0):

        # Check that there is an array to display
        if array is None:
            if self.array is None:
                raise Exception("An image array is required to display the trajectory")
            else:
                array = self.array

        # Display the position(s) on a single frame
        if frame is not None:
            tpmng.displayFrame(array[frame], self.trajectory, frame=frame)

        # Display the whole trajectory
        else:
            tpmng.displayTrajectory(array[show_frame], self.trajectory, track_ids=track_ids)

    # ----------------------------------------------------------
    # Return a specific selection of track(s) in a new dataframe
    def extract(self, track_ids=[0], as_dataframe=True):

        # Return a new dataframe
        if as_dataframe:
            return deepcopy( self.trajectory[ self.trajectory['particle'].isin(track_ids)] )

        # Return a list of array
        else:

            # Extract all tracks as arrays
            all_arrays = []
            for id in track_ids:
                current_track = self.trajectory[ self.trajectory['particle'] == id ]
                current_array = current_track[['frame','y','x']].to_numpy()
                all_arrays.append( np.copy(current_array) )

            return all_arrays

    # --------------------------------------
    # Save the selected trajectory into file
    def save(self, file_name=None, track_ids=None, default='.csv'):
        tpio.saveTrajectory(self.trajectory, filename=file_name, default=default, particle_ids=track_ids)
