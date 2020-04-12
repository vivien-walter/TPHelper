from copy import deepcopy
import numpy as np
import os
import pandas as pd
import sys
import trackpy as tp

import TPHelper.input_output as tpio
import TPHelper.intensity_profiler as tpint
import TPHelper.track_manager as tpmng
import TPHelper.trackpy_man as tpman
import TPHelper.trajectory_analysis as tpan

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
            raise Exception(
                "The function cannot be called as it requires a sequence of frame."
            )

    # Check the content of the dataframe
    if positions is not None:
        if "frame" in positions:
            if len(np.unique(positions["frame"])) <= 1:
                raise Exception(
                    "The function cannot be called as it requires multiple frames to be saved in the dataframe."
                )
        else:
            raise Exception(
                "The function cannot be called as it requires multiple frames to be saved in the dataframe."
            )


# -------------------------------
# Get the array from the instance
def _get_array(input, raw=False, frame=None, stack=False, display=False):

    # Input is an array
    if type(input) is np.ndarray:
        array = input

    # Input is a microImage instance
    else:
        try:
            # Return the array for display
            if display:

                # Select the frame if needed
                if frame is not None:
                    input.setFrame(frame)

                if raw:
                    array = input.frame.raw
                else:
                    array = input.frame.corrected

            # Return the array for calculations
            else:
                if raw:
                    array = input.source
                else:
                    array = input.array

        # Raise an exception if it does not work
        except:
            raise Exception(
                "A compatible NumPy array or microImage instance is required here as an input."
            )

    # Select a frame if required
    if frame is not None and len(array.shape) == 3:
        array = array[frame]

    # Check if there is multiple frames if required
    elif stack:
        _check_multiple_frames(array=array)

    return array


# -----------------------------
# Return the appropriate scales
def _get_scales(
    object, scale_input=None, space_scale=None, time_scale=None, get_fps=False
):

    # Get the time and space scale from the class
    input_space_scale = object.space_scale
    input_time_scale = object.time_scale

    # Get the time and space scale from an input
    if scale_input is not None:
        input_space_scale = extractSpaceCalibration(scale_input)
        input_time_scale = extractTimeCalibration(scale_input)

    # Correct the space scale input
    if input_space_scale is None and space_scale is None:
        space_scale = 1
    elif space_scale is not None:
        space_scale = space_scale
    else:
        space_scale = input_space_scale

    # Correct the time scale input
    if input_time_scale is None and time_scale is None:
        time_scale = 1
    elif time_scale is not None:
        time_scale = time_scale
    else:
        time_scale = input_time_scale

    # Convert in FPS if needed
    if get_fps:
        time_scale = 1 / time_scale

    return space_scale, time_scale


# ---------------------------------------------
# Check the input for the TrackManager instance
def _check_input(input, display_input=None, space_scale=None, time_scale=None):

    # Case if the input is a pandas Dataframe
    if isinstance(input, pd.DataFrame):
        dataframe = input
        display = None
        output_space_scale = None
        output_time_scale = None

    # Case if the input is an XML file
    elif os.path.isfile(str(input)):
        dataframe = tpio.loadTrajectory(input)
        display = None
        output_space_scale = None
        output_time_scale = None

    # Case if the input is a TrackingSession instance
    else:
        try:
            # Get the dataframe
            dataframe = input.tracks
            display = input.input

            output_space_scale = extractSpaceCalibration(display)
            output_time_scale = extractTimeCalibration(display)

        # Raise an exception if it does not work
        except:
            raise Exception(
                "A compatible Pandas DataFrame or TrackingSession instance is required here as an input."
            )

    # Update the array if needed
    if display_input is not None:
        display = display_input

    # Update the scales if needed
    if space_scale is not None:
        output_space_scale = space_scale

    if time_scale is not None:
        output_time_scale = time_scale

    return dataframe, display, output_space_scale, output_time_scale


##-\-\-\-\-\-\-\
## TRACKPY CLASS
##-/-/-/-/-/-/-/


class TrackingSession:
    def __init__(
        self,
        diameter=41,
        dark_spots=False,
        search_range=None,
        load_file=None,
        input=None,
    ):

        # Keep the array in memory
        self.input = input
        self.spots = None
        self.tracks = None

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
        self.engine = "auto"

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
    def locate(self, input=None, store=True, frame=0, raw_input=False):

        # Retrieve the array
        if input is None:
            input = self.input
        array = _get_array(input, raw=raw_input, frame=frame)

        # Check odd numbers
        self.diameter = _nbr2odd(self.diameter)

        # Run TrackPy
        dataframe = tp.locate(
            array,
            self.diameter,
            minmass=self.minmass,
            maxsize=self.maxsize,
            separation=self.separation,
            noise_size=self.noise_size,
            smoothing_size=self.smoothing_size,
            threshold=self.threshold,
            invert=self.invert,
            percentile=self.percentile,
            topn=self.topn,
            preprocess=self.preprocess,
            max_iterations=self.max_iterations,
            characterize=self.characterize,
            engine=self.engine,
        )

        # Store in the instance
        if store:
            self.spots = deepcopy(dataframe)
            self.tracks = deepcopy(dataframe)

        return dataframe

    # -------------------------------------
    # Batch process all frames of the stack
    def batch(self, input=None, filter=True, store=True, raw_input=False):

        # Retrieve the array
        if input is None:
            input = self.input
        array = _get_array(input, raw=raw_input, stack=True)

        # Check odd numbers
        self.diameter = _nbr2odd(self.diameter)

        # Run TrackPy
        dataframe = tp.batch(
            array,
            self.diameter,
            minmass=self.minmass,
            maxsize=self.maxsize,
            separation=self.separation,
            noise_size=self.noise_size,
            smoothing_size=self.smoothing_size,
            threshold=self.threshold,
            invert=self.invert,
            percentile=self.percentile,
            topn=self.topn,
            preprocess=self.preprocess,
            max_iterations=self.max_iterations,
            characterize=self.characterize,
            engine=self.engine,
        )

        # Store in the instance
        if store:
            self.spots = deepcopy(dataframe)
            self.tracks = deepcopy(dataframe)

        # Filter the trajectory
        if filter:
            dataframe = self.filter(dataframe, store=store)

        return dataframe

    # -----------------------------------------------------------
    # Filter the collected collection of points into a trajectory
    def filter(self, dataframe=None, store=True):

        # Retrieve the dataframe
        if dataframe is None:
            dataframe = self.spots

        # Connect positions together
        dataframe = tp.link(
            dataframe,
            self.search_range,
            memory=self.memory,
            adaptive_stop=self.adaptive_stop,
            adaptive_step=self.adaptive_step,
            neighbor_strategy=self.neighbor_strategy,
            link_strategy=self.link_strategy,
        )

        # Remove spurious trajectory
        if self.filter_stubs is not None:
            dataframe = tp.filtering.filter_stubs(
                dataframe, threshold=self.filter_stubs
            )

        # Regenerate the index
        dataframe = dataframe.reset_index(drop=True)

        # Store in the instance
        if store:
            self.tracks = deepcopy(dataframe)

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
    def preview(
        self, input=None, positions=None, frame=None, show_frame=0, raw_input=False
    ):

        # Get the array
        if input is None:
            input = self.input
        array = _get_array(input, raw=raw_input, frame=show_frame, display=True)

        # Get the dataframe
        if positions is None:
            positions = self.tracks

        # Select the type of display to apply
        tpmng.displayDataframe(positions, array, frame=frame, show_frame=show_frame)

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
    def __init__(self, input, display_input=None, space_scale=None, time_scale=None):

        # Extract the informations from the input
        positions, display, space_scale, time_scale = _check_input(
            input,
            display_input=display_input,
            space_scale=space_scale,
            time_scale=time_scale,
        )

        # Initialize the object
        self.positions = positions
        self.display = display

        # Set the scales
        self.space_scale = space_scale
        self.time_scale = time_scale

    ##-\-\-\-\-\-\-\
    ## PATH SELECTION
    ##-/-/-/-/-/-/-/

    # ---------------------------------
    # List all the tracks in the object
    def listTracks(self):
        return np.copy(self.positions["particle"].unique())

    # -------------------
    # Re-index the tracks
    def resetID(self):
        self.positions = tpmng.renumberList(self.positions)

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

        _check_multiple_frames(positions=self.positions)
        self.positions = tpmng.mergeTracks(self.positions, track_id1, track_id2)

    # ----------------------
    # Split a track into two
    def split(self, track_id, split_after):

        _check_multiple_frames(positions=self.positions)
        self.positions = tpmng.splitTrack(self.positions, track_id, split_after)

    # -------------------------
    # Remove the selected track
    def remove(self, track_id):
        self.positions = tpmng.deleteTrack(self.positions, track_id)

    ##-\-\-\-\-\-\-\
    ## TRACK ANALYSIS
    ##-/-/-/-/-/-/-/

    # -----------------------------------
    # Compute the MSD on the given tracks
    def getMSD(
        self,
        track_ids=None,
        combine_all=True,
        fps=None,
        time_scale=None,
        space_scale=None,
        max_lagtime=None,
        scale_input=None,
    ):

        # Extract the positions for calculations
        if track_ids is not None:
            positions = self.extract(track_ids=track_ids, as_dataframe=True)
        else:
            positions = self.positions

        # Get the time and space scale from the input if given
        space_scale, input_fps = _get_scales(
            self,
            scale_input=scale_input,
            space_scale=space_scale,
            time_scale=time_scale,
            get_fps=True,
        )

        if fps is not None:
            input_fps = fps

        return tpan.computeMSD(
            positions,
            combine_all=combine_all,
            fps=input_fps,
            space_scale=space_scale,
            max_lagtime=max_lagtime,
        )

    # -------------------------------------------
    # Measure the diffusivity from collected MSDs
    def getDiffusivity(
        self,
        msd=None,
        dimensions=2,
        track_ids=None,
        combine_all=True,
        fps=None,
        space_scale=None,
        max_lagtime=None,
        scale_input=None,
        time_scale=None,
    ):

        # Calculate the MSD if not provided
        if msd is None:
            msd = self.getMSD(
                track_ids=track_ids,
                combine_all=combine_all,
                fps=fps,
                space_scale=space_scale,
                max_lagtime=max_lagtime,
                scale_input=scale_input,
                time_scale=time_scale,
            )

        return tpan.getDiffusivity(msd, dimensions=dimensions)

    # --------------------------------------------------
    # Verify the track by analysing all the informations
    def verify(
        self,
        array=None,
        signal_properties=False,
        histograms=True,
        highlight_ids=None,
        scale_input=None,
        track_ids=None,
        scan_window=50,
        profile_type="gaussian",
        line_angle=0,
        fit_type="radial",
        n_points=1000,
        space_scale=None,
        dark_spots=False,
        use_fit=True,
        fps=None,
        max_lagtime=None,
        time_scale=None,
        dimensions=2,
    ):

        # Extract the positions for calculations
        if track_ids is not None:
            positions = self.extract(track_ids=track_ids, as_dataframe=True)
        else:
            positions = self.positions

        # Extract array if needed
        if array is None:
            array = _get_array(self.display, stack=True)

        # Get the time and space scale from the input if given
        space_scale, input_fps = _get_scales(
            self,
            scale_input=scale_input,
            space_scale=space_scale,
            time_scale=time_scale,
            get_fps=True,
        )

        if fps is not None:
            input_fps = fps

        tpan.verifyTracks(
            positions,
            array,
            signal_properties=signal_properties,
            histograms=histograms,
            highlight_ids=highlight_ids,
            track_ids=track_ids,
            scan_window=scan_window,
            profile_type=profile_type,
            line_angle=line_angle,
            fit_type=fit_type,
            n_points=n_points,
            space_scale=space_scale,
            dark_spots=dark_spots,
            use_fit=use_fit,
            fps=input_fps,
            max_lagtime=max_lagtime,
            dimensions=dimensions,
        )

    ##-\-\-\-\-\-\-\-\-\
    ## INTENSITY ANALYSIS
    ##-/-/-/-/-/-/-/-/-/

    # -------------------------------------------
    # Extract the intensity profiles on each path
    def intensityProfile(
        self,
        array=None,
        track_ids=None,
        scan_window=50,
        profile_type="gaussian",
        line_angle=0,
        fit_type="radial",
        n_points=1000,
        space_scale=None,
        dark_spots=False,
    ):

        # Extract array if needed
        if array is None:
            array = _get_array(self.display, stack=True)

        # Extract the space scale
        if space_scale is None:
            space_scale = extractSpaceCalibration(self.display)

        return tpint.intensityProfile(
            self.positions,
            array,
            track_ids=track_ids,
            scan_window=scan_window,
            profile_type=profile_type,
            line_angle=line_angle,
            fit_type=fit_type,
            n_points=n_points,
            space_scale=space_scale,
            dark_spots=dark_spots,
        )

    # --------------------------------------------------
    # Calculate the properties of the intensity profiles
    def signalProperties(
        self,
        profiles=None,
        array=None,
        track_ids=None,
        scan_window=50,
        profile_type="gaussian",
        line_angle=0,
        fit_type="radial",
        n_points=1000,
        space_scale=None,
        use_fit=True,
        dark_spots=False,
        percentage=True,
    ):

        # Generate profiles if needed
        if profiles is None:
            profiles = self.intensityProfile(
                array=array,
                track_ids=track_ids,
                scan_window=scan_window,
                profile_type=profile_type,
                line_angle=line_angle,
                fit_type=fit_type,
                n_points=n_points,
                space_scale=space_scale,
                dark_spots=dark_spots,
            )

        return ip.signalProperties(
            profiles,
            use_fit=use_fit,
            dark_spots=dark_spots,
            fit_type=profile_type,
            percentage=percentage,
        )

    # --------------------------------------------------
    # Calculate the properties of the intensity profiles
    def integratedProfile(
        self,
        profiles=None,
        array=None,
        track_ids=None,
        scan_window=50,
        profile_type="gaussian",
        line_angle=0,
        fit_type="radial",
        n_points=1000,
        space_scale=None,
        use_fit=True,
        dark_spots=False,
    ):

        # Generate profiles if needed
        if profiles is None:
            profiles = self.intensityProfile(
                array=array,
                track_ids=track_ids,
                scan_window=scan_window,
                profile_type=profile_type,
                line_angle=line_angle,
                fit_type=fit_type,
                n_points=n_points,
                space_scale=space_scale,
                dark_spots=dark_spots,
            )

        return ip.integrateProfile(
            profiles, use_fit=use_fit, dark_spots=dark_spots, fit_type=profile_type
        )

    ##-\-\-\-\-\-\
    ## OUTPUT DATA
    ##-/-/-/-/-/-/

    # ------------------------------------
    # Display a specific path and/or frame
    def show(
        self,
        frame=None,
        track_ids=None,
        display_input=None,
        show_frame=0,
        raw_input=False,
    ):

        # Adjust the display
        if frame is not None:
            show_frame = frame

        # Get the display array
        if display_input is None:
            display_input = self.display
        array = _get_array(display_input, raw=raw_input, display=True, frame=show_frame)

        # Display the position(s) on a single frame
        if frame is not None:
            tpmng.displayFrame(array, self.positions, frame=frame)

        # Display the whole trajectory
        else:
            tpmng.displayTrajectory(array, self.positions, track_ids=track_ids)

    # ----------------------------------------------------------
    # Return a specific selection of track(s) in a new dataframe
    def extract(self, track_ids=[0], as_dataframe=True):

        # Return a new dataframe
        if as_dataframe:
            return deepcopy(self.positions[self.positions["particle"].isin(track_ids)])

        # Return a list of array
        else:

            # Extract all tracks as arrays
            all_arrays = []
            for id in track_ids:
                current_track = self.positions[self.positions["particle"] == id]
                current_array = current_track[["frame", "y", "x"]].to_numpy()
                all_arrays.append(np.copy(current_array))

            return all_arrays

    # --------------------------------------
    # Save the selected trajectory into file
    def save(self, file_name=None, track_ids=None, default=".csv"):
        tpio.saveTrajectory(
            self.positions, filename=file_name, default=default, particle_ids=track_ids
        )


##-\-\-\-\-\-\-\-\
## PUBLIC FUNTIONS
##-/-/-/-/-/-/-/-/

# --------------------------------------
# Extract the array from the given input
def extractArray(input, raw=False, frame=None, stack=False, display=False):
    return _get_array(input, raw=raw, frame=frame, stack=stack, display=display)


# ------------------------------------
# Extract the positions from the input
def extractPositions(input):

    # Return directly if it is a dataframe
    if isinstance(input, pd.DataFrame):
        positions = input
        array = None

    # Extract if it is a TrackingSession
    elif isinstance(input, TrackingSession):
        positions = input.tracks
        array = input.array

    # Extract if it is a TrackManager class
    elif isinstance(input, TrackManager):
        positions = input.tracks
        array = input.display

    # Raise an error
    else:
        raise Exception(
            "A compatible Pandas DataFrame, TrackingSession or TrackManager instance is required here as an input."
        )

    return positions, array


# ---------------------------------------
# Extract the space calibration if exists
def extractSpaceCalibration(input):

    # Try to get the space unit from the input
    try:
        space_scale = input.space_scale

    except:
        space_scale = None

    return space_scale


# --------------------------------------
# Extract the time calibration if exists
def extractTimeCalibration(input):

    # Try to get the space unit from the input
    try:
        time_scale = input.time_scale

    except:
        time_scale = None

    return time_scale
