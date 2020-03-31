import numpy as np
import trackpy as tp

##-\-\-\-\-\-\-\-\-\
## PRIVATE FUNCTIONS
##-/-/-/-/-/-/-/-/-/

# -------------------------
# Select the required track
def _select_track(dataframe, track_id):
    return dataframe[dataframe['particle'] == track_id]

# ----------------------------------------
# Return all the frames in multiple tracks
def _return_frames(dataframe, all_tracks):

    # Read the frames listed in all tracks
    all_frames = []
    for track in all_tracks:
        all_frames.append( track['frame'].to_numpy() )

    # Concatenate the list of all frames
    all_frames = np.concatenate( all_frames )

    return np.unique( all_frames )

# -----------------------------
# Return the list of all tracks
def _return_tracks(dataframe):
    return np.unique( dataframe['particle'].to_numpy() )

##-\-\-\-\-\-\-\
## TRACK DISPLAY
##-/-/-/-/-/-/-/

# ------------------------------------
# Display the points on a single frame
def displayFrame(array, dataframe, frame=0):

    # Deal with multiple frame arrays
    if len(array.shape) == 3:
        array = array[frame]

    # Select all the particles on the given frame(s)
    if 'frame' in dataframe:
        dataframe = dataframe[ dataframe['frame']==frame ]

    tp.annotate(dataframe, array)

# ------------------------
# Display the trajectories
def displayTrajectory(array, dataframe, track_ids=None):

    # Initialize the display options
    colorby = 'particle'
    label = True

    # Select the tracks to display
    if track_ids is not None:
        dataframe = dataframe[ dataframe['particle'].isin(track_ids) ]

        # Edit the options if needed
        if len(track_ids) == 1:
            colorby = 'frame'
            label = False

    tp.plot_traj(dataframe, superimpose=array, colorby=colorby, label=label)

# ------------------------------------
# Display the content of the dataframe
def displayDataframe(dataframe, array, frame=None, show_frame=0):

    # Display a single frame
    if frame is not None or 'frame' not in dataframe:
        displayFrame(array, dataframe, frame=frame)

    # Display a trajectory
    else:
        displayTrajectory(array[show_frame], dataframe)

##-\-\-\-\-\-\-\-\-\-\
## TRACK MODIFICATIONS
##-/-/-/-/-/-/-/-/-/-/

# ----------------------
# Renumber the particles
def renumberList(dataframe):

    # Build the dictionary
    old_nbr = dataframe['particle'].unique()
    new_nbr = np.arange(old_nbr.shape[0])

    # Replace all values
    dataframe['particle'] = dataframe['particle'].replace(old_nbr, new_nbr)

    return dataframe

# ----------------
# Merge two tracks
def mergeTracks(dataframe, track_id1, track_id2):

    # Extract both tracks
    track1 = _select_track(dataframe, track_id1)
    track2 = _select_track(dataframe, track_id2)

    # Get the list of all frames
    all_frames = _return_frames( dataframe, [track1, track2] )

    # Process all frames
    for frame in all_frames:

        # Remove the doublons
        if len(track1[track1['frame'] == frame]) == 1 and len(track2[track2['frame'] == frame]) == 1:
            dataframe = dataframe.drop( track2[track2['frame'] == frame].index )

        # Convert the second track into the 1st one
        elif len(track1[track1['frame'] == frame]) == 0:
            dataframe.at[track2[track2['frame'] == frame].index[0], 'particle'] = track_id1

    return dataframe

# -------------
# Split a track
def splitTrack(dataframe, track_id, split_after):

    # Extract the track
    old_track = _select_track(dataframe, track_id)

    # Get the list of track numbers
    all_tracks = _return_tracks(dataframe)
    new_track_id = np.amax(all_tracks) + 1

    # Get the list of all frames
    all_frames = _return_frames( dataframe, [old_track] )

    # Process all frames
    for frame in all_frames:

        # Change the track number
        if frame > split_after:
            object_index = old_track[old_track['frame'] == frame].index[0]
            print(object_index)
            dataframe.at[object_index, 'particle'] = new_track_id

    return dataframe

# -------------------------
# Delete the selected track
def deleteTrack(dataframe, track_id):

    # Extract the track
    old_track = _select_track(dataframe, track_id)

    # Delete the track
    dataframe = dataframe.drop( old_track.index )

    return dataframe
