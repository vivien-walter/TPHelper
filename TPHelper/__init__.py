import TPHelper.intensity_profiler as ip
import TPHelper.trackpy_class as tc

##-\-\-\-\-\-\-\-\-\-\-\-\-\
## COMMAND LINE INTERACTIVITY
##-/-/-/-/-/-/-/-/-/-/-/-/-/

# ------------------------------------
# Start a trackpy optimization session
def startSession(diameter=41, dark_spots=False, search_range=None, load_file=None, input=None):
    return tc.TrackingSession(diameter=diameter, dark_spots=dark_spots, search_range=search_range, load_file=load_file, input=input)

# -----------------------------
# Start a track manager session
def startManager(input, display_input=None):
    return tc.TrackManager(input, display_input=display_input)

##-\-\-\-\-\-\-\-\-\-\-\-\
## TRACK AND IMAGE ANALYSIS
##-/-/-/-/-/-/-/-/-/-/-/-/

# --------------------------------
# Calculate the intensity profiles
def intensityProfile(positions, input=None, track_ids=None, scan_window=50, profile_type='gaussian', line_angle=0, fit_type='radial', n_points=1000, space_scale=None, dark_spots=False):

    # Get the positions and other informations from class
    positions, array = tc.extractPositions(positions)

    # Extract the array from the input
    if input is not None:
        array = tc.extractArray(input, stack=True)

        # Extract the space scale
        if space_scale is None:
            space_scale = tc.extractSpaceCalibration(input)

    # Perform the calculation
    return ip.intensityProfile(positions, array, track_ids=track_ids, scan_window=scan_window, profile_type=profile_type, line_angle=line_angle, fit_type=fit_type, n_points=n_points, space_scale=space_scale, dark_spots=dark_spots)

# -------------------------------------------------
# Compute the contrast, noise and SNR of the signal
def signalProperties(profiles, use_fit=True, dark_spots=False, fit_type='gaussian'):
    return ip.signalProperties(profiles, use_fit=use_fit, dark_spots=dark_spots, fit_type=fit_type)

# ---------------------------------------------
# Integrate the intensity on the given profiles
def integrateProfile(profiles, use_fit=True, dark_spots=False, fit_type='gaussian'):
    return ip.integratedIntensity(profiles, use_fit=use_fit, dark_spots=dark_spots, fit_type=fit_type)
