import bottleneck as bn
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit


##-\-\-\-\-\-\-\
## MATH FUNCTIONS
##-/-/-/-/-/-/-/

# -----------------
# Gaussian function
def _gaussian(x, A, x0, w, y0):
    return A * np.exp(-(x-x0)**2 / w**2) + y0

# -------------
# Sinc function
def _sinc(x, A, x0, w, y0):
    return A * np.sinc((x-x0) / w) + y0

##-\-\-\-\-\-\-\-\-\
## PRIVATE FUNCTIONS
##-/-/-/-/-/-/-/-/-/

# ---------------------------------------------------
# Reshape a stack of arrays before concatenating them
def _reshape_arrays(array_list):

    # Get the size list
    if isinstance(array_list[0], np.ndarray):
        size_list = [x.shape[0] for x in array_list]
    else:
        return np.array(array_list)

    # Resize if required
    if len(np.unique(size_list)) != 1:

        # Get the new size to use
        min_size = min(size_list)

        # Loop over all arrays
        new_arrays=[]
        for array in array_list:

            #Resize the array
            if array.shape[0] != min_size:
                to_remove = int( (array.shape[0] - min_size)/2 )
                array = array[to_remove:-to_remove]

            new_arrays.append(np.copy(array))

        array_list = new_arrays

    return np.array(array_list)

# ----------------------------------
# Convert the window size to a tuple
def _convert_window_size(window):

    if type(window) == int:
        window = [window, window]
    elif window is not None:
        window = list(window)

    return window

# -------------------------
# Get the correct crop size
def _get_crop_size(array_shape, center, crop_window):

    # Calculate the current limits
    xLeft, xRight = center[0] - crop_window[0], center[0] + crop_window[0]
    yTop, yBottom = center[1] - crop_window[1], center[1] + crop_window[1]

    # Calculate the new limits
    xLeft, xRight = max([xLeft, 0]), min([xRight, array_shape[0]])
    yTop, yBottom = max([yTop, 0]), min([yBottom, array_shape[1]])

    # Calculate the new window size
    x1, x2 = center[0] - xLeft, xRight - center[0]
    y1, y2 = center[1] - yTop, yBottom - center[1]

    return [ min([x1,x2]), min([y1,y2]) ]

# ----------------------------
# Center the array and crop it
def _crop_center(array, center, crop_window=None):

    # Only crop is different than None
    if crop_window is not None:

        # Get the actual window size
        crop_window = _get_crop_size(array.shape, center, crop_window)

        # Get the limits
        y1, y2, x1, x2 = center[0] - crop_window[0], center[0] + crop_window[0] + 1, center[1] - crop_window[1], center[1] + crop_window[1] + 1

        # Crop the array
        array = array[y1:y2,x1:x2]

    return array

# -------------------------------------
# Do a Gaussian fit on the input values
def _gaussian_fit(x,y, n_points=1000, dark_spots=False):

    # Initialize parameters
    y0Init = (y[0] + y[-1]) / 2
    center_index = np.argmax(y)

    if dark_spots:
        AInit = np.amin(y) - y0Init
    else:
        AInit = np.amax(y) - y0Init

    x0Init = x[center_index]
    wInit = x[ np.argmin( abs(((AInit+y0Init)/2)-y) ) ]
    wInit = (wInit - x0Init) / (np.sqrt(2*np.log(10)))

    # Do the fit
    parameters, covariances = curve_fit(_gaussian, x, y, p0=[AInit, x0Init, wInit, y0Init])

    # Prepare the range
    xMin, xMax = x[0], x[-1]
    step = (xMax-xMin) / n_points
    distance = np.arange(xMin, xMax+step, step)

    # Create the fitted profile
    intensity = _gaussian(distance, *parameters)

    return intensity, distance, parameters, np.sqrt(np.diag(covariances))

# ---------------------------------
# Do a sinc fit on the input values
def _sinc_fit(x,y, n_points=1000, dark_spots=False):

    # Initialize parameters
    y0Init = (y[0] + y[-1]) / 2
    center_index = np.argmax(y)

    if dark_spots:
        AInit = np.amin(y) - y0Init
    else:
        AInit = np.amax(y) - y0Init

    x0Init = x[center_index]
    wInit = x[ np.argmin( abs(((AInit+y0Init)/2)-y) ) ]
    wInit = (wInit-x0Init) * 4 /np.pi

    # Do the fit
    parameters, covariances = curve_fit(_sinc, x, y, p0=[AInit, x0Init, wInit, y0Init])

    # Prepare the range
    xMin, xMax = x[0], x[-1]
    step = (xMax-xMin) / n_points
    distance = np.arange(xMin, xMax+step, step)

    # Create the fitted profile
    intensity = _sinc(distance, *parameters)

    return intensity, distance, parameters, np.sqrt(np.diag(covariances))

# ------------------------------
# Fit with the required function
def _select_fit_function(raw_distance, raw_intensity, fit_function='gaussian', n_points=1000, dark_spots=False):

    # Do the required fit on the profile
    if fit_function == 'gaussian':
        intensity, distance, params, errs = _gaussian_fit(raw_distance, raw_intensity, n_points=n_points, dark_spots=dark_spots)

    elif fit_function == 'sinc':
        intensity, distance, params, errs = _sinc_fit(raw_distance, raw_intensity, n_points=n_points, dark_spots=dark_spots)

    return intensity, distance, params, errs

# -------------------------------------
# Compute a radial profile in the array
def _compute_radial_profile(array, center, mirror=True):

    # Get the radius range
    y, x = np.indices((array.shape))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int)

    # Calculate the elements
    tbin = np.bincount(r.ravel(), array.ravel())
    nr = np.bincount(r.ravel())

    # Compute the radial profile
    radialprofile = tbin / nr

    # Generate the distance array
    distance = np.unique(r)

    # Mirror the profile
    if mirror:
        radialprofile = np.concatenate([ radialprofile[::-1],radialprofile[1:] ])
        distance = np.concatenate([ -1*distance[::-1],distance[1:] ])

    return radialprofile, distance

# ---------------
# Get line limits
def _line_generation(center, angle, array_shape):

    # Extract the position
    y, x = center

    # Normalise the angle
    angle = -1 * (((angle+90) % 180) - 90)

    # Deal with the vertical case
    if abs(angle) == 90:
        p1, p2 = (0,int(center[1])), (int(array_shape[0]-1),int(center[1]))

    # Deal with the other angles
    else:

        # Get the angle in radian
        angle = angle * np.pi / 180

        # Left point
        xPLeft = 0
        yPLeft = center[0] + np.tan(angle) * center[1]

        # Recalculate if yPLeft is out of boundary
        if yPLeft < 0 or yPLeft > array_shape[0]-1:
            if yPLeft < 0:
                yPLeft = 0
            else:
                yPLeft = array_shape[0]-1
            xPLeft = center[1] - (yPLeft - center[0]) / np.tan(angle)

        # Right point
        xPRight = array_shape[1]-1
        yPRight = center[0] - np.tan(angle) * (xPRight - center[1])

        # Recalculate if yPRight is out of boundary
        if yPRight < 0 or yPRight > array_shape[0]-1:
            if yPRight < 0:
                yPRight = 0
            else:
                yPRight = array_shape[0]-1
            xPRight = center[1] - (yPRight-center[0]) / np.tan(angle)

        p1, p2 = (int(yPLeft), int(xPLeft)), (int(yPRight), int(xPRight))

    return p1, p2

# ---------------------------------
# Plot the profile on a single line
def _compute_line_profile(array, center, angle):

    # Get the line position
    pLeft, pRight = _line_generation(center, angle, array.shape)

    # Generate the coordinates
    numberPoints = int(np.hypot(pRight[1] - pLeft[1], pRight[0] - pLeft[0]))
    x, y = ( np.linspace(pLeft[1], pRight[1], numberPoints) , np.linspace(pLeft[0], pRight[0], numberPoints) )

    # Get the profile
    intensity = array[y.astype(np.int),x.astype(np.int)]

    # Generate the radius
    distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2) * np.sign(x - center[1])

    return {'intensity':intensity, 'distance':distance}

# -------------------
# Get the raw profile
def _compute_raw_profile(array, type='radial', line_angle=0, center=None, space_scale=None):

    # Calculate the center for the profile
    if center is None:
        center = ( int((array.shape[0]-1)/2) , int((array.shape[1]-1)/2))

    # Compute the radial profile of intensity
    if type == 'radial':
        intensity, distance = _compute_radial_profile(array, center)

    # Return a line
    if type == 'line':
        intensity, distance = _compute_line_profile(array, center, line_angle)

    # Prepare the output
    output = {'intensity':intensity, 'distance':distance}

    # Add the space scale
    if space_scale is not None:
        output['distance_px'] = np.copy(output['distance'])
        output['distance'] = output['distance'] / space_scale

    return output

# ---------------------------
# Return the fitted intensity
def _fit_profile(array, fit_function='gaussian', fit_type='radial', line_angle=0, center=None, n_points=1000, space_scale=None, dark_spots=1000):

    # Calculate the center for the profile
    raw_output = _compute_raw_profile(array, type=fit_type, line_angle=line_angle, center=center)
    raw_intensity, raw_distance = raw_output['intensity'], raw_output['distance']

    # Do the required fit on the profile
    intensity, distance, params, errs = _select_fit_function(raw_distance, raw_intensity, fit_function=fit_function, n_points=n_points, dark_spots=dark_spots)

    # Prepare the output
    output = {'raw_intensity':raw_intensity, 'raw_distance':raw_distance, 'intensity':intensity, 'distance':distance, 'parameters':params, 'errors':errs}

    # Add the space scale
    if space_scale is not None:
        output['distance_px'] = np.copy(output['distance'])
        output['distance'] = output['distance'] / space_scale

        output['raw_distance_px'] = np.copy(output['raw_distance'])
        output['raw_distance'] = output['raw_distance'] / space_scale

        output['parameters_px'] = np.copy(output['parameters'])
        output['parameters'][2] = output['parameters'][2] / space_scale

        output['errors_px'] = np.copy(output['errors'])
        output['errors'][2] = output['errors'][2] / space_scale

    return output

# -----------------------------------------
# Compute the required average on the array
def _compute_average(array, type='mean'):

    # Compute the required type of average
    if type == 'mean':
        intensity = bn.nanmean(array)
    elif type == 'median':
        intensity = bn.nanmedian(array)
    elif type == 'std':
        intensity = bn.nanstd(array)

    return {'intensity':intensity}

# -----------------------------
# Extract the intensity profile
def _get_profile(array, center, profile_type='gaussian', scan_window=[50,50], line_angle=0, fit_type='radial', n_points=1000, space_scale=None, dark_spots=False):

    # Crop the array on the given scan window
    cropped_array = _crop_center(np.copy(array), center, crop_window=scan_window)

    # Extract the intensity profile by fit
    if profile_type.lower() in ['gaussian', 'sinc']:
        output = _fit_profile(cropped_array, fit_function=profile_type.lower(), fit_type=fit_type, line_angle=line_angle, n_points=n_points, space_scale=space_scale, dark_spots=dark_spots)

    # Extract the single spot intensity
    elif profile_type.lower() == 'spot':
        output = {'intensity':array[center[0],center[1]]}

    # Extract the intensity on the given line
    elif profile_type.lower() in ['line','radial']:
        output = _compute_raw_profile(cropped_array, type=profile_type.lower(), line_angle=line_angle, space_scale=space_scale)

    # Extract the mean intensity in the given window
    elif profile_type.lower() in ['mean','median','std']:
        output = _compute_average(cropped_array, type=profile_type.lower())

    # Raise an error if type is unknown
    else:
        raise Exception('The selected profile type is not valid.')

    return output

# ---------------------------------------
# Swap dictionary and array in the output
def _swap_dict_array(data):

    # Get the keys
    dict_keys = data[0].keys()

    # Generate a new dict
    output = {}
    for key in dict_keys:

        # Process all frames
        crt_array = []
        for frame_dict in data:
            crt_array.append( frame_dict[key] )

        # Convert into a numpy array
        crt_array = _reshape_arrays(crt_array)

        output[key] = np.copy(crt_array)

    return output

# --------------------------------------
# Calculate the noise value on the array
def _noise_calculation(array, parameters):

    # Calculate the noise
    raw_noise = np.std(array[1:] - array[:-1], axis=1, ddof=1)

    # Get all the values
    noise_values = (raw_noise[:-1] + raw_noise[1:])/2

    # Complete with the limits
    noise_values = np.insert(noise_values, 0, raw_noise[0])
    noise_values = np.append(noise_values, raw_noise[-1])

    # Get the relevant parameters
    background = array[:,3]

    # Calculate the noise value
    noise_values = noise_values/background

    return noise_values

# --------------------------------------------------------------
# Calculate the contrast value on the array using fit parameters
def _contrast_calculation_fit(array):

    # Get the relevant parameters
    amplitude = array[:,0]
    background = array[:,3]

    # Calculate the contrast
    contrast = amplitude / background

    return contrast

# --------------------------------------------------------------
# Calculate the contrast value on the array using min/max values
def _contrast_calculation_extremum(array, parameters, dark_spots=False):

    # Get the extremum
    if dark_spots:
        amplitude = np.amin(array,axis=1)
    else:
        amplitude = np.amax(array,axis=1)

    # Get the relevant parameters
    background = parameters[:,3]

    # Calculate the contrast
    contrast = amplitude / background

    return contrast

# ---------------------------------------
# Process the arrays to measure the noise
def _process_noise(track_profile, dark_spots=False, fit_type='gaussian', n_points=1000):

    # Extract the intensity arrays
    if 'raw_intensity' in track_profile.keys():
        intensities = track_profile['raw_intensity']
    else:
        intensities = track_profile['intensity']

    # Get the fit parameters
    if 'parameters' in track_profile.keys():
        parameters = track_profile['parameters']
    else:

        # Extract fit parameters
        parameters =[]
        for i, int_value in enumerate(intensities):
            dist, int, params, err = _select_fit_function(distances[i], int_value, fit_function=fit_type, n_points=n_points, dark_spots=dark_spots)
            parameters.append(params)
        parameters = np.array(parameters)

    # Calculate the noise
    noise = _noise_calculation(intensities, parameters)

    return noise

# ------------------------------------------
# Process the arrays to measure the contrast
def _process_signal(track_profile, use_fit=True, dark_spots=False, fit_type='gaussian'):

    # Extract the intensity arrays
    if 'raw_intensity' in track_profile.keys():
        intensities = track_profile['raw_intensity']
    else:
        intensities = track_profile['intensity']
    distances = track_profile['distance']

    # Get the fit parameters
    if 'parameters' in track_profile.keys():
        parameters = track_profile['parameters']
    else:

        # Extract fit parameters
        parameters =[]
        for i, int_value in enumerate(intensities):
            dist, int, params, err = _select_fit_function(distances[i], int_value, fit_function=fit_type, dark_spots=dark_spots)
            parameters.append(params)
        parameters = np.array(parameters)

    # Calculate the contrast using the fitted amplitude
    if use_fit:
        contrast = _contrast_calculation_fit(parameters)

    # Calculate the contrast using the max/min amplitude
    else:
        contrast = _contrast_calculation_extremum(intensities, parameters, dark_spots=dark_spots)

    return contrast

# ----------------------------------------------
# Extract the profile and compute the properties
def _get_signal_properties(track_profile, use_fit=True, dark_spots=False, fit_type='gaussian'):

    # Return an error
    if 'distance' not in track_profile.keys():
        raise Exception('A 2D profile is required to extract the signal properties.')

    # Compute the contrast
    contrast = _process_signal(track_profile, use_fit=use_fit, dark_spots=dark_spots, fit_type=fit_type)

    # Compute the noise values
    noise = _process_noise(track_profile)

    # Calculate the signal-to-noise ratio
    snr = contrast / noise

    return contrast, noise, snr

# -------------------------------------
# Integrate the given set of parameters
def _integration_on_fit(parameters, fit_type, limits=[-1000,1000]):

    # Do the integration
    if fit_type == 'gaussian':
        integration, _ = quad(_gaussian, -np.inf, np.inf, args=(parameters[0], parameters[1], parameters[2], 0))

    elif fit_type == 'sinc':
        integration, _ = quad(_sinc, -np.inf, np.inf, args=(parameters[0], parameters[1], parameters[2], 0))

    return integration

# --------------------------------------------
# Calculate the integraton from fitted profile
def _integrate_fitted_profile(distances, parameters, dark_spots=False, fit_type='gaussian'):

    # Integrate each profile
    integrated_profiles = []
    for i, parameter_set in enumerate(parameters):
        integrated_profiles.append( _integration_on_fit(parameter_set, fit_type, limits=[distances[i][0], distances[i][-1]]) )
    integrated_profiles = np.array(integrated_profiles)

    return integrated_profiles

# ---------------------------------
# Integrate point to point profiles
def _integrate_intensity_profile(intensities, distances, parameters):

    # Integrate each profile
    integrated_profiles = []
    for i, int_value in enumerate(intensities):
        intensity = int_value - parameters[0][3]
        integrated_profiles.append( np.trapz(intensity, x=distances[i]) )
    integrated_profiles = np.array(integrated_profiles)

    return integrated_profiles

# ------------------------------------------------
# Compute the integrated intensity of each profile
def _get_integrated_intensity(track_profile, use_fit=True, dark_spots=False, fit_type='gaussian'):

    # Return an error
    if 'distance' not in track_profile.keys():
        raise Exception('A 2D profile is required to extract the signal properties.')

    # Extract the intensity arrays
    if 'raw_intensity' in track_profile.keys():
        intensities = track_profile['raw_intensity']
        distances = track_profile['raw_distance']
    else:
        intensities = track_profile['intensity']
        distances = track_profile['distance']

    # Get the fit parameters
    if 'parameters' in track_profile.keys():
        parameters = track_profile['parameters']
    else:
        # Extract fit parameters
        parameters =[]
        for i, int_value in enumerate(intensities):
            dist, int, params, err = _select_fit_function(distances[i], int_value, fit_function=fit_type, dark_spots=dark_spots)
            parameters.append(params)
        parameters = np.array(parameters)

    # Extract the profile to integrate from a fit
    if use_fit:
        integration = _integrate_fitted_profile(distances, parameters, dark_spots=dark_spots, fit_type=fit_type)

    # Extract the profile to integrate from the raw profile
    else:
        integration = _integrate_intensity_profile(intensities, distances, parameters)

    return integration

##-\-\-\-\-\-\-\-\
## PUBLIC FUNCTIONS
##-/-/-/-/-/-/-/-/

# ------------------------------------------------------------
# Extract the intensity profiles of the particle on each track
def intensityProfile(positions, array, track_ids=None, scan_window=50, profile_type='gaussian', line_angle=0, fit_type='radial', n_points=1000, space_scale=None, dark_spots=False):

    # Conver the input parameters
    scan_window = _convert_window_size(scan_window)

    # List all tracks
    track_list = positions['particle'].unique()

    # Make the list of tracks to process
    if track_ids is None:
        track_ids = track_list

    # Process all particles
    track_profiles = {}
    for id in track_ids:

        # Extract the particle position as function of the time
        crt_data = positions[positions['particle'] == id][['frame','y','x']].to_numpy()

        # Process all frames
        crt_profiles = []
        for t, y, x in crt_data:

            # Collect the profile
            t,y,x = int(t), int(y), int(x)

            output = _get_profile(array[t], (y,x), scan_window=scan_window, profile_type=profile_type, line_angle=line_angle, fit_type=fit_type, n_points=n_points, space_scale=space_scale, dark_spots=dark_spots)
            output['frame'] = t
            output['position'] = (y,x)

            crt_profiles.append( output )

        # Invert dict and arrays
        crt_profiles = _swap_dict_array(crt_profiles)

        # Append the track to the list
        track_profiles[id] = crt_profiles

    return track_profiles

# -----------------------------
# Extract the signal properties
def signalProperties(profiles, use_fit=True, dark_spots=False, fit_type='gaussian'):

    # Loop over all the profiles
    properties = {}
    for track_id in profiles.keys():
        contrast, noise, snr = _get_signal_properties(profiles[track_id], use_fit=use_fit, dark_spots=dark_spots, fit_type=fit_type)
        properties[track_id] = {'contrast':contrast, 'noise':noise, 'snr':snr}

    return properties

# --------------------------------
# Integrate the intensity profiles
def integratedIntensity(profiles, use_fit=True, dark_spots=False, fit_type='gaussian'):

    # Loop over all the profiles
    integrated_intensity = {}
    for track_id in profiles.keys():
        integrated_intensity[track_id] = _get_integrated_intensity(profiles[track_id], use_fit=use_fit, dark_spots=dark_spots, fit_type=fit_type)

    return integrated_intensity
