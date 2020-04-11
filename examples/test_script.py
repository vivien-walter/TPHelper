"""This script provides an example on how to use the TPHelper wrapper for TrackPy,
to set a particle trajectory detection and then to manage the tracks.

It requires either:
- an image stack (e.g. test_image.tif)
- an image folder containing multiple frames (e.g. test_folder/)"""

# Import the required external module(s)
import matplotlib.pyplot as plt
import microImage as mim

# Import the functions from the module
from TPHelper import startSession, startManager, intensityProfile, signalProperties, integrateProfile

# Open the image
"""Path to the file to open. Replace with the path to an existing file or folder."""
test_file = '/path/to/folder/test_image.tif'
imageArray = mim.openImage(test_file)

"""If required, use the space here to edit the image using the functions in microImage.
Visit https://github.com/vivien-walter/microImage/tree/master/examples to see how to
use the module microImage."""
imageArray = mim.backgroundCorrection(imageArray, signed_bits=True, average='median')
correctedArray = mim.contrastCorrection(imageArray, min=600, max=1200)

## ----------------------------------
## PART 1 - Extracting the trajectory

# Start the TPHelper session
tpSettings = startSession(input=imageArray, diameter=41)

# Show the list of all attributes
tpSettings.showParameters()

# Edit some of the attributes
tpSettings.memory = 3
tpSettings.filter_stubs = 40

# Batch process the image
trajectory = tpSettings.batch(filter=True, store=True)

# Check if the detection worked
tpSettings.preview(array=correctedArray)

# Save the settings in a file
tpSettings.save(file_name='test_settings.json')

## ----------------------------------
## PART 2 - Edition of the trajectory

# Start the TPHelper manager
tpManager = startManager(trajectory, input=correctedArray)

# List all tracks and display the list
print( tpManager.listTracks() )

# Display a specific track
tpManager.show(track_ids=[40])

# Remove the useless track
tpManager.remove(40)

# Reset the particle index
tpManager.resetID()

# Split the 6th track after the 11th frame
tpManager.split(5,10)

# Merge two tracks
tpManager.merge(10,12)

# Save the trajectory in a XML file
tpManager.save(file_name='test_trajectory.xml', track_ids=None)

## ----------------------------------
## PART 3 - Analysis of the intensity

"""In this example, we will use the TPHelper function for the analysis. Please keep
in mind that the TrackManager instance also have these functions embedded in it.
Refer to the online documentation to see how to use them."""

# Extract the selected tracks from the manager
trajectory = tpManager.extract(track_ids=[0,1,2,4,5])

# Calculate the intensity profiles using a sinc fit
profiles = intensityProfile(trajectory, input=imageArray, profile_type='sinc', space_scale=46.21)

# Select the profile of the track with the ID = 4
track4_profile = profiles[4]

# Plot the profile of the first frame
intensity = track4_profile['intensity'][0]
distance = track4_profile['distance'][0]

plt.plot(distance, intensity)
plt.show()

# Extract the contrast, noise and signal-to-noise ratios from the fitted profiles
properties = signalProperties(profiles, use_fit=True)

# Display the mean value of the contrast of the track with the ID = 5
track5_contrasts = properties[4]['contrast']
print(track5_contrasts.mean())

# Calculate the integrated intensity on each raw profile
integrations = integrateProfile(profiles, use_fit=False)

# Display the standard deviation of the integrated intensity of the track with the ID = 0
track0_integration = integrations[0]
print(track0_integration.std())
