"""This script provides an example on how to use the TPHelper wrapper for TrackPy,
to set a particle trajectory detection and then to manage the tracks.

It requires either:
- an image stack (e.g. test_image.tif)
- an image folder containing multiple frames (e.g. test_folder/)"""

# Import the required external module(s)
import matplotlib.pyplot as plt
import microImage as mim

# Import the functions from the module
from TPHelper import startSession, startManager

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
tpSettings = startSession(array=imageArray, diameter=41)

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
tpManager = startManager(trajectory, array=correctedArray)

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
