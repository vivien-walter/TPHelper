from datetime import datetime
import json
import os
import pandas as pd
import xml.etree.ElementTree as ET

from TPHelper.track_manager import _return_tracks, _return_frames, _select_track

##-\-\-\-\-\-\-\-\-\
## PRIVATE FUNCTIONS
##-/-/-/-/-/-/-/-/-/

# --------------------------
# Generate a date-based name
def _generate_name():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


# --------------------------------------
# Check that the extension is authorized
def _check_extensions(file, extensions=[".json"]):

    file_name, file_extension = os.path.splitext(file)

    # Save in the new list on if the extension is authorized
    if file_extension in extensions:
        return file

    # Raise an error if not valid
    else:
        raise Exception("The extension is not valid.")


# ------------------------------------------
# Return the list of attributes of the class
def _get_attribute_list():
    return [
        "diameter",
        "minmass",
        "maxsize",
        "separation",
        "noise_size",
        "smoothing_size",
        "threshold",
        "invert",
        "percentile",
        "topn",
        "preprocess",
        "max_iterations",
        "characterize",
        "engine",
        "search_range",
        "memory",
        "adaptive_stop",
        "adaptive_step",
        "neighbor_strategy",
        "link_strategy",
        "filter_stubs",
    ]


# -------------------------------------------------
# Convert settings in class instance to a dictonary
def _settings2dict(setting_object):

    # Retrieve the list of settings to save
    setting_list = _get_attribute_list()

    # Save all the settings in a dictionnary
    setting_dict = {}
    for setting in setting_list:
        setting_dict[setting] = getattr(setting_object, setting)

    return setting_dict


# --------------------------------------
# Convert the trajectory to a XML format
def _traj2XML(positions, calibration=None):

    # Initialize the attribute dictionary
    data_attributes = {"generationDateTime": "None", "from": "TPHelper"}

    # Extract informations from the trajectory
    particle_list = _return_tracks(positions)
    data_attributes["nTracks"] = str(len(particle_list))

    # Get the space and time calibration
    if calibration is None:
        data_attributes["spaceUnits"] = "pixels"
        data_attributes["frameInterval"] = "1.0"
        data_attributes["timeUnits"] = "frames"

    # Prepare the data and its attributes
    data = ET.Element("Tracks")
    for attribute in data_attributes.keys():
        data.set(attribute, data_attributes[attribute])

    # Loop over all the particles in the trajectory
    for particle_id in particle_list:

        # Get the number of frames
        crt_particle = _select_track(positions, particle_id)
        frame_list = _return_frames(positions, [crt_particle])
        item_attributes = {"nSpots": str(len(frame_list))}

        # Prepare the item and its attributes
        item = ET.SubElement(data, "particle")
        for attribute in item_attributes.keys():
            item.set(attribute, item_attributes[attribute])

        # Fill the tree with all the items
        for frame in frame_list:

            # Generate a temp subitem for the array
            tmp_subitem = ET.SubElement(item, "detection")
            tmp_subitem.set("t", str(frame))

            # Get the value from the trajectory
            crt_frame = crt_particle.loc[crt_particle["frame"] == frame]
            tmp_subitem.set("x", str(float(crt_frame["x"])))
            tmp_subitem.set("y", str(float(crt_frame["y"])))

            tmp_subitem.set("z", "0.0")

    return ET.tostring(data)


# ----------------------------------
# Convert a XML file to a trajectory
def _xml2Traj(filename):

    # Extract the tree from the file
    trajXML = ET.parse(filename).getroot()

    # Initialize the lists
    particle_id = []
    frame_nbr = []
    x_position = []
    y_position = []

    # Loop over all particles
    for id, track in enumerate(trajXML):

        # Loop over all positions
        for position in track:
            particle_id.append(int(id))
            frame_nbr.append(int(position.attrib["t"]))
            x_position.append(float(position.attrib["x"]))
            y_position.append(float(position.attrib["y"]))

    # Generate the DataFrame
    trajectory = pd.DataFrame(
        {"y": y_position, "x": x_position, "frame": frame_nbr, "particle": particle_id}
    )

    return trajectory


# ---------------------------
# Save the data in a XML file
def _saveXML(positions, file_name, calibration=None):

    # Convert position into an .xml file
    dataXML = _traj2XML(positions, calibration=calibration)

    # Save in file
    fileToSave = open(file_name, "w")
    fileToSave.write(dataXML.decode("utf-8"))
    fileToSave.close()


# ---------------------------
# Save the data in a CSV file
def _saveCSV(positions, file_name):

    # Save in a file
    positions.to_csv(file_name)


##-\-\-\-\-\-\-\-\-\-\
## I/O TRACKPY SETTINGS
##-/-/-/-/-/-/-/-/-/-/

# --------------------------------
# Save the settings in a JSON file
def saveSettings(settings, name=None):

    # Retrieve the informations from the dictionnary
    setting_dict = _settings2dict(settings)

    # Set the file name
    if name is not None:
        name = _check_extensions(name)
    else:
        now = datetime.now()
        name = _generate_name() + "_TPsettings.json"

    # Save the dictionnary in a file
    with open(name, "w") as fp:
        json.dump(setting_dict, fp)


# ----------------------------------
# Load the settings from a JSON file
def loadSettings(name):

    # Check the file
    name = _check_extensions(name)

    # Retrieve the dictonary
    with open(name, "r") as fp:
        data = json.load(fp)

    return data


##-\-\-\-\-\-\-\
## I/O TRAJECTORY
##-/-/-/-/-/-/-/

# --------------------------------
# Save the trajectories in file(s)
def saveTrajectory(dataframe, filename=None, default=".csv", particle_ids=None):

    # Generate the file name
    if filename is not None:
        filename = _check_extensions(
            filename, extensions=[".csv", ".txt", ".dat", ".xml"]
        )
    else:
        filename = _generate_name() + "_trajectory" + default

    # Select the tracks to save
    if particle_ids is not None:
        dataframe = dataframe[dataframe["particle"].isin(particle_ids)]

    # Save as a XML file
    if os.path.splitext(filename)[1] == ".xml":
        _saveXML(dataframe, filename)

    # Save as a CSV file
    else:
        _saveCSV(dataframe, filename)


# ---------------------------------------------
# Load a trajectory into a TrackManager session
def loadTrajectory(filename):
    return _xml2Traj(filename)
