import TPHelper.trackpy_class as tc

##-\-\-\-\-\-\-\-\-\-\-\-\-\
## COMMAND LINE INTERACTIVITY
##-/-/-/-/-/-/-/-/-/-/-/-/-/

# ------------------------------------
# Start a trackpy optimization session
def startSession(diameter=41, dark_spots=False, search_range=None, load_file=None, array=None):
    return tc.TrackingSession(diameter=diameter, dark_spots=dark_spots, search_range=search_range, load_file=load_file, array=array)

# -----------------------------
# Start a track manager session
def startManager(positions, array=None):
    return tc.TrackManager(positions, array=array)
