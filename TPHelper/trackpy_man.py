##-\-\-\-\-\-\
## DESCRIPTIONS
##-/-/-/-/-/-/

description_dictionary = {
'diameter' : """This may be a single number or a tuple giving the feature’s extent in each dimension,
useful when the dimensions do not have equal resolution (e.g. confocal microscopy).
The tuple order is the same as the image shape, conventionally (z, y, x) or (y, x).
The number(s) must be odd integers. When in doubt, round up.""",
'minmass' : """The minimum integrated brightness. This is a crucial parameter for eliminating spurious features.
Recommended minimum values are 100 for integer images and 1 for float images. Defaults to 0 (no filtering)""",
'maxsize' : """Maximum radius-of-gyration of brightness, default None""",
'separation' : """Minimum separation between features.
Default is diameter + 1. May be a tuple, see diameter for details.""",
'noise_size' : """Width of Gaussian blurring kernel,
in pixels Default is 1. May be a tuple, see diameter for details.""",
'smoothing_size' : """The size of the sides of the square kernel used in boxcar (rolling average) smoothing,
in pixels Default is diameter. May be a tuple, making the kernel rectangular.""",
'threshold' : """Clip bandpass result below this value. Thresholding is done on the already background-subtracted image.
By default, 1 for integer images and 1/255 for float images.""",
'invert' : """This will be deprecated. Use an appropriate PIMS pipeline to invert a Frame or FramesSequence.
Set to True if features are darker than background. False by default.""",
'percentile' : """Features must have a peak brighter than pixels in this percentile.
This helps eliminate spurious peaks.""",
'topn' : """Return only the N brightest features above minmass.
If None (default), return all features above minmass.""",
'preprocess' : """Set to False to turn off bandpass preprocessing.""",
'max_iterations' : """max number of loops to refine the center of mass, default 10""",
'characterize' : """Compute “extras”: eccentricity, signal, ep. True by default.""",
'engine' : """{‘auto’, ‘python’, ‘numba’}""",
'search_range' : """the maximum distance features can move between frames, optionally per dimension""",
'memory' : """The maximum number of frames during which a feature can vanish,
then reappear nearby, and be considered the same particle.
0 by default.""",
'adaptive_stop' : """If not None, when encountering an oversize subnet, retry by progressively reducing search_range until the subnet is solvable.
If search_range becomes <= adaptive_stop, give up and raise a SubnetOversizeException.""",
'adaptive_step' : """Reduce search_range by multiplying it by this factor.""",
'neighbor_strategy' : """Algorithm used to identify nearby features. Default ‘KDTree’.""",
'link_strategy' : """Algorithm used to resolve subnetworks of nearby particles ‘auto’ uses hybrid (numba+recursive)
if available ‘drop’ causes particles in subnetworks to go unlinked""",
'filter_stubs' : """Filter out trajectories with few points. They are often spurious.
Minimum number of points (video frames) to survive.""",
}

##-\-\-\-\-\-\-\
## DISPLAY VALUES
##-/-/-/-/-/-/-/

# ------------------------------------------------------
# Show all the values, either as a list or a sorted grid
def showValues(object, group_by=True):

    # Initialize text
    printedText = ""

    # Set separator
    if group_by:
        separator = "\t"
    else:
        separator = "\n"

    # Title 1
    if group_by:
        printedText += """Object properties:
------------------\n"""

    printedText += "self.diameter: " + str(object.diameter) + separator
    printedText += "self.minmass: " + str(object.minmass) + "\n"
    printedText += "self.maxsize: " + str(object.maxsize) + separator
    printedText += "self.separation: " + str(object.separation) + "\n"
    printedText += "self.percentile: " + str(object.percentile) + separator
    printedText += "self.invert: " + str(object.invert) + "\n"

    # Title 2
    if group_by:
        printedText += """\nFilter properties:
------------------\n"""

    printedText += "self.noise_size: " + str(object.noise_size) + separator
    printedText += "self.smoothing_size: " + str(object.smoothing_size) + "\n"
    printedText += "self.threshold: " + str(object.threshold) + separator
    printedText += "self.preprocess: " + str(object.preprocess) + "\n"
    printedText += "self.topn: " + str(object.topn) + "\n"

    # Title 3
    if group_by:
        printedText += """\nOther parameters:
-----------------\n"""

    printedText += "self.characterize: " + str(object.characterize) + separator
    printedText += "self.engine: " + str(object.engine) + "\n"

    # Title 4
    if group_by:
        printedText += """\nTrajectory parameters:
-----------------\n"""

    printedText += "self.search_range: " + str(object.search_range) + separator
    printedText += "self.memory: " + str(object.memory) + "\n"
    printedText += "self.adaptive_stop: " + str(object.adaptive_stop) + separator
    printedText += "self.adaptive_step: " + str(object.adaptive_step) + "\n"
    printedText += "self.neighbor_strategy: " + str(object.neighbor_strategy) + separator
    printedText += "self.link_strategy: " + str(object.link_strategy) + "\n"
    printedText += "self.filter_stubs: " + str(object.filter_stubs) + "\n"

    # Display the text
    print(printedText)

##-\-\-\-\-\-\-\-\-\-\
## DISPLAY DESCRIPTION
##-/-/-/-/-/-/-/-/-/-/

# ----------------------------------------------
# Show the description of a specific parameteter
def showDescription(name, value):

    print("""-----------------------
self."""+name+""":
-----------------------
Current value: """+str(value) + """
-----------------------\n""")

    print( description_dictionary[name] )
    print('-----------------------')
