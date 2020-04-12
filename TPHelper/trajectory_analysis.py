import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import trackpy as tp

import TPHelper.intensity_profiler as tpip

##-\-\-\-\-\-\-\
## MATH FUNCTIONS
##-/-/-/-/-/-/-/

# ---------
# Power law
def _powerlaw(x, A, k):
    return A * (x ** k)


##-\-\-\-\-\-\-\-\-\
## PRIVATE FUNCTIONS
##-/-/-/-/-/-/-/-/-/

# ----------------------------------------------------
# Calculate the max lagtime to use for better accuracy
def _calculate_max_lagtime(trajectory):

    # Loop over all the particles
    number_frames = []
    for particle_id in trajectory["particle"].unique():

        # Extract the frames specific particle trajectory
        dataframe = trajectory[trajectory["particle"].isin([particle_id])]
        frames = dataframe["frame"].to_numpy()

        number_frames.append(frames.shape[0])

    # Calculate the average max number of frames
    max_lagtime = np.mean(number_frames)

    return int(max_lagtime / 2)


# --------------------------------------------------
# Calculate the MSD of the ensemble of the particles
def _compute_general_msd(trajectory, space_scale, fps, max_lagtime=100):

    # Calculate the MSD
    msd = tp.emsd(trajectory, space_scale, fps, max_lagtime=max_lagtime)

    # Format into a 2-D numpy array
    msd = np.array([msd.index, msd.to_numpy()])

    return msd.T


# ------------------------------------------
# Calculate the MSD of each individual track
def _compute_individual_msd(trajectory, space_scale, fps, max_lagtime=100):

    # Calculate the MSD
    msd = tp.imsd(trajectory, space_scale, fps, max_lagtime=max_lagtime)

    # Format the output into a dictionnary
    msd_dict = {}
    for particle_id in msd.keys():

        # Extract the array
        msd_array = np.array([msd.index, msd[particle_id].to_numpy()])
        msd_dict[particle_id] = np.copy(msd_array.T)

    return msd_dict


# ----------------------------
# Fit the MSD with a power law
def _fit_MSD(msd):

    # Initialize the parameters
    kInit = 1
    AInit = msd[-1, 1] / msd[-1, 0]

    # Do the fit
    params, cov = curve_fit(_powerlaw, msd[:, 0], msd[:, 1], p0=[AInit, kInit])

    return params, np.sqrt(np.diag(cov))


# ------------------------
# Calculate the value of D
def _calculate_diffusitivity(msd, dimensions=2):

    # Fit the MSD
    parameters, errors = _fit_MSD(msd)

    # Calculate the diffusivity
    diffusivity = parameters[0] / (2 * dimensions)

    return diffusivity, parameters, errors


# ----------------------------------------
# Display the result of the track analysis
def _display_analysis(
    intensities,
    msd,
    diffusivity,
    frames,
    histograms=True,
    signal_properties=False,
    highlight_ids=None,
):

    # Display properties
    if signal_properties:
        figure, axs = plt.subplots(3, 2, figsize=(6, 9))
        intensity_name = "Contrast"
    else:
        figure, axs = plt.subplots(2, 2, figsize=(6, 6))
        intensity_name = "∫intensity"

    # Color arrays for bars
    track_names = list(diffusivity.keys())
    colors = ["blue"] * len(track_names)
    if highlight_ids is not None:
        for track_id in highlight_ids:
            colors[track_names.index(track_id)] = "red"

    # 1 - Display the MSDs over time
    for track_ids in msd.keys():

        if highlight_ids is None:
            axs[0][0].plot(msd[track_ids][:, 0], msd[track_ids][:, 1], "-")
        else:
            if int(track_ids) in highlight_ids:
                axs[0][0].plot(
                    msd[track_ids][:, 0], msd[track_ids][:, 1], "r-", linewidth=2
                )
            else:
                axs[0][0].plot(
                    msd[track_ids][:, 0], msd[track_ids][:, 1], "k--", linewidth=0.5
                )

    axs[0][0].set_ylabel("MSD")
    axs[0][0].set_xlabel("Lag time")

    # 2 - Display the intensity over time
    all_i = []
    avg_i = []
    for track_ids in intensities.keys():

        if signal_properties:
            i_values = intensities[track_ids]["contrast"]
        else:
            i_values = intensities[track_ids]
        avg_i.append(np.mean(i_values))
        for i in i_values:
            all_i.append(i)

        if highlight_ids is None:
            axs[0][1].plot(frames[track_ids], i_values, "-")
        else:
            if int(track_ids) in highlight_ids:
                axs[0][1].plot(frames[track_ids], i_values, "r-", linewidth=2)
            else:
                axs[0][1].plot(frames[track_ids], i_values, "k--", linewidth=0.5)

    axs[0][1].set_ylabel(intensity_name)
    axs[0][1].set_xlabel("Frame")

    # 3 - Display the D
    if histograms:
        values_D = [diffusivity[x] for x in diffusivity.keys()]
        axs[1][0].hist(values_D, density=True, color="blue")

        axs[1][0].set_ylabel("#")
        axs[1][0].set_xlabel("D")

        if highlight_ids is not None:
            for track_id in highlight_ids:
                axs[1][0].axvline(x=diffusivity[track_id], color="red", linewidth=2)

    else:

        # Get all the values
        track_names = list(diffusivity.keys())
        values_D = [diffusivity[x] for x in track_names]

        axs[1][0].bar(track_names, values_D, color=colors)

        axs[1][0].set_ylabel("D")
        axs[1][0].set_xlabel("Track IDs")

    # 4 - Display the average I values
    if histograms:
        axs[1][1].hist(all_i, color="blue", density=True)

        # Highlight tracks if required
        if highlight_ids is not None:
            for track_id in highlight_ids:
                if signal_properties:
                    i_values = intensities[track_ids]["contrast"]
                else:
                    i_values = intensities[track_ids]
                axs[1][1].hist(i_values, density=True, color="red")

        axs[1][1].set_ylabel("#")
        axs[1][1].set_xlabel(intensity_name)

    else:
        track_names = list(diffusivity.keys())

        axs[1][1].bar(track_names, avg_i, color=colors)

        axs[1][1].set_ylabel("<" + intensity_name + ">")
        axs[1][1].set_xlabel("Track IDs")

    # Optional graphs
    if signal_properties:

        # Extract the data
        avg_noise = []
        all_noise = []
        avg_snr = []
        all_snr = []
        for track_ids in intensities.keys():

            noise_values = intensities[track_ids]["noise"]
            avg_noise.append(np.mean(noise_values))
            for i in noise_values:
                all_noise.append(i)

            snr_values = intensities[track_ids]["snr"]
            avg_snr.append(np.mean(snr_values))
            for i in snr_values:
                all_snr.append(i)

        # 5 - Display the noise values
        if histograms:
            axs[2][0].hist(all_noise, density=True, color="blue")

            # Highlight tracks if required
            if highlight_ids is not None:
                for track_id in highlight_ids:
                    i_values = intensities[track_ids]["noise"]
                    axs[2][0].hist(i_values, density=True, color="red")

            axs[2][0].set_ylabel("#")
            axs[2][0].set_xlabel("Noise")

        else:
            track_names = list(diffusivity.keys())
            axs[2][0].bar(track_names, avg_noise, color=colors)

            axs[2][0].set_ylabel("<Noise>")
            axs[2][0].set_xlabel("Track IDs")

        # 6 - Display the SNR values
        if histograms:
            axs[2][1].hist(all_snr, density=True, color="blue")

            # Highlight tracks if required
            if highlight_ids is not None:
                for track_id in highlight_ids:
                    i_values = intensities[track_ids]["snr"]
                    axs[2][1].hist(i_values, density=True, color="red")

            axs[2][1].set_ylabel("#")
            axs[2][1].set_xlabel("SNR")

        else:
            track_names = list(diffusivity.keys())
            axs[2][1].bar(track_names, avg_snr, color=colors)

            axs[2][1].set_ylabel("<SNR>")
            axs[2][1].set_xlabel("Track IDs")

    plt.tight_layout()
    plt.show()


##-\-\-\-\-\-\-\-\
## PUBLIC FUNCTIONS
##-/-/-/-/-/-/-/-/

# ---------------------------------
# Compute the MSD on all the tracks
def computeMSD(trajectory, combine_all=True, fps=1, space_scale=1, max_lagtime=None):

    # Calculate the max lagtime is needed
    if max_lagtime is None:
        max_lagtime = _calculate_max_lagtime(trajectory)

    # Return the MSD of the collection of particles
    if combine_all:
        msd = _compute_general_msd(
            trajectory, space_scale, fps, max_lagtime=max_lagtime
        )
    else:
        msd = _compute_individual_msd(
            trajectory, space_scale, fps, max_lagtime=max_lagtime
        )

    return msd


# -----------------------------------------------
# Calculate the diffusivity from the MSD measured
def getDiffusivity(msd, dimensions=2):

    # Calculate D on a single MSD
    if isinstance(msd, np.ndarray):
        fit_results = {}
        (
            diffusivity,
            fit_results["parameters"],
            fit_results["errors"],
        ) = _calculate_diffusitivity(msd, dimensions=dimensions)

    # Calculate D on a collection of MSD
    else:

        # Initialize
        diffusivity = {}
        fit_results = {}

        # Process all tracks
        for track_id in msd.keys():

            # Calculate the MSD
            current_msd = msd[track_id]
            current_fit = {}
            (
                current_D,
                current_fit["parameters"],
                current_fit["errors"],
            ) = _calculate_diffusitivity(current_msd, dimensions=dimensions)

            # Save the results
            diffusivity[track_id] = current_D
            fit_results[track_id] = current_fit

    return diffusivity, fit_results


# ----------------------------------------------------------------------------
# Start a global verification of all the tracks collected and their properties
def verifyTracks(
    positions,
    array,
    signal_properties=False,
    histograms=True,
    highlight_ids=None,
    track_ids=None,
    scan_window=50,
    profile_type="gaussian",
    line_angle=0,
    fit_type="radial",
    n_points=1000,
    space_scale=None,
    dark_spots=False,
    use_fit=True,
    percentage=True,
    fps=1,
    max_lagtime=None,
    dimensions=2,
):

    # Calculate the intensity profiles
    profiles = tpip.intensityProfile(
        positions,
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

    # Get the frames for each track
    frames = {}
    for track_ids in profiles.keys():
        frames[track_ids] = np.copy(profiles[track_ids]["frame"])

    # Extract the value(s) to display
    if signal_properties:
        intensities = tpip.signalProperties(
            profiles,
            use_fit=use_fit,
            dark_spots=dark_spots,
            fit_type=profile_type,
            percentage=percentage,
        )
    else:
        intensities = tpip.integratedIntensity(
            profiles, use_fit=use_fit, dark_spots=dark_spots, fit_type=profile_type
        )

    # Calculate the MSD
    if space_scale is None:
        space_scale = 1
    msd = computeMSD(
        positions,
        combine_all=False,
        fps=fps,
        space_scale=space_scale,
        max_lagtime=max_lagtime,
    )

    # Calculate the diffusitivity
    diffusivity, _ = getDiffusivity(msd, dimensions=dimensions)

    # Display the informations
    _display_analysis(
        intensities,
        msd,
        diffusivity,
        frames,
        histograms=histograms,
        signal_properties=signal_properties,
        highlight_ids=highlight_ids,
    )
