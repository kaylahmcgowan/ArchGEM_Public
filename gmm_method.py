#!/usr/bin/env python3
"""
Usage:
 python /home/kaylahbreanne.mcgowan/ARCHGEM2025/gmm_method.py /home/kaylahbreanne.mcgowan/ARCHGEM/datasets/L1/O3/L1_O3a_scattered_light_gspy.csv --savedir /home/kaylahbreanne.mcgowan/ARCHGEM2025_PAPER/L1/O3a/gmm_method --channel L1:GDS-CALIB_STRAIN --tdur 8

This script reads a CSV file containing candidate event times (with a column named "GPStime"),
retrieves TimeSeries data for each event, computes a Q–transform, fits a Gaussian Mixture Model (GMM)
to the high–energy (Energy > 70) time–frequency data, and produces a multi–panel figure along with a CSV
of computed parameters.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from gwpy.timeseries import TimeSeries

# Set default fonts (smaller than before).
plt.rc('font', family='serif', size=12)

# ---------------- Helper Functions ------------------
def filter_points_by_time(times, freqs, threshold=1, mode="lowest"):
    """
    Given two arrays (times and freqs), if two points occur within
    'threshold' seconds of each other, keep only one point according to mode.
    
    mode="lowest": keep the one with the lower frequency.
    mode="highest": keep the one with the highest frequency.
    
    Returns:
      kept_times, kept_freqs, discarded_times, discarded_freqs
    """
    kept_times = []
    kept_freqs = []
    discarded_times = []
    discarded_freqs = []
    for t, f in zip(times, freqs):
        if not kept_times:
            kept_times.append(t)
            kept_freqs.append(f)
        else:
            if t - kept_times[-1] < threshold:
                if mode == "lowest":
                    if f < kept_freqs[-1]:
                        discarded_times.append(kept_times[-1])
                        discarded_freqs.append(kept_freqs[-1])
                        kept_times[-1] = t
                        kept_freqs[-1] = f
                    else:
                        discarded_times.append(t)
                        discarded_freqs.append(f)
                elif mode == "highest":
                    if f > kept_freqs[-1]:
                        discarded_times.append(kept_times[-1])
                        discarded_freqs.append(kept_freqs[-1])
                        kept_times[-1] = t
                        kept_freqs[-1] = f
                    else:
                        discarded_times.append(t)
                        discarded_freqs.append(f)
                else:
                    raise ValueError("mode must be either 'lowest' or 'highest'")
            else:
                kept_times.append(t)
                kept_freqs.append(f)
    return (np.array(kept_times), np.array(kept_freqs),
            np.array(discarded_times), np.array(discarded_freqs))


def gmm_method(chan, event_time, tdur, n_components):
    """
    Retrieves TimeSeries data, computes a Q–transform, filters for high energy (>70),
    and fits a Gaussian Mixture Model (GMM) to the filtered time–frequency data.
    The GMM centroids are then filtered (keeping the one with the lowest frequency when centroids are too close).
    
    Returns:
      X: 2D array of filtered [Time, Frequency] points.
      probabilities: Maximum cluster assignment probability for each point.
      centroids_filtered: DataFrame with filtered centroid positions.
      qspecgram: The Q–transform object (for plotting the spectrogram).
      centroids_discarded: DataFrame with discarded centroids.
    """
    tstart = event_time - tdur
    tend = event_time + tdur
    TS = TimeSeries.get(chan, tstart, tend)
    qspecgram = TS.q_transform(qrange=(4,150), frange=(10,100),
                               outseg=(tstart, tend), fres=0.01)
    # Build a DataFrame of all time-frequency-energy values.
    times = qspecgram.times.value.repeat(len(qspecgram.frequencies.value))
    frequencies = np.tile(qspecgram.frequencies.value, len(qspecgram.times.value))
    energies = qspecgram.value.flatten()
    df = pd.DataFrame({"Time": times, "Frequency": frequencies, "Energy": energies})
    filtered_df = df[df["Energy"] > 70]
    if filtered_df.shape[0] < 2:
        print("Insufficient filtered data in GMM method (Energy > 70).")
        return np.array([]), np.array([]), pd.DataFrame(), qspecgram, pd.DataFrame()
    X = filtered_df[["Time", "Frequency"]].values
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(X)
    probabilities = gmm.predict_proba(X).max(axis=1)
    # For GMM, we choose to keep the centroid with the lowest frequency when centroids are too close.
    centroids_df = pd.DataFrame(gmm.means_, columns=["Time", "Frequency"]).sort_values("Time")
    kept_times, kept_freqs, discarded_times, discarded_freqs = filter_points_by_time(
        centroids_df["Time"].values, centroids_df["Frequency"].values, threshold=1, mode="lowest")
    centroids_filtered = pd.DataFrame({"Time": kept_times, "Frequency": kept_freqs})
    centroids_discarded = pd.DataFrame({"Time": discarded_times, "Frequency": discarded_freqs})
    return X, probabilities, centroids_filtered, qspecgram, centroids_discarded


# ---------------- Main Analysis Function ------------------
def arch_analysis(chan, event_time, tdur, n_components, savedir=None):
    """
    Fetches TimeSeries data, computes the Q-transform, fits a Gaussian Mixture Model,
    and produces a multi-panel figure showing:
      - The Q-transform spectrogram.
      - The maximum frequency per time with the GMM centroids overlaid.
      - The selected (sorted) peak locations from the GMM centroids.
      - The velocity vs. frequency computed from the centroids.
    
    Also computes and prints several parameters and exports them to CSV.
    """
    tstart = event_time - tdur
    tend = event_time + tdur

    # Create (and change to) the saving directory if provided.
    if savedir:
        save_dir = os.path.join(savedir, str(int(event_time)))
        os.makedirs(save_dir, exist_ok=True)
        os.chdir(save_dir)
    else:
        print('No directory given; results will be saved in the current working directory.')

    # Retrieve GMM results.
    # (Note: qspecgram is computed here; we do not use any find_peaks processing.)
    n_points = 9 if n_components is None else n_components
    X, probabilities, centroids_filtered, qspecgram, centroids_discarded = gmm_method(chan, event_time, tdur, n_components)
    
    if centroids_filtered.empty:
        print("No valid GMM centroids found. Exiting analysis.")
        return

    # Reconstruct the filtered DataFrame (using the same Q-transform) to compute maximum frequency vs. time.
    times_all = qspecgram.times.value.repeat(len(qspecgram.frequencies.value))
    freqs_all = np.tile(qspecgram.frequencies.value, len(qspecgram.times.value))
    energies_all = qspecgram.value.flatten()
    df_all = pd.DataFrame({"Time": times_all, "Frequency": freqs_all, "Energy": energies_all})
    filtered_df_all = df_all[df_all["Energy"] > 70]
    max_freq_at_time = filtered_df_all.groupby("Time")["Frequency"].max()
    
    # Build a sorted table of selected peaks from the GMM centroids.
    t_fmax_selected = centroids_filtered.sort_values("Time")[["Time", "Frequency"]].values
    if t_fmax_selected.shape[0] < 2:
        print("Not enough GMM centroids to compute time differences.")
        return
    # Compute time differences.
    t_fmax_sorted = t_fmax_selected  # already sorted by time
    dt = np.ediff1d(t_fmax_sorted[:, 0])
    
    # Calculate analysis parameters.
    f_scat = 1 / (np.average(dt))
    f_max_avg = np.average(t_fmax_sorted[:, 1])
    l = 0.000001064  # laser wavelength in meters
    x_surf = (f_scat / 1.77) * l
    v_surf = f_scat * l
    std_freq = np.std(t_fmax_sorted[:, 1])
    std_dt = np.std(dt)
    
    # Print computed values.
    print(f"{f_scat:.3f} Average frequency of motion [Hz]")
    print(f"{f_max_avg:.3f} Average maximum frequency of arches [Hz]")
    print(f"{x_surf:.3e} Scattering surface movement distance [m]")
    print(f"{v_surf:.3e} Velocity of the scattering surface [m/s]")
    print(f"{std_freq:.3f} Standard deviation of maximum frequencies")
    print(f"{std_dt:.3f} Standard deviation of time intervals between arches")
    
    # Export key variables to CSV.
    variables = {
        'gps_time': event_time,
        'f_scat': f_scat,
        'f_max_avg': f_max_avg,
        'x_surf': x_surf,
        'v_surf': v_surf,
    }
    df_vars = pd.DataFrame(list(variables.items()), columns=['Variable Name', 'Value'])
    output_file = f"output_{event_time}.csv"
    df_vars.to_csv(output_file, index=False)
    print(f"Variables exported to {output_file}")

    # Compute velocities for the selected peaks.
    if t_fmax_sorted.shape[0] > 1:
        velocities_selected = []
        for i in range(len(t_fmax_sorted) - 1):
            dt_step = dt[i]
            f_scat_i = 1 / dt_step
            v_surf_i = f_scat_i * l
            velocities_selected.append([v_surf_i, t_fmax_sorted[i, 1]])
        velocities_selected = np.array(velocities_selected)
    else:
        velocities_selected = np.array([])

    # ------------- Plotting Section (Multi-panel Figure) -------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # ----- Subplot 1 (Top Left): Q Transform Spectrogram -----
    ax1 = axs[0, 0]
    t_vals = qspecgram.times.value
    f_vals = qspecgram.frequencies.value
    dt_med = np.median(np.diff(t_vals))
    df_med = np.median(np.diff(f_vals))
    t_edges = np.concatenate(([t_vals[0] - dt_med/2], t_vals + dt_med/2))
    f_edges = np.concatenate(([f_vals[0] - df_med/2], f_vals + df_med/2))
    im_spec = ax1.pcolormesh(t_edges, f_edges, qspecgram.value.T, shading='auto',
                             cmap='viridis', vmax=100)
    ax1.set_title('Q Transform', fontsize=14)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Frequency (Hz)', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_ylim(10, 100)
    cbar1 = fig.colorbar(im_spec, ax=ax1)
    cbar1.set_label('Normalized energy', fontsize=12)
    
    # ----- Subplot 2 (Top Right): Maximum Frequency vs Time with GMM Centroids -----
    ax2 = axs[0, 1]
    ax2.plot(max_freq_at_time.index - tstart, max_freq_at_time.values, '.', color='mediumblue', label='Max frequency')
    ax2.plot(centroids_filtered["Time"] - tstart, centroids_filtered["Frequency"], 'x',
             color='yellow', markersize=10, label='GMM Centroids')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Frequency (Hz)', fontsize=12)
    ax2.set_title('Max Frequency vs Time', fontsize=14)
    ax2.legend(fontsize=10)
    
    # ----- Subplot 3 (Bottom Left): Selected GMM Centroid Locations -----
    ax3 = axs[1, 0]
    ax3.plot(t_fmax_sorted[:, 0] - tstart, t_fmax_sorted[:, 1], 'o', color='navy',
             markeredgecolor='black', markersize=8)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Frequency (Hz)', fontsize=12)
    ax3.set_title('Selected GMM Centroids', fontsize=14)
    ax3.grid(True)
    
    # ----- Subplot 4 (Bottom Right): Velocity vs Maximum Frequency -----
    ax4 = axs[1, 1]
    if velocities_selected.size > 0 and velocities_selected.ndim == 2:
        ax4.scatter(velocities_selected[:, 0], velocities_selected[:, 1],
                    marker='x', color='crimson', s=50, label='Cycle')
        ax4.set_xlabel('Average Velocity (m/s)', fontsize=12)
        ax4.set_ylabel('Frequency (Hz)', fontsize=12)
        ax4.set_title('Velocity vs Frequency', fontsize=14)
        ax4.legend(fontsize=10)
        ax4.grid(True)
    else:
        ax4.text(0.5, 0.5, 'No valid velocity data', ha='center', va='center', fontsize=12)
        ax4.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(f"combined_plots_{event_time}.png")
    plt.close()
    print(f"Combined figure saved as combined_plots_{event_time}.png")
    
    return


# ---------------- Main Script ------------------
def main(args):
    # Read the CSV file containing candidate events.
    try:
        arch_candidates = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error reading CSV file {args.csv_file}: {e}")
        sys.exit(1)

    # Create the results directory if it does not exist.
    os.makedirs(args.savedir, exist_ok=True)

    # Iterate through each row in the dataframe and run arch_analysis.
    for index, row in arch_candidates.iterrows():
        chan = args.channel  # Use the channel provided as an argument.
        event_time = row['GPStime']  # Assumes the CSV has a column named 'GPStime'
        tdur = args.tdur
        n_components = args.n_components

        # Convert event_time to string for folder name.
        event_time_str = str(int(event_time))
        expected_folder_path = os.path.join(args.savedir, event_time_str)
        print(f"Processing event time {event_time_str}...")
        if os.path.isdir(expected_folder_path):
            print(f"Folder for event time {event_time_str} already exists. Skipping.")
            continue

        try:
            arch_analysis(chan, event_time, tdur, n_components, savedir=args.savedir)
            print(f"Processed event time {event_time_str}.\n")
        except Exception as e:
            print(f"An error occurred while processing event time {event_time_str}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ARCH analysis on a CSV file containing event times (GPStime column) using GMM only."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV file containing candidate events."
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default="./archgem_results",
        help="Directory where results will be saved (default: ./archgem_results)"
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="L1:GDS-CALIB_STRAIN",
        help="Channel to use (default: L1:GDS-CALIB_STRAIN)"
    )
    parser.add_argument(
        "--tdur",
        type=float,
        default=8.0,
        help="Duration (in seconds) around the event time to analyze (default: 8)"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=9,
        help="Number of components for the Gaussian Mixture Model (default: 9)"
    )
    args = parser.parse_args()
    main(args)
