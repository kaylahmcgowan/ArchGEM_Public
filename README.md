# ARCHGEM: Scattered Light Arch Analysis Tool

ARCHGEM is a dual-method pipeline for identifying and analyzing arch-shaped features in LIGO time-frequency data. It combines two complementary methods: a peak-finding algorithm and a Gaussian Mixture Model (GMM)-based clustering technique. This tool helps in characterizing scattered light noise by extracting statistics such as scattering frequency, surface velocity, and arch properties.

---

## 🚀 Features

* **Find Peaks Method**: Uses maxima in time-frequency energy to identify arches.
* **GMM Method**: Machine learning-based clustering of time-frequency points.
* **Combined Diagnostics Plot**: Shows spectrogram, cluster confidence, and velocity-frequency relations.
* **Robust Error Handling**: Skips invalid events and logs all failures.
* **Merged Summary Output**: Per-event outputs + combined summary including an averaged result.

---

## 📦 Installation & Requirements

ARCHGEM requires Python 3.7+ and the following packages:

```bash
pip install numpy pandas matplotlib scikit-learn gwpy
```

---

## 📁 Input Format

CSV file with column:

```csv
GPStime
1259274047
1259274999
...
```

---

## 🧪 Usage

### Step 1: Extract event times from CSV for HTCondor submission

> ⚠️ Update `extract_event_times.sh` to point to your specific `.csv` file.

```bash
cut -d',' -f1 L1_O3a_scattered_light_gspy.csv | tail -n +2 > event_times.txt
chmod +x extract_event_times.sh
./extract_event_times.sh
```

This creates a plain `event_times.txt` file for HTCondor to queue jobs.

### Step 2: Run ARCHGEM analysis

#### 🔁 Condor Submission:

```bash
condor_submit job.submit
```

#### 🧪 Local Example (single event):

```bash
python execute_archgem.py \
  --event_time 1259274047 \
  --file_path L1_O3a_scattered_light_gspy.csv \
  --savedir RESULTS/L1/O3a/ \
  --channel L1:GDS-CALIB_STRAIN \
  --tdur 8 \
  --n_components 9
```

#### 🧪 Local Full Run (multiple events):

```bash
python final_combined_script_with_plot.py \
  L1_O3a_scattered_light_gspy.csv \
  --savedir RESULTS/L1/O3a/ \
  --channel L1:GDS-CALIB_STRAIN \
  --tdur 8
```

---

## 📤 Output Structure

```
/output_directory/
├── combined_summary.csv          # All event results with avg row per event
├── summary_log.csv              # Processing log with error status
├── 1259274047/
│   ├── combined_plot_1259274047.png
│   ├── output_1259274047.csv   # Combined method results for this event
│   └── findpeaks_1259274047.csv (optional)
│   └── gmm_1259274047.csv      (optional)
...
```

---

## 📊 Output Columns (combined\_summary.csv)

* `event_time`: GPS timestamp
* `method`: one of `find_peaks`, `gmm`, or `average`
* `f_scat`: Estimated scattering frequency \[Hz]
* `f_max_avg`: Average max frequency of peaks \[Hz]
* `x_surf`: Estimated surface displacement \[m]
* `v_surf`: Estimated velocity of scattering surface \[m/s]

---

## 🛠 Troubleshooting

All failed events are logged in `summary_log.csv` with a descriptive error message. Example:

```csv
1259274999,0,10,no,no,"No valid peaks found using Find Peaks."
```

---

## 🔬 Applications

* Detector commissioning: correlate arches with hardware changes
* Data quality: improve glitch classification (e.g. Gravity Spy)
* Astrophysics: rule out scattering noise in transient candidate analysis

---

## 📬 Citation

Please cite our forthcoming paper or contact [Kaylah McGowan](mailto:kaylahbreanne.mcgowan@ligo.org) for updates.

---

## 📁 Repository Tree

```bash
ARCHGEM/
├── fp_in.py
├── gmm_in.py
├── run_archgem.py
├── execute_archgem.py
├── final_combined_script_with_plot.py
├── run_archgem.sh
├── job.submit
├── extract_event_times.sh
├── README.md
├── /output/...
```


