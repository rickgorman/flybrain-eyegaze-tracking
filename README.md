# flybrain-eyegaze-tracking

A webcam gaze tracker that steers an on-screen fruit fly through a simulation of the real *Drosophila* male central nervous system connectome.

```text
Webcam → face/iris features (in the browser) → personal gaze calibration
       → gaze direction relative to the fly → fly neural simulation
       → trained motor readout → movement of the on-screen fly
```

## How it works

**In the browser.** [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) extracts iris positions, eye-expression blendshapes and head pose from your webcam. A short calibration fits a personal regularized regression from those features to a point on the screen, and a separate validation pass reports its error. Video never leaves the browser.

**In a local Python process.** The browser sends only the normalized direction from the fly to your gaze point (`dx`, `dy`). `brain_controller.py` turns that into firing-rate drive on four existing visual-projection populations (LC4 and LPLC2, left and right), runs a leaky integrate-and-fire simulation of **162,517 neurons and 6,138,378 connections** for 80 ms, and averages the membrane voltage of **1,310 descending neurons** over the last 40 ms. A linear readout turns those voltages into a velocity. The fly moves only by that velocity — the target coordinates never reach the readout.

Training changes only the readout. The connectome weights stay fixed. Each update resets neural state, so the same input always gives the same output.

### What it is not

This is an engineered controller built around real connectivity. It is **not** a validated biological fly, and it does not understand webcam images. The input-channel-to-direction assignment, neuron dynamics and motor decoder are modeling choices; neurotransmitter signs are predictions. Gaze accuracy comes from the camera, feature extraction and your calibration — the connectome does not make gaze tracking more accurate.

## Getting started

The connectome data and the simulator are not included in this repository. The steps below download and build them locally (about 1.1 GB of downloads).

### Requirements

- Python 3.12+
- Google Chrome
- A webcam
- Git and `curl`
- ~2 GB free disk space

### 1. Get the simulator

The leaky integrate-and-fire engine comes from [TheMrRaGe/flybrain](https://github.com/TheMrRaGe/flybrain), pinned to the commit this controller was built and tested against:

```bash
git clone https://github.com/TheMrRaGe/flybrain.git upstream/flybrain
git -C upstream/flybrain checkout 9aaed09db9aef1c961a5aff82e47982e04b5e516
```

### 2. Install Python dependencies

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy pandas pyarrow pytest
```

### 3. Download the connectome

The [Male CNS v1.0](https://janelia-flyem.github.io/male-cns/download/) connectivity tables from HHMI Janelia FlyEM (CC BY 4.0). The weights file is about 1 GB.

```bash
mkdir -p data/raw
cd data/raw
BASE=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
curl -fLO "$BASE/body-annotations-male-cns-v1.0-minconf-0.5.feather"
curl -fLO "$BASE/body-neurotransmitters-male-cns-v1.0.feather"
curl -fLO "$BASE/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
shasum -a 256 -c <<'EOF'
2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2  body-annotations-male-cns-v1.0-minconf-0.5.feather
95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621  body-neurotransmitters-male-cns-v1.0.feather
e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1  connectome-weights-male-cns-v1.0-minconf-0.5.feather
EOF
cd ../..
```

### 4. Build the simulation graph

Keeps traced, typed neurons and connections with at least five synapses:

```bash
mkdir -p data/processed
.venv/bin/python upstream/flybrain/scripts/build_creature.py \
  --data data/raw \
  --weights data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --whole --min-weight 5 \
  --out data/processed/male-cns-v1.0-traced-typed-min5.npz
```

The result should report 162,517 neurons and 6,138,378 connections. Its SHA-256 is `55daea10e1beeb2fcc19f2883400cd7c6b4d3595f204117ecd2eaac05b3dd8f9`; the controller records the graph hash in its checkpoint and refuses a checkpoint trained on a different graph.

### 5. Train the fly controller

```bash
.venv/bin/python brain_controller.py train
```

This fits the linear readout on 144 simulated directions, checks it against 36 held-out directions, and saves `data/processed/gaze-controller.npz`. It takes under a minute. You can also retrain from the web page.

### 6. Run

```bash
.venv/bin/python gaze_server.py
```

Open **http://127.0.0.1:8766** in Chrome. Put the window on the screen you intend to use, then:

1. **Start camera** and sit comfortably with both eyes visible.
2. **Calibrate** by looking at the targets. This fits your personal gaze map.
3. **Check accuracy** on separate targets that were not used for fitting.
4. Look around the field. The fly follows your estimated gaze.

**Pointer test** sends the mouse position through the same neural controller, so you can check the fly's movement without depending on webcam accuracy. **Pause** stops movement; **Stop camera** releases the webcam. Recalibrate after moving the window to another display, changing the camera, or changing your seating position. Losing the camera, blinking, pausing or losing the backend stops the fly.

The server listens on `127.0.0.1` only. No video is recorded, and the operating-system cursor is never controlled. Your calibration is saved in the browser's local storage; **Reset calibration** clears it.

## Tests

```bash
.venv/bin/python -m pytest -q
node web/test-gaze.mjs
```

The Python tests use a tiny synthetic connectome, but still need the simulator from step 1.

## Layout

- `web/` — browser app: camera, calibration, validation and the field the fly moves in
- `web/vendor/` — MediaPipe Tasks Vision bundle, WebAssembly runtime and face landmarker model
- `gaze_server.py` — local HTTP server: static files plus the status, step, train and pause API
- `brain_controller.py` — input encoding, simulation window, readout training and checkpoint validation
- `tests/` — controller and server tests

## Credits

- **Connectome:** Male CNS Connectome v1.0 — HHMI Janelia FlyEM Project Team, the Cambridge Drosophila Connectomics Group (MRC Laboratory of Molecular Biology) and Google Research Connectomics. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Not redistributed here.
- **Simulator:** [TheMrRaGe/flybrain](https://github.com/TheMrRaGe/flybrain), an experimental community leaky integrate-and-fire simulator. Not redistributed here; step 1 fetches it.
- **Model parameters:** Shiu et al., "A Drosophila computational brain model reveals sensorimotor processing", *Nature* 634, 210 (2024), [doi:10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9).
- **Face tracking:** [MediaPipe](https://github.com/google-ai-edge/mediapipe) by Google, Apache License 2.0 (see `web/vendor/LICENSE`).
