# mapping

[![Documentation Status](https://readthedocs.org/projects/mapping-fyst/badge/?version=latest)](https://mapping-fyst.readthedocs.io/en/latest/?badge=latest)

This package is forked from <https://github.com/KristinChengWu/mapping> with small modifications.

``mapping`` is a Python software package for simulating on-sky data acquisition and forecasting scanning performance.
The ``mapping`` package allows for tools that:

* Configuring camera instruments with user-defined and/or pre-defined detector arrays.
* Representing different scan patterns in terms of its offsets as well as its telescope motion.
* Determining optimal times for observing a particular source.

While default options are specific to PrimeCam and the Fred Young Submillimetre Telescope (FYST),
the package can also be used for other telescopes and cameras.

See the full documentation here:

<https://mapping-fyst.readthedocs.io/en/latest/>

To interact with the [tutorial notebooks](tutorials) without installation, you can use this:

<https://mybinder.org/v2/gh/KristinChengWu/mapping/HEAD>

Below are Overleaf reports of our investigations.

* [Effect of Field Rotation on Polarimetric Measurements](https://www.overleaf.com/read/xbwfsngdnvyc): Summarizes our investigation on the implications of field rotation in our scans.
* [Pong Scan Motion Analysis](https://www.overleaf.com/read/wjtvgybbkmbj): Analyzes the motion of the “Curvy Pong” scan pattern. Highlights how speed, acceleration, and jerk changes with specific parameters and how it compares with limitations to the telescope.
* [Daisy Motion and Area Coverage Analysis](https://www.overleaf.com/read/xmphnjnfkkvg): A similar motion analysis for the Daisy pattern, which is optimized for point-sources. Also talks about how much area coverage one might expect.

## Installation

For local installations,

```bash
git clone git@github.com:ccatobs/scan_patterns
cd scan_patterns
pip install .
```

Below is the original instruction.

For local installations, you can install the latest version

```bash
pip install git+https://github.com/KristinChengWu/mapping.git#egg=mapping
```

or upgrade an existing version.

```bash
pip install -U git+https://github.com/KristinChengWu/mapping.git#egg=mapping
```

For use in Docker or other self-contained python projects use the following line in the requirements.txt file:

```bash
-e git+https://github.com/KristinChengWu/mapping.git#egg=mapping
```
