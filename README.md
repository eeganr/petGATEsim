# PET Simulation Codebase

This is a bunch of my work in developing simulation/data analysis software for PET imaging.

## Installation

This whole repository runs best on Stanford Sherlock, so I recommend cloning it there if you're using the scripts and such. Overall though, setup should be pretty straightforward. Python 3.12 is ideal as well.

All the requirements to run the currently-used files are in requirements.txt and can be installed with:
```bash
ml python/3.12
pip install --user -r requirements.txt
```
(first line only if using Sherlock). Also, the randoms correction/data processing toolkit requires its own separate module. I recommend running:
```bash
cd randoms
pip install --user .
```

## Simulation

The files needed to simulate are all in macros and some scripts in the parent folder. I'll describe them briefly.

### Simulation Run Scripts

The scripts that are run when making a new simulation in order from highest level to lowest are:

1) **submit.sh** &mdash; Generates multiple jobs, one for each simulation. We usually run multiple simulations (I do up to 120 regularly) at the same time on Sherlock if we need a lot of data. Otherwise, the simulations are pretty slow. This queues jobs stored in customgen.sh, and ensures that the files generated from the simulations will be discrete.
2) **customgen.sh** &mdash; The actual script that gets queued. Defines geometry file, phantom, time to run simulation, etc.
3) **runtrain.sh** &mdash; Here is where the actual GATE simulation is queued. In this file you'll find the output directory for the files. Otherwise, it mostly just takes information from customgen.

### Macros

Most of these are just specific phantoms, but there's a few that aren't. In particular:

- **simu_pet.mac** &mdash; This is the parent macro that all the other macros are run within. You may need to poke around in here to add additional macros that for instance control physical regions if you need that. The geometry file is passed in from the runtrain script as {camerafile} and the phantom as {sourcefile}. But, for instance, if you want a water background volume that may need to be manually added. Also, data output format is controlled here. We mainly just run off singles and don't bother with the other generated files, but if one wants to change that one can.
- **digitizer.mac** &mdash; This handles the digitizer, evidently. In particular, this has energy filtering information, energy/time blurring. It also has coincidence sorting information, but we generally don't use that and just bundle coincidences ourselves (theirs is somewhat unreliable and doesn't lend itself well to customization).
- **Geometry.mac** &mdash; This is just the actual PET scanner. I think it's the GenII system.

The phantom macros are varied but a few good examples are:
- **scatterphantom.mac** &mdash; It's the [NEMA PET Scatter Phantom](https://www.spect.com/our-products/nema-scatter-phantom)'s actual activity information.
- **scatterphantomvol.mac** &mdash; It's the corresponding volume file that needs to be loaded before initialization in the simu_pet.mac file to ensure you get the plastic and water where they're supposed to go.

## Data Processing and Analysis

### Note on Python/C++ Bindings

The backend of how the data processing and analysis works is in the */randoms* folder. They're essentially python bindings of C++ functions, just because there's so much data that has to be sifted through, often not in an easily vectorized fashion. This folder doesn't have to be touched much, as nearly all the functions defined in randoms.cpp have corresponding python bindings pre-written in the python file. See the installation section for how to install the randoms module.

### Python Analysis Functions

Each analysis python file has an accompanying script. Unfortunately some of the parameters are at the top of the python file, others are in the script, just depending on what's more consistent between runs and what you might need to run with differing parameters.
You should probably run aggregate.py first, the rest are just depending on what you need.
1) **aggregate.py** &mdash; A lot of things have to be done through just a raw pass of the data. Each simulation that you run will generate a separate Singles.dat file, and they all are combed through by this python file, with corresponding .sh script which tells the file which of the Singles files to look through (they should be labeled Singles1, Singles2, ...). There's also a pretty self-explanatory config at the top of the python file. It'll generate a bunch of files in a folder you define. In particular:

    - **[NAME].lm** &mdash; all the coincidences in the 10 parameter listmode format (x1, y1, z1, TOF, # scatters, x2, y2, z2, crystalID1, crystalID2)
    - **[NAME]_delay.lm** &mdash; all the delayed window coincidences, also listmode
    - **[NAME]_actual.lm** &mdash; all of the coincidences that are <u>actually</u> randoms (aka ground truth randoms).
    - **coin_lor.npy** &mdash; all the coincidences for a given LOR in a DETS x DETS matrix (row i column j is coincidences for LOR with ends at detector i and detector j, matrix is symmetric) that can be loaded with np.load
    - **actuals.npy** &mdash; similar to above but for ground truth random coincidences 
    - **dw_nums.npy** &mdash; similar to above but number of delayed coincidences at each LOR
    - **scatters.npy** &mdash; similar to above but just number of scatters at each LOR
    - **prompts_count.npy** &mdash; just total number of prompts for a given detector (is a 1D array) 
    - **prompts_count.npy** &mdash; just total number of singles for a given detector (is a 1D array)

    The goal is to basically make it so that this is mostly enough stuff to run whatever other script one might need.
2) **randoms_agg.py** &mdash; generates additional files specifically for the use of the Singles-Prompts randoms correction technique.
3) **split.py** &mdash; Splits the big listmode files into separate files (e.g. 0_1_coin.lm) for coincidences on each LOR
4) **combine.py** &mdash; The inverse of split!
5) **correct.py** &mdash; Applies delayed window randoms correction, singles-prompts, and ground truth correction to the SPLIT files. (So if you use this, first split the data, then correct it, then recombine it).
6) **tag.py** &mdash; tags a listmode file with each line's random/scatter fraction (scatter count becomes scatter fraction for given LOR, crystalID2 becomes randoms fraction).
7) **3param.py** &mdash; turns a listmode file into a 3parameter format file (crystalID1, crystalID2, TOF). Also edits timing information to be compatable with Dr. Nasir's scripts.
8) **write_lut.py** &mdash; This one's an odd one out since it actually does look at all the Singles.dat files again. This just writes a lookup table of each line being a coincidence and whether it's a random, scatter, or true coincidence. It's just 4 parameters, (crystalID1, crystalID2, TOF, type) where:
$$ \mathrm{type} = 
\begin{cases} 
    0 & \text{if true} \\
    1 & \text{if scatter} \\
    2 & \text{if random}
\end{cases} $$

## Quick Reference Checklist

If you're running a simulation, here's my quick reference checklist of things to do:

1) **submit.sh** &mdash; ensure number of simulations is correct
2) **customgen.sh** &mdash; ensure phantom macro is correct, customize time for each simulation, check time to allot for simulation to complete on Sherlock is enough
3) **runtrain.sh** &mdash; change output folder for data (it will overwrite previous data otherwise)
4) **digitizer.mac** &mdash; ensure coincidence window, energy filtering, centroids, etc are ok
5) **simu_pet.mac** &mdash; check the volume information of the phantom (e.g. watercylinder.mac accompanying the cylinder)
Then just run `sh customgen.sh` .

Afterwards, depending on your needs, run the appropriate analysis in the analysis folder.



