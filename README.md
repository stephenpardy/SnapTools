# SnapTools

![SnapTools's Travis CI Status](https://travis-ci.org/stephenpardy/SnapTools.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/stephenpardy/SnapTools/badge.svg?branch=master)](https://coveralls.io/github/stephenpardy/SnapTools?branch=master)

Bespoke tools for working with isolated or interacting galaxy snapshots produced by Gadget (or gadget family).
Does not work for Cosmological snapshots.

## Install:
```
pip install git+https://github.com/stephenpardy/SnapTools.git
```
PyPI distro coming soon.

## Requires:

numpy>=1.10.0

matplotlib

multiprocess

h5py

## How to use:

```python
from snaptools import snapshot
snap = snapshot.Snapshot('tests/galaxies0.hdf5')
print(snap)  # get info about the snapshot
print(snap.pos['stars'].mean(axis=0))  # get a rough center of mass of the simulation
```

Updated tutorials coming soon. In the meantime, see https://github.com/stephenpardy/Offsets_Notebooks for some older examples (not all may work with this version).

TODO
- [ ] 100% testing
- [ ] Add better plotting methods
- [ ] Add interface to YT.