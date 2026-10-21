Preprocessing: raw overnight ECG and skin temperature to 10-minute masters to nightly tables.

`00_master_<family>.py` builds one feature-extraction design x analysis window (see `doc/DATA.md`);
`01`-`06` align labels, impute and clean, in that order. Run through `scripts/reproduce.sh` (stage 1).
