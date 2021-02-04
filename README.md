# reactive_ff
Run [./unpack.sh](unpack.sh) to unpack tar files.

Use [./write_random_log_files.py](write_random_log_files.py) to create QM input files from a random subset of existing xyz files, while using [./make_batch_input_from_xyz.py](make_batch_input_from_xyz.py) for the full set (not updated to the orca input).

[./normal_mode_sampling.py](normal_mode_sampling.py) does normal mode sampling, using the Q-Chem dataset.
This creates the `xyz` folder, which can be extracted from the [./archives](archives).

Use [./parse_molpro.py](parse_molpro.py) and [./parse_orca.py](parse_orca.py) for parsers.
