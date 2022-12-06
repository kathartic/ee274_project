# Analysis

The scripts in this folder are meant to compare the performance of our custom
compressors against the given PNG.

## `file_comparison` instructions

The `file_comparison` script calculates the compression ratio for a given
compressor. Assuming you cloned the repo into `~/ee274_project`, you can run:

```bash
$ cd ~/ee274_project
$ python analysis/file_comparison.py \
    -f test_data/kodim03.png \
    -c filtered_zlib
```

Where `-f` controls test file, and the compressor type is passed in via `-c`
flag. Allowed compressor types are:

- PNG filters + zlib: `filtered_zlib` or `filteredzlib`
- PNG filters + zstd: `filtered_zstd` or `filteredzstd`
- PNG filters + 3rd-order Markov arithmetic: `arithmetic3` or `filtered_arithmetic3`
- PNG filters + 4th-order Markov arithmetic: `arithmetic4` or `filtered_arithmetic4`

Note all compressors passed in via CLI are case-insensitive. You can also
optionally control verbosity by passing in the `-v` flag, e.g.

```bash
$ python analysis/file_comparison.py \
    -f test_data/stanford-logo.png \
    -c filtered_zstd \
    -v
```

which will print more logs. You can always run the script with the `--help` flag
for more:

```bash
$ python analysis/file_comparison.py --help
```

### filter type handling

In the PNG specification, filter types are prepended to each scanline (see
RFC 2083, section 4.1.3). However, as part of our experimentation we've played
with instead encoding the filter types as their own block.

To encode filter types as their own block, pass in the `-s` flag, e.g.

```bash
$ python analysis/file_comparison.py \
    -f test_data/kodim01.png \
    -c arithmetic3 \
    -s
```
