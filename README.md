# EE274 final project fall 2022
The goal of our final project is to explore lossless image compression via
PNG.

See [SCL](https://github.com/kedartatwawadi/stanford_compression_library) for
original README.

## Getting started
- Create conda environment and install required packages:
    ```
    conda create --name ee274_project python=3.8.2
    conda activate ee274_project
    python -m pip install -r requirements.txt
    ```
- Add path to the repo to `PYTHONPATH`:
    ```
    export PYTHONPATH=$PYTHONPATH:<path_to_repo>
    ``` 

- **Run unit tests**

  To run all tests:
    ```
    find . -name "*.py" -exec py.test -s -v {} +
    ```

  To run a single test
  ```
  py.test -s -v core/data_stream_tests.py
  ```

## Contributor cookbook

To add your own compressor that plays with PNG, the workflow may go something
like this:

1.  Add a compressor in the folder `png_compressors/` that inherits from
    `CoreEncoder` (more details in `png_compressors/README.md`).
2.  Add the name of your compressor to `analysis/file_comparison.py`'s
    `get_encoder()` function.
3.  Compare your compressor's performance to others by running
    `analysis/file_comparison.py` as a script (more details in
    `analysis/README.md`.
 
