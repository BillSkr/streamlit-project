# Software Technology 2025 Python Pipeline
Example computational pipeline for scRNA-seq data (software technology project 2025)

This is the source code for an example bioinformatics pipeline regarding the processing and analysis of scRNA-seq data using Scanpy

To use, follow the steps below:

## Python requirements

- Python version >= 3.9

- Create virtual environment

  - for windows:

    ```
    python -m venv venv

    venv\Scripts\activate

    pip install -r requirements.txt
    ```

  - for linux:

    ```
    python -m venv venv

    venv/bin/activate

    pip install -r requirements.txt
    ```

  - for conda:

    ```
    conda create -n <your_env_name>

    conda activate <your_env_name>

    conda install --file requirements.txt
    ```

  - for uv:

    ```
    uv venv --python 3.10

    uv pip install -r requirements.txt
    ```

   

## How to run

- Open console.

- Activate virtual environment.

- Run the following command:

  ```
  code .
  ```

- NOTE: if you are using uv, you don't have to activate the venv through the console; just type 

  ```
  code .
  ```

# The data used for this experiment can be found here:

[GSE148822](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148822)

