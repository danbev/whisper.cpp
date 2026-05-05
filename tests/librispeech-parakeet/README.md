# parakeet.cpp/tests/librispeech

[LibriSpeech](https://www.openslr.org/12) is a standard dataset for
training and evaluating automatic speech recognition systems.

This directory contains a set of tools to evaluate the recognition
performance of parakeet.cpp on LibriSpeech corpus.

## Quick Start

1. (Pre-requirement) Compile `parakeet-cli` and prepare the Parakeet
   model in `ggml` format.

   ```
   $ # Execute the commands below in the project root dir.
   $ cmake -B build
   $ cmake --build build --config Release
   ```

2. Set up the environment to compute WER score.

   ```
   $ pip install -r requirements.txt
   ```

   For example, if you use `virtualenv`, you can set up it as follows:

   ```
   $ python3 -m venv venv
   $ . venv/bin/activate
   $ pip install -r requirements.txt
   ```

3. Run the benchmark test.

   ```
   $ python run_eval.py --download --cli ../../build/bin/parakeet-cli
   ```

   On Windows with a Visual Studio build, the CLI path usually includes the
   configuration directory:

   ```
   > python run_eval.py --download --cli ..\..\build\bin\Release\parakeet-cli.exe
   ```

   For a CUDA build, pass the CUDA build's `parakeet-cli` explicitly:

   ```
   > python run_eval.py --download --force --cli ..\..\build-cuda\bin\Release\parakeet-cli.exe
   ```

   `parakeet-cli` loads the model once for all pending LibriSpeech files.
   Without `--force`, existing `*.flac.txt` transcript files are reused.

## Makefile runner

The checked-in `Makefile`/`eval.mk` runner uses `make` plus commands such as
`wget`, `tar`, `mv`, and `rm`. It invokes the configured `parakeet-cli` once
per audio file and then runs `eval.py`.

## How-to guides

### How to change the inference parameters

With `run_eval.py`, pass Parakeet CLI options after `--`.

```
$ python run_eval.py --download --cli ../../build/bin/parakeet-cli -- --threads 8 --no-flash-attn
```

With the Makefile runner, create `eval.conf` and override variables.

```
PARAKEET_MODEL = parakeet-tdt-0.6b-v3
PARAKEET_FLAGS = --no-prints --threads 8 --output-txt
```

Check out `eval.mk` for more details.
