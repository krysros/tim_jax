# README

Solution of Simply Supported Rectangular Plates under Sinusoidal Load using Automatic Differentiation (AD).

## Theory

See ch. 5, p. 105 of [Theory of Plates and Shells](Theory_of_Plates_and_Shells.bib) by S. Timoshenko and S. Woinowsky-Krieger.

## Installation

```console
python -m pip install -r requirements.txt
```

## Setup (optional)

```bash
export JAX_ENABLE_X64=True
export JAX_PLATFORM_NAME=cpu
```

or

```powershell
$env:JAX_ENABLE_X64 = "True"
$env:JAX_PLATFORM_NAME = "cpu"
```

## Tests

```console
pytest
```

## Calculations

```console
python main.py
```