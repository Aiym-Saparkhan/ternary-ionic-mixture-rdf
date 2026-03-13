# ternary-ionic-mixture-rdf

This project calculates radial distribution functions (RDF) for a ternary ionic mixture using the Yukawa potential and the Ornstein-Zernike equation with HNC closure.

## System
The ternary ionic mixture consists of:
- D+
- He2+
- C6+

## Method
The code uses:
- Yukawa potential
- Ornstein-Zernike equation
- HNC approximation

## Output
The program calculates:
- g_DD(r)
- g_HeHe(r)
- g_CC(r)
- g_DHe(r)
- g_DC(r)
- g_HeC(r)

It also saves:
- RDF plot
- HNC convergence plot
- CSV table with RDF values
- CSV table with peak values

## Main file
- `main.py`

## Parameters
The default parameters are:
- Gamma = 0.8
- kappa = 1.0
- x = [0.70, 0.20, 0.10]

## Run
Run the script in Python:
```python
python main.py
