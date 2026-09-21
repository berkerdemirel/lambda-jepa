# Encoder log-det block-Gram figure (bottom-right cell)

Reproduces `figures/figure-hier-qqt_encoder_bottomright.png`: the
(λ_L2=0.1, λ_det on W1=-0.2) cell — empirical | theory | balanced-init-no-reg
— from the encoder ($W_1$) log-determinant regularizer experiment.

## Requirements
- Python 3.9+
- numpy
- scikit-learn
- seaborn
- matplotlib

Install with:
    pip install numpy scikit-learn seaborn matplotlib

## Run
    python test_reg_encoder.py

This runs the full sweep (λ_L2 ∈ {0, 0.1}, λ_det_W1 ∈ {0, -0.2}, 30,000
full-batch gradient-descent steps per cell) and writes two figures to
`figures/`:
- `figure-hier-qqt_encoder.png` — full 2x6 grid over all (λ_L2, λ_det) cells
- `figure-hier-qqt_encoder_bottomright.png` — just the (0.1, -0.2) cell

Takes roughly 1-2 minutes on a laptop CPU (pure numpy, no GPU needed).

## Files
- `test_reg_encoder.py` — main script (task setup, training loop, theory
  construction, plotting)
- `utils.py` — `get_lambda_balanced` (constructs λ-balanced weight
  factorizations), `BlindColours` (colormap/palette), and other helpers
- `networks/linear_network.py` — `LinearNetworkreg`, the two-layer linear
  network with L2 decay + log-det regularization on either layer
