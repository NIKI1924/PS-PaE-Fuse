# Global-skill checkpoint training record

## Identity

- Role: frozen global-skill expert shared by all three phase-router runs.
- Public asset: `global_skill_seed123_best_epoch16.pt`.
- SHA-256: `9f1f1ce692d6437bc6c382b1af239bdac80b067da8b241e14c1ed81cfcf07016`.
- Size: 117,475,500 bytes.
- Historical path:
  `/home/xrx/wenqiong/ov2_dem_c32_s123_u10fix/best_v3_spf.pt`.

## Launch and data

The model was retrained from scratch after rebuilding the corrected temporal and
U10 cache:

```text
train_v3_spfdem.py \
  --cache-dir /vol2/xrx/temporal_cache \
  --output-dir /home/xrx/wenqiong/ov2_dem_c32_s123_u10fix \
  --batch-size 64 --gpu 1 --epochs 35 \
  --var-scale-z500 900 --base-ch 32 --seed 123
```

- Training set: 7,840 patches of 128 x 128, cropped by 16 pixels at evaluation.
- Validation set: 1,988 patches.
- Extreme oversampling: 6,760 extreme patches repeated three times, yielding
  21,360 samples per epoch.
- Parameters: 277,670.
- Regression loss: latitude-weighted Gaussian CRPS.
- Event losses: focal loss (`alpha=0.75`, `gamma=2`) and Tversky loss
  (`alpha=0.3`, `beta=0.7`, weight 0.5).
- Event-weight schedule: 0.1 for epochs 1--3, 0.5 for epochs 4--25 and 0.8 from
  epoch 26.
- Best validation objective: 0.2984 at epoch 16. That saved state is the frozen
  checkpoint used in downstream evaluation.

The process continued through epoch 27, after which the worker was terminated
with return code 137. No later completed epoch improved on epoch 16, and the
already-saved epoch-16 state was subsequently loaded successfully with no
missing or unexpected parameters. The full per-epoch record is
`logs/global_skill_training_history.csv`.

