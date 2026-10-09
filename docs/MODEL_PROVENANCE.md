# Upstream model provenance

This record identifies the model artifacts and initialization data used to
generate the 2021--2022 GraphCast and FuXi arrays evaluated in the paper.

## GraphCast

- Released checkpoint filename: `GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz`
- Training period represented by the release: ERA5, 1979--2017.
- Checkpoint SHA-256: `4e8cf600f3f402ebb430483e782b94648cef57bb757dec24dd7b11760d3fbacc`
- Initialization source: ARCO ERA5,
  `gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`.
- Initialization frames: `init-6h` and `init` for each 00 UTC start.
- Rollout: autoregressive 6-h steps; 24--168 h daily leads were retained, with
  the paper evaluating 72, 120 and 168 h.
- Historical run manifest:
  `/vol2/xrx/modela_v2/graphcast/2022/_run_config_demo2022_s11.json`.
- Generation scripts: `archive/server_scripts/run_graphcast_modela_*.py`.

## FuXi

- Released cascade checkpoint: official short- and medium-range ONNX models.
- `short.onnx` SHA-256: `8f23532f6f43dbc2d107ddf202c8a25afd7985c982d35465db096e73c34f0d3c`.
- `medium.onnx` SHA-256: `45fb061b23e604f74d5a77e780e7f3ca0c07313dca09a0b2b067b8e1e526fd8c`.
- Initialization source: WeatherBench 2 ERA5,
  `weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr`.
- Initialization frames: `init-6h` and `init` for each 00 UTC start.
- Channel handling: official 70-channel order and time encoding. Relative
  humidity was clipped as an ERA5 fraction to `[0, 1]` and converted to percent
  before inference.
- Cascade: short model for steps 1--20, medium model for steps 21--28.
- Historical run manifest:
  `/vol2/xrx/modela_v2/fuxi/2022/_run_config_v3_shard0.json`.
- Generation scripts: `archive/server_scripts/run_fuxi_modela_year*.py`.

The absolute paths are preserved only to make the original executions
traceable. They are site-specific and should be replaced when reproducing the
rollouts elsewhere.

