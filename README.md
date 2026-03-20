# cs217-final-proj

PyTorch vs NKI kernel experiments on AWS Trainium 2, with a profiler that runs experiments, captures NEFF/NTFF artifacts, and optionally ingests into InfluxDB for neuron-profile.

## Setup

Use a preinstalled Neuron environment (e.g. `/opt/aws_neuron_...`). No separate requirements.txt; ensure `torch`, `torch_xla`, `neuronxcc`, and `neuron-profile` (aws-neuronx-tools) are available.

## Experiment layout

Each experiment lives under `experiments/<name>/` with two entrypoints:

- **`run_torch.py`** — PyTorch-only implementation (standard ops on Neuron). Must set `NEURON_FRAMEWORK_DEBUG=1` and run so the compiler emits a NEFF in the experiment directory.
- **`run_nki.py`** — Same op via an NKI custom kernel (PyTorch + NKI). Same as above; for NKI the profiler also sets `XLA_IR_DEBUG=1` and `XLA_HLO_DEBUG=1`.

Both are runnable standalone: `python run_torch.py` / `python run_nki.py` from the experiment directory. (Names avoid shadowing the `torch` package.)

## Overall GNN Experiment Structure
The GNNs that we experiment with all have the following structure:
- Graphs consist of nodes and edges
- Nodes have "feature vectors"
Each GNN has the following as a fundamental layer:
- For each node:
  - Collect the feature vector for each incoming edge
  - Aggregate these feature vectors somehow (max, sum, avg, etc.)
How this is done depends on how we choose to represent the graph structure. In our experiments, we generate random graphs by:
- Generating src: (num_edges,), dst: (num_edges,), where edge 0 is between `src[0]` and `dst[0]`.
- Generating the node features. Generate x: (num_nodes, feature_dim)

## Experiments Index (Quick TOC)
Use these READMEs as the running table of contents for what each experiment tests:

- `experiments/graphsage_gather_scatter_mean/` — first faster GraphSAGE neighbor-aggregation (v1): on-device gather fused with NKI; compare report shows about `1.09x` speedup.
- `experiments/graphsage_gather_scatter_mean_v2/` — v2 “Option 2” accumulation strategy; exploration / tuning (does not claim the main end-to-end speedup).
- `experiments/graphsage_gather_scatter_mean_v3/` — v3 mask-loading variant; exploration / tuning (does not claim the main end-to-end speedup).
- `experiments/graphsage_gather_scatter_mean_v4/` — v4 src-block reordering; exploration / tuning (does not claim the main end-to-end speedup).
- `experiments/graphsage_gather_scatter_mean_v5/` — full 2-layer GraphSAGE end-to-end: only the neighbor aggregation is swapped to NKI while linear layers + ReLU stay in PyTorch.
- `experiments/graphsage_gather_scatter_mean_exaggerated/` — same neighbor-aggregation core, but with heavier default shapes; includes the speedup-matrix harness.
- `experiments/gcn_nki/` — 2-layer GCN with NKI aggregation + PyTorch MLP blocks.
- `experiments/gcn_fx_rewrite/` — 2-layer GCN where aggregation is auto-rewritten via `torch.fx` to an NKI kernel call.
- `experiments/gcn_fx_toggle/` — side-by-side profiling for FX rewrite vs a toggle (see README for modes).
- `experiments/gin_nki/` — 2-layer GIN with NKI aggregation.

## Torch vs NKI Results (Average `total_time`)
Device-side kernel timings from each experiment’s `artifacts/compare_report.json` (time source: `total_time`):

| Experiment | NKI avg time (s) | Torch avg time (s) | Speedup (Torch / NKI) |
|---|---:|---:|---:|
| `scatter_add` | 0.000139 | 0.000159 | 1.141x |
| `graphsage_gather_scatter_mean` | 0.001737 | 0.001900 | 1.094x |
| `gcn_nki` | 0.003510 | 0.005187 | 1.478x |
| `gin_nki` | 0.003525 | 0.005223 | 1.482x |

## Profiler

**InfluxDB defaults:** Create or edit `profiler_influx.json` in the repo root with your InfluxDB settings so you don't need to pass `--db-endpoint` / `--db-org` / `--db-bucket` every time:

```json
{
  "db_endpoint": "http://localhost:8086",
  "db_org": "joshdelg-cs217",
  "db_bucket": "cs217"
}
```

Authentication (token) is handled by the `influx` CLI or `INFLUX_TOKEN`; see InfluxDB docs.

From the repo root:

```bash
# Profile both implementations and compare (ingest to InfluxDB if profiler_influx.json is set)
python -m profiler profile <experiment_name> --mode compare

# Multiple trials and average timings
python -m profiler profile <experiment_name> --mode compare --trials 5

# Profile only torch or only nki
python -m profiler profile <experiment_name> --mode torch
python -m profiler profile <experiment_name> --mode nki

# Ingest captured profiles into InfluxDB (you run InfluxDB separately)
python -m profiler profile <experiment_name> --mode compare --ingest \
  --db-endpoint http://localhost:8086 --db-org myorg --db-bucket profiles
```

- **Artifacts** are written to `experiments/<name>/artifacts/torch/` and `experiments/<name>/artifacts/nki/` (each contains `model.neff` and `profile.ntff`). In compare mode, `artifacts/compare_report.json` has averaged timings.
- **NEFF discovery**: After running a script, the profiler finds the newest `*.neff` under the experiment dir. Optional `profiler_config.json` in the experiment dir can override with `"torch"` / `"nki"` keys or `neff_path` / `neff_dir`.

## Example

`experiments/example/` contains minimal `run_torch.py` and `run_nki.py` stubs. On a Trainium instance with the Neuron env activated:

```bash
python -m profiler profile example --mode compare
```

**Viewing profiles:** Start the viewer with the **same** InfluxDB org (and endpoint) so it can see the ingested data:

```bash
neuron-profile view --db-endpoint http://localhost:8086 --db-org joshdelg-cs217
```

Then open http://localhost:3001. If you don’t pass `--db-org`, the UI may show an empty table because it’s not querying your org.

Alternatively, open the NEFF/NTFF files directly:  
`neuron-profile view -n experiments/example/artifacts/torch/model.neff -s experiments/example/artifacts/torch/profile.ntff`

### Profile warnings you may see

- **Missing DMA metadata / "DMA engine X queue Y is invalid"** — The profiler already runs `neuron-profile capture` with `--enable-dge-notifs` and `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1`. If you still see this, you may be viewing an **old** NTFF (captured before that change). Re-run compare to generate new profiles: `python -m profiler profile example --mode compare`. New captures should have full DMA metadata.

- **Missing compiler metrics in NEFF** — The profiler expects optional metadata (e.g. compiler version, high-level metrics) that newer compilers write into the NEFF. Your NEFF was built with whatever compiler is in your current env (e.g. the DLAMI’s venv). **This is only a warning:** the profile and timeline are still valid; you just don’t get some extra summary metrics in the UI.  
  **On an AWS instance:** The Neuron stack (neuronx-cc, aws-neuronx-tools, torch-neuronx) is often pinned to a specific SDK release. The compiler used by PyTorch XLA/NKI may be slightly older than what the installed `neuron-profile` expects for “compiler metrics,” so the warning is common and not a sign of a broken setup. To reduce or remove it you can: (1) Use a newer DLAMI or Neuron DLC that ships a matching/newer compiler and tools, or (2) Update the Neuron packages in your venv if your image allows it (`pip install -U neuronx-cc torch-neuronx`, then re-run your experiment and capture). If you’re on a fixed image, you can safely ignore the warning.

- **Profiler UI deprecation** — AWS is moving to **Neuron Explorer** for profiling. The current neuron-profile UI still works; for the new UI and features, see [Neuron Explorer](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/index.html).
