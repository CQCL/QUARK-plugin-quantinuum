# QUARK-plugin-quantinuum

This is a QUARK plugin that implements access to running QUARK benchmarks on Quantinuum devices.

Access is provided through [Quantinuum Nexus](https://nexus.quantinuum.com), and use of the Quantinuum backends
requires setting up a Nexus account. Please see the Nexus documentation for more information on the available devices.

Once you have a Nexus account, authentication is provided through the qnexus python package. With this plugin installed, simply run `qnx login` in the terminal before running your QUARK benchmark configuration.

Example benchmark configurations can be found in [examples](examples/).

Please note that running a benchmark on Nexus requires a `nexus_upload` step followed by `nexus_run`. The reason for this split is to allow running an uploaded benchmark multiple times with different configurations without needing multiple uploads. See below for more details.

Run a benchmark config using:

```terminal
quark -c <config_file>
```

## Registered QUARK Modules

The modules registered by this plugin and their configuration APIs are documented below. Pipeline step names are the names to use in QUARK configuration files.


### Nexus Upload

Pipeline step name: `nexus_upload`

Defined in [nexus_upload.py](src/quark_plugin_quantinuum/miscellaneous/nexus_upload.py).

Uploads pytket circuits to Quantinuum Nexus and passes Nexus circuit references to a later Nexus execution module.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `project_name` | `str` | `"Quark Benchmarking"` | Nexus project name used for uploaded circuits. The project is created if it does not already exist. |

Example:

```yaml
- "free_fermion": {output_circuit_type: "pytket", n_shots: 100}
- "nexus_upload": {project_name: "Quark Benchmarking"}
```

### Nexus Run Backend

Pipeline step name: `nexus_run`

Defined in [quantinuuum_nexus.py](src/quark_plugin_quantinuum/backends/quantinuuum_nexus.py).

Compiles uploaded Nexus circuits for a Quantinuum backend, executes the compiled circuits, and returns counts for benchmark postprocessing.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `device` | `str` | Required | Quantinuum backend device name to use through Nexus, for example `"H2-1LE"`. |
| `compile_optimization_level` | `int` | `1` | Nexus compilation optimization level. |
| `n_shots` | `int` | `100` | Stored module setting for shot count. Shot counts are currently provided by the uploaded benchmark input. |
| `project_name_override` | `Optional[str]` | `None` | Optional Nexus project name for compile and execute jobs. Uses the upload project when omitted. |

Example:

```yaml
- "free_fermion": {output_circuit_type: "pytket", n_shots: 100}
- "nexus_upload": {project_name: "Quark Benchmarking"}
- "nexus_run": {device: "H2-1LE", compile_optimization_level: 1}
```

### Aer Simulator Backend

Pipeline step name: `aer_simulator`

Defined in [aer_simulator.py](src/quark_plugin_quantinuum/backends/aer_simulator.py).

For testing purposes, runs simulation benchmarks locally with `qiskit_aer.AerSimulator` and returns counts for downstream benchmark postprocessing.

| Parameter    | Type    | Default | Description                                                                                               |
|--------------|---------|---------|-----------------------------------------------------------------------------------------------------------|
| `noise_rate` | `float` | `0.0`   | Depolarizing error rate applied to two-qubit gates in the local Aer noise model. The default is no noise. |

Example:

```yaml
- "free_fermion": {output_circuit_type: "qiskit", n_shots: 100}
- "aer_simulator": {noise_rate: 0.001}
```


### Free Fermion Benchmark

Pipeline step name: `free_fermion`

Defined in [free_fermion.py](src/quark_plugin_quantinuum/benchmarks/free_fermion/free_fermion.py).

This module is only used for testing purposes and is not up-to-date. For actual benchmarking purposes, please use the Free Fermion benchmark from [QUARK-plugin-hamiltonian-simulation](https://github.com/Quantinuum/QUARK-plugin-hamiltonian-simulation).
