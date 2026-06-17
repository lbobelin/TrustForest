# TrustForest
 Typed Computation Graphs for Uncertainty-Aware Trust Fusion in Cybersecurity

Library based on computation graph forest - that you can express as json files - acting as a semantic intermediary layer between raw metascore and high level decision indicators, and ensure that your high level indicator are semantically correct. 

# TrustForest basic usage

## 1. JSON forest specifications

`TrustForestEngine` can be built from a JSON specification:

```python
engine = TrustForestEngine.from_json_file("forest_vcdb_example.json")
```

A forest specification contains:

- `nodes`: fusion nodes with `id`, `inputs`, `input_types`, `output_type`, and registered `fusion` function name.
- `outputs`: optional default list of nodes to return when calling `evaluate_outputs()`.

An example is provided in `forest_vcdb_example.json`.

## 2. Selectable output nodes

The engine supports multi-output evaluation:

```python
outputs = engine.evaluate_many(
    ["actor_reliability", "attr_criticality", "global_incident_trust"],
    leaves,
)
```

or, using the JSON-configured default outputs:

```python
outputs = engine.evaluate_outputs(leaves)
```

You can also use:

```python
result = engine.evaluate("global_incident_trust", leaves)
```

## 3. Fusion function registry

Fusion functions can be exposed to JSON specs with:

```python
@register_fusion_function("my_fusion")
def my_fusion(inputs):
    ...
```

or registered dynamically:

```python
register_fusion("my_fusion", my_fusion)
```

Existing fusion functions in `lib.py` are registered automatically.

## Experiments

The existing UEBA and CTI scripts were run successfully with the library. The VCDB script imports and starts correctly; it stops only because no `data/VERIS_sample/*.json` files are included in this archive.
