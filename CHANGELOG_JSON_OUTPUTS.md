# TrustForest code changes

## 1. JSON forest specifications

`TrustForestEngine` can now be built from a JSON specification:

```python
engine = TrustForestEngine.from_json_file("forest_vcdb_example.json")
```

A forest specification contains:

- `nodes`: fusion nodes with `id`, `inputs`, `input_types`, `output_type`, and registered `fusion` function name.
- `outputs`: optional default list of nodes to return when calling `evaluate_outputs()`.

An example is provided in `forest_vcdb_example.json`.

## 2. Selectable output nodes

The engine now supports multi-output evaluation:

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

The previous single-output API is preserved:

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

## Compatibility checks

The existing UEBA and CTI scripts were run successfully with the modified library. The VCDB script imports and starts correctly; it stops only because no `data/VERIS_sample/*.json` files are included in this archive.
