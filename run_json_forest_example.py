from lib import TrustForestEngine, ProbabilisticScore

engine = TrustForestEngine.from_json_file("forest_vcdb_example.json")

leaves = {
    "raw_actor_field": ProbabilisticScore(1.0),
    "raw_attribute_field": ProbabilisticScore(1.0),
}
leaves["raw_actor_field"].value = "Organized crime"
leaves["raw_attribute_field"].value = "Medical"

# New API: evaluate configured outputs in one shared pass.
outputs = engine.evaluate_outputs(leaves)
for node_id, value in outputs.items():
    print(f"{node_id}: {value}")

# Backward-compatible API still works.
print("single:", engine.evaluate("global_incident_trust", leaves))
