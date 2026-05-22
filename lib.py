import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable, Iterable

# =====================================================================
# 1. THE RIGID TRUST TYPE SYSTEM
# =====================================================================

class TrustDimension(ABC):
    """Abstract base class for all strict trust metrics."""
    @abstractmethod
    def validate(self) -> bool:
        pass

class MissingDimension(TrustDimension):
    """Represents a missing or unavailable dimension in the graph."""
    def validate(self) -> bool:
        return True
    
    def __repr__(self):
        return "MissingDimension()"

class DstTriplet(TrustDimension):
    """Dempster-Shafer Theory Triplet: Belief, Disbelief, Uncertainty."""
    def __init__(self, belief: float, disbelief: float, uncertainty: float):
        self.b = float(belief)
        self.d = float(disbelief)
        self.u = float(uncertainty)

    def validate(self) -> bool:
        total = self.b + self.d + self.u
        return abs(total - 1.0) < 0.001 and (0.0 <= self.b <= 1.0) and (0.0 <= self.d <= 1.0) and (0.0 <= self.u <= 1.0)

    def __repr__(self):
        return f"DstTriplet(b={self.b:.3f}, d={self.d:.3f}, u={self.u:.3f})"

class ProbabilisticScore(TrustDimension):
    """Standard scalar probability score bound between 0 and 1."""
    def __init__(self, value: float):
        self.value = float(value)

    def validate(self) -> bool:
        return 0.0 <= self.value <= 1.0

    def __repr__(self):
        return f"ProbabilisticScore(val={self.value:.3f})"


# =====================================================================
# 2. THE MOLECULAR COMPUTATION GRAPH INTERFACE
# =====================================================================

class FusionNode:
    """A logical node in the forest that aggregates dimensions dynamically."""
    def __init__(self, node_id: str, expected_input_types: List[str], output_type: str, fuse_func):
        self.id = node_id
        self.expected_input_types = expected_input_types
        self.output_type = output_type
        self.fuse_func = fuse_func

    def execute(self, inputs: List[TrustDimension]) -> TrustDimension:
        # Strict validation of runtime signatures
        for idx, item in enumerate(inputs):
            if isinstance(item, MissingDimension):
                continue
            if item.typeId() != self.expected_input_types[idx]:
                raise TypeError(f"Node '{self.id}' expected {self.expected_input_types[idx]} at position {idx}, got {item.typeId()}")
        
        res = self.fuse_func(inputs)
        if not res.validate():
            raise ValueError(f"Node '{self.id}' produced an invalid trust dimension state.")
        return res

# Dynamically patch type identifiers into the classes for easy clean checking
DstTriplet.typeId = lambda self: "DST_TRIPLET"
ProbabilisticScore.typeId = lambda self: "PROBA_SCORE"
MissingDimension.typeId = lambda self: "MISSING"

class QualitativeLabel(TrustDimension):
    """Encapsulates a strict qualitative string tag from a controlled vocabulary."""
    def __init__(self, label: str, allowed_vocabulary: List[str]):
        self.label = str(label)
        self.vocab = allowed_vocabulary

    def validate(self) -> bool:
        # Sécurité : l'étiquette doit impérativement faire partie du vocabulaire autorisé
        return self.label in self.vocab

    def __repr__(self):
        return f"QualitativeLabel('{self.label}')"

# Enregistrement du type dans le dictionnaire global
QualitativeLabel.typeId = lambda self: "QUALITATIVE_LABEL"


# =====================================================================
# 2B. FUSION FUNCTION REGISTRY
# =====================================================================

FUSION_FUNCTION_REGISTRY: Dict[str, Callable[[List[TrustDimension]], TrustDimension]] = {}


def register_fusion_function(name: Optional[str] = None):
    """Decorator used to expose a fusion function to JSON forest specifications.

    Example
    -------
    @register_fusion_function("dst_combine")
    @register_fusion_function("root_global_fusion")
def root_global_fusion(inputs):
        ...
    """
    def decorator(func: Callable[[List[TrustDimension]], TrustDimension]):
        FUSION_FUNCTION_REGISTRY[name or func.__name__] = func
        return func
    return decorator


def register_fusion(name: str, func: Callable[[List[TrustDimension]], TrustDimension]) -> None:
    """Registers a fusion function programmatically for JSON loading."""
    FUSION_FUNCTION_REGISTRY[name] = func


def get_registered_fusion(name: str) -> Callable[[List[TrustDimension]], TrustDimension]:
    """Returns a registered fusion function by name."""
    try:
        return FUSION_FUNCTION_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(FUSION_FUNCTION_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown fusion function '{name}'. Available functions: {available}") from exc


class TrustForestEngine:
    """Central processing core managing typed topology validation and evaluation.

    The public ``evaluate`` method is kept for backward compatibility and returns
    a single target node. ``evaluate_many`` is the new analyst-facing API: it
    evaluates several output nodes in one pass and reuses one memoization cache,
    so shared sub-graphs are computed only once.
    """
    def __init__(self, output_nodes: Optional[List[str]] = None):
        self.nodes: Dict[str, FusionNode] = {}
        self.edges: Dict[str, List[str]] = {}  # parent_id -> list of child_ids
        self.output_nodes: List[str] = list(output_nodes or [])

    def register_node(self, node: FusionNode, children_ids: List[str]):
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node id '{node.id}'.")
        self.nodes[node.id] = node
        self.edges[node.id] = list(children_ids)

    def set_outputs(self, output_nodes: Iterable[str]) -> None:
        """Sets the default output nodes used by evaluate_outputs()."""
        self.output_nodes = list(output_nodes)

    def validate_graph_integrity(self):
        """Performs structural and semantic static type checking across the forest."""
        for parent_id, children in self.edges.items():
            if parent_id not in self.nodes:
                raise ValueError(f"Unknown parent node '{parent_id}' in edge table.")
            parent_node = self.nodes[parent_id]
            if len(children) != len(parent_node.expected_input_types):
                raise ValueError(
                    f"Structure mismatch: Node '{parent_id}' expects "
                    f"{len(parent_node.expected_input_types)} children but got {len(children)}"
                )

            for idx, child_id in enumerate(children):
                if child_id in self.nodes:  # Intermediate nodes
                    child_out = self.nodes[child_id].output_type
                    expected = parent_node.expected_input_types[idx]
                    if child_out != expected:
                        raise TypeError(
                            f"Static Type Error: Arc {child_id} -> {parent_id} "
                            f"type mismatch ({child_out} vs {expected})"
                        )

        for out in self.output_nodes:
            if out not in self.nodes:
                # Leaves are valid outputs, but flag accidental typos early by allowing
                # them only at evaluation time if leaf_data contains the corresponding key.
                continue

    def evaluate(self, target_node_id: str, leaf_data: Dict[str, TrustDimension], cache=None) -> TrustDimension:
        """Demand-driven pull evaluation with explicit node memoization.

        Parameters
        ----------
        target_node_id:
            ID of the requested node or leaf.
        leaf_data:
            Mapping from leaf identifiers to concrete TrustDimension objects.
        cache:
            Optional dictionary shared across multiple evaluations.
        """
        if cache is None:
            cache = {}

        if target_node_id in cache:
            return cache[target_node_id]

        # Base case: if it is a raw data leaf, pull it out.
        if target_node_id not in self.nodes:
            result = leaf_data.get(target_node_id, MissingDimension())
            cache[target_node_id] = result
            return result

        children = self.edges[target_node_id]
        child_inputs = [self.evaluate(child_id, leaf_data, cache) for child_id in children]

        node = self.nodes[target_node_id]
        result = node.execute(child_inputs)
        cache[target_node_id] = result
        return result

    def evaluate_many(
        self,
        target_node_ids: Optional[Iterable[str]],
        leaf_data: Dict[str, TrustDimension],
    ) -> Dict[str, TrustDimension]:
        """Evaluates several output nodes using a shared cache.

        This supports analyst-facing workflows where intermediate dimensions are
        needed in addition to, or instead of, a single high-level fused score.
        """
        targets = list(target_node_ids if target_node_ids is not None else self.output_nodes)
        if not targets:
            raise ValueError("No output nodes specified. Pass targets or configure engine.output_nodes.")

        cache: Dict[str, TrustDimension] = {}
        return {target: self.evaluate(target, leaf_data, cache) for target in targets}

    def evaluate_outputs(self, leaf_data: Dict[str, TrustDimension]) -> Dict[str, TrustDimension]:
        """Evaluates the default output nodes configured on the engine."""
        return self.evaluate_many(None, leaf_data)

    def to_json_dict(self) -> Dict[str, Any]:
        """Serializes the graph topology to a JSON-compatible dictionary.

        Fusion functions are represented by their registered names when possible.
        """
        reverse_registry = {func: name for name, func in FUSION_FUNCTION_REGISTRY.items()}
        nodes = []
        for node_id, node in self.nodes.items():
            nodes.append({
                "id": node_id,
                "inputs": self.edges.get(node_id, []),
                "input_types": node.expected_input_types,
                "output_type": node.output_type,
                "fusion": reverse_registry.get(node.fuse_func, getattr(node.fuse_func, "__name__", "<unregistered>")),
            })
        return {"nodes": nodes, "outputs": self.output_nodes}

    def to_json_file(self, path: str) -> None:
        """Writes the graph specification to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2)

    @classmethod
    def from_json_dict(
        cls,
        spec: Dict[str, Any],
        fusion_registry: Optional[Dict[str, Callable[[List[TrustDimension]], TrustDimension]]] = None,
        validate: bool = True,
    ) -> "TrustForestEngine":
        """Builds an engine from a declarative JSON-compatible specification.

        Expected schema
        ---------------
        {
          "nodes": [
            {
              "id": "global_trust",
              "inputs": ["actor_reliability", "attr_criticality"],
              "input_types": ["DST_TRIPLET", "DST_TRIPLET"],
              "output_type": "DST_TRIPLET",
              "fusion": "root_global_fusion"
            }
          ],
          "outputs": ["actor_reliability", "global_trust"]
        }
        """
        registry = dict(FUSION_FUNCTION_REGISTRY)
        if fusion_registry:
            registry.update(fusion_registry)

        engine = cls(output_nodes=spec.get("outputs", []))
        for node_spec in spec.get("nodes", []):
            missing = {"id", "inputs", "input_types", "output_type", "fusion"} - set(node_spec)
            if missing:
                raise ValueError(f"Invalid node specification; missing fields: {sorted(missing)}")
            fusion_name = node_spec["fusion"]
            if fusion_name not in registry:
                available = ", ".join(sorted(registry)) or "<none>"
                raise KeyError(f"Unknown fusion function '{fusion_name}'. Available functions: {available}")

            node = FusionNode(
                node_id=node_spec["id"],
                expected_input_types=list(node_spec["input_types"]),
                output_type=node_spec["output_type"],
                fuse_func=registry[fusion_name],
            )
            engine.register_node(node, list(node_spec["inputs"]))

        if validate:
            engine.validate_graph_integrity()
        return engine

    @classmethod
    def from_json_file(
        cls,
        path: str,
        fusion_registry: Optional[Dict[str, Callable[[List[TrustDimension]], TrustDimension]]] = None,
        validate: bool = True,
    ) -> "TrustForestEngine":
        """Builds an engine from a JSON forest specification file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json_dict(json.load(f), fusion_registry=fusion_registry, validate=validate)


# =====================================================================
# 3. CONCRETE MATHEMATICAL FUSION UTILITIES
# =====================================================================

@register_fusion_function("dempster_shafer_combine")
def dempster_shafer_combine(t1: DstTriplet, t2: DstTriplet) -> DstTriplet:
    """Core implementation of the classical Dempster's rule of combination."""
    conflict = t1.b * t2.d + t1.d * t2.b
    if abs(conflict - 1.0) < 0.0001:
        # Total conflict fallback: transition fully into epistemic uncertainty
        return DstTriplet(0.0, 0.0, 1.0)
    
    normalization = 1.0 - conflict
    b_joint = (t1.b * t2.b + t1.b * t2.u + t1.u * t2.b) / normalization
    d_joint = (t1.d * t2.d + t1.d * t2.u + t1.u * t2.d) / normalization
    u_joint = (t1.u * t2.u) / normalization
    return DstTriplet(b_joint, d_joint, u_joint)

@register_fusion_function("fuse_ueba_anomaly")
def fuse_ueba_anomaly(inputs: List[TrustDimension]) -> DstTriplet:
    val = inputs[0]
    if isinstance(val, MissingDimension):
        return DstTriplet(0.0, 0.0, 1.0) # Absence de log UEBA = Incertitude totale
    
    # On prend le score scalaire [0, 1] généré par l'algorithme UEBA (ex: Autoencoder)
    anomaly_score = val.value
    
    if anomaly_score > 0.8:
        # Un score très élevé génère de la suspicion, mais on préserve 
        # une forte incertitude (u) car on ne connaît pas l'intention de l'utilisateur
        return DstTriplet(0.5, 0.0, 0.5) 
    else:
        return DstTriplet(0.0, 0.7, 0.3) # Comportement normal
# =====================================================================
# 4. EXPERIMENTATION IMPLEMENTATION (VERIS COMPLIANT SYNTHESIS)
# =====================================================================

# Define node operations targeting structured JSON configurations
@register_fusion_function("fuse_actor_reliability")
def fuse_actor_reliability(inputs: List[TrustDimension]) -> DstTriplet:
    val = inputs[0]
    if isinstance(val, MissingDimension):
        return DstTriplet(0.0, 0.0, 1.0) # Data missing means full uncertainty
    
    # Simple semantic mapping from string attributes to DST values
    actor_type = val.value if hasattr(val, 'value') else str(val)
    if "Organized crime" in actor_type or "State-sponsored" in actor_type:
        return DstTriplet(0.8, 0.0, 0.2) # High malicious intent proof
    elif "Internal" in actor_type:
        return DstTriplet(0.3, 0.4, 0.3)
    return DstTriplet(0.0, 0.0, 1.0)

@register_fusion_function("fuse_attribute_criticality")
def fuse_attribute_criticality(inputs: List[TrustDimension]) -> DstTriplet:
    val = inputs[0]
    if isinstance(val, MissingDimension):
        return DstTriplet(0.0, 0.0, 1.0)
    
    data_type = val.value if hasattr(val, 'value') else str(val)
    if "Medical" in data_type or "Credentials" in data_type:
        return DstTriplet(0.9, 0.0, 0.1) # Severe technical compromise
    return DstTriplet(0.2, 0.5, 0.3)

@register_fusion_function("root_global_fusion")
def root_global_fusion(inputs: List[TrustDimension]) -> DstTriplet:
    t1 = inputs[0] if not isinstance(inputs[0], MissingDimension) else DstTriplet(0.0, 0.0, 1.0)
    t2 = inputs[1] if not isinstance(inputs[1], MissingDimension) else DstTriplet(0.0, 0.0, 1.0)
    return dempster_shafer_combine(t1, t2)

@register_fusion_function("fuse_asset_importance")
def fuse_asset_importance(inputs: List[TrustDimension]) -> DstTriplet:
    val = inputs[0]
    
    # 1. Gestion de l'omission : si l'actif n'est pas catégorisé, incertitude totalepip install matplotlib
    if isinstance(val, MissingDimension):
        return DstTriplet(0.0, 0.0, 1.0)
    
    label = val.label
    
    # 2. Mapping sémantique du qualitatif vers le quantitatif (DST)
    if label == "CRITICAL":
        # Croyance absolue que l'impact sera majeur, aucune incertitude
        return DstTriplet(1.0, 0.0, 0.0)
    elif label == "HIGH":
        return DstTriplet(0.8, 0.0, 0.2)
    elif label == "MEDIUM":
        # On est au milieu : on augmente l'incertitude (u=0.5) car l'impact est flou
        return DstTriplet(0.3, 0.2, 0.5)
    elif label == "LOW":
        # Forte croyance que l'actif n'est pas important (Disbelief)
        return DstTriplet(0.0, 0.9, 0.1)
        
    # Sécurité si le validateur a été contourné
    return DstTriplet(0.0, 0.0, 1.0)

if __name__ == "__main__":
    print("Initializing TrustForest Engine and building structural topology...")
    engine = TrustForestEngine()

    # Instantiate nodes conforming strictly to semantic parameters
    actor_node = FusionNode("actor_reliability", ["PROBA_SCORE"], "DST_TRIPLET", fuse_actor_reliability)
    attr_node = FusionNode("attr_criticality", ["PROBA_SCORE"], "DST_TRIPLET", fuse_attribute_criticality)
    root_node = FusionNode("global_incident_trust", ["DST_TRIPLET", "DST_TRIPLET"], "DST_TRIPLET", root_global_fusion)

    # Register topology layout
    engine.register_node(actor_node, ["raw_actor_field"])
    engine.register_node(attr_node, ["raw_attribute_field"])
    engine.register_node(root_node, ["actor_reliability", "attr_criticality"])

    # Perform structural compilation check
    engine.validate_graph_integrity()
    print("Static validation passed. Topology is sound.")

    # -----------------------------------------------------------------
    # SIMULATING VERIS DB RECORDS (VCDB JSON EXTRACTIONS)
    # -----------------------------------------------------------------
    print("\n--- STARTING EXPERIMENTAL PIPELINE ---")
    
    # 1. Nominal Record (Fully filled JSON)
    veris_record_nominal = {
        "actor": {"external": {"variety": ["Organized crime"]}},
        "attribute": {"confidentiality": {"data": {"variety": ["Medical"]}}}
    }

    # 2. Asymmetric Record (Missing actor context entirely)
    veris_record_asymmetric = {
        "attribute": {"confidentiality": {"data": {"variety": ["Medical"]}}}
    }

    # Helper to map simulated logs to library leaf structures
    def build_leaves(record: dict) -> dict:
        leaves = {}
        if "actor" in record:
            leaves["raw_actor_field"] = ProbabilisticScore(1.0)
            leaves["raw_actor_field"].value = record["actor"]["external"]["variety"][0] 
        if "attribute" in record:
            leaves["raw_attribute_field"] = ProbabilisticScore(1.0)
            leaves["raw_attribute_field"].value = record["attribute"]["confidentiality"]["data"]["variety"][0]
        return leaves

    # RUN Nominal Evaluation
    print("\n[EXP 1] Processing Complete System Record:")
    leaves_nominal = build_leaves(veris_record_nominal)
    res_nominal = engine.evaluate("global_incident_trust", leaves_nominal)
    print(f"Final Global Threat Evaluation Result: {res_nominal}")

    # RUN Asymmetric Evaluation
    print("\n[EXP 2] Processing Damaged/Asymmetric System Record (Missing Actor Field):")
    leaves_asymmetric = build_leaves(veris_record_asymmetric)
    res_asymmetric = engine.evaluate("global_incident_trust", leaves_asymmetric)
    print(f"Final Global Threat Evaluation Result: {res_asymmetric}")
    print("\nValidation successful: Notice how uncertainty (u) expanded organically without breaking computing workflow framework structures.")