import os
import json
import glob
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from lib import (
    TrustForestEngine, FusionNode, ProbabilisticScore, 
    QualitativeLabel, DstTriplet, root_global_fusion,
    fuse_actor_reliability, fuse_attribute_criticality
)

def init_experiment_engine() -> TrustForestEngine:
    """Initializes the graph topology for VERIS experiments."""
    engine = TrustForestEngine()
    
    # Declaration of nodes according to the proposed architecture
    actor_node = FusionNode("actor_reliability", ["PROBA_SCORE"], "DST_TRIPLET", fuse_actor_reliability)
    attr_node = FusionNode("attr_criticality", ["PROBA_SCORE"], "DST_TRIPLET", fuse_attribute_criticality)
    root_node = FusionNode("global_incident_trust", ["DST_TRIPLET", "DST_TRIPLET"], "DST_TRIPLET", root_global_fusion)

    # Registration of edges (Links)
    engine.register_node(actor_node, ["raw_actor_field"])
    engine.register_node(attr_node, ["raw_attribute_field"])
    engine.register_node(root_node, ["actor_reliability", "attr_criticality"])
    
    engine.validate_graph_integrity()
    return engine

def get_first_scavenged_string(data_structure) -> Optional[str]:
    """Navigates recursively through the VERIS structures to extract the first valid string."""
    if isinstance(data_structure, str):
        return data_structure
    if isinstance(data_structure, list) and len(data_structure) > 0:
        return get_first_scavenged_string(data_structure[0])
    if isinstance(data_structure, dict):
        # If it's a dictionary, we look for a text value or a ‘variety’ key
        if "variety" in data_structure:
            return get_first_scavenged_string(data_structure["variety"])
        for val in data_structure.values():
            res = get_first_scavenged_string(val)
            if res:
                return res
    return None

def map_json_to_leaves(record: Dict) -> Dict:
    """Converts the VERIS JSON into type-safe sheets for the engine in an ultra-robust manner."""
    leaves = {}
    
    # Secure Extraction of the Actor
    try:
        actor_data = record.get("actor", {}).get("external", {}).get("variety", None)
        if not actor_data: # Fallback if it's not an external actor
            actor_data = record.get("actor", {}).get("internal", {}).get("variety", None)
            
        actor_str = get_first_scavenged_string(actor_data)
        if actor_str:
            leaves["raw_actor_field"] = ProbabilisticScore(1.0)
            leaves["raw_actor_field"].value = actor_str
    except Exception:
        pass

    # Secure Extraction of the Attribute (CIA Triad / Confidentiality)
    try:
        # Resolves the issue where 'data' or 'confidentiality' is a list or nested dictionary
        conf_data = record.get("attribute", {}).get("confidentiality", {}).get("data", None)
        attr_str = get_first_scavenged_string(conf_data)
        
        if attr_str:
            leaves["raw_attribute_field"] = ProbabilisticScore(1.0)
            leaves["raw_attribute_field"].value = attr_str
    except Exception:
        pass
        
    return leaves

def run_pipeline() -> Tuple[List[float], List[float], List[float], List[float]]:
    engine = init_experiment_engine()
    
    # Paths to your VCDB sample
    search_path = os.path.join("data", "VERIS_sample", "*.json")
    json_files = glob.glob(search_path)
    
    if not json_files:
        print(f"[-] No JSON files found in {search_path}. Please check the directory structure.")
        return [], [], [], []

    print(f"[+] Launch of analyses on {len(json_files)} VERIS files...")

    # Tables for storing metrics for charts
    nominal_uncertainties = []
    nominal_beliefs = []
    degraded_uncertainties = []
    degraded_beliefs = []

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            try:
                record = json.load(f)
            except json.JSONDecodeError:
                continue

            # --- 1.  NOMINAL RUN (Raw data from the database) ---
            leaves_nominal = map_json_to_leaves(record)
            # We only keep records that contain at least one piece of useful information.
            if not leaves_nominal:
                continue
                
            res_nominal = engine.evaluate("global_incident_trust", leaves_nominal)
            if isinstance(res_nominal, DstTriplet):
                nominal_uncertainties.append(res_nominal.u)
                nominal_beliefs.append(res_nominal.b)

            # --- 2.  DEGRADED RUN(The actor is intentionally removed to test asymmetry) ---
            record_degraded = record.copy()
            if "actor" in record_degraded:
                del record_degraded["actor"] # Retention Anomaly Injection
                
            leaves_degraded = map_json_to_leaves(record_degraded)
            res_degraded = engine.evaluate("global_incident_trust", leaves_degraded)
            if isinstance(res_degraded, DstTriplet):
                degraded_uncertainties.append(res_degraded.u)
                degraded_beliefs.append(res_degraded.b)

    return nominal_uncertainties, nominal_beliefs, degraded_uncertainties, degraded_beliefs

def generate_plots(nom_u: List[float], nom_b: List[float], deg_u: List[float], deg_b: List[float]):
    """Generates curves and distribution histograms for academic purposes."""
    print("[+] Generating performance plots for the paper...")
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- GRAPH 1 : Evolution and Distribution of Uncertainty (u) ---
    ax1.hist(nom_u, bins=20, alpha=0.6, label='Nominal Run (Full Data)', color='#2ca02c')
    ax1.hist(deg_u, bins=20, alpha=0.6, label='Degraded Run (Missing Actor)', color='#d62728')
    ax1.set_title("Axiomatic Uncertainty Propagation ($u$)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epistemic Uncertainty Metric ($u$)")
    ax1.set_ylabel("Number of VERIS Incidents")
    ax1.legend(loc='upper right')

    # --- GRAPH 2 : Stability of the Belief (b) against the Compensatory Effect ---
    # Sorting the data for a proper visualization in a cumulative/trend curve
    nom_b_sorted = sorted(nom_b)
    deg_b_sorted = sorted(deg_b)
    
    ax2.plot(nom_b_sorted, label='Nominal Trust Belief ($b$)', color='#1f77b4', linewidth=2)
    ax2.plot(deg_b_sorted, label='Degraded Trust Belief ($b$)', color='#ff7f0e', linestyle='--', linewidth=2)
    ax2.set_title("Non-Compensatory Belief Resilience ($b$)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Sorted Incident Evaluation Index")
    ax2.set_ylabel("Belief Value ($b$)")
    ax2.legend(loc='lower right')

    plt.tight_layout()
    
    # Save in high definition for LaTeX (PDF vectorial or PNG 300 DPI)
    plot_path_png = "veris_experimental_results.png"
    plot_path_pdf = "veris_experimental_results.pdf"
    plt.savefig(plot_path_png, dpi=300)
    plt.savefig(plot_path_pdf)
    
    print(f"[+] Graph saved successfully :\n    -> {plot_path_png}\n    -> {plot_path_pdf}")

if __name__ == "__main__":
    nom_u, nom_b, deg_u, deg_b = run_pipeline()
    if nom_u:
        generate_plots(nom_u, nom_b, deg_u, deg_b)
    else:
        print("[-] Failed to run the experiment : No data collected.")