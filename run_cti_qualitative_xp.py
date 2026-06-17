import random
import matplotlib.pyplot as plt
from typing import List, Dict

from lib import (
    TrustForestEngine, FusionNode, QualitativeLabel, 
    DstTriplet, dempster_shafer_combine, MissingDimension
)

# =====================================================================
# 1. MERGE FUNCTIONS FOR CTI QUALITATIVE DATA
# =====================================================================

def fuse_source_trust(inputs: List[any]) -> DstTriplet:
    """Translates the qualitative label from the MISP (Admiralty) into a DST triplet."""
    source_label = inputs[0]
    if isinstance(source_label, MissingDimension):
        return DstTriplet(0.0, 0.0, 1.0) # Unknown source = Total uncertainty
    
    lbl = source_label.label
    if lbl == "A_Completely_Reliable":
        return DstTriplet(0.8, 0.0, 0.2)
    elif lbl == "B_Usually_Reliable":
        return DstTriplet(0.6, 0.1, 0.3)
    elif lbl == "C_Fairly_Reliable":
        return DstTriplet(0.4, 0.2, 0.4)
    else: # D_Not_Usually_Reliable
        return DstTriplet(0.1, 0.6, 0.3)

def fuse_context_severity(inputs: List[any]) -> DstTriplet:
    """Fuses the TLP (distribution) and threat type (cyber severity)."""
    tlp_node = inputs[0]
    type_node = inputs[1]
    
    b, d, u = 0.0, 0.0, 1.0
    
    if not isinstance(tlp_node, MissingDimension):
        if tlp_node.label in ["RED", "AMBER"]: # Restricted distribution = Potential critical incident
            b += 0.4
        else:
            d += 0.2
            
    if not isinstance(type_node, MissingDimension):
        if type_node.label == "APT_Targeted_Malware":
            b += 0.5
        elif type_node.label == "Generic_Botnet":
            b += 0.2
            d += 0.2
        else: # Osint_Scrape
            d += 0.6
            
    total = b + d + 0.2
    return DstTriplet(b/total, d/total, 0.2/total)

def root_cti_evaluation(inputs: List[any]) -> DstTriplet:
    """Final fusion by the Dempster-Shafer rule."""
    t1 = inputs[0] if not isinstance(inputs[0], MissingDimension) else DstTriplet(0.0, 0.0, 1.0)
    t2 = inputs[1] if not isinstance(inputs[1], MissingDimension) else DstTriplet(0.0, 0.0, 1.0)
    return dempster_shafer_combine(t1, t2)

# =====================================================================
# 2. SIMULATOR OF PUBLIC MISP / STIX FLOWS
# =====================================================================

def generate_public_misp_stream(count: int = 3000) -> List[Dict]:
    stream = []
    for _ in range(count):
        #  Typical IoC categories found in public CTI dumps
        category = random.choices([0, 1, 2], weights=[0.70, 0.20, 0.10], k=1)[0]
        
        if category == 0: # 1. OSINT background noise (Low confidence, TLP:CLEAR)
            stream.append({
                "source": random.choice(["C_Fairly_Reliable", "D_Not_Usually_Reliable"]),
                "tlp": "CLEAR",
                "type": "Osint_Scrape"
            })
        elif category == 1: # 2. Standard Commercial Traffic (Medium Trust, Botnets)
            stream.append({
                "source": "B_Usually_Reliable",
                "tlp": "AMBER",
                "type": "Generic_Botnet"
            })
        else: # 3. Alert from a state-sponsored CERT (Absolute trust, APT cyber-spying)
            stream.append({
                "source": "A_Completely_Reliable",
                "tlp": "RED",
                "type": "APT_Targeted_Malware"
            })
    return stream

# =====================================================================
# 3. BENCHMARK PIPELINE
# =====================================================================

if __name__ == "__main__":
    engine = TrustForestEngine()
    
    # Vocabulaires qualitatifs stricts
    v_source = ["A_Completely_Reliable", "B_Usually_Reliable", "C_Fairly_Reliable", "D_Not_Usually_Reliable"]
    v_tlp = ["RED", "AMBER", "GREEN", "CLEAR"]
    v_type = ["APT_Targeted_Malware", "Generic_Botnet", "Osint_Scrape"]
    
    # Enregistrement des nœuds
    node_src = FusionNode("node_src", ["QUALITATIVE_LABEL"], "DST_TRIPLET", fuse_source_trust)
    node_ctx = FusionNode("node_ctx", ["QUALITATIVE_LABEL", "QUALITATIVE_LABEL"], "DST_TRIPLET", fuse_context_severity)
    node_root = FusionNode("root", ["DST_TRIPLET", "DST_TRIPLET"], "DST_TRIPLET", root_cti_evaluation)
    
    engine.register_node(node_src, ["leaf_src"])
    engine.register_node(node_ctx, ["leaf_tlp", "leaf_type"])
    engine.register_node(node_root, ["node_src", "node_ctx"])
    
    misp_data = generate_public_misp_stream(3000)
    
    results_b = []
    results_u = []
    
    for ioc in misp_data:
        leaves = {
            "leaf_src": QualitativeLabel(ioc["source"], v_source),
            "leaf_tlp": QualitativeLabel(ioc["tlp"], v_tlp),
            "leaf_type": QualitativeLabel(ioc["type"], v_type)
        }
        
        # Intentional insertion of missing data (e.g., malformed stream or omitted TLP)
        if random.random() < 0.08:
            del leaves["leaf_tlp"]
            
        res = engine.evaluate("root", leaves)
        if isinstance(res, DstTriplet):
            results_b.append(res.b)
            results_u.append(res.u)

    #  Extraction and sorting of results for graphical analysis
    sorted_idx = sorted(range(len(results_b)), key=lambda k: results_b[k])
    b_sorted = [results_b[i] for i in sorted_idx]
    u_sorted = [results_u[i] for i in sorted_idx]
    
    plt.figure(figsize=(9, 4.5))
    plt.plot(b_sorted, label="CTI Trust Belief ($b$)", color="darkgreen", linewidth=2.5)
    plt.fill_between(range(len(u_sorted)), b_sorted, [b + u for b, u in zip(b_sorted, u_sorted)], 
                     color="green", alpha=0.15, label="CTI Uncertainty ($u$)")
    
    plt.title("Qualitative CTI (MISP/STIX) Attribute Fusion Profile", fontsize=12, fontweight="bold")
    plt.xlabel("Processed Public Threat Indicators (Sorted by relevance)")
    plt.ylabel("DST Mass Distribution")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.savefig("cti_qualitative_results.pdf")
    print("[+] CTI Qualitative experiment completed successfully (cti_qualitative_results.pdf).")