import random
import matplotlib.pyplot as plt
from typing import List, Dict

# Import des composants de votre bibliothèque hypothétique
from lib import (
    TrustForestEngine, FusionNode, QualitativeLabel, 
    DstTriplet, dempster_shafer_combine, MissingDimension
)

# =====================================================================
# 1. FONCTIONS DE FUSION POUR DONNÉES QUALITATIVES
# =====================================================================

def fuse_source_trust(inputs: List[any]) -> DstTriplet:
    """Traduit le label qualitatif de l'Amirauté (MISP) en triplet DST."""
    source_label = inputs[0]
    if isinstance(source_label, MissingDimension):
        return DstTriplet(0.0, 0.0, 1.0) # Source inconnue = Incertitude totale
    
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
    """Fusionne le TLP (diffusion) et le Type de menace (gravité informatique)."""
    tlp_node = inputs[0]
    type_node = inputs[1]
    
    b, d, u = 0.0, 0.0, 1.0
    
    if not isinstance(tlp_node, MissingDimension):
        if tlp_node.label in ["RED", "AMBER"]: # Diffusion restreinte = Incident critique potentiel
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
    """Fusion finale par la règle de Dempster-Shafer."""
    t1 = inputs[0] if not isinstance(inputs[0], MissingDimension) else DstTriplet(0.0, 0.0, 1.0)
    t2 = inputs[1] if not isinstance(inputs[1], MissingDimension) else DstTriplet(0.0, 0.0, 1.0)
    return dempster_shafer_combine(t1, t2)

# =====================================================================
# 2. SIMULATEUR DE FLUX MISP / STIX PUBLIC
# =====================================================================

def generate_public_misp_stream(count: int = 3000) -> List[Dict]:
    stream = []
    for _ in range(count):
        # 3 catégories d'IoC typiques trouvés dans les dumps de CTI publics
        category = random.choices([0, 1, 2], weights=[0.70, 0.20, 0.10], k=1)[0]
        
        if category == 0: # 1. Bruit de fond OSINT (Faible confiance, TLP:CLEAR)
            stream.append({
                "source": random.choice(["C_Fairly_Reliable", "D_Not_Usually_Reliable"]),
                "tlp": "CLEAR",
                "type": "Osint_Scrape"
            })
        elif category == 1: # 2. Flux Commercial standard (Confiance moyenne, Botnets)
            stream.append({
                "source": "B_Usually_Reliable",
                "tlp": "AMBER",
                "type": "Generic_Botnet"
            })
        else: # 3. Alerte d'un CERT étatique (Confiance absolue, APT cyber-espionnage)
            stream.append({
                "source": "A_Completely_Reliable",
                "tlp": "RED",
                "type": "APT_Targeted_Malware"
            })
    return stream

# =====================================================================
# 3. PIPELINE DE BENCHMARK
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
        
        # Injection volontaire de données manquantes (ex: flux mal formaté ou TLP omis)
        if random.random() < 0.08:
            del leaves["leaf_tlp"]
            
        res = engine.evaluate("root", leaves)
        if isinstance(res, DstTriplet):
            results_b.append(res.b)
            results_u.append(res.u)

    # Extraction et tri des résultats pour analyse graphique
    sorted_idx = sorted(range(len(results_b)), key=lambda k: results_b[k])
    b_sorted = [results_b[i] for i in sorted_idx]
    u_sorted = [results_u[i] for i in sorted_idx]
    
    plt.figure(figsize=(9, 4.5))
    plt.plot(b_sorted, label="IoC Actionable Threat Belief ($b$)", color="darkgreen", linewidth=2.5)
    plt.fill_between(range(len(u_sorted)), b_sorted, [b + u for b, u in zip(b_sorted, u_sorted)], 
                     color="green", alpha=0.15, label="CTI Conflicting Uncertainty ($u$)")
    
    plt.title("Qualitative CTI (MISP/STIX) Attribute Fusion Profile", fontsize=12, fontweight="bold")
    plt.xlabel("Processed Public Threat Indicators (Sorted by Actionability)")
    plt.ylabel("DST Mass Distribution")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.savefig("cti_qualitative_results.pdf")
    print("[+] Expérience CTI Qualitative terminée avec succès (cti_qualitative_results.pdf).")