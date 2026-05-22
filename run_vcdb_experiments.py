import os
import json
import glob
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Import du moteur et des types depuis votre bibliothèque lib.py
from lib import (
    TrustForestEngine, FusionNode, ProbabilisticScore, 
    QualitativeLabel, DstTriplet, root_global_fusion,
    fuse_actor_reliability, fuse_attribute_criticality
)

def init_experiment_engine() -> TrustForestEngine:
    """Initialise la topologie du graphe pour les expérimentations VERIS."""
    engine = TrustForestEngine()
    
    # Déclaration des nœuds selon l'architecture proposée
    actor_node = FusionNode("actor_reliability", ["PROBA_SCORE"], "DST_TRIPLET", fuse_actor_reliability)
    attr_node = FusionNode("attr_criticality", ["PROBA_SCORE"], "DST_TRIPLET", fuse_attribute_criticality)
    root_node = FusionNode("global_incident_trust", ["DST_TRIPLET", "DST_TRIPLET"], "DST_TRIPLET", root_global_fusion)

    # Enregistrement des liaisons (Arêtes)
    engine.register_node(actor_node, ["raw_actor_field"])
    engine.register_node(attr_node, ["raw_attribute_field"])
    engine.register_node(root_node, ["actor_reliability", "attr_criticality"])
    
    engine.validate_graph_integrity()
    return engine

def get_first_scavenged_string(data_structure) -> Optional[str]:
    """Navigue récursivement dans les structures VERIS pour extraire la première string valide."""
    if isinstance(data_structure, str):
        return data_structure
    if isinstance(data_structure, list) and len(data_structure) > 0:
        return get_first_scavenged_string(data_structure[0])
    if isinstance(data_structure, dict):
        # Si c'est un dictionnaire, on cherche une valeur textuelle ou une clé 'variety'
        if "variety" in data_structure:
            return get_first_scavenged_string(data_structure["variety"])
        for val in data_structure.values():
            res = get_first_scavenged_string(val)
            if res:
                return res
    return None

def map_json_to_leaves(record: Dict) -> Dict:
    """Traduit de manière ultra-robuste le JSON VERIS en feuilles typées pour le moteur."""
    leaves = {}
    
    # Extraction sécurisée de l'Acteur
    try:
        actor_data = record.get("actor", {}).get("external", {}).get("variety", None)
        if not actor_data: # Fallback si ce n'est pas un acteur externe
            actor_data = record.get("actor", {}).get("internal", {}).get("variety", None)
            
        actor_str = get_first_scavenged_string(actor_data)
        if actor_str:
            leaves["raw_actor_field"] = ProbabilisticScore(1.0)
            leaves["raw_actor_field"].value = actor_str
    except Exception:
        pass

    # Extraction sécurisée de l'Attribut (Triade CIA / Confidentialité)
    try:
        # Résout le problème où 'data' ou 'confidentiality' est une liste ou un dict imbriqué
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
    
    # Chemins d'accès vers votre échantillon VCDB
    search_path = os.path.join("data", "VERIS_sample", "*.json")
    json_files = glob.glob(search_path)
    
    if not json_files:
        print(f"[-] Aucun fichier JSON trouvé dans {search_path}. Vérifiez l'arborescence.")
        return [], [], [], []

    print(f"[+] Lancement des analyses sur {len(json_files)} fichiers VERIS...")

    # Tableaux pour stocker les métriques pour les graphiques
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

            # --- 1. RUN NOMINAL (Données brutes de la base) ---
            leaves_nominal = map_json_to_leaves(record)
            # On ne garde que les enregistrements qui ont au moins une info exploitable
            if not leaves_nominal:
                continue
                
            res_nominal = engine.evaluate("global_incident_trust", leaves_nominal)
            if isinstance(res_nominal, DstTriplet):
                nominal_uncertainties.append(res_nominal.u)
                nominal_beliefs.append(res_nominal.b)

            # --- 2. RUN DEGRADED (On ampute volontairement l'acteur pour tester l'asymétrie) ---
            record_degraded = record.copy()
            if "actor" in record_degraded:
                del record_degraded["actor"] # Injection d'anomalie de rétention
                
            leaves_degraded = map_json_to_leaves(record_degraded)
            res_degraded = engine.evaluate("global_incident_trust", leaves_degraded)
            if isinstance(res_degraded, DstTriplet):
                degraded_uncertainties.append(res_degraded.u)
                degraded_beliefs.append(res_degraded.b)

    return nominal_uncertainties, nominal_beliefs, degraded_uncertainties, degraded_beliefs

def generate_plots(nom_u: List[float], nom_b: List[float], deg_u: List[float], deg_b: List[float]):
    """Génère des courbes et histogrammes de distribution académiques."""
    print("[+] Génération des graphiques de performance pour l'article...")
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- GRAPH 1 : Évolution et Distribution de l'Incertitude (u) ---
    ax1.hist(nom_u, bins=20, alpha=0.6, label='Nominal Run (Full Data)', color='#2ca02c')
    ax1.hist(deg_u, bins=20, alpha=0.6, label='Degraded Run (Missing Actor)', color='#d62728')
    ax1.set_title("Axiomatic Uncertainty Propagation ($u$)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epistemic Uncertainty Metric ($u$)")
    ax1.set_ylabel("Number of VERIS Incidents")
    ax1.legend(loc='upper right')

    # --- GRAPH 2 : Stabilité de la Croyance (b) contre l'effet Compensatoire ---
    # Tri des données pour une visualisation propre en courbe cumulative/tendance
    nom_b_sorted = sorted(nom_b)
    deg_b_sorted = sorted(deg_b)
    
    ax2.plot(nom_b_sorted, label='Nominal Trust Belief ($b$)', color='#1f77b4', linewidth=2)
    ax2.plot(deg_b_sorted, label='Degraded Trust Belief ($b$)', color='#ff7f0e', linestyle='--', linewidth=2)
    ax2.set_title("Non-Compensatory Belief Resilience ($b$)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Sorted Incident Evaluation Index")
    ax2.set_ylabel("Belief Value ($b$)")
    ax2.legend(loc='lower right')

    plt.tight_layout()
    
    # Sauvegarde en haute définition pour LaTeX (PDF vectoriel ou PNG 300 DPI)
    plot_path_png = "veris_experimental_results.png"
    plot_path_pdf = "veris_experimental_results.pdf"
    plt.savefig(plot_path_png, dpi=300)
    plt.savefig(plot_path_pdf)
    
    print(f"[+] Graphique sauvegardé avec succès :\n    -> {plot_path_png}\n    -> {plot_path_pdf}")

if __name__ == "__main__":
    nom_u, nom_b, deg_u, deg_b = run_pipeline()
    if nom_u:
        generate_plots(nom_u, nom_b, deg_u, deg_b)
    else:
        print("[-] Échec de l'expérimentation : Aucune donnée collectée.")