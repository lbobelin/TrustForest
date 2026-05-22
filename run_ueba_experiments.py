import random
import matplotlib.pyplot as plt
from typing import List, Dict

# Import de votre bibliothèque
from lib import (
    TrustForestEngine, FusionNode, ProbabilisticScore, 
    QualitativeLabel, DstTriplet, dempster_shafer_combine,
    MissingDimension
)

# =====================================================================
# 1. FONCTIONS DE FUSION DÉDIÉES À L'UEBA
# =====================================================================

def fuse_ueba_signal(inputs: List[any]) -> DstTriplet:
    """Projette le score ML brut [0,1] en triplet DST."""
    val = inputs[0]
    if isinstance(val, MissingDimension):
        return DstTriplet(0.0, 0.0, 1.0) # Capteur ML hors-ligne = Incertitude totale
    
    score = val.value
    if score > 0.8:
        # Forte suspicion d'anomalie, mais on garde de l'incertitude 
        # car le ML ne connaît pas l'intention de l'utilisateur.
        return DstTriplet(0.6, 0.0, 0.4)
    elif score < 0.3:
        return DstTriplet(0.0, 0.8, 0.2) # Comportement sain prouvé
    return DstTriplet(0.1, 0.1, 0.8)

def fuse_context_risk(inputs: List[any]) -> DstTriplet:
    """Combine le rôle de l'utilisateur (Qualitatif) et l'heure (Probabiliste)."""
    user_tier = inputs[0]
    hours_score = inputs[1]
    
    # Par défaut, si tout est manquant
    b, d, u = 0.0, 0.0, 1.0
    
    if not isinstance(user_tier, MissingDimension):
        if user_tier.label == "CRITICAL": # Profil à haut risque (ex: Admin)
            b += 0.3
        elif user_tier.label == "LOW":
            d += 0.4
            
    if not isinstance(hours_score, MissingDimension):
        if hours_score.value < 0.3: # Activité nocturne suspecte
            b += 0.4
        else:
            d += 0.3
            
    # Normalisation pour créer un triplet DST valide
    total = b + d + 0.3 # 0.3 d'incertitude incompressible pour le contexte
    return DstTriplet(b/total, d/total, 0.3/total)

def root_security_decision(inputs: List[any]) -> DstTriplet:
    """Fusionne le signal ML et le contexte RH/Horaire."""
    t1 = inputs[0] if not isinstance(inputs[0], MissingDimension) else DstTriplet(0.0, 0.0, 1.0)
    t2 = inputs[1] if not isinstance(inputs[1], MissingDimension) else DstTriplet(0.0, 0.0, 1.0)
    return dempster_shafer_combine(t1, t2)

# =====================================================================
# 2. GÉNÉRATEUR DE TÉLÉMÉTRIE FLUX UEBA SYNTHÉTIQUE
# =====================================================================

def generate_ueba_logs(count: int = 5000) -> List[Dict]:
    logs = []
    vocab = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    
    for _ in range(count):
        scenario = random.choices([0, 1, 2], weights=[0.85, 0.10, 0.05], k=1)[0]
        
        if scenario == 0: # 1. Comportement nominal standard
            logs.append({
                "ueba_score": random.uniform(0.0, 0.25),
                "user_label": random.choice(["MEDIUM", "LOW"]),
                "hours": random.uniform(0.8, 1.0) # Pleine journée
            })
        elif scenario == 1: # 2. Anomalie bénigne (Faux positif ML classique)
            logs.append({
                "ueba_score": random.uniform(0.85, 0.99), # Alerte rouge ML
                "user_label": "LOW", # Simple utilisateur
                "hours": random.uniform(0.8, 1.0) # En journée, comportement métier explicable
            })
        else: # 3. Vraie Attaque (Exfiltration nocturne par un compte à privilèges)
            logs.append({
                "ueba_score": random.uniform(0.90, 1.0),
                "user_label": "CRITICAL",
                "hours": random.uniform(0.0, 0.1) # Milieu de la nuit
            })
            
    return logs

# =====================================================================
# 3. PIPELINE D'ÉVALUATION
# =====================================================================

if __name__ == "__main__":
    # Initialisation du moteur
    engine = TrustForestEngine()
    
    node_ueba = FusionNode("node_ueba", ["PROBA_SCORE"], "DST_TRIPLET", fuse_ueba_signal)
    node_ctx = FusionNode("node_ctx", ["QUALITATIVE_LABEL", "PROBA_SCORE"], "DST_TRIPLET", fuse_context_risk)
    node_root = FusionNode("root", ["DST_TRIPLET", "DST_TRIPLET"], "DST_TRIPLET", root_security_decision)
    
    engine.register_node(node_ueba, ["leaf_ueba"])
    engine.register_node(node_ctx, ["leaf_user", "leaf_hours"])
    engine.register_node(node_root, ["node_ueba", "node_ctx"])
    
    # Génération des données
    dataset = generate_ueba_logs(5000)
    
    # Listes pour analyse graphique
    final_beliefs = []
    final_uncertainties = []
    raw_ueba_scores = []
    
    vocab = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    for log in dataset:
        # Encodage strict dans les types de la bibliothèque
        leaves = {
            "leaf_ueba": ProbabilisticScore(log["ueba_score"]),
            "leaf_user": QualitativeLabel(log["user_label"], vocab),
            "leaf_hours": ProbabilisticScore(log["hours"])
        }
        
        # Simulation d'une panne de capteur aléatoire sur 5% des paquets (Asymétrie)
        if random.random() < 0.05:
            del leaves["leaf_ueba"]
            
        res = engine.evaluate("root", leaves)
        
        if isinstance(res, DstTriplet):
            final_beliefs.append(res.b)
            final_uncertainties.append(res.u)
            raw_ueba_scores.append(log["ueba_score"] if "leaf_ueba" in leaves else -0.05)

    # =====================================================================
    # 4. GÉNÉRATION DE LA FIGURE POUR LE PAPIER
    # =====================================================================
    plt.style.use('default')
    plt.figure(figsize=(9, 5))
    
    # Tri des données selon la croyance finale pour voir la courbe de décision
    sorted_indices = sorted(range(len(final_beliefs)), key=lambda k: final_beliefs[k])
    b_plot = [final_beliefs[i] for i in sorted_indices]
    u_plot = [final_uncertainties[i] for i in sorted_indices]
    
    plt.plot(b_plot, label="Final Calibrated Threat Belief ($b$)", color="red", linewidth=2.5)
    plt.fill_between(range(len(u_plot)), b_plot, [b + u for b, u in zip(b_plot, u_plot)], 
                     color="gray", alpha=0.3, label="Epistemic Uncertainty Range ($u$)")
    
    plt.title("UEBA Anomaly Contextualization via TrustForest", fontsize=12, fontweight="bold")
    plt.xlabel("Simulated Telemetry Events (Sorted by Risk)")
    plt.ylabel("Mathematical Mass Assignment")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.savefig("ueba_experimental_results.pdf")
    plt.savefig("ueba_experimental_results.png", dpi=300)
    print("[+] Expérimentation UEBA terminée. Graphiques vectoriels générés (ueba_experimental_results.pdf).")