import random
import matplotlib.pyplot as plt
from typing import List, Dict

from lib import (
    TrustForestEngine, FusionNode, ProbabilisticScore, 
    QualitativeLabel, DstTriplet, dempster_shafer_combine,
    MissingDimension
)

# =====================================================================
# 1. FUSION FUNCTIONS DEDICATED TO UEBA
# =====================================================================

def fuse_ueba_signal(inputs: List[any]) -> DstTriplet:
    """Projects the raw ML score [0,1] into a DST triplet."""
    val = inputs[0]
    if isinstance(val, MissingDimension):
        return DstTriplet(0.0, 0.0, 1.0) # Offline ML sensor = Total uncertainty
    
    score = val.value
    if score > 0.8:
        # Strong suspicion of anomaly, but we keep some uncertainty
        # because the ML doesn't know the user's intent.
        return DstTriplet(0.6, 0.0, 0.4)
    elif score < 0.3:
        return DstTriplet(0.0, 0.8, 0.2) # Proven safe behavior
    return DstTriplet(0.1, 0.1, 0.8)

def fuse_context_risk(inputs: List[any]) -> DstTriplet:
    """Combine the user's role (Qualitative) and the time of day (Probabilistic)."""
    user_tier = inputs[0]
    hours_score = inputs[1]
    
    # By default, if everything is missing
    b, d, u = 0.0, 0.0, 1.0
    
    if not isinstance(user_tier, MissingDimension):
        if user_tier.label == "CRITICAL": # High-risk profile (e.g., Admin)
            b += 0.3
        elif user_tier.label == "LOW":
            d += 0.4
            
    if not isinstance(hours_score, MissingDimension):
        if hours_score.value < 0.3: # Suspicious nighttime activity
            b += 0.4
        else:
            d += 0.3
            
    # Standardization to create a valid DST triplet
    total = b + d + 0.3 # 0.3 of unavoidable uncertainty for the context
    return DstTriplet(b/total, d/total, 0.3/total)

def root_security_decision(inputs: List[any]) -> DstTriplet:
    """Fuses the ML signal and the RH/contextual information."""
    t1 = inputs[0] if not isinstance(inputs[0], MissingDimension) else DstTriplet(0.0, 0.0, 1.0)
    t2 = inputs[1] if not isinstance(inputs[1], MissingDimension) else DstTriplet(0.0, 0.0, 1.0)
    return dempster_shafer_combine(t1, t2)

# =====================================================================
# 2.  SYNTHETIC UEBA FLUX TELEMETRY GENERATOR
# =====================================================================

def generate_ueba_logs(count: int = 5000) -> List[Dict]:
    logs = []
    vocab = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    
    for _ in range(count):
        scenario = random.choices([0, 1, 2], weights=[0.85, 0.10, 0.05], k=1)[0]
        
        if scenario == 0: # 1. Standard nominal behavior
            logs.append({
                "ueba_score": random.uniform(0.0, 0.25),
                "user_label": random.choice(["MEDIUM", "LOW"]),
                "hours": random.uniform(0.8, 1.0) # Full day
            })
        elif scenario == 1: # 2. Benign Anomaly (Classic False Positive ML)
            logs.append({
                "ueba_score": random.uniform(0.85, 0.99), # Red ML Alert
                "user_label": "LOW", # Simple User
                "hours": random.uniform(0.8, 1.0) # During the day, explainable business behavior
            })
        else: # 3. Real Attack (Nighttime Exfiltration by a Privileged Account)
            logs.append({
                "ueba_score": random.uniform(0.90, 1.0),
                "user_label": "CRITICAL",
                "hours": random.uniform(0.0, 0.1) # Middle of the night
            })
            
    return logs

# =====================================================================
# 3. EVALUATION PIPELINE
# =====================================================================

if __name__ == "__main__":
    # Engine Initialization
    engine = TrustForestEngine()
    
    node_ueba = FusionNode("node_ueba", ["PROBA_SCORE"], "DST_TRIPLET", fuse_ueba_signal)
    node_ctx = FusionNode("node_ctx", ["QUALITATIVE_LABEL", "PROBA_SCORE"], "DST_TRIPLET", fuse_context_risk)
    node_root = FusionNode("root", ["DST_TRIPLET", "DST_TRIPLET"], "DST_TRIPLET", root_security_decision)
    
    engine.register_node(node_ueba, ["leaf_ueba"])
    engine.register_node(node_ctx, ["leaf_user", "leaf_hours"])
    engine.register_node(node_root, ["node_ueba", "node_ctx"])
    
    # Data Generation
    dataset = generate_ueba_logs(5000)
    
    # Lists for graphical analysis
    final_beliefs = []
    final_uncertainties = []
    raw_ueba_scores = []
    
    vocab = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    for log in dataset:
        # Strict typing in library types
        leaves = {
            "leaf_ueba": ProbabilisticScore(log["ueba_score"]),
            "leaf_user": QualitativeLabel(log["user_label"], vocab),
            "leaf_hours": ProbabilisticScore(log["hours"])
        }
        
        # Simulation of a random sensor failure on 5% of the packets (Asymmetry)
        if random.random() < 0.05:
            del leaves["leaf_ueba"]
            
        res = engine.evaluate("root", leaves)
        
        if isinstance(res, DstTriplet):
            final_beliefs.append(res.b)
            final_uncertainties.append(res.u)
            raw_ueba_scores.append(log["ueba_score"] if "leaf_ueba" in leaves else -0.05)

    # =====================================================================
    # 4. FIGURE GENERATION FOR THE PAPER
    # =====================================================================
    plt.style.use('default')
    plt.figure(figsize=(9, 5))
    
    # Sorting the data according to the final belief to visualize the decision curve
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
    print("[+] UEBA experimentation completed. Vector graphics generated (ueba_experimental_results.pdf).")