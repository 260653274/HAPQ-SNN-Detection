import json
import glob

files = [
    "experiments/gen1_hapq/ablation/gen1_ablation_prune_only.json",
    "experiments/gen1_hapq/ablation/gen1_ablation_quant_w.json",
    "experiments/gen1_hapq/ablation/gen1_ablation_quant_wu.json",
    "experiments/gen1_hapq/ablation/gen1_ablation_full.json",
    "experiments/gen1_hapq/ablation/gen1_ablation_hapq.json"
]

for fpath in files:
    try:
        with open(fpath, "r") as f:
            data = json.load(f)
        
        candidate = data.get("best_candidate", {})
        layers = candidate.get("layers", [])
        
        if not layers:
            print(f"{fpath}: No layers found")
            continue
            
        b_w = [l.get("b_w", 8) for l in layers]
        b_u = [l.get("b_u", 12) for l in layers]
        active_blocks = [l.get("active_blocks", 1) for l in layers]
        
        max_blocks = max(active_blocks) if active_blocks else 0
        avg_bw = sum(b_w)/len(b_w) if b_w else 0
        avg_bu = sum(b_u)/len(b_u) if b_u else 0
        
        print(f"{fpath}:")
        print(f"  Max Active Blocks: {max_blocks}")
        print(f"  Avg b_w: {avg_bw:.2f}")
        print(f"  Avg b_u: {avg_bu:.2f}")
        print(f"  First layer active: {active_blocks[0]}")
        
    except Exception as e:
        print(f"Error reading {fpath}: {e}")
