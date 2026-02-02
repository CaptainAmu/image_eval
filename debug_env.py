#!/usr/bin/env python3
"""
Step-by-step env check for compute_image_metrics.py.
Run on the same machine/env as your Slurm job: conda activate image_eval && python debug_env.py
"""
import sys

def step(name, fn):
    print(f"\n--- Step: {name} ---")
    try:
        fn()
        print("  OK")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

# 1. Python & path
step("Python version", lambda: print(f"  {sys.version}"))

# 2. torch
step("import torch", lambda: __import__("torch"))
step("torch.cuda", lambda: print(f"  cuda available: {__import__('torch').cuda.is_available()}"))

# 3. transformers (for CLIP and LAION)
step("transformers", lambda: print(f"  version: {__import__('transformers').__version__}"))

# 4. Other deps (pandas, PIL)
step("pandas", lambda: __import__("pandas"))
step("PIL", lambda: __import__("PIL.Image"))

print("\n--- Done. If any step FAILed, fix that step first (see comments in script). ---")
