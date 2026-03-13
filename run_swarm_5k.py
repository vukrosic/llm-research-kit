import subprocess
import os
import sys
import time
from configs.ablation_configs import ABLATION_CONFIGS

def main():
    # Target subset: Attention Variations (20) + New batch (17) + Sweeps (~20)
    # Total should be around 58.
    
    # We identify them by the prefix "attn_" and being "new"
    # Actually, let's just run everything that's NOT in the completed list.
    
    completed = []
    if os.path.exists("completed_exps.txt"):
        with open("completed_exps.txt", "r") as f:
            completed = [line.strip() for line in f if line.strip()]
            
    # The "58 new ablations" are the Attention Variations, New Batch, and Sweeps.
    attention_variations = [
        "attn_q_norm_only", "attn_k_norm_only", "attn_no_qk_norm", "attn_qk_layernorm",
        "attn_scale_0_5", "attn_scale_2_0", "attn_softcap_10", "attn_softcap_30", "attn_softcap_50",
        "attn_window_64", "attn_window_128", "attn_window_256", "attn_act_relu", "attn_act_squared_relu",
        "attn_act_gelu", "attn_mqa_bias", "attn_gqa_8", "attn_gqa_2", "attn_sandwich_norm2",
        "attn_baseline_original"
    ]
    new_batch = [
        "attn_hilo_f90", "attn_hilo_f75", "attn_hilo_f50", "attn_pool_k2", "attn_pool_k4", "attn_pool_k8",
        "attn_shared_qkv", "attn_poly2", "attn_poly3", "attn_hilo_pool", "attn_window64_pool2",
        "attn_softcap8_relu", "attn_scale15_poly2", "attn_shared_qkv_norm", "attn_gqa_4",
        "attn_deepnorm_scale", "attn_small_embed_init"
    ]
    # Sweeps
    window_sweeps = [f"attn_window_sweep_{i*32}" for i in range(1, 11)]
    softcap_sweeps = [f"attn_softcap_sweep_{i*10 + 5}" for i in range(1, 6)]
    scale_sweeps = [f"attn_scale_sweep_{i*0.5}" for i in [2, 3, 4, 5, 6]]
    
    target_exps = attention_variations + new_batch + window_sweeps + softcap_sweeps + scale_sweeps
    
    # Check count
    # 20 + 17 + 10 + 5 + 5 = 57. 
    # Wait, the user said "around 58". Close enough.
    
    print(f"🚀 Found {len(target_exps)} target experiments.")
    
    # Ensure tokens = 5000
    TOKENS = 5000
    OUTPUT_DIR = "./ablation_results"
    
    for i, exp in enumerate(target_exps):
        print(f"\n[{i+1}/{len(target_exps)}] Starting experiment: {exp}")
        
        # We run in a separate process to clear memory perfectly
        cmd = [
            sys.executable, "run_ablations.py",
            "--tokens", str(TOKENS),
            "--experiments", exp,
            "--compile",
            "--output_dir", OUTPUT_DIR
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        start_time = time.time()
        process = subprocess.Popen(cmd)
        process.wait()
        
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print(f"✅ {exp} completed successfully in {elapsed:.1f}s")
            # Log to completed_exps_5k.txt
            with open("completed_exps_5k.txt", "a") as f:
                f.write(f"{exp}\n")
        else:
            print(f"❌ {exp} failed with exit code {process.returncode}")
            # Try once more without compile if it failed (likely OOM in compile)
            print(f"🔄 Retrying {exp} WITHOUT compilation...")
            cmd_no_compile = [
                sys.executable, "run_ablations.py",
                "--tokens", str(TOKENS),
                "--experiments", exp,
                "--no-compile",
                "--output_dir", OUTPUT_DIR
            ]
            process = subprocess.Popen(cmd_no_compile)
            process.wait()
            if process.returncode == 0:
                print(f"✅ {exp} completed (no-compile fallback)")
                with open("completed_exps_5k.txt", "a") as f:
                    f.write(f"{exp} (no-compile)\n")
            else:
                print(f"💀 {exp} failed permanently.")
        
        # Cool down
        time.sleep(2)

    print("\n🏁 All experiments finished!")

if __name__ == "__main__":
    main()
