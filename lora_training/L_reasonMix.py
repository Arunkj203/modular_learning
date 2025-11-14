
from .lora_training import phase_train
import traceback

def main():
    
    # ---------------------------
    # SVAMP DATASETS
    # ---------------------------

    svamp_phase1 = [
        "Dataset/SVAMP_train_phase1_analysis.json"
    ]

    svamp_phase2 = [
        "Dataset/SVAMP_train_phase2_execution_batch1_70.json",
        "Dataset/SVAMP_train_phase2_execution_batch2_140.json",
        "Dataset/SVAMP_train_phase2_execution_batch3_210.json",
        "Dataset/SVAMP_train_phase2_execution_batch4_280.json",
        "Dataset/SVAMP_train_phase2_execution_batch5_350.json",
    ]

    svamp_phase3 = [
        "Dataset/SVAMP_train_phase3_execution_batch1_70.json",
        "Dataset/SVAMP_train_phase3_execution_batch2_140.json",
        "Dataset/SVAMP_train_phase3_execution_batch3_210.json",
        "Dataset/SVAMP_train_phase3_execution_batch4_280.json",
        "Dataset/SVAMP_train_phase3_execution_batch5_350.json",
    ]

    # ---------------------------
    # GSM8K DATASETS
    # ---------------------------

    gsm_phase1 = [
        "Dataset/GSM8K_train_phase1_analysis_batch1_200.json",
    ]

    gsm_phase2 = [
        "Dataset/GSM8K_train_phase2_execution_batch1_200.json",

        
    ]

    gsm_phase3 = [
        "Dataset/GSM8K_train_phase3_execution_batch1_200.json",
    ]


    # ---------------------------
    # TRAINING — Phase 1
    # ---------------------------
    try:
        print("\n=== Running Phase 1 Training (ReasonMix) ===")
        phase_train(
            files=svamp_phase1 + gsm_phase1,
            phase=1,
            adapter_name="L_reasonMix_phase1"
        )
        print("[Phase 1] Completed successfully.\n")

    except Exception as e:
        print(f"[Phase 1 ERROR] Training failed: {e}\n")
        traceback.print_exc()



    # ---------------------------
    # TRAINING — Phase 2
    # ---------------------------
    try:
        print("\n=== Running Phase 2 Training (ReasonMix) ===")
        phase_train(
            files=svamp_phase2 + gsm_phase2,
            phase=2,
            adapter_name="L_reasonMix_phase2"
        )
        print("[Phase 2] Completed successfully.\n")

    except Exception as e:
        print(f"[Phase 2 ERROR] Training failed: {e}\n")
        traceback.print_exc()



    # ---------------------------
    # TRAINING — Phase 3
    # ---------------------------
    try:
        print("\n=== Running Phase 3 Training (ReasonMix) ===")
        phase_train(
            files=svamp_phase3 + gsm_phase3,
            phase=3,
            adapter_name="L_reasonMix_phase3"
        )
        print("[Phase 3] Completed successfully.\n")

    except Exception as e:
        print(f"[Phase 3 ERROR] Training failed: {e}\n")
        traceback.print_exc()

    print("\nAll phase models of ReasonMix trained successfully.")


if __name__ == "__main__":
    main()





'''
svamp_phase1_files = [
        "Dataset/SVAMP_train_phase1_analysis.json"
    ]

    gsm8k_phase1_files = [
        "Dataset/GSM8K_train_phase1_analysis_batch1_200.json",
        "Dataset/GSM8K_train_phase1_analysis_batch2_400.json",
        "Dataset/GSM8K_train_phase1_analysis_batch3_600.json",
        "Dataset/GSM8K_train_phase1_analysis_batch4_800.json",
        "Dataset/GSM8K_train_phase1_analysis_batch5_1000.json",
        "Dataset/GSM8K_train_phase1_analysis_batch6_1200.json",
        "Dataset/GSM8K_train_phase1_analysis_batch7_1400.json",
        "Dataset/GSM8K_train_phase1_analysis_batch8_1600.json",
        "Dataset/GSM8K_train_phase1_analysis_batch9_1800.json",
        "Dataset/GSM8K_train_phase1_analysis_batch10_2000.json",
    ]
'''