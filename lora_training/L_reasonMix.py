
from .lora_training import phase_train

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
    
    # ReasonMix Phase 1 = SVAMP + GSM
    phase_train(
        files=svamp_phase1 + gsm_phase1,
        phase=1,
        adapter_name="L_reasonMix_phase1"
    )


    # ---------------------------
    # TRAINING — Phase 2
    # ---------------------------

    phase_train(
        files=svamp_phase1 + gsm_phase2,
        phase=2,
        adapter_name="L_reasonMix_phase2"
    )


    # ---------------------------
    # TRAINING — Phase 3
    # ---------------------------
    phase_train(
        files=svamp_phase1 + gsm_phase3,
        phase=3,
        adapter_name="L_reasonMix_phase3"
    )

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