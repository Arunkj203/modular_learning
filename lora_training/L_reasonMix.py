
from .lora_training import phase_train,load_phase_model
import traceback
import os
# Main code file
from ..solve import *


def test_phase(dataset,dataset_name,total_batches):
  
    adapter = "L_reasonMix"
    PHASE1_ADAPTER = "L_reasonMix_phase1"
    PHASE2_ADAPTER = "L_reasonMix_phase2"
    PHASE3_ADAPTER = "L_reasonMix_phase3"

    model1,tokenizer = load_phase_model(PHASE1_ADAPTER)

    output_dir = os.path.join("lora_training","L_reasonMix")

    print(f"\n=== Running  Testing (ReasonMix) with {dataset_name} ===")
    print("\n=== Running Phase 1 Testing (ReasonMix) ===")
    generate_phase1_analysis(dataset,dataset_name, model1, tokenizer, output_dir=output_dir)

    model2,tokenizer = load_phase_model(PHASE2_ADAPTER)
    print("\n=== Running Phase 2 Testing (ReasonMix) ===")
    generate_phase2_execution_batch(f"{dataset_name}_test_phase1_analysis.json", model2, tokenizer,adapter,total_batches, output_dir=output_dir)
       
    
    model3,tokenizer = load_phase_model(PHASE3_ADAPTER)
    print("\n=== Running Phase 3 Testing (ReasonMix) ===")
    for i in range(1,total_batches+1):
        print(f"Phase 3 Batch {i} processing....")
        generate_phase3_execution(f"{dataset_name}_test_phase2_execution_batch{i}_{i*100}.json", model3,tokenizer,adapter, output_dir=output_dir)
    
    print(f"\nAll phase models of ReasonMix tested for {dataset_name} successfully.")


if __name__ == "__main__":
    
    svamp = list(load_dataset("ChilleD/SVAMP", split="test"))[:5]
    print(f"Loaded SVAMP test: {len(svamp)} samples")


    # SVAMP:
    test_phase(svamp,"svamp",1) 

    gsm8k = list(load_dataset("gsm8k", "main", split="test"))[:5]
    print(f"Loaded GSM8k test: {len(gsm8k)} samples")



    # GSM8k:
    # test_phase(gsm8k,"gsm8k",1)


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


# def train_phase():
#     # ---------------------------
#     # SVAMP DATASETS
#     # ---------------------------

#     svamp_phase1 = [
#         "Dataset/SVAMP_train_phase1_analysis.json"
#     ]

#     svamp_phase2 = [
#         "Dataset/SVAMP_train_phase2_execution_batch1_70.json",
#         "Dataset/SVAMP_train_phase2_execution_batch2_140.json",
#         "Dataset/SVAMP_train_phase2_execution_batch3_210.json",
#         "Dataset/SVAMP_train_phase2_execution_batch4_280.json",
#         "Dataset/SVAMP_train_phase2_execution_batch5_350.json",
#     ]

#     svamp_phase3 = [
#         "Dataset/SVAMP_train_phase3_execution_batch1_70.json",
#         "Dataset/SVAMP_train_phase3_execution_batch2_140.json",
#         "Dataset/SVAMP_train_phase3_execution_batch3_210.json",
#         "Dataset/SVAMP_train_phase3_execution_batch4_280.json",
#         "Dataset/SVAMP_train_phase3_execution_batch5_350.json",
#     ]

#     # ---------------------------
#     # GSM8K DATASETS
#     # ---------------------------

#     gsm_phase1 = [
#         "Dataset/GSM8K_train_phase1_analysis_batch1_200.json",
#     ]

#     gsm_phase2 = [
#         "Dataset/GSM8K_train_phase2_execution_batch1_200.json",

        
#     ]

#     gsm_phase3 = [
#         "Dataset/GSM8K_train_phase3_execution_batch1_200.json",
#     ]


#     # ---------------------------
#     # TRAINING — Phase 1
#     # ---------------------------
#     try:
#         print("\n=== Running Phase 1 Training (ReasonMix) ===")
#         phase_train(
#             files=svamp_phase1 + gsm_phase1,
#             phase=1,
#             adapter_name="L_reasonMix_phase1"
#         )
#         print("[Phase 1] Completed successfully.\n")

#     except Exception as e:
#         print(f"[Phase 1 ERROR] Training failed: {e}\n")
#         traceback.print_exc()



#     # ---------------------------
#     # TRAINING — Phase 2
#     # ---------------------------
#     try:
#         print("\n=== Running Phase 2 Training (ReasonMix) ===")
#         phase_train(
#             files=svamp_phase2 + gsm_phase2,
#             phase=2,
#             adapter_name="L_reasonMix_phase2"
#         )
#         print("[Phase 2] Completed successfully.\n")

#     except Exception as e:
#         print(f"[Phase 2 ERROR] Training failed: {e}\n")
#         traceback.print_exc()



#     # ---------------------------
#     # TRAINING — Phase 3
#     # ---------------------------
#     try:
#         print("\n=== Running Phase 3 Training (ReasonMix) ===")
#         phase_train(
#             files=svamp_phase3 + gsm_phase3,
#             phase=3,
#             adapter_name="L_reasonMix_phase3"
#         )
#         print("[Phase 3] Completed successfully.\n")

#     except Exception as e:
#         print(f"[Phase 3 ERROR] Training failed: {e}\n")
#         traceback.print_exc()

#     print("\nAll phase models of ReasonMix trained successfully.")
