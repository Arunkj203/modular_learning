# Project Structure

```
modular_learning/
├── __init__.py
├── .gitignore
├── config.py
├── main.py
├── model_config.py
├── project_structure.md
├── README.md
├── requirements.txt
├── solve.py
├── test_code.py
├── logs/
│   └── SVAMP_train.txt
├── Mock_Dataset/
│   ├── test_data.json
│   └── train_data.json
├── others/
│   ├── __init__.py
│   ├── Calc_token.py
│   ├── suggestions.txt
│   └── workflow.txt
├── phase_1/
│   ├── __init__.py
│   ├── analyze_problems.py
│   ├── phase_1_main.py
│   └── preprocess.py
├── Phase 2: Reflective Primitive Planning
    ├── Step 1: Retrieve primitives
    ├── Step 2: Evaluate sufficiency  ← meta-reasoning
    ├── Step 3: Generate missing + sequence  ← repair & planning
    └── Step 4: Register new primitives
├── phase_3/
│   ├── __init__.py
│   ├── lora_training.py
│   ├── phase_3_main.py
│   └── synthetic_data.py
├── phase_4/
│   ├── __init__.py
│   └── phase_4_main.py
├── primitve_storage/
└── results/
    ├── primitive_library_1.json
    ├── primitive_summary.txt
    ├── sample_result.txt
    └── svamp_primitives.txt
```

## Directory Descriptions

- **logs/**: Log files from training and other processes
- **Mock_Dataset/**: Mock dataset files for testing and training
- **others/**: Miscellaneous utility files and documentation
- **phase_1/**: Initial problem analysis and preprocessing phase
- **phase_2/**: Primitive generation phase
- **phase_3/**: LoRA training and evaluation phase
- **phase_4/**: Final evaluation phase
- **primitve_storage/**: Storage for primitive data
- **results/**: Output files and results from different phases
