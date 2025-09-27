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
├── test_lora_code.py
├── utils.py
├── Mock_Dataset/
│   ├── test_data.json
│   └── train_data.json
├── others/
│   ├── __init__.py
│   ├── Calc_token.py
│   └── workflow.txt
├── phase_1/
│   ├── __init__.py
│   ├── analyze_problems.py
│   ├── phase_1_main.py
│   └── preprocess.py
├── phase_2/
│   ├── __init__.py
│   ├── generate_primitive.py
│   └── phase_2_main.py
├── phase_3/
│   ├── __init__.py
│   ├── lora_training.py
│   ├── phase_3_main.py
│   └── synthetic_data.py
├── phase_4/
│   ├── __init__.py
│   ├── evaluate.py
│   └── phase_4_main.py
└── results/
    ├── primitive_library_1.json
    ├── primitive_summary.txt
    └── svamp_primitives.txt
```

## Directory Descriptions

- **Mock_Dataset/**: Mock dataset files for testing and training
- **others/**: Miscellaneous utility files and documentation
- **phase_1/**: Initial problem analysis and preprocessing phase
- **phase_2/**: Primitive generation phase
- **phase_3/**: LoRA training and evaluation phase
- **phase_4/**: Final evaluation phase
- **results/**: Output files and results from different phases
