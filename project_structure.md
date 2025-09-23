# Project Structure

```
modular_learning/
├── config.py
├── main.py
├── README.md
├── requirements.txt
├── task_controller.py
├── utils.py
├── models_config/
│   └── model_config.py
├── others/
│   ├── Calc_token.py
│   └── workflow.txt
├── phase_1/
│   ├── analyze_problems.py
│   ├── phase_1_main.py
│   └── preprocess.py
├── phase_2/
│   ├── generate_primitive.py
│   ├── phase_2_main.py
│   └── retrieve_primitives.py
├── phase_3/
│   ├── evaluate_lora.py
│   ├── lora_training.py
│   ├── phase_3_main.py
│   └── synthetic_data.py
├── phase_4/
│   ├── evaluate.py
│   ├── phase_4_main.py
└── results/
    ├── primitive_library_1.json
    ├── primitive_summary.txt
    └── svamp_primitives.txt
```

## Directory Descriptions

- **models_config/**: Configuration files for different models
- **others/**: Miscellaneous utility files and documentation
- **phase_1/**: Initial problem analysis and preprocessing phase
- **phase_2/**: Primitive generation phase
- **phase_3/**: LoRA training and evaluation phase
- **phase_4/**: Final evaluation phase
- **results/**: Output files and results from different phases
