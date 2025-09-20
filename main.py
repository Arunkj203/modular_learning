# Main code file

from phase_1.phase_1_main import run_phase1
from phase_2.phase_2_main import run_phase2
from phase_3.phase_3_main import run_phase3
from phase_4.phase_4_main import run_phase4



from config import *
from datasets import load_dataset


# Load SVAMP dataset
dataset = load_dataset(DATASET_SVAMP)


for problem in dataset['train']:

    '''  Phase 1: Problem Analysis'''

    processed, analysis = run_phase1(problem, dataset_name="SVAMP")


    '''  Phase 2: Primitive Generation  '''

    primitive_sequence , new_primitives_to_train = run_phase2(processed["question"], analysis)


    '''  Phase 3: Primitive Training and Testing  '''

    res_phase_3 = run_phase3()

    '''  Phase 4: Problem Solving '''

    res_phase_4 = run_phase4()

    print("Solutions:", res_phase_4)
