import ipdb
import json
import tqdm
from collections import defaultdict

from utils import Tools
from example_usage import infill_one_step, infill_two_steps


INSTRUCTION = '''\
    <insert>

<|/ file |>'''

INSTRUCTION_2 = '''\
    # step-by-step solution description:
    # 1. <insert>
    
    # solution code here:
    <insert>

<|/ file |>'''

class CodeGenerator:
    def __init__(self, source_path, target_path, exp_type):
        self.source_path = source_path
        self.target_path = target_path
        self.exp_type = exp_type

    def generate(self):
        source_lines = list(Tools.load_jsonl(self.source_path))
        prompts = []
        problem_descriptions = []        
        for line in source_lines:
            prompt = line['prompt']
            problem_descriptions.append(prompt)
            if self.exp_type == 'one_step':
                prompts.append(prompt + INSTRUCTION)
            elif self.exp_type == 'two_steps':
                prompts.append(prompt + INSTRUCTION_2)
        
        # prompts = prompts[:1]
        
        if self.exp_type == 'one_step':
            generated_lines, clone_num = infill_one_step(
                prompts
            )
        elif self.exp_type == 'two_steps':
            generated_lines, clone_num = infill_two_steps(
                prompts
            )
        
        results = defaultdict(list)
        for i in range(len(generated_lines)):
            problem_desc = problem_descriptions[i//clone_num]
            generated = generated_lines[i]
            if not generated.startswith(problem_desc):
                ipdb.set_trace()
            generated = generated[len(problem_desc):][:-len('<|/ file |>')]
            results[problem_desc].append(generated)
        
        result_list = []
        for problem_desc in results.keys():
            result_list.append({
                'prompt': problem_desc,
                'samples': results[problem_desc]
            })
        
        Tools.dump_jsonl(self.target_path, result_list)

    def aggregate(self, file_paths, target_path):
        results_dict = defaultdict(list)
        for file_path in file_paths:
            lines = list(Tools.load_jsonl(file_path))
            for line in lines:
                prompt = line['prompt']
                samples = line['samples']
                results_dict[prompt].extend(samples)
        result_list = []
        for prompt in results_dict.keys():
            result_list.append({
                'prompt': prompt,
                'samples': results_dict[prompt]
            })
        Tools.dump_jsonl(target_path, result_list)

if __name__ == "__main__":
    generator = CodeGenerator(
        source_path="data/hmeval/delTestCase_problems.jsonl",
        target_path="incoder_temp0.8_num50_2.jsonl",
        exp_type="two_steps"
    )
    generator.generate()
    # generator.aggregate(['incoder_temp0.8_num50_2.jsonl', 'incoder_temp0.8_num50_1.jsonl'], 'incoder_temp0.8_num50.jsonl')
