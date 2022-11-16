from utils import Tools

class CodeViewer:
    def __init__(self, sample_path):
        self.sample_path = sample_path
    
    def view(self):
        samples = list(Tools.load_jsonl(self.sample_path))
        for sample in samples:
            print(sample['samples'][0])
            input()

class Trimer:
    def __init__(self, sample_path, new_sample_path):
        self.lines = list(Tools.load_jsonl(sample_path))
        self.new_sample_path = new_sample_path
    
    def trim(self):
        new_lines = []
        for line in self.lines:
            sample = line['samples'][0]
            prompt = line['prompt']
            new_sample = sample[len(prompt):][:-len('<|/ file |>')]
            new_lines.append({
                'prompt': prompt,
                'samples': [new_sample]
            })
        Tools.dump_jsonl(self.new_sample_path, new_lines)


if __name__ == "__main__":
    # CodeViewer(sample_path="data/hmeval/sampled/incoder_temp0_sample_num1_raw.jsonl").view()
    Trimer(sample_path="data/hmeval/sampled/incoder_temp0_sample_num1_raw.jsonl",
           new_sample_path="data/hmeval/sampled/incoder_temp0_sample_num1_raw_trim.jsonl").trim()