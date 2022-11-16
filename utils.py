import json

class Tools:
    @staticmethod
    def load_jsonl(path):
        with open(path, "r", encoding='utf8') as f:
            for line in f:
                yield json.loads(line)
    
    @staticmethod
    def dump_jsonl(path, data):
        with open(path, "w", encoding='utf8') as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
