from typing import List

import torch
import tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import ipdb
import tqdm

tokenizers_version = tuple(int(n) for n in tokenizers.__version__.split('.'))
if tokenizers_version < (0, 12, 1):
    print("warning: Your tokenizers version looks old and you will likely have formatting issues. We recommend installing tokenizers >= 0.12.1")

# set BIG_MODEL to use the 6.7B parameter model
BIG_MODEL = True

# use a GPU
CUDA = True

# print intermediate outputs of infilling
VERBOSE = False

if BIG_MODEL:
    model_name = "facebook/incoder-6B"

    # the arguments added below will load a half precision version of the model,
    # which requires less RAM than loading the full float32 version.  this 
    # should fit in ~16GB of RAM
    # NOTE: half precision should *not* be used if you plan to fine-tune the
    # model. You'll need full precision and a lot of GPU memory. We have not
    # tested fine-tuning in `transformers` (the model was trained in fairseq)
    if CUDA:
        kwargs = dict(
            revision="float16", 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        kwargs = dict(
            low_cpu_mem_usage=True,
        )
else:
    model_name = "facebook/incoder-1B"
    kwargs = {}


# signals the start of a document
BOS = "<|endoftext|>"
# signals the end of a generated infill
EOM = "<|endofmask|>"
PAD = "[pad]"

print("loading model")
model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("loading complete")

tokenizer.add_special_tokens({"pad_token": PAD})
tokenizer.padding_side = 'right'
model.resize_token_embeddings(len(tokenizer))

if CUDA:
    # if you plan to fine-tune the model, you should not use half precision.
    model = model.half().cuda()

def make_sentinel(i):
    # signals (1) a location to insert an infill and (2) the start of the infill generation
    return f"<|mask:{i}|>"

def _generate(inputs: List[str], max_to_generate: int, temperature: float, num_return: int):
    """
    Do standard left-to-right completion of the prefix `input` by sampling from the model
    """
    # tokenized = tokenizer(inputs, return_tensors="pt", padding='max_length', max_length=100)
    tokenized = tokenizer(inputs, return_tensors="pt")
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    if CUDA:
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
    max_length = max_to_generate + input_ids.size(1)
    if max_length > 2048:
        print("warning: max_length {} is greater than the context window {}".format(max_length, 2048))
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, do_sample=True, top_p=0.95, temperature=temperature,
                                max_length=max_length, num_return_sequences=num_return, attention_mask=attention_mask)
    # pass clean_up_tokenization_spaces=False to avoid removing spaces before punctuation, e.g. "from ." -> "from."
    decoded_strs = tokenizer.batch_decode(output, clean_up_tokenization_spaces=False)

    # manually remove the BOS and PAD tokens
    handled_strs = []
    for hypo_str in decoded_strs:
        # while hypo_str.startswith(PAD):
        #     hypo_str = hypo_str[len(PAD):]
        if hypo_str.startswith(BOS):
            hypo_str = hypo_str[len(BOS):]
        handled_strs.append(hypo_str)
    
    return handled_strs

def batch_generate(inputs: List[str], max_to_generate: int, temperature: float, num_return: int):
    """
    控制显存占用, 每次生成batch_size个句子
    """
    # if num_return > 50:
    #     raise ValueError("num_return must be <= 80")
    batch_size = 1
    print("batch_size:", batch_size)
        
    result = []
    for i in tqdm.tqdm(range(0, len(inputs), batch_size)):
        result.extend(_generate(inputs[i:i+batch_size], max_to_generate=max_to_generate, temperature=temperature, num_return=num_return))
    return result

def build_prompt_step1_from_parts(parts: List[str], extra_sentinel: bool=True):
    """
    extra_sentinel: bool. we recommend setting this to True, as it makes it
        easier for the model to end generated infills. See the footnote in 
        section 2.2 of our paper for details.
    """
    prompt = ""
    for sentinel_ix, part in enumerate(parts):
        prompt += part
        if extra_sentinel or (sentinel_ix < len(parts) - 1):
            prompt += make_sentinel(sentinel_ix)
    return prompt

def handle_infill_EOM(infills: List[str]):
    flag = False
    handled_infills = []
    for infill in infills:
        if EOM not in infill:
            flag = True
            infill += EOM
        handled_infills.append(infill[:infill.index(EOM) + len(EOM)])
    if flag:
        print("warning: some infills were not ended")
    return handled_infills

def combine_parts_and_infills_two_steps(contexts: List[str], infills_step_1: List[str], infills_step_2: List[str], num_return: List[int]):
    combined_text = []
    for i in range(len(infills_step_2)):
        text = ''
        parts = contexts[i//(num_return[0]*num_return[1])].split('<insert>')
        text += parts[0]
        text += infills_step_1[i//num_return[1]][:-len(EOM)]
        text += parts[1]
        text += infills_step_2[i][:-len(EOM)]
        text += parts[2]
        combined_text.append(text)
    return combined_text

def infill_two_steps(contexts: List[str], max_to_generate: List[int]=[100, 150], temperature: List[float]=[0.8, 0.8], num_return: List[int]=[50, 2]):
    prompts_step_1 = []  # length N
    for context in contexts:
        prompts_step_1.append(build_prompt_step1_from_parts(context.split("<insert>")))
    
    prompts_step_1 = [prompt + make_sentinel(0) for prompt in prompts_step_1]
    
    completions_step_1 = batch_generate(
        prompts_step_1,
        max_to_generate=max_to_generate[0],
        temperature=temperature[0],
        num_return=num_return[0]
    )  # length N*num_return[0]

    # length N*num_return[0]
    infill_content_step_1 = [completions_step_1[i][len(prompts_step_1[i//num_return[0]]):] for i in range(len(completions_step_1))]
    infill_content_for_step_2 = handle_infill_EOM(infill_content_step_1)
    
    prompts_step_2 = [prompts_step_1[i//num_return[0]] + infill_content_for_step_2[i] + make_sentinel(1) for i in range(len(completions_step_1))]

    completions_step_2 = batch_generate(
        prompts_step_2,
        max_to_generate=max_to_generate[1],
        temperature=temperature[1],
        num_return=num_return[1]
    )  # length N*num_return[0]*num_return[1]
    
    infill_content_step_2 = [completions_step_2[i][len(prompts_step_2[i//(num_return[1])]):] for i in range(len(completions_step_2))]
    infill_content_for_step_3 = handle_infill_EOM(infill_content_step_2)
    
    final_generated_content = combine_parts_and_infills_two_steps(contexts, infill_content_for_step_2, infill_content_for_step_3, num_return)
    clone_num = num_return[0]*num_return[1]
    return final_generated_content, clone_num

def combine_parts_and_infills_one_step(contexts: List[str], infills_step_1: List[str], num_return: int):
    combined_text = []
    for i in range(len(infills_step_1)):
        text = ''
        parts = contexts[i//num_return].split('<insert>')
        text += parts[0]
        text += infills_step_1[i][:-len(EOM)]
        text += parts[1]
        combined_text.append(text)
    return combined_text

def infill_one_step(contexts: List[str], max_to_generate: int=150, temperature: float=0.8, num_return: int=50):
    prompts_step_1 = []  # length N
    for context in contexts:
        prompts_step_1.append(build_prompt_step1_from_parts(context.split("<insert>")))
    
    prompts_step_1 = [prompt + make_sentinel(0) for prompt in prompts_step_1]
    
    completions_step_1 = batch_generate(
        prompts_step_1,
        max_to_generate=max_to_generate,
        temperature=temperature,
        num_return=num_return
    )  # length N*num_return[0]
    
    # length N*num_return[0]
    infill_content_step_1 = [completions_step_1[i][len(prompts_step_1[i//num_return]):] for i in range(len(completions_step_1))]
    infill_content_for_step_2 = handle_infill_EOM(infill_content_step_1)
    
    final_generated_content = combine_parts_and_infills_one_step(contexts, infill_content_for_step_2, num_return)
    clone_num = num_return
    return final_generated_content, clone_num

if __name__ == "__main__":
   #     example = '''\
# def <insert>
#     """ Count the number of occurrences of each word in the file. """
#     <insert>
# <|/ file |>'''
#     result = infill_two_steps([example])
    example = '''\
def count_word():
    """ Count the number of occurrences of each word in the file. """
    <insert>
<|/ file |>'''
    result = infill_one_step([example])
    ipdb.set_trace()
