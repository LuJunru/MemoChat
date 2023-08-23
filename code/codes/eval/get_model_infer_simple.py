import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from optimum.bettertransformer import BetterTransformer
import torch
import os
import json
from tqdm import tqdm
import ray

q_pre = "<s>\n"
qa_link = "\n"
a_pos = "\n</s>"
MaxLen = 2048
TarLen = 512
TaskTarLen = {
    "chatting_dialogsum": MaxLen,
    "chatting_alpacagpt4": MaxLen,
    "writing_topiocqa": TarLen // 2,
    "writing_dialogsum": TarLen,
    "retrieval_dialogsum": 32,
    "retrieval_topiocqa": 32
}

def get_gpu_memory(ray_num_gpus):
    """Get available memory for each GPU."""
    gpu_memory = []
    for gpu_id in range(ray_num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory

def run_eval(model_path, model_id, question_file, answer_file, num_gpus, load_in_8bit, ray_num_gpus):
    assert num_gpus % ray_num_gpus == 0

    # split question file into num_gpus files
    ques_jsons = []
    with open(os.path.expanduser(question_file), "r") as ques_file:
        for line in ques_file:
            ques_jsons.append(line)

    chunk_size = len(ques_jsons) // (num_gpus // ray_num_gpus)
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
        get_answers_func = ray.remote(num_gpus=ray_num_gpus)(
            get_model_answers
        ).remote
        ans_handles.append(
            get_answers_func(
                model_path, model_id, ques_jsons[i: i + chunk_size], ray_num_gpus, load_in_8bit
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    with open(os.path.expanduser(answer_file), "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")

@torch.inference_mode()
def get_model_answers(model_path, model_id, question_jsons, ray_num_gpus, load_in_8bit):
    model_path = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, truncation_side='left')

    available_gpu_memory = get_gpu_memory(ray_num_gpus)
    gpu_memory_dict = {i: str(int(available_gpu_memory[i] * 0.85)) + "GiB" for i in range(ray_num_gpus)}
    gpu_memory_dict["cpu"] = "0GiB"

    if "t5" in model_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", max_memory=gpu_memory_dict, load_in_8bit=load_in_8bit
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", max_memory=gpu_memory_dict, load_in_8bit=load_in_8bit
        )

    # Initialize with BetterTransformer, injecting Flash-Attention
    model = BetterTransformer.transform(model)

    # turn on eval mode to stop batch normalizarion & dropout, can work together with torch.inference_mode
    model = model.eval()

    ans_jsons = []
    for i, line in enumerate(tqdm(question_jsons)):
        ques_json = json.loads(line)
        idx = ques_json["question_id"]
        qs = q_pre + ques_json["text"] + qa_link

        task_type = ques_json["type"]
        if "t5" in model_path:
            input_ids = tokenizer([qs], max_length=MaxLen, truncation=True, add_special_tokens=False).input_ids
            target_len = TaskTarLen[task_type]
        else:
            input_ids = tokenizer([qs], max_length=(MaxLen - TarLen), truncation=True, add_special_tokens=False).input_ids
            target_len = min(len(input_ids[0]) + TaskTarLen[task_type], MaxLen)

        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.2,
            max_length=target_len
        )
        if "t5" in model_path:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        print(outputs)

        ans_jsons.append(
            {
                "question_id": idx,
                "text": outputs,
                "model_id": model_id,
                "metadata": {},
            }
        )
    return ans_jsons

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--ray-num-gpus", type=int, default=1)
    parser.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()

    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(num_gpus=args.num_gpus)

    run_eval(
        args.model_path,
        args.model_id,
        args.question_file,
        args.answer_file,
        args.num_gpus,
        args.load_in_8bit,
        args.ray_num_gpus
    )
