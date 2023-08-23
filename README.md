# MemoChat
MemoChat: Tuning LLMs to Use Memos for Consistent Long-Range Open-domain Conversation 

## Environment
We provide [core_requirement.txt](core_requirement.txt) for your convenience.

## Model Weights
The initial models we used are [fastchat models (v1.3)](https://lmsys.org/blog/2023-03-30-vicuna/). Below are the model weights of our fine-tuned version. Our models are built upon Fastchat modles, thus we adopt same cc-by-nc-sa-4.0 license.

| Name | Share Link |
| --- | --- |
| MemoChat-Fastchat-T5-3B | https://huggingface.co/Junrulu/MemoChat-Fastchat-T5-3B |
| MemoChat-Vicuna-7B | https://huggingface.co/Junrulu/MemoChat-Vicuna-7B |
| MemoChat-Vicuna-13B | https://huggingface.co/Junrulu/MemoChat-Vicuna-13B |
| MemoChat-Vicuna-33B | https://huggingface.co/Junrulu/MemoChat-Vicuna-33B |

## Workflow
`RootPath` is the absolute path of this repo. Download initial models and put them in [model](model) folder.
### Instruction Tuning
```
Run `bash code/scripts/tuning.sh RootPath`. Intermediate evaluation are included in this script as well.
```

### MemoChat Testing
```
Run `bash code/scripts/memochat.sh RootPath` for pipeline testing with fine-tuned models. 
Run `bash code/scripts/memochat_gpt.sh RootPath` for pipeline testing with GPT3.5 API.
Run `bash code/scripts/llm_judge.sh RootPath` for GPT4 judge (openai api is required).
```

### Our Results
We provide our prediction results [here](https://drive.google.com/file/d/1jGNhT3iPXEA8B2fXHZ2Einy1AMre-8xB/view?usp=sharing).

## Acknowledgement
We thank [Vicuna project](https://github.com/lm-sys/FastChat/tree/main) for their great work.

## Citation
```
@misc{lu2023memochat,
      title={MemoChat: Tuning LLMs to Use Memos for Consistent Long-Range Open-Domain Conversation}, 
      author={Junru Lu and Siyu An and Mingbao Lin and Gabriele Pergola and Yulan He and Di Yin and Xing Sun and Yunsheng Wu},
      year={2023},
      eprint={2308.08239},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
