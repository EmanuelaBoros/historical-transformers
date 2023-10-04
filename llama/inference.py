# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time
from tqdm import tqdm
import torch
from transformers import LlamaTokenizer

from inference.safety_utils import get_safety_checker
from inference.model_utils import load_model, load_peft_model
from ft_datasets.hipe_dataset import PROMPT_DICT
import jsonlines
def main(
        model_name,
        peft_model: str = None,
        quantization: bool = False,
        max_new_tokens=100,  # The maximum numbers of tokens to generate
        prompt_file: str = None,
        seed: int = 42,  # seed value for reproducibility
        do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
        min_length: int = None,  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
        use_cache: bool = True,
        # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
        top_p: float = 1.0,
        # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature: float = 1.0,  # [optional] The value used to modulate the next token probabilities.
        top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
        length_penalty: int = 1,
        # [optional] Exponential penalty to the length that is used with beam-based generation.
        enable_azure_content_safety: bool = False,  # Enable safety check with Azure content safety api
        enable_sensitive_topics: bool = False,  # Enable check for sensitive topics using AuditNLG APIs
        enable_salesforce_content_safety: bool = True,  # Enable safety check with Salesforce safety flan t5
        max_padding_length: int = None,  # the max padding length to be used with tokenizer padding the prompts.
        use_fast_kernels: bool = False,
        # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
        **kwargs
):

    safe_model_name = model_name.replace("/", "_")
    safe_peft_model = peft_model.replace("/", "_") if peft_model else "None"
    # Constructing the filename using hyperparameters:
    output_filename = os.path.join('data/hipe/results/',
        f"model_{safe_model_name}_peft_{safe_peft_model}_max_new_tokens_{max_new_tokens}_" +\
        f"seed_{seed}_do_sample_{do_sample}_min_length_{min_length}_" +\
        f"use_cache_{use_cache}_top_p_{top_p}_temperature_{temperature}_" +\
        f"max_padding_{max_padding_length}.jsonl")

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    safety_checker = get_safety_checker(enable_azure_content_safety,
                                        enable_sensitive_topics,
                                        enable_salesforce_content_safety,
                                        )

    # Read data
    data_test = []
    for input_file in ['data/hipe/HIPE-2022-v2.1-hipe2020-test-fr_universal.jsonl']:
        with jsonlines.open(input_file, 'r') as f:
            for line in f:
                data_test.append(line)

    for ann in tqdm(data_test, total=len(data_test)):
        if ann.get("tokens", "") == "":
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)

        # user_prompt = prompt + ann["tokens"]

        # # Safety check of the user prompt
        # safety_results = [check(user_prompt) for check in safety_checker]
        # are_safe = all([r[1] for r in safety_results])
        # if are_safe:
        #     print("User prompt deemed safe.")
        #     print(f"User prompt:\n{user_prompt}")
        # else:
        #     print("User prompt deemed unsafe.")
        #     for method, is_safe, report in safety_results:
        #         if not is_safe:
        #             print(method)
        #             print(report)
        #     print("Skipping the inference as the prompt is not safe.")
        #     sys.exit(1)  # Exit the program with an error status

        # Convert the prompt to model's input format
        batch = tokenizer(prompt, padding='max_length', truncation=True, max_length=max_padding_length,
                          return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}

        # Generate output
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )

        e2e_inference_time = (time.perf_counter() - start) * 1000
        e2e_inference_time_minutes = e2e_inference_time / 60000  # Convert ms to minutes
        # print(f"The inference time is {e2e_inference_time_minutes:.2f} minutes")

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if prompt in output_text:
            output_text = output_text.replace(prompt, '').strip()
        # Append the output_text to a jsonlines file
        with jsonlines.open(output_filename, mode='a') as writer:
            writer.write({"tokens": ann["tokens"],
                          "tags": ann["tags"],
                          "pred_tags": output_text,
                          "inference_time_mins": e2e_inference_time_minutes})



if __name__ == "__main__":
    fire.Fire(main)
