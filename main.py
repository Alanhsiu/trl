# %% [markdown]
# ### Import Libraries

# %%
import sys
current_base_path = '/home/b09901066/trl'
sys.path.append(f"{current_base_path}/vc")
sys.path.append(f"{current_base_path}/CLAPS")

import importlib
import torch
import os
import math
import numpy as np
import random
import time
import json
from tqdm import tqdm
from types import SimpleNamespace
from datetime import datetime
from typing import List, Tuple
import string


import vc
importlib.reload(vc)
from vc.trainer_encodec_vc_inference import (
    pack_inputs_v2,
    get_ar_prediction_audio_batch
)
from vc.encodec_model.nar_bart_model import NARBartForConditionalGeneration

from transformers import BartForConditionalGeneration, AutoTokenizer
from trl import (
    DPOTrainer,
    DPOConfig,
    AutoModelForSeq2SeqLMWithValueHead,
    create_reference_model
)
from datasets import Dataset

from dpo_eval import (
    get_reward_claps,
    get_reward_asr,
    eval_dpo_claps_asr_batch,
    convert_array_to_tensor_format
)
from CLAPS.inference import load_model

import argparse
from faster_whisper import WhisperModel
from torch.utils.tensorboard import SummaryWriter
import logging

# %%
# Define paths and device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# Timestamp
now = datetime.now()
ts = now.strftime("%m%d-%H%M")
print("timestamp:", ts)

# Output paths
model_output_dir = os.path.join(current_base_path, "model_output", ts)
agent_output_dir = os.path.join(current_base_path, "output", ts)
os.makedirs(model_output_dir, exist_ok=True)
os.makedirs(agent_output_dir, exist_ok=True)

# Initialize TensorBoard SummaryWriter
log_dir = f"{model_output_dir}/tensorboard_logs"
writer = SummaryWriter(log_dir=log_dir)

# %% [markdown]
# ### Utility Functions

# %%
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# %%
import shutil
from pathlib import Path
import soundfile as sf

def generate_output_batch(
        ar_model, 
        nar_model, 
        ar_tokenizer, 
        nar_tokenizer, 
        clap_model,
        asr_model,
        accelerator,
        src_encodec: list, 
        instruction: list, 
        transcription: list,
        args_predict: SimpleNamespace, 
        episode_counter: int = 0, 
        base_path: str = current_base_path, 
        temperature: float = 1.0
) -> tuple[list[float], list[str]]:
    audio_list, decode_ar_list = get_ar_prediction_audio_batch(
        args_predict, ar_model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, episode_counter, temperature=temperature
    )
    reward_list, tokenized_decode_ar_list = [], []
    valid_audio_paths = []

    # temp_dir_path = Path("/dev/shm/temp_audio_files")
    # temp_dir_path.mkdir(parents=True, exist_ok=True)

    for i, audio in enumerate(audio_list):
        if audio is not None:
            tensor_audio = convert_array_to_tensor_format(audio)
            if tensor_audio[0].shape[0] == 1:
                tensor_audio[0] = tensor_audio[0].squeeze(0)
            clap_reward = get_reward_claps(
                clap_model=clap_model, accelerator=accelerator, prompts=instruction[i], wavs=tensor_audio
            )
            output_path_ckpt = args_predict.output_path.replace(".wav", f"_generate_{episode_counter}_item_{i}.wav")
            sf.write(output_path_ckpt, np.ravel(audio), samplerate=24000)
            asr_reward = get_reward_asr(file_path=output_path_ckpt, asr_model=asr_model, ground_truth=transcription[i])

            # final_reward = clap_reward * asr_reward
            # final_reward = clap_reward*0.9 + asr_reward*0.1
            final_reward = clap_reward*0.6 + asr_reward*0.4
            reward_list.append(final_reward)
            valid_audio_paths.append(output_path_ckpt)
        else:
            reward_list.append(0)

    for decode_ar in decode_ar_list:
        list_decode_ar = decode_ar.flatten().tolist()
        filtered_decode_ar_list = list_decode_ar[2:-1]
        decode_ar_tokens = ar_tokenizer.convert_ids_to_tokens(filtered_decode_ar_list)
        tokenized_decode_ar = ar_tokenizer.convert_tokens_to_string(decode_ar_tokens)
        tokenized_decode_ar_list.append(tokenized_decode_ar)

    return reward_list, tokenized_decode_ar_list

def extract_data_from_json(file_path: str, extract_number: int, asr_model) -> Tuple[List[list], List[str], List[list]]:
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    with open(f"{current_base_path}/dataset/emotional_instructions.json", 'r') as f:
        emotional_instructions = json.load(f)
    
    all_src_encodec = []
    all_instruction = []
    all_transcription = []
    count = 0
    
    for item in data:
        src_encodec = item["src_encodec"]
        instruction = random.choice(emotional_instructions)
        # transcription = item["transcription"]
        transcription = ''.join([char for char in item["transcription"] if char not in string.punctuation]).lower().strip()
        if len(src_encodec[0]) <= 100:
            valid_transcriptions = ['neighboring fields', 'but i will be in a minute', 'will you', 'why is it', 'what more do you want', 'why not', 'the wind wakened me', 'yes', 'hush pearl hush', 'good night husband', 'what music', 'shut up will you', 'here he is', 'that was strange']
            if transcription not in valid_transcriptions:
                continue
            
            all_src_encodec.append(src_encodec)
            all_instruction.append(instruction)
            all_transcription.append(transcription)
        
            count += 1
            print(f"Count: {count}, Transcription: {transcription}")

        if count == extract_number:
            break
        
    # all_transcription = [''.join([char for char in transcription if char not in string.punctuation]).lower().strip() for transcription in all_transcription]

    return all_src_encodec, all_instruction, all_transcription

def train_model(
        model,
        model_ref,
        ar_tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        model_output_dir: str,
        beta: float,
        resume_from_checkpoint: bool,
        model_checkpoint: str,
        learning_rate: float = 5e-07,
        num_train_epochs: int = 200,
        max_length: int = 1024*9,
        max_prompt_length: int = 1024*9,
        max_target_length: int = 1024*9,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        seed: int = 42,
        iteration: int = 0
) -> None:
    training_args = DPOConfig(
        beta=beta,
        output_dir=model_output_dir,
        resume_from_checkpoint=model_checkpoint if resume_from_checkpoint else None,
        seed=seed,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        max_target_length=max_target_length,
        evaluation_strategy="steps",
        save_steps=5000,
        logging_dir=f"{model_output_dir}/logs"
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        tokenizer=ar_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss_type="sigmoid", # "sigmoid", "hinge", "ipo", "kto_pair", "bco_pair"
        # log_fn=logging.info,  # Pass logging function
        # writer=writer,        # Pass TensorBoard writer
    )
    trainer.train()

    print(f"Trainer State: {trainer.state}")
    print(f"Trainer State Global Step: {trainer.state.global_step}, Trainer State Log History: {trainer.state.log_history}")
    
    for log in trainer.state.log_history:
        if "train_loss" in log:
            iteration_loss = log["train_loss"]
            writer.add_scalar("Loss/Iteration", iteration_loss, iteration)
            logging.info(f"Iteration: {iteration}, Loss: {iteration_loss}")
            print(f"Iteration: {iteration}, Loss: {iteration_loss}")
        else:
            print(f"Log: {log}")

    model.config.to_json_file(f"{model_output_dir}/config.json")


# %%
def process_data_batch(
    sample_size: int, 
    ar_model, 
    nar_model, 
    ar_tokenizer, 
    nar_tokenizer, 
    clap_model,
    asr_model,
    accelerator,
    selected_src_encodec: List[list], 
    selected_instruction: List[str],
    selected_transcription: List[str],
    args_predict, 
    base_path: str = current_base_path, 
    temperature: float = 1.0, 
    iteration: int = 0,
    prev_eval_avg: float = 0,
    strategy: str = "top_bottom_percent"  # Default to original strategy. Options: "max_min", "top_bottom_percent", "above_below_average", "above_prev_eval"
) -> Tuple[List[str], List[str], List[str], List[float], List[float], List[float]]:
    # Ensure sample size is valid
    if sample_size < 2:
        raise ValueError("Parameter 'sample_size' must be greater than 1.")

    chosen, rejected, prompts, chosen_rewards, rejected_rewards, average_rewards = [], [], [], [], [], []

    disable_tqdm = not os.isatty(1)
    for i in tqdm(range(len(selected_src_encodec)), desc="Processing Data", disable=disable_tqdm):
        rewards, tokenized_outputs = [], []
        size_of_packed_input = (
            len(selected_src_encodec[i][0]) +
            len(ar_tokenizer(selected_instruction[i])["input_ids"][1:-1]) +
            3
        )
        if 4 < size_of_packed_input <= 1024:
            selected_src_encodec_list = [selected_src_encodec[i]] * sample_size
            selected_instruction_list = [selected_instruction[i]] * sample_size
            selected_transcription_list = [selected_transcription[i]] * sample_size
            rewards, tokenized_outputs = generate_output_batch(
                ar_model=ar_model, 
                nar_model=nar_model, 
                ar_tokenizer=ar_tokenizer, 
                nar_tokenizer=nar_tokenizer,
                src_encodec=selected_src_encodec_list,
                instruction=selected_instruction_list, 
                transcription=selected_transcription_list,
                clap_model=clap_model,
                asr_model=asr_model,
                accelerator=accelerator,
                args_predict=args_predict,
                episode_counter=f"data_{i}",
                base_path=base_path, 
                temperature=temperature
            )

        valid_rewards = [r for r in rewards if r is not None]
        valid_outputs = [tokenized_outputs[j] for j in range(len(rewards)) if rewards[j] is not None]

        if len(valid_rewards) >= 2:
            average_reward = np.mean(valid_rewards)
            print(f"Average reward for data index {i}: {average_reward}")

            if strategy == "max_min":
                # Original max-min strategy
                max_reward_index = np.argmax(valid_rewards)
                min_reward_index = np.argmin(valid_rewards)
                chosen_outputs = [valid_outputs[max_reward_index]]
                rejected_outputs = [valid_outputs[min_reward_index]]
                chosen_rewards_part = [valid_rewards[max_reward_index]]
                rejected_rewards_part = [valid_rewards[min_reward_index]]

            elif strategy == "top_bottom_percent":
                # Select top and bottom 20% of rewards
                twenty_percent_num = max(1, math.ceil(len(valid_rewards) * 0.2))
                max_indices = np.argsort(valid_rewards)[-twenty_percent_num:]
                min_indices = np.argsort(valid_rewards)[:twenty_percent_num]

                chosen_outputs = [valid_outputs[j] for j in max_indices]
                rejected_outputs = [valid_outputs[j] for j in min_indices]
                chosen_rewards_part = [valid_rewards[j] for j in max_indices]
                rejected_rewards_part = [valid_rewards[j] for j in min_indices]

            elif strategy == "above_below_average":
                # Select rewards above and below the average
                threshold = 0.05
                chosen_outputs = [valid_outputs[j] for j in range(len(valid_rewards)) if valid_rewards[j] > average_reward + threshold]
                rejected_outputs = [valid_outputs[j] for j in range(len(valid_rewards)) if valid_rewards[j] < average_reward - threshold]

                # Sort and trim to ensure balanced chosen and rejected outputs
                chosen_outputs = [x for _, x in sorted(zip(valid_rewards, chosen_outputs), reverse=True)]
                rejected_outputs = [x for _, x in sorted(zip(valid_rewards, rejected_outputs))]

                min_length = min(len(chosen_outputs), len(rejected_outputs))
                chosen_outputs = chosen_outputs[:min_length]
                rejected_outputs = rejected_outputs[:min_length]

                chosen_rewards_part = [valid_rewards[j] for j in range(len(valid_rewards)) if valid_rewards[j] > average_reward][:min_length]
                rejected_rewards_part = [valid_rewards[j] for j in range(len(valid_rewards)) if valid_rewards[j] < average_reward][:min_length]

            elif strategy == "above_prev_eval":
                # Select rewards above and below a previous evaluation average
                chosen_outputs = [valid_outputs[j] for j in range(len(valid_rewards)) if valid_rewards[j] > prev_eval_avg]
                rejected_outputs = [valid_outputs[j] for j in range(len(valid_rewards)) if valid_rewards[j] < prev_eval_avg]

                # Sort and trim
                chosen_outputs = [x for _, x in sorted(zip(valid_rewards, chosen_outputs), reverse=True)]
                rejected_outputs = [x for _, x in sorted(zip(valid_rewards, rejected_outputs))]

                min_length = min(len(chosen_outputs), len(rejected_outputs))
                if min_length == 0:
                    chosen_outputs = [valid_outputs[np.argmax(valid_rewards)]]
                    rejected_outputs = [valid_outputs[np.argmin(valid_rewards)]]
                    chosen_rewards_part = [valid_rewards[np.argmax(valid_rewards)]]
                    rejected_rewards_part = [valid_rewards[np.argmin(valid_rewards)]]
                else:
                    chosen_outputs = chosen_outputs[:min_length]
                    rejected_outputs = rejected_outputs[:min_length]
                    chosen_rewards_part = [valid_rewards[j] for j in range(len(valid_rewards)) if valid_rewards[j] > prev_eval_avg][:min_length]
                    rejected_rewards_part = [valid_rewards[j] for j in range(len(valid_rewards)) if valid_rewards[j] < prev_eval_avg][:min_length]

            obs_input = pack_inputs_v2(ar_tokenizer, selected_src_encodec[i], selected_instruction[i])
            tokenize_input = ar_tokenizer.convert_ids_to_tokens(obs_input)
            tokenize_input_str = ar_tokenizer.convert_tokens_to_string(tokenize_input)
            prompts.extend([tokenize_input_str] * len(chosen_outputs))
            average_rewards.append(average_reward)

            chosen.extend(chosen_outputs)
            rejected.extend(rejected_outputs)
            chosen_rewards.extend(chosen_rewards_part)
            rejected_rewards.extend(rejected_rewards_part)
        else:
            print(f"Not enough valid rewards for data index {i}")

    if len(selected_src_encodec) == 1:
        chosen *= 2
        rejected *= 2
        prompts *= 2
        chosen_rewards *= 2
        rejected_rewards *= 2
        average_rewards *= 2    

    return chosen, rejected, prompts, chosen_rewards, rejected_rewards, average_rewards


def generate_data(ar_model, 
                  ar_tokenizer, 
                  nar_model, 
                  nar_tokenizer, 
                  clap_model,
                  asr_model,
                  accelerator,
                  selected_src_encodec: List[list], 
                  selected_instruction: List[str],
                  selected_transcription: List[str],
                  args_predict: SimpleNamespace, 
                  sample_size: int, 
                  iteration: int, 
                  agent_output_dir: str, 
                  base_path: str = current_base_path, 
                  temperature: float = 1.0
) -> Tuple[dict, List[float], List[float]]:
    """
    Generates data for the dataset and saves info to a JSON file.
    Returns:
        tuple:
            data_for_dataset (dict): A dictionary containing the data for the dataset.
            chosen_rewards (List[float]): A list of rewards for the chosen outputs.
            rejected_rewards (List[float]): A list of rewards for the rejected outputs.
    """
    chosen, rejected, prompts, chosen_rewards, rejected_rewards, average_rewards = process_data_batch(
        sample_size=sample_size,
        ar_model=ar_model,
        nar_model=nar_model,
        ar_tokenizer=ar_tokenizer,
        nar_tokenizer=nar_tokenizer,
        selected_src_encodec=selected_src_encodec,
        selected_instruction=selected_instruction,
        selected_transcription=selected_transcription,
        args_predict=args_predict,
        base_path=base_path,
        temperature=temperature,
        iteration = iteration,
        clap_model=clap_model,
        asr_model=asr_model,
        accelerator=accelerator
    )

    data = {
        "prompt": prompts,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "average_rewards": average_rewards
    }

    with open(f"{agent_output_dir}/data_iter_{iteration}.json", "w") as outfile:
        json.dump(data, outfile, indent=4)

    data_for_dataset = {key: data[key] for key in ["prompt", "chosen", "rejected"]}

    return data_for_dataset, chosen_rewards, rejected_rewards

def train_iteration(model, 
                    model_checkpoint,
                    iteration, 
                    data_size, 
                    sample_size, 
                    ar_model, 
                    ar_tokenizer,
                    nar_model, 
                    nar_tokenizer,
                    all_src_encodec, 
                    all_instruction, 
                    all_transcription,
                    args_predict, 
                    agent_output_dir,
                    model_output_dir_base, 
                    clap_model,
                    asr_model,
                    accelerator,
                    beta = 0.1, 
                    temperature = 1.0,
                    base_path=current_base_path,
                    resume_from_checkpoint = False,
                    learning_rate = 5e-07,
                    num_train_epochs = 100,
                    max_length = 1024*9,
                    max_prompt_length = 1024*9,
                    max_target_length = 1024*9,
                    per_device_train_batch_size = 1,
                    gradient_accumulation_steps = 1,
                    seed = 42,
):
    """
    Executes one training iteration: generates data, trains the model, and saves the output.
    """

    selected_src_encodec = all_src_encodec[:data_size]
    selected_instruction = all_instruction[:data_size]
    selected_transcription = all_transcription[:data_size]

    data_for_dataset, chosen_rewards, rejected_rewards = generate_data(ar_model=model,
                                                                        ar_tokenizer=ar_tokenizer,
                                                                        nar_model=nar_model,
                                                                        nar_tokenizer=nar_tokenizer,
                                                                        selected_src_encodec=selected_src_encodec,
                                                                        selected_instruction=selected_instruction,
                                                                        selected_transcription=selected_transcription,
                                                                        args_predict=args_predict,
                                                                        sample_size=sample_size,
                                                                        iteration=iteration,
                                                                        agent_output_dir=agent_output_dir,
                                                                        base_path=base_path,
                                                                        temperature=temperature,
                                                                        clap_model=clap_model,
                                                                        asr_model=asr_model,
                                                                        accelerator=accelerator)

    dataset = Dataset.from_dict(data_for_dataset)
    dataset_dict = dataset.train_test_split(test_size=0.1, shuffle=True, seed=seed)
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["test"]

    model_output_dir = f"{model_output_dir_base}/iter_{iteration}"
    os.makedirs(model_output_dir, exist_ok=True)

    model_ref = create_reference_model(model)
    
    train_model(
        model=model,
        model_ref=model_ref,
        ar_tokenizer=ar_tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_output_dir=model_output_dir,
        beta=beta,
        resume_from_checkpoint=resume_from_checkpoint,
        model_checkpoint=model_checkpoint,
        learning_rate = learning_rate,
        num_train_epochs = num_train_epochs,
        max_length = max_length,
        max_prompt_length = max_prompt_length,
        max_target_length = max_target_length,
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        seed = seed,
        iteration=iteration
    )

    return f"{model_output_dir}/dpo_model", chosen_rewards, rejected_rewards

# %% [markdown]
# ### ASR Model Configuration

# %%
asr_model_size = "large-v1" ## tiny, medium, large-v1, large-v2

if torch.cuda.is_available():
    asr_model = WhisperModel(asr_model_size, device="cuda", compute_type="float16")
    print(asr_model_size, "cuda", "float16")
else:
    asr_model = WhisperModel(asr_model_size, device="cpu", compute_type="int8")
    print(asr_model_size, "cpu", "int8")

# %% [markdown]
# ### Hyperparameters

# %%
# Seed
seed = 42

# Arguments
args_predict = SimpleNamespace(output_path=f"{current_base_path}/output/{ts}/example.wav", seed=seed, device=device)
ar_checkpoint = "lca0503/speech-chatgpt-base-ar-v2-epoch10-wotrans"
nar_checkpoint = "lca0503/speech-chatgpt-base-nar-v2-epoch4-wotrans"

# Model and Iterations
model_checkpoint = ar_checkpoint
sample_size = 10
num_iterations = 1000
all_data_size = 14
train_selected_indices = list(range(int(all_data_size * 0.8)))
data_size_per_iteration = len(train_selected_indices)

# Training Configuration
beta = 0.1
learning_rate = 5e-07
num_train_epochs = 3
max_length = 1024 * 9
max_prompt_length = 1024 * 9
max_target_length = 1024 * 9
per_device_train_batch_size = 8
gradient_accumulation_steps = 1

# Evaluation Configuration
eval_train = True
eval_test = True
eval_train_indices = train_selected_indices
eval_test_indices = list(range(int(all_data_size * 0.8), all_data_size))
eval_train_data_len = 1000
eval_test_data_len = len(eval_test_indices)
num_eval = 10
eval_frequency = 1

# Load data
original_src_encodec, original_instruction, original_transcription = extract_data_from_json('dataset/transformed_soxdata_encodec.json', all_data_size, asr_model)

# %% [markdown]
# ### CLAPS Model Configuration

# %%
sr = 24000
text_enc_name = "google/flan-t5-large"
text_enc_dim = 1024
text_blstm_dim = 256
speech_enc_name = "wavlm"
speech_enc_dim = 768
speech_blstm_dim = 256
rep_dim = 512
sub_dim = 0
n_sub = 1
ckpt_pth = f'{current_base_path}/CLAPS/pretrained/7d/cp_claps_blstm_m_50k_v3/cp_0045000'
project_dir = "cp_claps"

a = argparse.Namespace(
    sr=sr,
    text_enc_name=text_enc_name,
    text_enc_dim=text_enc_dim,
    text_blstm_dim=text_blstm_dim,
    speech_enc_name=speech_enc_name,
    speech_enc_dim=speech_enc_dim,
    speech_blstm_dim=speech_blstm_dim,
    rep_dim=rep_dim,
    sub_dim=sub_dim,
    n_sub=n_sub,
    ckpt_pth=ckpt_pth,
    project_dir=project_dir
)

# %% [markdown]
# # Main Functions

# %% [markdown]
# ### Load Models

# %%
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_checkpoint, return_dict=True)
ar_model = BartForConditionalGeneration.from_pretrained(ar_checkpoint)
ar_tokenizer = AutoTokenizer.from_pretrained(ar_checkpoint)
# ar_tokenizer.pad_token = ar_tokenizer.eos_token
nar_model = NARBartForConditionalGeneration.from_pretrained(nar_checkpoint)
nar_tokenizer = AutoTokenizer.from_pretrained(nar_checkpoint)
clap_model, accelerator = load_model(a)

# %% [markdown]
# ### Logging Start

# %%
log_path = f'{model_output_dir}/log_training.log'
print(f"Logging to: {log_path}")

# Set up logging
logging.basicConfig(
    filename=log_path,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Log Configuration Parameters
logging.info(
    "\n========== Configuration Parameters ==========\n"
    f"Model Configuration:\n"
    f"  ASR Model Size           : {asr_model_size}\n"
    f"  AR Model Checkpoint      : {ar_checkpoint}\n"
    f"  NAR Model Checkpoint     : {nar_checkpoint}\n"
    f"  Device                   : {device}\n"
    f"\nTraining Configuration:\n"
    f"  Number of Iterations     : {num_iterations}\n"
    f"  Data Size Per Iteration  : {data_size_per_iteration}\n"
    f"  Sample Size              : {sample_size}\n"
    f"  Learning Rate            : {learning_rate}\n"
    f"  Beta                     : {beta}\n"
    f"  Training Epochs          : {num_train_epochs}\n"
    f"  Max Sequence Length      : {max_length}\n"
    f"  Max Prompt Length        : {max_prompt_length}\n"
    f"  Max Target Length        : {max_target_length}\n"
    f"  Per Device Batch Size    : {per_device_train_batch_size}\n"
    f"  Gradient Accumulation    : {gradient_accumulation_steps}\n"
    f"  Random Seed              : {seed}\n"
    # f"\nPaths:\n"
    # f"  Model Output Directory   : {model_output_dir}\n"
    # f"  Agent Output Directory   : {agent_output_dir}\n"
    # f"  Base Path                : {current_base_path}\n"
    f"\nEvaluation Configuration:\n"
    f"  Train Data Length        : {eval_train_data_len}\n"
    f"  Test Data Length         : {eval_test_data_len}\n"
    f"  Train Indices            : {eval_train_indices}\n"
    f"  Test Indices             : {eval_test_indices}\n"
    f"  Train Evaluation Enabled : {eval_train}\n"
    f"  Test Evaluation Enabled  : {eval_test}\n"
    f"  Number of Evaluations    : {num_eval}\n"
    f"=============================================="
)
# Log Evaluation Data Details
if eval_train or eval_test:
    output_lines = []

    if eval_train:
        output_lines.append("\n========== Train Evaluation Data ==========")
        for i in eval_train_indices:
            output_lines.append(f"  Train Index {i:4d} -> Instruction: '{original_instruction[i]}' -> Transcription: '{original_transcription[i]}'")
        output_lines.append("===========================================")

    if eval_test:
        output_lines.append("\n========== Test Evaluation Data ==========")
        for i in eval_test_indices:
            output_lines.append(f"  Test Index {i:4d}  -> Instruction: '{original_instruction[i]}' -> Transcription: '{original_transcription[i]}'")
        output_lines.append("===========================================")

    logging.info("\n" + "\n".join(output_lines))


# %% [markdown]
# ### Initial Setup

# %%
# Start time
total_start_time = time.time()

# Prepare data for training
if train_selected_indices:
    batch_src_encodec = [original_src_encodec[i] for i in train_selected_indices]
    batch_instruction = [original_instruction[i] for i in train_selected_indices]
    batch_transcription = [original_transcription[i] for i in train_selected_indices]
    logging.info(f"Processing data from selected indices: {train_selected_indices}")
else:
    start_idx, end_idx = 0, data_size_per_iteration
    batch_src_encodec = original_src_encodec[start_idx:end_idx]
    batch_instruction = original_instruction[start_idx:end_idx]
    batch_transcription = original_transcription[start_idx:end_idx]
    logging.info(f"Processing data from index {start_idx} to {end_idx}")


# %%
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ["WANDB__SERVICE"] = "disabled"

# %% [markdown]
# ### Start training iterations

# %%
disable_tqdm = not os.isatty(1)

def evaluate_iteration(eval_type, iteration, eval_indices, eval_data_len, is_baseline=False):
    metrics, rewards, all_claps_rewards, all_asr_rewards = eval_dpo_claps_asr_batch(
        nar_model=nar_model,
        ar_tokenizer=ar_tokenizer,
        nar_tokenizer=nar_tokenizer,
        trained_model=model,
        args_predict=args_predict,
        all_src_encodec=original_src_encodec,
        all_instruction=original_instruction,
        all_ground_truth=original_transcription,
        iteration=iteration,
        num_evaluations=num_eval,
        eval_data_len=eval_data_len,
        selected_indices=eval_indices,
        device=device,
        clap_model=clap_model,
        asr_model=asr_model,
        accelerator=accelerator
    )

    prefix = "Baseline/" if is_baseline else f"Eval/{eval_type}_"
    step = 0 if is_baseline else iteration

    logging.info(f"{'Baseline' if is_baseline else 'Trained Model'} Iteration {iteration} {eval_type} Set Evaluation:")
    logging.info(f"EVAL: Cosine_Sim metrics {eval_type} Set for iteration {iteration}: {metrics}")
    logging.info(f"EVAL: Cosine_Sim score {eval_type} Set for iteration {iteration}: {rewards}")

    reward_list = [np.mean([r for r in reward_group if r is not None]) if reward_group else None for reward_group in rewards]
    filtered_reward_list = [r for r in reward_list if r is not None]
    avg_reward = np.mean(filtered_reward_list) if filtered_reward_list else None
    avg_claps_rewards = np.mean(all_claps_rewards)
    avg_asr_rewards = np.mean(all_asr_rewards)

    logging.info(f"EVAL: {'Baseline' if is_baseline else 'Trained model'} Cosine_Sim score list on {eval_type} set: {reward_list}")
    logging.info(f"EVAL: {'Baseline' if is_baseline else 'Trained model'} average Cosine_Sim score on {eval_type} set: {avg_reward}")
    logging.info(f"EVAL: {'Baseline' if is_baseline else 'Trained model'} average CLAPS rewards on {eval_type} set: {avg_claps_rewards}")
    logging.info(f"EVAL: {'Baseline' if is_baseline else 'Trained model'} average ASR rewards on {eval_type} set: {avg_asr_rewards}")

    # Log rewards to TensorBoard
    if avg_reward is not None:
        writer.add_scalar(f"{prefix}Avg_Cosine_Sim", avg_reward, step)
    if avg_claps_rewards is not None:
        writer.add_scalar(f"{prefix}Avg_CLAPS_Reward", avg_claps_rewards, step)
    if avg_asr_rewards is not None:
        writer.add_scalar(f"{prefix}Avg_ASR_Reward", avg_asr_rewards, step)
        
logging.info("Evaluating initial model performance before training begins.")

if eval_train:
    evaluate_iteration("Train", iteration=-1, eval_indices=eval_train_indices, eval_data_len=eval_train_data_len, is_baseline=True)

if eval_test:
    evaluate_iteration("Test", iteration=-1, eval_indices=eval_test_indices, eval_data_len=eval_test_data_len, is_baseline=True)


for iteration in tqdm(range(num_iterations), desc="Training Iterations", disable=disable_tqdm):
    logging.info(f"-----------Starting iteration {iteration}-----------")

    resume = False

    model_checkpoint, chosen_rewards, rejected_rewards = train_iteration(
        model=model,
        model_checkpoint=model_checkpoint,
        iteration=iteration,
        data_size=data_size_per_iteration,
        sample_size=sample_size,
        ar_model=ar_model,
        ar_tokenizer=ar_tokenizer,
        nar_model=nar_model,
        nar_tokenizer=nar_tokenizer,
        all_src_encodec=batch_src_encodec,
        all_instruction=batch_instruction,
        all_transcription=batch_transcription,
        args_predict=args_predict,
        agent_output_dir=agent_output_dir,
        model_output_dir_base=model_output_dir,
        temperature=1.0,
        beta=beta,
        base_path=current_base_path,
        resume_from_checkpoint=resume,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        max_target_length=max_target_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        clap_model=clap_model,
        asr_model=asr_model,
        accelerator=accelerator
    )

    logging.info(f"Chosen rewards for iteration {iteration}: {chosen_rewards}")
    logging.info(f"Rejected rewards for iteration {iteration}: {rejected_rewards}")
    
    # Log chosen and rejected rewards to TensorBoard
    if chosen_rewards:
        writer.add_scalar("Train/Avg_Chosen_Reward", np.mean(chosen_rewards), iteration)
    if rejected_rewards:
        writer.add_scalar("Train/Avg_Rejected_Reward", np.mean(rejected_rewards), iteration)


    if (iteration + 1) % eval_frequency == 0:
        if eval_train:
            evaluate_iteration("Train", iteration, eval_train_indices, eval_train_data_len)
        if eval_test:
            evaluate_iteration("Test", iteration, eval_test_indices, eval_test_data_len)

    logging.info(f"-----------Finished iteration {iteration}-----------")

total_end_time = time.time()
total_time_taken = total_end_time - total_start_time
logging.info(f"Total time taken for the entire process: {total_time_taken:.2f} seconds")
# Close TensorBoard SummaryWriter
writer.close()

# Transcriptions: 'neighboring fields', 'but i will be in a minute', 'will you', 'why is it', 'what more do you want', 'why not', 'the wind wakened me', 'yes', 'hush pearl hush', 'good night husband', 'what music', 'shut up will you', 'here he is', 'that was strange'



