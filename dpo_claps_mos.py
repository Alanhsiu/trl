#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[ ]:


import sys
sys.path.append("/work/b0990106x/trl/vc")
import importlib
import vc
importlib.reload(vc)
import torch
from vc.trainer_encodec_vc_inference import pack_inputs_v2, get_ar_prediction_get_audio, get_ar_prediction_audio_batch
from types import SimpleNamespace
from transformers import BartForConditionalGeneration, AutoTokenizer
from datasets import Dataset
from trl import DPOTrainer, DPOConfig, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
from vc.encodec_model.nar_bart_model import NARBartForConditionalGeneration
from datetime import datetime
import os
import numpy as np
from dpo_eval import get_reward_claps ,eval_dpo_claps_batch, convert_array_to_tensor_format
from dpo_eval import get_reward_mos, process_and_get_mos_reward, eval_dpo_mos
import json
from tqdm import tqdm
import time
from typing import List, Tuple
import random
import argparse
import soundfile as sf
import math

sys.path.append('/work/b0990106x/trl/CLAPS')
from CLAPS.inference import load_model


# ### Utility Functions

# In[2]:


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# In[3]:


def generate_output_batch(
        model,
        ar_model, 
        nar_model, 
        ar_tokenizer, 
        nar_tokenizer, 
        clap_model,
        accelerator,
        src_encodec: list, 
        instruction: list, 
        args_predict: SimpleNamespace, 
        episode_counter: int = 0, 
        base_path: str = "/work/b0990106x/trl", 
        temperature: float = 1.0
) -> tuple[float, str]:
    '''
    Generates output from AR model, synthesize the audio, and evaluate the audio using NISQA.
    Returns:
        tuple:
            reward(float): The reward of the audio.
            tokenized_decode_ar(str): The tokenized output of the AR model - first layer.
    '''
    # Generate predictions using the AR model
    audio_list, decode_ar_list = get_ar_prediction_audio_batch(
        args_predict, model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, episode_counter, temperature=temperature
    )
    # extract the instruction from the list 
    reward_list,tokenized_decode_ar_list = [], []

    for i, audio in enumerate(audio_list): 
        # audio ---> tensor([])
        if audio is not None:
            tensor_audio = convert_array_to_tensor_format(audio)
            if tensor_audio[0].shape[0]==1:
                tensor_audio[0] = tensor_audio[0].squeeze(0)
            reward_claps = get_reward_claps(clap_model=clap_model, accelerator=accelerator, prompts = instruction[i], wavs = tensor_audio)
            
            output_path_ckpt = args_predict.output_path.replace(".wav", f"_generate_{episode_counter}_item_{i}.wav")
            sf.write(output_path_ckpt, np.ravel(audio), samplerate=24000)
            reward_mos = get_reward_mos(output_path=output_path_ckpt, base_path=base_path)  
            
            reward = (reward_claps + reward_mos/5) / 2
        else: 
            reward = 0
        reward_list.append(reward)
    
    for decode_ar in decode_ar_list:
        list_decode_ar = decode_ar.flatten().tolist()   
        filtered_decode_ar_list = list_decode_ar[2:-1]
        decode_ar_tokens = ar_tokenizer.convert_ids_to_tokens(filtered_decode_ar_list)
        tokenized_decode_ar = ar_tokenizer.convert_tokens_to_string(decode_ar_tokens)
        tokenized_decode_ar_list.append(tokenized_decode_ar)
        
    return reward_list, tokenized_decode_ar_list

def extract_data_from_json(file_path: str) -> Tuple[List[list], List[str], List[list]]:
    """
    Loads data from a JSON file and extracts 'src_encodec', 'instruction', and 'tgt_encodec'.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        tuple:
            all_src_encodec (List[list]): A list containing the 'src_encodec' data from each item in the JSON file.
            all_instruction (List[str]): A list containing the 'instruction' data from each item in the JSON file.
            all_tgt_encodec (List[list]): A list containing the 'tgt_encodec' data from each item in the JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    all_src_encodec = [item["src_encodec"] for item in data]
    all_instruction = [item["instruction"] for item in data]

    return all_src_encodec, all_instruction

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
        seed: int = 42
) -> None:
    '''
    Train the DPO model and save the model.

    Args:
        model(AutoModelForSeq2SeqLMWithValueHead): The DPO model.
        model_ref(AutoModelForCausalLM): The reference model.
        ar_tokenizer(AutoTokenizer): The tokenizer.
        train_dataset(Dataset): The training dataset.
        val_dataset(Dataset): The validation dataset.
        model_output_dir(str): The output directory for the model.
        beta(float): The beta value.
        resume_from_checkpoint(bool): Whether to resume from a checkpoint.
        model_checkpoint(str): The path to the model

    Returns:
        None
    '''

    training_args = DPOConfig(
        beta = beta,
        output_dir = model_output_dir,
        resume_from_checkpoint = model_checkpoint if resume_from_checkpoint else None,
        seed = seed,
        per_device_train_batch_size = per_device_train_batch_size,
        num_train_epochs = num_train_epochs,
        gradient_accumulation_steps = gradient_accumulation_steps,
        learning_rate = learning_rate,
        max_length = max_length,
        max_prompt_length = max_prompt_length,
        max_target_length = max_target_length,
        evaluation_strategy="steps",
        save_steps = 5000,
        logging_dir = f"{model_output_dir}/logs"
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        tokenizer=ar_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    # Train the model
    trainer.train()

    # Save the model
    # trainer.save_model(f"{model_output_dir}/dpo_model")
    model.config.to_json_file(f"{model_output_dir}/config.json")
    # ar_tokenizer.save_pretrained(f"{model_output_dir}/dpo_model")


# In[4]:


def process_data_batch(sample_size: int, 
                       model,
                        ar_model, 
                        nar_model, 
                        ar_tokenizer, 
                        nar_tokenizer, 
                        clap_model,
                        accelerator,
                        selected_src_encodec: List[list], 
                        selected_instruction: List[str],
                        args_predict: SimpleNamespace, 
                        base_path: str = "/work/b0990106x/trl", 
                        temperature: float = 1.0, 
                        iteration: int = 0
) -> Tuple[List[str], List[str], List[str], List[float], List[float], List[float]]:
    # If sample size is 1, we cannot choose the best and worst outputs
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
            selected_src_encodec_list = [selected_src_encodec[i]]*sample_size
            selected_instruction_list = [selected_instruction[i]]*sample_size
            rewards, tokenized_outputs = generate_output_batch(
                model=model,
                ar_model=ar_model, 
                nar_model=nar_model, 
                ar_tokenizer=ar_tokenizer, 
                nar_tokenizer=nar_tokenizer,
                src_encodec = selected_src_encodec_list,
                instruction=selected_instruction_list, 
                clap_model=clap_model,
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
            average_rewards.append(average_reward)

            
            # choose first 20% of the data and last 20% of the data 
            twenty_percent_num = math.ceil(len(valid_rewards)/2 * 0.2)
            max_reward_indexs = np.argsort(valid_rewards)[-twenty_percent_num:]
            min_reward_indexs = np.argsort(valid_rewards)[:twenty_percent_num]
            
            chosen_outputs = [valid_outputs[j] for j in max_reward_indexs]
            rejected_outputs = [valid_outputs[j] for j in min_reward_indexs]

            obs_input = pack_inputs_v2(ar_tokenizer, selected_src_encodec[i], selected_instruction[i])
            tokenize_input = ar_tokenizer.convert_ids_to_tokens(obs_input)
            tokenize_input_str = ar_tokenizer.convert_tokens_to_string(tokenize_input)
            
            prompts.extend([tokenize_input_str] * len(chosen_outputs))
            chosen.extend(chosen_outputs)
            chosen_rewards.extend([valid_rewards[j] for j in max_reward_indexs])
            rejected.extend(rejected_outputs)
            rejected_rewards.extend([valid_rewards[j] for j in min_reward_indexs])
        else:
            print(f"Not enough valid rewards for data index {i}")

    # If there is only one data, we need to double the data because we need it for training set and validation set
    if len(selected_src_encodec) == 1:
        chosen *= 2
        rejected *= 2
        prompts *= 2
        chosen_rewards *= 2
        rejected_rewards *= 2
        average_rewards *= 2    
    
    return chosen, rejected, prompts, chosen_rewards, rejected_rewards, average_rewards

def generate_data(model,
                  ar_model, 
                  ar_tokenizer, 
                  nar_model, 
                  nar_tokenizer, 
                  clap_model,
                  accelerator,
                  selected_src_encodec: List[list], 
                  selected_instruction: List[str],
                  args_predict: SimpleNamespace, 
                  sample_size: int, 
                  iteration: int, 
                  agent_output_dir: str, 
                  base_path: str = "/work/b0990106x/trl", 
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
        model=model,
        ar_model=ar_model,
        nar_model=nar_model,
        ar_tokenizer=ar_tokenizer,
        nar_tokenizer=nar_tokenizer,
        selected_src_encodec=selected_src_encodec,
        selected_instruction=selected_instruction,
        args_predict=args_predict,
        base_path=base_path,
        temperature=temperature,
        iteration = iteration,
        clap_model=clap_model,
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
                    args_predict, 
                    agent_output_dir,
                    model_output_dir_base, 
                    clap_model,
                    accelerator,
                    beta = 0.1, 
                    temperature = 1.0,
                    base_path="/work/b0990106x/trl",
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
    # print(f"Iteration {iteration}")

    # ar_model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
    # ar_tokenizer = AutoTokenizer.from_pretrained(ar_checkpoint)
    # ar_tokenizer.pad_token = ar_tokenizer.eos_token
    # nar_model = NARBartForConditionalGeneration.from_pretrained(nar_checkpoint)
    # nar_tokenizer = AutoTokenizer.from_pretrained(nar_checkpoint)

    selected_src_encodec = all_src_encodec[:data_size]
    selected_instruction = all_instruction[:data_size]

    data_for_dataset, chosen_rewards, rejected_rewards = generate_data(model=model, 
                                                                    ar_model=ar_model,
                                                                    ar_tokenizer=ar_tokenizer,
                                                                    nar_model=nar_model,
                                                                    nar_tokenizer=nar_tokenizer,
                                                                    selected_src_encodec=selected_src_encodec,
                                                                    selected_instruction=selected_instruction,
                                                                    args_predict=args_predict,
                                                                    sample_size=sample_size,
                                                                    iteration=iteration,
                                                                    agent_output_dir=agent_output_dir,
                                                                    base_path=base_path,
                                                                    temperature=temperature,
                                                                    clap_model=clap_model,
                                                                    accelerator=accelerator)

    dataset = Dataset.from_dict(data_for_dataset)
    dataset_dict = dataset.train_test_split(test_size=0.1, shuffle=True, seed=seed)
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["test"]

    model_output_dir = f"{model_output_dir_base}/iter_{iteration}"
    os.makedirs(model_output_dir, exist_ok=True)

    # model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_checkpoint, return_dict=True)
    model_ref = create_reference_model(model)
    
    train_model(model=model,
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
                seed = seed)

    return f"{model_output_dir}/dpo_model", chosen_rewards, rejected_rewards


# ### Hyperparameters

# In[ ]:


# Load all data
selected_src_encodec, selected_instruction = extract_data_from_json('dpo_data/src_encodec.json')

# Define paths and device
base_path = "/work/b0990106x/trl"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define timestamp
now = datetime.now()
ts = now.strftime("%m%d-%H%M")
print("timestamp:", ts)

# Define paths
model_output_dir = os.path.join(base_path, "model_output", ts) # Location where the model are saved
agent_output_dir = os.path.join(base_path, "output", ts) # Path of saving the generated audio for reward model to evaluate
os.makedirs(model_output_dir, exist_ok=True)
os.makedirs(agent_output_dir, exist_ok=True)

seed = 42 # Training: seed

# Define arguments 
args_predict = SimpleNamespace(output_path=f"{base_path}/output/{ts}/example.wav", seed=seed, device=device)
ar_checkpoint = "lca0503/speech-chatgpt-base-ar-v2-epoch10-wotrans"
nar_checkpoint = "lca0503/speech-chatgpt-base-nar-v2-epoch4-wotrans"

# Models and Iterations
model_checkpoint = ar_checkpoint # Prepare: set the initial model checkpoint
# sample_size = 80 # Prepare Dataset: generate how many outputs to select max and min for chosen and rejected (original: 10)
sample_size = 40 # Prepare Dataset: generate how many outputs to select max and min for chosen and rejected (original: 10)
num_iterations = 1000  # Training: train how many iterations (original: 100)
# some short data: 8, 16, 65, 100, 102, 105, 112, 132, 140, 145
train_selected_indices = [8, 16, 65, 100, 102]
# train_selected_indices = random.sample(range(len(selected_src_encodec)), 5) # Training: train on selected data indicies from all_src_encodec
 # Training: train on selected data indicies from all_src_encodec
data_size_per_iteration = len(train_selected_indices) # Training: each iteration will train how many data

# Define Training Configuration
beta = 0.1 # Training: beta value for DPO
learning_rate = 5e-07 # Training: learning rate (original: 5e-07)
num_train_epochs = 3 # Training: number of training epochs
max_length = 1024*9 # Training: max length of the model
max_prompt_length = 1024*9 # Training: max length of the prompt
max_target_length = 1024*9 # Training: max length of the target
per_device_train_batch_size = 16 # Training: batch size (original: 8)
gradient_accumulation_steps = 1 # Training: gradient accumulation steps

# Evaluation Configuration
eval_train = False # Evaluation: evaluate on training data or not
eval_test = True # Evaluation: evaluate on testing data or not
eval_train_indices = train_selected_indices # Evaluation: evaluate on training data indicies from all_src_encodec
eval_test_indices = [105, 112, 132, 140, 145]
# eval_test_indices = random.sample(range(len(selected_src_encodec)), 5) # Evaluation: evaluate on testing data indicies from all_src_encodec
eval_train_data_len = 1000 # Evaluation: evaluate how many training data
eval_test_data_len = len(eval_test_indices) # Evaluation: evaluate how many testing data
num_eval = 1 # Evaluation: evaluate how many times per data (original: 10)
eval_frequency = 1 # Evaluation: evaluate every how many iterations
# Define temperature
# eval_selected_indices = random.sample(range(len(all_src_encodec)), eval_data_len) # Evaluation: select 10 data for evaluation
print(f"length of all_src_encodec: {len(selected_src_encodec)}") # ~ 9000 data
print(f"length of all_instruction: {len(selected_instruction)}") # ~ 9000 data


# In[ ]:


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
ckpt_pth=f'{base_path}/CLAPS/pretrained/7d/cp_claps_blstm_m_50k_v3/cp_0045000'
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
        n_sub=n_sub,  # Number of subspaces, if any
        ckpt_pth=ckpt_pth,  # Set your checkpoint path
        project_dir=project_dir  # Example project directory
    )

clap_model, accelerator = load_model(a)


# In[ ]:


print(f"num_iterations: {num_iterations}")
print(f"data_size_per_iteration: {data_size_per_iteration}")
print(f"sample_size: {sample_size}")
print(f"beta: {beta}")
print(f"learning_rate: {learning_rate}")
print(f"num_train_epochs: {num_train_epochs}")
print(f"ar_checkpoint: {ar_checkpoint}")
print(f"nar_checkpoint: {nar_checkpoint}")
print(f"args_predict: {args_predict}")
print(f"model_output_dir: {model_output_dir}")
print(f"agent_output_dir: {agent_output_dir}")
print(f"base_path: {base_path}")
print(f"device: {device}")
print(f"eval_train_data_len: {eval_train_data_len}")
print(f"eval_test_data_len: {eval_test_data_len}")
print(f"eval_train_indices: {eval_train_indices}")
print(f"eval_test_indices: {eval_test_indices}")
print(f"eval_train: {eval_train}")
print(f"eval_test: {eval_test}")
print(f"num_eval: {num_eval}")

# print training data
for i in train_selected_indices:
    print('training idx', i,':', selected_instruction[i])
    
# print evaluation data
if eval_test:
    for i in eval_test_indices:
        print('evaluation idx', i,':', selected_instruction[i])

if eval_train:
    for i in eval_train_indices:
        print('evaluation idx', i,':', selected_instruction[i])




# # Main Functions

# ### Load Models

# In[ ]:


model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_checkpoint, return_dict=True)
ar_model = BartForConditionalGeneration.from_pretrained(ar_checkpoint)
ar_tokenizer = AutoTokenizer.from_pretrained(ar_checkpoint)
# ar_tokenizer.pad_token = ar_tokenizer.eos_token
nar_model = NARBartForConditionalGeneration.from_pretrained(nar_checkpoint)
nar_tokenizer = AutoTokenizer.from_pretrained(nar_checkpoint)


# ### Logging Start

# In[ ]:


import logging

log_path = f'{model_output_dir}/log_training.log'
print(f"Logging to: {log_path}")

# Set up logging
logging.basicConfig(
    filename=log_path, 
    filemode='a', 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

logging.info(
    f"Parameters:\n"
    f"Prepare Data: sample_size: {sample_size}\n"
    f"Training: num_iterations: {num_iterations}\n"
    f"Training: data_size_per_iteration: {data_size_per_iteration}\n"
    f"Training: train_selected_indices: {train_selected_indices}\n"
    f"Training: beta: {beta}\n"
    f"Training: learning_rate: {learning_rate}\n"
    f"Training: num_train_epochs: {num_train_epochs}\n"
    f"Training: max_length: {max_length}\n"
    f"Training: max_prompt_length: {max_prompt_length}\n"
    f"Training: max_target_length: {max_target_length}\n"
    f"Training: per_device_train_batch_size: {per_device_train_batch_size}\n"
    f"Training: gradient_accumulation_steps: {gradient_accumulation_steps}\n"
    f"Training: seed: {seed}\n"
    f"Training: ar_checkpoint: {ar_checkpoint}\n"
    f"Training: nar_checkpoint: {nar_checkpoint}\n"
    f"Training: args_predict: {args_predict}\n"
    f"Training: model_output_dir: {model_output_dir}\n"
    f"Training: agent_output_dir: {agent_output_dir}\n"
    f"Training: base_path: {base_path}\n"
    f"Training: device: {device}\n"
    f"Evaluation: eval_train_data_len: {eval_train_data_len}\n"
    f"Evaluation: eval_test_data_len: {eval_test_data_len}\n"
    f"Evaluation: eval_train_indices: {eval_train_indices}\n"
    f"Evaluation: eval_test_indices: {eval_test_indices}\n"
    f"Evaluation: eval_train: {eval_train}\n"
    f"Evaluation: eval_test: {eval_test}\n"
    f"Evaluation: num_eval: {num_eval}"
)


# ### Initial Setup

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[ ]:


# Start time
total_start_time = time.time()
def evaluate_model(data_len, indices, log_prefix):
    # eval dpo claps
    original_model_metrics, original_model_rewards = eval_dpo_claps_batch(
        nar_model=nar_model,
        ar_tokenizer=ar_tokenizer,
        nar_tokenizer=nar_tokenizer,
        trained_model=model,
        args_predict=args_predict,
        all_src_encodec=selected_src_encodec,
        all_instruction=selected_instruction,
        iteration=-1,
        num_evaluations=num_eval,
        eval_data_len=data_len,
        selected_indices=indices,
        device=device,
        clap_model=clap_model,
        accelerator=accelerator
    )
    logging.info(f"{log_prefix} Set Evaluation:")
    logging.info(f"{log_prefix} model metrics: {original_model_metrics}")
    logging.info(f"{log_prefix} model rewards: {original_model_rewards}")

    # eval dpo mos
    original_model_metrics_mos, original_model_rewards_mos = eval_dpo_mos(
        nar_model=nar_model,
        ar_tokenizer=ar_tokenizer,
        nar_tokenizer=nar_tokenizer,
        trained_model=model,
        args_predict=args_predict,
        all_src_encodec=selected_src_encodec,
        all_instruction=selected_instruction,
        iteration=-1,
        num_evaluations=num_eval,
        eval_data_len=data_len,
        selected_indices=indices,
        device=device,
    )
    logging.info(f"{log_prefix} model MOS metrics: {original_model_metrics_mos}")
    logging.info(f"{log_prefix} model MOS rewards: {original_model_rewards_mos}")

    reward_list = process_rewards(original_model_rewards)
    logging.info(f"{log_prefix} Cosine_Sim score list: {reward_list}")

    reward_list_mos = process_rewards(original_model_rewards_mos)
    logging.info(f"{log_prefix} MOS score list: {reward_list_mos}")

    average_claps = compute_average_reward(reward_list, f"{log_prefix} average Claps")
    average_mos = compute_average_reward(reward_list_mos, f"{log_prefix} average MOS")

    weighted_reward = 0.5 * average_claps + 0.5 * average_mos / 5
    logging.info(f"{log_prefix} weighted average rewards: {weighted_reward}")

def process_rewards(rewards):
    reward_list = []
    for reward_batch in rewards:
        filtered_rewards = [r for r in reward_batch if r is not None]
        reward_list.append(np.mean(filtered_rewards) if filtered_rewards else None)
    return reward_list

def compute_average_reward(reward_list, log_message):
    filtered_reward_list = [r for r in reward_list if r is not None]
    if filtered_reward_list:
        avg_reward = np.mean(filtered_reward_list)
        logging.info(f"{log_message}: {avg_reward}")
        return avg_reward
    else:
        logging.info(f"{log_message}: None")
        return 0

# Run evaluations
if eval_train:
    evaluate_model(data_len=eval_train_data_len, indices=eval_train_indices, log_prefix="Original Model Train")

if eval_test:
    evaluate_model(data_len=eval_test_data_len, indices=eval_test_indices, log_prefix="Original Model Test")

    
# If train_selected_indices is not empty, we will use the selected indices for training
if train_selected_indices:
    batch_src_encodec = [selected_src_encodec[i] for i in train_selected_indices]
    batch_instruction = [selected_instruction[i] for i in train_selected_indices]
    logging.info(f"Processing data from selected indices: {train_selected_indices}")
else:
    start_idx = 0
    end_idx = data_size_per_iteration
    batch_src_encodec = selected_src_encodec[start_idx:end_idx] 
    batch_instruction = selected_instruction[start_idx:end_idx]
    logging.info(f"Processing data from index {start_idx} to {end_idx}")


# ### Start training iterations

# In[ ]:


disable_tqdm = not os.isatty(1)
def evaluate_model(data_len, indices, iteration, log_prefix):
    # eval dpo claps
    trained_model_metrics, trained_model_rewards = eval_dpo_claps_batch(
        nar_model=nar_model,
        ar_tokenizer=ar_tokenizer,
        nar_tokenizer=nar_tokenizer,
        trained_model=model,
        args_predict=args_predict,
        all_src_encodec=selected_src_encodec,
        all_instruction=selected_instruction,
        iteration=iteration,
        num_evaluations=num_eval,
        eval_data_len=data_len,
        selected_indices=indices,
        device=device,
        clap_model=clap_model,
        accelerator=accelerator
    )
    logging.info(f"Trained Model Iteration {iteration} {log_prefix} Set Evaluation: ")
    logging.info(f"EVAL: Cosine_Sim metrics {log_prefix} Set for iteration {iteration}: {trained_model_metrics}")
    logging.info(f"EVAL: Cosine_Sim score {log_prefix} Set for iteration {iteration}: {trained_model_rewards}")

    # eval dpo mos
    trained_model_metrics_mos, trained_model_rewards_mos = eval_dpo_mos(
        nar_model=nar_model,
        ar_tokenizer=ar_tokenizer,
        nar_tokenizer=nar_tokenizer,
        trained_model=model,
        args_predict=args_predict,
        all_src_encodec=selected_src_encodec,
        all_instruction=selected_instruction,
        iteration=iteration,
        num_evaluations=num_eval,
        eval_data_len=data_len,
        selected_indices=indices,
        device=device
    )
    logging.info(f"EVAL: MOS metrics {log_prefix} Set for iteration {iteration}: {trained_model_metrics_mos}")
    logging.info(f"EVAL: MOS score {log_prefix} Set for iteration {iteration}: {trained_model_rewards_mos}")

    # Process rewards
    reward_list = process_rewards(trained_model_rewards)
    logging.info(f"EVAL: Trained model Cosine_Sim score list on {log_prefix} set: {reward_list}")

    reward_list_mos = process_rewards(trained_model_rewards_mos)
    logging.info(f"EVAL: Trained model MOS score list on {log_prefix} set: {reward_list_mos}")

    # Compute average rewards
    compute_and_log_average_reward(reward_list, reward_list_mos, iteration, log_prefix)

def process_rewards(rewards):
    reward_list = []
    for reward_batch in rewards:
        filtered_rewards = [r for r in reward_batch if r is not None]
        reward_list.append(np.mean(filtered_rewards) if filtered_rewards else None)
    return reward_list

def compute_and_log_average_reward(reward_list, reward_list_mos, iteration, log_prefix):
    filter_reward_list = [r for r in reward_list if r is not None]
    if filter_reward_list:
        avg_reward = np.mean(filter_reward_list)
        logging.info(f"EVAL: Trained model average Cosine_Sim score on {log_prefix} set for iteration {iteration}: {avg_reward}")
    else:
        logging.info(f"EVAL: Trained model average Cosine_Sim score on {log_prefix} set for iteration {iteration}: None")

    filter_reward_list_mos = [r for r in reward_list_mos if r is not None]
    if filter_reward_list_mos:
        avg_reward_mos = np.mean(filter_reward_list_mos)
        logging.info(f"EVAL: Trained model average MOS score on {log_prefix} set for iteration {iteration}: {avg_reward_mos}")
    else:
        logging.info(f"EVAL: Trained model average MOS score on {log_prefix} set for iteration {iteration}: None")

    # Weighted reward calculation
    weighted_reward = 0.5 * np.mean(filter_reward_list) + 0.5 * np.mean(filter_reward_list_mos) / 5
    logging.info(f"EVAL: Trained model weighted average score on {log_prefix} set for iteration {iteration}: {weighted_reward}")

# Main iteration loop
for iteration in tqdm(range(num_iterations), desc="Training Iterations", disable=disable_tqdm):
    logging.info(f"-----------Starting iteration {iteration}-----------")

    # resume = iteration > 0 # resume from the previous checkpoint when iteration > 0
    resume = False

    # model_checkpoint is the model checkpoint from the previous iteration
    model_checkpoint, chosen_rewards, rejected_rewards = train_iteration(
        model,
        model_checkpoint,
        iteration=iteration,
        data_size=data_size_per_iteration,
        sample_size=sample_size,
        ar_model=ar_model,
        ar_tokenizer=ar_tokenizer,
        nar_model=nar_model,
        nar_tokenizer=nar_tokenizer,
        all_src_encodec=batch_src_encodec,
        all_instruction=batch_instruction,
        args_predict=args_predict,
        agent_output_dir=agent_output_dir,
        model_output_dir_base=model_output_dir,
        temperature=1.0,
        beta=beta,
        base_path=base_path,
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
        accelerator=accelerator
    )

    logging.info(f"Chosen rewards for iteration {iteration}: {chosen_rewards}")
    logging.info(f"Rejected rewards for iteration {iteration}: {rejected_rewards}")
    logging.info(f"Finished training iteration {iteration}")

    if (iteration + 1) % eval_frequency == 0:
        # Perform evaluation
        if eval_train:
            evaluate_model(data_len=eval_train_data_len, indices=eval_train_indices, iteration=iteration, log_prefix="Training")
        if eval_test:
            evaluate_model(data_len=eval_test_data_len, indices=eval_test_indices, iteration=iteration, log_prefix="Testing")

    logging.info(f"-----------Finished iteration {iteration}-----------")

total_end_time = time.time()

# Calculate total time taken
total_time_taken = total_end_time - total_start_time
logging.info(f"Total time taken for the entire process: {total_time_taken:.2f} seconds")

