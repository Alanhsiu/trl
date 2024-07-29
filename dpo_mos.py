#!/usr/bin/env python
# coding: utf-8

# # Training

# In[1]:


import sys
sys.path.append("/work/b0990106x/trl/vc")
import importlib
import vc
importlib.reload(vc)
import torch
from vc.trainer_encodec_vc_inference import get_ar_prediction_v3, pack_inputs_v2
from types import SimpleNamespace
from transformers import BartForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from NISQA.nisqa.NISQA_model import nisqaModel
from datasets import Dataset
from trl import DPOTrainer, DPOConfig, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
from vc.encodec_model.nar_bart_model import NARBartForConditionalGeneration
from datetime import datetime
import os
import numpy as np
from dpo_eval import get_reward, eval_dpo_mos
import json
from tqdm import tqdm
import time
from typing import List, Tuple


# In[ ]:


def generate_output(
        ar_model, 
        nar_model, 
        ar_tokenizer, 
        nar_tokenizer, 
        src_encodec: list, 
        instruction: list, 
        args_predict: SimpleNamespace, 
        episode_counter: int = 0, 
        base_path: str = "/work/b0990106x/trl", 
        temperature: float = 1.0
) -> tuple[float, str]:
    '''
    Generates output from AR model, synthesize the audio, and evaluate the audio using NISQA.

    Args:
        ar_model(BartForConditionalGeneration): AR model
        nar_model(NarbartForConditionalGeneration): NAR model
        ar_tokenizer(AutoTokenizer): AR tokenizer
        nar_tokenizer(AutoTokenizer): NAR tokenizer
        src_encodec(list): A list of inputs, where each input is a list of layers, and each layer is a list of v_token integers.
        instruction(list): A list of string of instructions.
        args_predict(SimpleNamespace): A SimpleNamespace object containing the arguments for the NISQA prediction.
        episode_counter(int): A counter that determine the name of the output audio.
        base_path(str): The path to the base directory.
        temperature(float): The temperature for the AR model.

    Returns:
        tuple:
            reward(float): The reward of the audio.
            tokenized_decode_ar(str): The tokenized output of the AR model - first layer.
    '''
    # Generate predictions using the AR model
    _, decode_ar, output_path_ckpt = get_ar_prediction_v3(
        args_predict, ar_model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, episode_counter, temperature=temperature
    )

    # Flatten the decoded AR output tensor and convert it to a list
    list_decode_ar = decode_ar.flatten().tolist()   

    # Evaluate the audio to get the reward
    reward = get_reward(output_path_ckpt, base_path)
    
    # Filter the decoded AR output to remove special tokens
    filtered_decode_ar_list = list_decode_ar[2:-1]

    # Convert the filtered token IDs back to tokens and then to a string
    decode_ar_tokens = ar_tokenizer.convert_ids_to_tokens(filtered_decode_ar_list)
    tokenized_decode_ar = ar_tokenizer.convert_tokens_to_string(decode_ar_tokens)

    return reward, tokenized_decode_ar



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
    all_tgt_encodec = [item["tgt_encodec"] for item in data]

    return all_src_encodec, all_instruction, all_tgt_encodec



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
        generate_during_eval = True,
        resume_from_checkpoint = model_checkpoint if resume_from_checkpoint else None,
        seed = seed,
        per_device_train_batch_size = per_device_train_batch_size,
        num_train_epochs = num_train_epochs,
        gradient_accumulation_steps = gradient_accumulation_steps,
        learning_rate = learning_rate,
        max_length = max_length,
        max_prompt_length = max_prompt_length,
        max_target_length = max_target_length
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
    trainer.save_model(f"{model_output_dir}/dpo_model")
    model.config.to_json_file(f"{model_output_dir}/dpo_model/config.json")
    ar_tokenizer.save_pretrained(f"{model_output_dir}/dpo_model")


# In[ ]:


def process_data(sample_size: int, 
                 ar_model, 
                 nar_model, 
                 ar_tokenizer, 
                 nar_tokenizer, 
                 all_src_encodec: List[list], 
                 all_instruction: List[str],
                 args_predict: SimpleNamespace, 
                 base_path: str = "/work/b0990106x/trl", 
                 temperature: float = 1.0, 
                 iteration: int = 0
) -> Tuple[List[str], List[str], List[str], List[float], List[float], List[float]]:
    """
    Process data to generate outputs, calculate rewards, and organize chosen and rejected data.

    Args:
        sample_size (int): The number of samples to generate for each data.
        ar_model (BartForConditionalGeneration): The AR model.
        nar_model (NarbartForConditionalGeneration): The NAR model.
        ar_tokenizer (AutoTokenizer): The AR tokenizer.
        nar_tokenizer (AutoTokenizer): The NAR tokenizer.
        all_src_encodec (List[list]): A list of src_encodec data.
        all_instruction (List[str]): A list of instruction data.
        args_predict (SimpleNamespace): A SimpleNamespace object containing the arguments for the NISQA prediction.
        base_path (str): The path to the base directory.
        temperature (float): The temperature for the AR model.

    Returns:
        tuple:
            chosen (List[str]): A list of chosen outputs.
            rejected (List[str]): A list of rejected outputs.
            prompts (List[str]): A list of prompts.
            chosen_rewards (List[float]): A list of rewards for the chosen outputs.
            rejected_rewards (List[float]): A list of rewards for the rejected outputs.
            average_rewards (List[float]): A list of average rewards.
    """
    # If sample size is 1, we cannot choose the best and worst outputs
    if sample_size < 2:
        raise ValueError("Parameter 'sample_size' must be greater than 1.")

    chosen, rejected, prompts, chosen_rewards, rejected_rewards, average_rewards = [], [], [], [], [], []

    for i in tqdm(range(len(all_src_encodec)), desc="Processing Data"):
        rewards, tokenized_outputs = [], []

        for j in tqdm(range(sample_size), desc="Processing Samples"):
            size_of_packed_input = (
                    len(all_src_encodec[i][0]) +
                    len(ar_tokenizer(all_instruction[i])["input_ids"][1:-1]) +
                    3
            )
            if 4 < size_of_packed_input <= 1024:
                reward, tokenized_decode_ar = generate_output(
                    ar_model=ar_model, 
                    nar_model=nar_model, 
                    ar_tokenizer=ar_tokenizer, 
                    nar_tokenizer=nar_tokenizer,
                    src_encodec = all_src_encodec[i],
                    instruction=all_instruction[i], 
                    args_predict=args_predict,
                    episode_counter=f"iter_{iteration}_data_{i}_episode_{j}",
                    base_path=base_path, 
                    temperature=temperature
                )
                rewards.append(reward)
                tokenized_outputs.append(tokenized_decode_ar)

        valid_rewards = [r for r in rewards if r is not None]
        valid_outputs = [tokenized_outputs[j] for j in range(len(rewards)) if rewards[j] is not None]

        if len(valid_rewards) >= 2:
            max_reward_index = np.argmax(valid_rewards)
            min_reward_index = np.argmin(valid_rewards)
            average_reward = np.mean(valid_rewards)
            chosen_output = valid_outputs[max_reward_index]
            rejected_output = valid_outputs[min_reward_index]

            obs_input = pack_inputs_v2(ar_tokenizer, all_src_encodec[i], all_instruction[i])
            tokenize_input = ar_tokenizer.convert_ids_to_tokens(obs_input)
            tokenize_input_str = ar_tokenizer.convert_tokens_to_string(tokenize_input)
            prompts.append(tokenize_input_str)

            chosen.append(chosen_output)
            chosen_rewards.append(valid_rewards[max_reward_index])
            rejected.append(rejected_output)
            rejected_rewards.append(valid_rewards[min_reward_index])
            average_rewards.append(average_reward)
        else:
            print(f"Not enough valid rewards for data index {i}")

    # If there is only one data, we need to double the data because we need it for training set and validation set
    if len(all_src_encodec) == 1:
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

    Args:
        ar_model (BartForConditionalGeneration): The AR model.
        ar_tokenizer (AutoTokenizer): The AR tokenizer.
        nar_model (NarbartForConditionalGeneration): The NAR model.
        nar_tokenizer (AutoTokenizer): The NAR tokenizer.
        selected_src_encodec (List[list]): A list of src_encodec data.
        selected_instruction (List[str]): A list of instruction data.
        args_predict (SimpleNamespace): A SimpleNamespace object containing the arguments for the NISQA prediction.
        sample_size (int): The number of samples to generate for each data.
        iteration (int): The iteration number.
        agent_output_dir (str): The output directory for the agent.
        base_path (str): The path to the base directory.
        temperature (float): The temperature for the AR model.
    
    Returns:
        tuple:
            data_for_dataset (dict): A dictionary containing the data for the dataset.
            chosen_rewards (List[float]): A list of rewards for the chosen outputs.
            rejected_rewards (List[float]): A list of rewards for the rejected outputs.
    """
    chosen, rejected, prompts, chosen_rewards, rejected_rewards, average_rewards = process_data(
        sample_size=sample_size,
        ar_model=ar_model,
        nar_model=nar_model,
        ar_tokenizer=ar_tokenizer,
        nar_tokenizer=nar_tokenizer,
        all_src_encodec=selected_src_encodec,
        all_instruction=selected_instruction,
        args_predict=args_predict,
        base_path=base_path,
        temperature=temperature,
        iteration = iteration
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



def train_iteration(model_checkpoint, 
                    iteration, 
                    data_size, 
                    sample_size, 
                    ar_checkpoint, 
                    nar_checkpoint, 
                    all_src_encodec, 
                    all_instruction, 
                    args_predict, 
                    agent_output_dir,
                    model_output_dir_base, 
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
                    seed = 42 
):
    """
    Executes one training iteration: generates data, trains the model, and saves the output.
    """
    # print(f"Iteration {iteration}")

    ar_model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
    ar_tokenizer = AutoTokenizer.from_pretrained(ar_checkpoint)
    ar_tokenizer.pad_token = ar_tokenizer.eos_token
    nar_model = NARBartForConditionalGeneration.from_pretrained(nar_checkpoint)
    nar_tokenizer = AutoTokenizer.from_pretrained(nar_checkpoint)

    selected_src_encodec = all_src_encodec[:data_size]
    selected_instruction = all_instruction[:data_size]

    data_for_dataset, chosen_rewards, rejected_rewards = generate_data(ar_model=ar_model,
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
                                                                        temperature=temperature)

    dataset = Dataset.from_dict(data_for_dataset)
    dataset_dict = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["test"]

    model_output_dir = f"{model_output_dir_base}/iter_{iteration}"
    os.makedirs(model_output_dir, exist_ok=True)

    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_checkpoint, return_dict=True)
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


# In[ ]:


# Load all data
all_src_encodec, all_instruction, all_tgt_encodec = extract_data_from_json('dpo_data/src_encodec.json')

# all_src_encodec = all_src_encodec[2:]
# all_instruction = all_instruction[2:]

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


# Define arguments 
args_predict = SimpleNamespace(output_path=f"{base_path}/output/{ts}/example.wav", seed=0, device=device)
ar_checkpoint = "lca0503/speech-chatgpt-base-ar-v2-epoch10-wotrans"
nar_checkpoint = "lca0503/speech-chatgpt-base-nar-v2-epoch4-wotrans"

# Models and Iterations
model_checkpoint = ar_checkpoint # Prepare: set the initial model checkpoint
sample_size = 10 # Prepare Dataset: generate how many outputs to select max and min for chosen and rejected
data_size_per_iteration = 10 # Training: each iteration will train how many data
num_iterations = 30  # Training: train how many iterations
train_selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # Training: train on selected data indicies from all_src_encodec

# Define Training Configuration
beta = 0.1 # Training: beta value for DPO
learning_rate = 1e-06 # Training: learning rate
num_train_epochs = 500 # Training: number of training epochs
max_length = 1024*9 # Training: max length of the model
max_prompt_length = 1024*9 # Training: max length of the prompt
max_target_length = 1024*9 # Training: max length of the target
per_device_train_batch_size = 5 # Training: batch size
gradient_accumulation_steps = 1 # Training: gradient accumulation steps
seed = 42 # Training: seed

# Evaluation Configuration
eval_train = True # Evaluation: evaluate on training data or not
eval_test = True # Evaluation: evaluate on testing data or not
eval_train_indices = train_selected_indices # Evaluation: evaluate on training data indicies from all_src_encodec
eval_test_indices = [10,11,12,13,14,15,16,17,18,19] # Evaluation: evaluate on testing data indicies from all_src_encodec
eval_train_data_len = 10 # Evaluation: evaluate how many training data
eval_test_data_len = 10 # Evaluation: evaluate how many testing data
num_eval = 10 # Evaluation: evaluate how many times per data

# Define temperature
# eval_selected_indices = random.sample(range(len(all_src_encodec)), eval_data_len) # Evaluation: select 10 data for evaluation
print(f"length of all_src_encodec: {len(all_src_encodec)}") # ~ 9000 data
print(f"length of all_instruction: {len(all_instruction)}") # ~ 9000 data


# In[ ]:


print(f"num_iterations: {num_iterations}")
print(f"data_size_per_iteration: {data_size_per_iteration}")
print(f"sample_size: {sample_size}")
print(f"beta: {beta}")
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

print(all_instruction[0:2])


# In[ ]:


import logging
# Set up logging
logging.basicConfig(
    filename=f'{model_output_dir}/log_training.log', 
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

# Start time
total_start_time = time.time()
if eval_train:
    original_model_metrics, original_model_rewards = eval_dpo_mos(ar_checkpoint=ar_checkpoint,
                                                                    nar_checkpoint=nar_checkpoint,
                                                                    trained_model_checkpoint=ar_checkpoint, # original model
                                                                    args_predict=args_predict,
                                                                    all_src_encodec=all_src_encodec,
                                                                    all_instruction=all_instruction,
                                                                    iteration = -1,
                                                                    num_evaluations = num_eval,
                                                                    eval_data_len=eval_train_data_len,
                                                                    selected_indices=eval_train_indices,
                                                                    device=device,
                                                                    )
    logging.info(f"Original Model Train Set Evaluation: ")
    logging.info(f"Original model metrics on training set: {original_model_metrics}")
    logging.info(f"Original model rewards on training set: {original_model_rewards}")
    reward_list = []
    for rewards in original_model_rewards:
        filter_rewards = [r for r in rewards if r is not None]
        reward_list.append(np.mean(filter_rewards))
    logging.info(f"Original model reward list on training set: {reward_list}")
    filter_reward_list = [r for r in reward_list if r is not None]
    logging.info(f"Original model average rewards on training set: {np.mean(filter_reward_list)}")
    

if eval_test:
    original_model_metrics, original_model_rewards = eval_dpo_mos(ar_checkpoint=ar_checkpoint,
                                                                    nar_checkpoint=nar_checkpoint,
                                                                    trained_model_checkpoint=ar_checkpoint, # original model
                                                                    args_predict=args_predict,
                                                                    all_src_encodec=all_src_encodec,
                                                                    all_instruction=all_instruction,
                                                                    iteration = -1,
                                                                    num_evaluations = num_eval,
                                                                    eval_data_len=eval_test_data_len,
                                                                    selected_indices=eval_test_indices,
                                                                    device=device,
                                                                    )
    logging.info(f"Original Model Test Set Evaluation: ")
    logging.info(f"Original model metrics on testing set: {original_model_metrics}")
    logging.info(f"Original model rewards on testing set: {original_model_rewards}")
    reward_list = []
    for rewards in original_model_rewards:
        filter_rewards = [r for r in rewards if r is not None]
        reward_list.append(np.mean(rewards))
    logging.info(f"Original model reward list on testing set: {reward_list}")
    filter_reward_list = [r for r in reward_list if r is not None]  
    logging.info(f"Original model average rewards on testing set: {np.mean(filter_reward_list)}")

    
# If train_selected_indices is not empty, we will use the selected indices for training
if train_selected_indices:
    batch_src_encodec = [all_src_encodec[i] for i in train_selected_indices]
    batch_instruction = [all_instruction[i] for i in train_selected_indices]
    logging.info(f"Processing data from selected indices: {train_selected_indices}")
else:
    start_idx = 0
    end_idx = data_size_per_iteration
    batch_src_encodec = all_src_encodec[start_idx:end_idx] 
    batch_instruction = all_instruction[start_idx:end_idx]
    logging.info(f"Processing data from index {start_idx} to {end_idx}")

for iteration in tqdm(range(num_iterations), desc="Training Iterations"):
    logging.info(f"-----------Starting iteration {iteration}-----------")
    
    resume = iteration > 0 # resume from the previous checkpoint when iteration > 0

    logging.info(f"Starting iteration {iteration}")
    
    # model_checkpoint is the model checkpoint from the previous iteration
    # chosen_rewards and rejected_rewards are the rewards of the data
    model_checkpoint, chosen_rewards, rejected_rewards = train_iteration(model_checkpoint=model_checkpoint,
                                iteration=iteration,
                                data_size=data_size_per_iteration,
                                sample_size=sample_size,
                                ar_checkpoint=ar_checkpoint,
                                nar_checkpoint=nar_checkpoint,
                                all_src_encodec=batch_src_encodec,
                                all_instruction=batch_instruction,
                                args_predict=args_predict,
                                agent_output_dir=agent_output_dir,
                                model_output_dir_base=model_output_dir,
                                temperature = 1.0,
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
                                seed=seed
                                )

    logging.info(f"Chosen rewards for iteration {iteration}: {chosen_rewards}")
    logging.info(f"Rejected rewards for iteration {iteration}: {rejected_rewards}")
    logging.info(f"Finished training iteration {iteration}")

    # Evaluate the result of the current iteration
    if eval_train:
        trained_model_metrics, trained_model_rewards = eval_dpo_mos(ar_checkpoint=ar_checkpoint,
                                                                    nar_checkpoint=nar_checkpoint,
                                                                    trained_model_checkpoint=model_checkpoint,
                                                                    args_predict=args_predict,
                                                                    all_src_encodec=all_src_encodec,
                                                                    all_instruction=all_instruction,
                                                                    iteration = iteration,
                                                                    num_evaluations = num_eval,
                                                                    eval_data_len=eval_train_data_len,
                                                                    selected_indices=eval_train_indices,
                                                                    device=device
                                                                    )
        logging.info(f"Trained Model Iteration {iteration} Train Set Evaluation: ")
        logging.info(f"EVAL: MOS metrics Training Set for iteration {iteration}: {trained_model_metrics}")
        logging.info(f"EVAL: MOS score Training Set for iteration {iteration}: {trained_model_rewards}")
        reward_list = []
        for rewards in trained_model_rewards:
            filter_rewards = [r for r in rewards if r is not None]
            reward_list.append(np.mean(filter_rewards))
        logging.info(f"EVAL: Trained model MOS score list on training set: {reward_list}")
        filter_reward_list = [r for r in reward_list if r is not None]
        logging.info(f"EVAL: Trained model average MOS score on training set: {np.mean(filter_reward_list)}")

    if eval_test:
        trained_model_metrics, trained_model_rewards = eval_dpo_mos(ar_checkpoint=ar_checkpoint,
                                                                    nar_checkpoint=nar_checkpoint,
                                                                    trained_model_checkpoint=model_checkpoint,
                                                                    args_predict=args_predict,
                                                                    all_src_encodec=all_src_encodec,
                                                                    all_instruction=all_instruction,
                                                                    iteration = iteration,
                                                                    num_evaluations = num_eval,
                                                                    eval_data_len=eval_test_data_len,
                                                                    selected_indices=eval_test_indices,
                                                                    device=device
                                                                    )
        logging.info(f"Trained Model Iteration {iteration} Test Set Evaluation: ")
        logging.info(f"EVAL: MOS metrics Testing Set for iteration {iteration}: {trained_model_metrics}")
        logging.info(f"EVAL: MOS score Testing Set for iteration {iteration}: {trained_model_rewards}")
        reward_list = []
        for rewards in trained_model_rewards:
            filter_rewards = [r for r in rewards if r is not None]
            reward_list.append(np.mean(filter_rewards))
        logging.info(f"EVAL: Trained model MOS score list on testing set: {reward_list}")
        filter_reward_list = [r for r in reward_list if r is not None]  
        logging.info(f"EVAL: Trained model average MOS score on testing set: {np.mean(filter_reward_list)}")

total_end_time = time.time()

# Calculate total time taken
total_time_taken = total_end_time - total_start_time
logging.info(f"Total time taken for the entire process: {total_time_taken:.2f} seconds")

