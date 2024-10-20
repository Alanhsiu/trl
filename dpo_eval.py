import sys
import importlib
import torch
from types import SimpleNamespace
from transformers import BartForConditionalGeneration, AutoTokenizer
from NISQA.nisqa.NISQA_model import nisqaModel
from datasets import load_from_disk
from vc.trainer_encodec_vc_inference import get_ar_prediction_v3, get_ar_prediction_for_data, get_ar_prediction_get_audio, get_ar_prediction_audio_batch
from vc.encodec_model.nar_bart_model import NARBartForConditionalGeneration
import os
import numpy as np
import json
from typing import List, Tuple
import argparse
import random

sys.path.append('/work/b0990106x/trl/CLAPS')
from CLAPS.inference import infer

# Print current path
sys.path.append("/work/b0990106x/trl/vc")
import vc
importlib.reload(vc)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

def load_models_and_tokenizers(ar_checkpoint, nar_checkpoint):
    nar_model = NARBartForConditionalGeneration.from_pretrained(nar_checkpoint)
    ar_tokenizer = AutoTokenizer.from_pretrained(ar_checkpoint)
    nar_tokenizer = AutoTokenizer.from_pretrained(nar_checkpoint)
    ar_tokenizer.pad_token = ar_tokenizer.eos_token
    return nar_model, ar_tokenizer, nar_tokenizer

def convert_array_to_tensor_format(audio_array):
    tensor_audio = torch.tensor(audio_array, dtype=torch.float32)    
    return [tensor_audio]

def get_reward_mos(output_path, base_path):
    args_nisqa = {
        "mode": "predict_file",
        "pretrained_model": f"{base_path}/NISQA/weights/nisqa.tar",
        "deg": output_path,
        "data_dir": None,
        "output_dir": f"{base_path}/NISQA/result/",
        "csv_file": None,
        "csv_deg": None,
        "num_workers": 0,
        "bs": 1,
        "ms_channel": None,
        'ms_max_segments':3000
    }
    args_nisqa["tr_bs_val"] = args_nisqa["bs"]
    args_nisqa["tr_num_workers"] = args_nisqa["num_workers"]
    nisqa = nisqaModel(args_nisqa)
    try:
        prediction = nisqa.predict()
        reward = float(prediction["mos_pred"].iloc[0])
        # print("Reward:", reward)
        return reward
    except Exception as e:
        print("Error:", e)
        print("get_reward function end ___________________________")
        # raise
        return None

def get_reward_length(token_list):
    # count the length of the token list
    # print(token_list)
    token_list = [int(token.split("_")[2]) for token in token_list]
    length = len(token_list)

    score_level = [10, 50, 100, 200, 300]
    if length < score_level[0]:
        reward = 5
    elif length < score_level[1]:
        reward = 4 + ((score_level[1] - length) / (score_level[1] - score_level[0]))
    elif length < score_level[2]:
        reward = 3 + ((score_level[2] - length) / (score_level[2] - score_level[1]))
    elif length < score_level[3]:
        reward = 2 + ((score_level[3] - length) / (score_level[3] - score_level[2]))
    elif length < score_level[4]:
        reward = 1 + ((score_level[4] - length) / (score_level[4] - score_level[3]))
    else:
        reward = 0

    return reward, length

def get_reward_claps(clap_model, 
                     accelerator,
                     prompts, 
                     wavs, 
                     sr = 24000,
                     text_enc_name = "google/flan-t5-large",
                     text_enc_dim = 1024,
                     text_blstm_dim = 256,
                     speech_enc_name = "wavlm",
                     speech_enc_dim = 768,
                     speech_blstm_dim = 256,
                     rep_dim = 512,
                     sub_dim = 0,
                     n_sub = 1,
                     ckpt_pth='/work/b0990106x/trl/CLAPS/pretrained/7d/cp_claps_blstm_m_50k_v3/cp_0045000',
                     project_dir = "cp_claps"
                     ):
    # the function infer 
    # prompt 
    # wavs tensor([[]])
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

    cosine_sim = infer(clap_model, accelerator, prompts, wavs)
    reward = cosine_sim.item()

    return reward

def get_reward_even_token(token_list):
    token_list = [int(token.split("_")[2]) for token in token_list]
    # print("token_list",token_list)
    even_count = 0
    for token in token_list:
        if token % 2 == 0:
            even_count += 1
    reward = (even_count/len(token_list)) * 5
    percent = even_count/len(token_list)
    total = even_count
    return reward, percent, total

def process_and_get_mos_reward(model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, args_predict, episode_counter=0, base_path="/work/b0990106x/trl", temperature = 1.0):
    _, _, output_path_ckpt = get_ar_prediction_v3(args_predict, model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, episode_counter, temperature = temperature)
    reward = get_reward_mos(output_path_ckpt, base_path)
    return reward

# def process_and_get_claps_reward(model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, args_predict, episode_counter=0, base_path="/work/b0990106x/trl", temperature = 1.0):
#     decode_ar, wav = get_ar_prediction_get_audio(
#         args_predict, model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, episode_counter, temperature=temperature
#     )
#     tensor_wav = convert_array_to_tensor_format(wav)
#     if tensor_wav[0].shape[0]==1:
#         tensor_wav[0] = tensor_wav[0].squeeze(0)

#     reward = get_reward_claps(prompts = instruction, wavs = tensor_wav)
#     return reward

def process_and_get_claps_reward_batch(model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, args_predict, clap_model, accelerator, episode_counter=0, base_path="/work/b0990106x/trl", temperature = 1.0):
    audio_list, _ = get_ar_prediction_audio_batch(
        args_predict, model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, episode_counter, temperature=temperature
    )
    
    reward_list = []

    # audio_list might consist of None values
    
    for i, audio in enumerate(audio_list): 
        if audio is not None:
        # audio ---> tensor([])
            tensor_audio = convert_array_to_tensor_format(audio)
            if tensor_audio[0].shape[0]==1:
                tensor_audio[0] = tensor_audio[0].squeeze(0)
            # print(tensor_audio)
            reward = get_reward_claps(clap_model=clap_model, accelerator=accelerator, prompts = instruction[i], wavs = tensor_audio)
        else:
            reward = 0 
        reward_list.append(reward)
    
    return reward_list

def process_and_get_even_reward(model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, args_predict, episode_counter=0, base_path="/work/b0990106x/trl", temperature = 1.0):
    decode_ar = get_ar_prediction_for_data(args_predict, model, ar_tokenizer, src_encodec, instruction)
    list_decode_ar = decode_ar.flatten().tolist()
    filtered_decode_ar_list = list_decode_ar[2:-1]
    decode_ar_tokens = ar_tokenizer.convert_ids_to_tokens(filtered_decode_ar_list)
    tokenized_decode_ar = ar_tokenizer.convert_tokens_to_string(decode_ar_tokens)
    reward, even_percent, even_total = get_reward_even_token(decode_ar_tokens)
    return reward, even_percent, even_total

def process_and_get_length_reward(model, ar_tokenizer, src_encodec, instruction, args_predict, temperature = 1.0):
    decode_ar = get_ar_prediction_for_data(args_predict, model, ar_tokenizer, src_encodec, instruction)
    list_decode_ar = decode_ar.flatten().tolist()
    filtered_decode_ar_list = list_decode_ar[2:-1]
    decode_ar_tokens = ar_tokenizer.convert_ids_to_tokens(filtered_decode_ar_list)
    reward, length = get_reward_length(decode_ar_tokens)
    return reward, length

def calculate_metrics(rewards):
    metrics = {
        "mean": np.mean(rewards),
        "median": np.median(rewards),
        "std_dev": np.std(rewards),
        "variance": np.var(rewards),
        "min": np.min(rewards),
        "max": np.max(rewards),
        "25th_percentile": np.percentile(rewards, 25),
        "75th_percentile": np.percentile(rewards, 75),
        "rewards": rewards
    }
    return metrics

def eval_dpo_token_length(
    ar_checkpoint, # checkpoint for the AR model
    trained_model_checkpoint, # path checkpoint for the trained model
    args_predict, # arguments for the prediction
    all_src_encodec,
    all_instruction,
    eval_data_len=1000,
    num_evaluations = 10,
    selected_indices=None,  # Add this parameter
    device = "cuda" if torch.cuda.is_available() else "cpu"
):
    ar_tokenizer = AutoTokenizer.from_pretrained(ar_checkpoint)
    ar_tokenizer.pad_token = ar_tokenizer.eos_token
    
    trained_model = BartForConditionalGeneration.from_pretrained(trained_model_checkpoint, return_dict=True)
    trained_model.to(device)

    all_data_metrics = []

    data_indices = selected_indices if selected_indices is not None else range(len(all_instruction))
    data_len = len(data_indices)
    target_rewards = min(eval_data_len, data_len)

    count_rewards = 0
    i = 0

    while count_rewards < target_rewards:
        if i >= data_len:
            print("Exceeded initial data length.")
            break
        
        idx = data_indices[i]
        instruction = all_instruction[idx]
        src_encodec = all_src_encodec[idx]
        size_of_packed_input = (len(src_encodec[0]) + len(ar_tokenizer(instruction)["input_ids"][1:-1]) + 3)
        
        if size_of_packed_input <= 1024 and size_of_packed_input > 4:
            rewards = []
            output_lengths = []

            for _ in range(num_evaluations):
                reward, length = process_and_get_length_reward(
                    model = trained_model, 
                    ar_tokenizer = ar_tokenizer, 
                    src_encodec = src_encodec, 
                    instruction = instruction,
                    args_predict=args_predict,
                    temperature=1.0
                )
                rewards.append(reward)
                output_lengths.append(length)
                
            filtered_rewards = [r for r in rewards if r is not None]
            filtered_output_lengths = [r for r in output_lengths if r is not None]
            metrics = {
                "average_output_length": np.mean(filtered_output_lengths),
                "mean": np.mean(filtered_output_lengths),
                "median": np.median(filtered_output_lengths),
                "std_dev": np.std(filtered_output_lengths),
                "variance": np.var(filtered_output_lengths),
                "min": np.min(filtered_output_lengths),
                "max": np.max(filtered_output_lengths),
                "25th_percentile": np.percentile(filtered_output_lengths, 25),
                "75th_percentile": np.percentile(filtered_output_lengths, 75),
                "rewards": rewards,
                "output_lengths": output_lengths
            }
            all_data_metrics.append({
                "idx": idx,
                "metrics": metrics
            })
            count_rewards += 1
        else:
            print(f"Skipping data point {idx} due to insufficient packed input size.")
        i += 1

    return all_data_metrics

# def eval_dpo_even_token(
#     ar_checkpoint, # checkpoint for the AR model
#     nar_checkpoint, # checkpoint for the NAR model
#     trained_model_checkpoint, # path checkpoint for the trained model
#     args_predict, # arguments for the prediction
#     all_src_encodec,
#     all_instruction,
#     iteration,
#     eval_data_len=1000,
#     num_evaluations = 10,
#     selected_indices=None,  # Add this parameter
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     ):
    
#     nar_model, ar_tokenizer, nar_tokenizer = load_models_and_tokenizers(ar_checkpoint, nar_checkpoint)
#     trained_model = BartForConditionalGeneration.from_pretrained(trained_model_checkpoint, return_dict=True)
#     trained_model.to(device)

#     all_data_metrics = []

#     data_indices = selected_indices if selected_indices is not None else range(len(all_instruction))
#     data_len = len(data_indices)
#     target_rewards = min(eval_data_len, data_len)

#     count_rewards = 0
#     i = 0

#     while count_rewards < target_rewards:
#         if i >= data_len:
#             print("Exceeded initial data length.")
#             break
        
#         idx = data_indices[i]
#         instruction = all_instruction[idx]
#         src_encodec = all_src_encodec[idx]
#         size_of_packed_input = (len(src_encodec[0]) + len(ar_tokenizer(instruction)["input_ids"][1:-1]) + 3)
        
#         if size_of_packed_input <= 1024 and size_of_packed_input > 4:
#             rewards = []
#             even_percent_list = []
#             even_total_list = []

#             for j in range(num_evaluations):
#                 reward, even_percent, even_total = process_and_get_even_reward(
#                     trained_model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, args_predict,
#                     episode_counter=f"eval_{iteration}_data_{idx}_{j}"
#                 )
#                 rewards.append(reward)
#                 even_percent_list.append(even_percent)
#                 even_total_list.append(even_total)

#             filtered_rewards = [r for r in rewards if r is not None]
#             filtered_even_total = [r for r in even_total_list if r is not None]
#             metrics = {
#                 "mean": np.mean(filtered_rewards),
#                 "median": np.median(filtered_rewards),
#                 "std_dev": np.std(filtered_rewards),
#                 "variance": np.var(filtered_rewards),
#                 "min": np.min(filtered_rewards),
#                 "max": np.max(filtered_rewards),
#                 "25th_percentile": np.percentile(filtered_rewards, 25),
#                 "75th_percentile": np.percentile(filtered_rewards, 75),
#                 "rewards": rewards,
#                 "even_percent": even_percent_list,
#                 "even_total": even_total_list,
#                 "even_total_avg": np.mean(filtered_even_total)
#             }
#             all_data_metrics.append({
#                 "idx": idx,
#                 "metrics": metrics
#             })
#             count_rewards += 1
#         else:
#             print(f"Skipping data point {idx} due to insufficient packed input size.")
#         i += 1

#     return all_data_metrics
    
#     # filtered_rewards = [r for r in trained_model_rewards if r is not None]

#     # filtered_trained_model_rewards = [r for r in trained_model_rewards if r is not None]

#     # trained_model_metrics = calculate_metrics(filtered_trained_model_rewards)

#     # return trained_model_metrics, trained_model_rewards, trained_model_even_percent, trained_model_even_total

def eval_dpo_mos(
            # ar_checkpoint, # checkpoint for the AR model
            # nar_checkpoint, # checkpoint for the NAR model
            # trained_model_checkpoint, # path checkpoint for the trained model
            nar_model,
            ar_tokenizer,
            nar_tokenizer,
            trained_model,
            args_predict, # arguments for the prediction
            all_src_encodec,
            all_instruction,
            iteration, # can be number or text
            num_evaluations = 10,
            eval_data_len=1000,
            selected_indices=None,  # Add this parameter
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ):
    # load models and tokenizer
    # nar_model, ar_tokenizer, nar_tokenizer = load_models_and_tokenizers(ar_checkpoint, nar_checkpoint)
    # trained_model = BartForConditionalGeneration.from_pretrained(trained_model_checkpoint, return_dict=True)
    trained_model.to(device)
    # List for storing rewards
    all_data_metrics = []
    all_rewards = []

    # Get data indices
    # If selected_indices is None, use all data
    # eval_data_len controls the number of data points to evaluate unless the data is exhausted
    data_indices = selected_indices if selected_indices is not None else range(len(all_instruction))
    data_len = len(data_indices)
    count_rewards = 0
    target_rewards = min(eval_data_len, data_len)  

    i = 0
    while count_rewards < target_rewards:
        if i >= data_len:
            print("Exceeded initial data length.")
            break
        
        idx = data_indices[i]
        instruction = all_instruction[idx]
        src_encodec = all_src_encodec[idx]
        size_of_packed_input = (len(src_encodec[0]) + len(ar_tokenizer(instruction)["input_ids"][1:-1]) + 3)
        
        if size_of_packed_input <= 1024 and size_of_packed_input > 4:
            rewards = []
            for j in range(num_evaluations):
                # Process with trained model
                trained_model_reward = process_and_get_mos_reward(trained_model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, args_predict, episode_counter=f"eval_{iteration}_data_{idx}_{j}")
                # print(f"Trained model reward: {trained_model_reward}")
                rewards.append(trained_model_reward)

            filtered_trained_model_rewards = [r for r in rewards if r is not None]
            if filtered_trained_model_rewards != []:
                trained_model_metrics = calculate_metrics(filtered_trained_model_rewards)
                all_data_metrics.append({
                    "idx": idx,
                    "metrics": trained_model_metrics
                })
                all_rewards.append(rewards)
                count_rewards += 1
            else: 
                all_data_metrics.append({
                    "idx": idx,
                    "metrics": None
                })
                all_rewards.append(None)
        else:
            print(f"Skipping data point {idx} due to insufficient packed input size.")
        i += 1
    
    return all_data_metrics, all_rewards

# def eval_dpo_claps(
#         ar_checkpoint, # checkpoint for the AR model
#         nar_checkpoint, # checkpoint for the NAR model
#         trained_model_checkpoint, # path checkpoint for the trained model
#         args_predict, # arguments for the prediction
#         all_src_encodec,
#         all_instruction,
#         iteration, # can be number or text
#         num_evaluations = 10,
#         eval_data_len=1000,
#         selected_indices=None,  # Add this parameter
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         ):
#     # load models and tokenizer
#     nar_model, ar_tokenizer, nar_tokenizer = load_models_and_tokenizers(ar_checkpoint, nar_checkpoint)
#     trained_model = BartForConditionalGeneration.from_pretrained(trained_model_checkpoint, return_dict=True)
#     trained_model.to(device)
#     # List for storing rewards
#     all_data_metrics = []
#     all_rewards = []

#     # Get data indices
#     # If selected_indices is None, use all data
#     # eval_data_len controls the number of data points to evaluate unless the data is exhausted
#     data_indices = selected_indices if selected_indices is not None else range(len(all_instruction))
#     data_len = len(data_indices)
#     count_rewards = 0
#     target_rewards = min(eval_data_len, data_len)  
    
#     i = 0
#     while count_rewards < target_rewards:
#         if i >= data_len:
#             print("Exceeded initial data length.")
#             break
        
#         idx = data_indices[i]
#         instruction = all_instruction[idx]
#         src_encodec = all_src_encodec[idx]
#         size_of_packed_input = (len(src_encodec[0]) + len(ar_tokenizer(instruction)["input_ids"][1:-1]) + 3)
        
#         if size_of_packed_input <= 1024 and size_of_packed_input > 4:
#             rewards = []
            
#             for j in range(num_evaluations):
#                 # Process with trained model
#                 set_seed(j+42)
#                 trained_model_reward = process_and_get_claps_reward(trained_model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, args_predict, episode_counter=f"eval_{iteration}_data_{idx}_{j}")
#                 print(f"Trained model reward: {trained_model_reward}")
#                 rewards.append(trained_model_reward)

#             filtered_trained_model_rewards = [r for r in rewards if r is not None]
#             if filtered_trained_model_rewards != []:
#                 trained_model_metrics = calculate_metrics(filtered_trained_model_rewards)
#                 all_data_metrics.append({
#                     "idx": idx,
#                     "metrics": trained_model_metrics
#                 })
#                 all_rewards.append(rewards)
#                 count_rewards += 1
#             else: 
#                 all_data_metrics.append({
#                     "idx": idx,
#                     "metrics": None
#                 })
#                 all_rewards.append(None)
#         else:
#             print(f"Skipping data point {idx} due to insufficient packed input size.")
#         i += 1
    
#     return all_data_metrics, all_rewards

def eval_dpo_claps_batch(
        # ar_checkpoint, # checkpoint for the AR model
        # nar_checkpoint, # checkpoint for the NAR model
        nar_model,
        ar_tokenizer,
        nar_tokenizer,
        trained_model,
        # trained_model_checkpoint, # path checkpoint for the trained model
        args_predict, # arguments for the prediction
        all_src_encodec,
        all_instruction,
        iteration, # can be number or text
        clap_model,
        accelerator,
        num_evaluations = 10,
        eval_data_len=1000,
        selected_indices=None,  # Add this parameter
        device = "cuda" if torch.cuda.is_available() else "cpu",
        ):
    # load models and tokenizer
    # nar_model, ar_tokenizer, nar_tokenizer = load_models_and_tokenizers(ar_checkpoint, nar_checkpoint)
    # trained_model = BartForConditionalGeneration.from_pretrained(trained_model_checkpoint, return_dict=True)
    trained_model.to(device)
    # List for storing rewards
    all_data_metrics = []
    all_rewards = []

    # Get data indices
    # If selected_indices is None, use all data
    # eval_data_len controls the number of data points to evaluate unless the data is exhausted
    data_indices = selected_indices if selected_indices is not None else range(len(all_instruction))
    data_len = len(data_indices)
    count_rewards = 0
    target_rewards = min(eval_data_len, data_len)  

    i = 0
    while count_rewards < target_rewards:
        if i >= data_len:
            print("Exceeded initial data length.")
            break
        
        idx = data_indices[i]
        instruction = all_instruction[idx]
        src_encodec = all_src_encodec[idx]
        size_of_packed_input = (len(src_encodec[0]) + len(ar_tokenizer(instruction)["input_ids"][1:-1]) + 3)
        
        batch_src_encodec = [src_encodec] * num_evaluations
        batch_instruction = [instruction] * num_evaluations
        if size_of_packed_input <= 1024 and size_of_packed_input > 4:
            rewards = process_and_get_claps_reward_batch(trained_model, nar_model, ar_tokenizer, nar_tokenizer, batch_src_encodec, batch_instruction, args_predict, episode_counter=f"eval_{iteration}_data_{idx}", clap_model=clap_model, accelerator=accelerator)
            filtered_trained_model_rewards = [r for r in rewards if r is not None]
            if filtered_trained_model_rewards != []:
                trained_model_metrics = calculate_metrics(filtered_trained_model_rewards)
                all_data_metrics.append({
                    "idx": idx,
                    "metrics": trained_model_metrics
                })
                all_rewards.append(rewards)
                count_rewards += 1
            else: 
                all_data_metrics.append({
                    "idx": idx,
                    "metrics": None
                })
                all_rewards.append(None)
        else:
            print(f"Skipping data point {idx} due to insufficient packed input size.")
        i += 1
    
    return all_data_metrics, all_rewards
