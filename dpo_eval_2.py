import sys
sys.path.append("/work/b0990106x/trl/vc")
import importlib
import vc
importlib.reload(vc)
import torch
from vc.trainer_encodec_vc_inference import get_ar_prediction_v3
from types import SimpleNamespace
from transformers import BartForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from NISQA.nisqa.NISQA_model import nisqaModel
from datasets import load_from_disk
from trl import DPOTrainer, DPOConfig, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
from vc.encodec_model.nar_bart_model import NARBartForConditionalGeneration
from datetime import datetime
import os
import json

# Define paths and device
base_path = "/work/b0990106x/trl"
device = "cuda" if torch.cuda.is_available() else "cpu"
ts = "beta05"
model_file = "beta05"

agent_output_dir = f"{base_path}/output/{ts}"

if not os.path.exists(agent_output_dir):
    os.makedirs(agent_output_dir)

# Prepare Model
all_src_encodec_layers = []
all_src_encodec = []
all_instruction = []

layer_len = 8
# data_len = len(dataset)

args_predict = SimpleNamespace(output_path=f"{base_path}/output/{ts}/example.wav", seed=0, device=device)
agent_input_dir = f"{base_path}/data-encodec"
test_dataset = load_from_disk(agent_input_dir)
data_len = len(test_dataset)
print("data_len:", data_len)

for i in range(layer_len):
    all_src_encodec_layers.append(test_dataset[f"src_encodec_{i}"])

for i in range(data_len):
    src_encodec = []
    for j in range(layer_len):
        src_encodec.append(all_src_encodec_layers[j][i])
    all_src_encodec.append(src_encodec)

    all_instruction.append(test_dataset["instruction"][i])

ar_checkpoint = "lca0503/speech-chatgpt-base-ar-v2-epoch10-wotrans"
nar_checkpoint = "lca0503/speech-chatgpt-base-nar-v2-epoch4-wotrans"

nar_model = NARBartForConditionalGeneration.from_pretrained(nar_checkpoint)
ar_tokenizer = AutoTokenizer.from_pretrained(ar_checkpoint)
nar_tokenizer = AutoTokenizer.from_pretrained(nar_checkpoint)
ar_tokenizer.pad_token = ar_tokenizer.eos_token

# Function to calculate reward
import time
def get_reward(output_path):
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
    }
    args_nisqa["tr_bs_val"] = args_nisqa["bs"]
    args_nisqa["tr_num_workers"] = args_nisqa["num_workers"]
    nisqa = nisqaModel(args_nisqa)
    try:
        prediction = nisqa.predict()
        reward = float(prediction["mos_pred"].iloc[0])
        print("Reward:", reward)
        return reward
    except Exception as e:
        print("Error:", e)
        print("get_reward function end ___________________________")
        return None
    
def process_and_get_scores(model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, episode_counter=0):
    temp, decode_ar,output_path_ckpt = get_ar_prediction_v3(args_predict, model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, episode_counter)
    list_decode_ar = decode_ar.flatten().tolist()
    time.sleep(0.5)
    reward = get_reward(output_path_ckpt)
    return reward

# Load trained model
trained_model = BartForConditionalGeneration.from_pretrained(f"{base_path}/model_output/{model_file}/dpo_model")
model = BartForConditionalGeneration.from_pretrained(ar_checkpoint, return_dict=True)

# Initialize lists to store rewards
trained_model_rewards = []

i = 0 
count_rewards = 0
target_rewards = 1000

while count_rewards < target_rewards:
    if i >= data_len:
        print("Exceeded initial data length.")
        break
    instruction = all_instruction[i]
    src_encodec = all_src_encodec[i]
    size_of_packed_input = (len(src_encodec[0]) + len(ar_tokenizer(instruction)["input_ids"][1:-1]) + 3)
    
    if size_of_packed_input <= 1024 and size_of_packed_input > 4:
        # Process with trained model
        trained_model.to(device)
        trained_model_reward = process_and_get_scores(trained_model, nar_model, ar_tokenizer, nar_tokenizer, src_encodec, instruction, episode_counter=i)
        trained_model_rewards.append(trained_model_reward)

        count_rewards += 1
    else:
        print(f"Skipping data point {i} due to insufficient packed input size.")
    i += 1


print("Trained Model Rewards:", trained_model_rewards)

import numpy as np
# Filter out None values
filtered_trained_model_rewards = [r for r in trained_model_rewards if r is not None]

# Calculate and print metrics
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
    }
    return metrics

trained_model_metrics = calculate_metrics(filtered_trained_model_rewards)

# print("Old Model Rewards:", old_model_rewards)
# print("Trained Model Rewards:", trained_model_rewards)
print("Trained Model Metrics:", trained_model_metrics)
metrics = {
    "trained_model": trained_model_metrics,
    "trained_model_rewards": trained_model_rewards,
}

with open(f"{base_path}/output/{ts}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

