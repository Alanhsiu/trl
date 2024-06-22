# Test Chosen vs Rejected Reward
import json
import sys
sys.path.append("/work/b0990106x/trl/vc") 
from vc.trainer_encodec_vc_inference import get_ar_prediction_v2
import os 
from types import SimpleNamespace
from NISQA.nisqa.NISQA_model import nisqaModel
from tqdm import tqdm
import torch
from vc.encodec_model.nar_bart_model import NARBartForConditionalGeneration
from transformers import AutoTokenizer, BartForConditionalGeneration
from datasets import load_from_disk
import numpy as np

# load the model
ar_checkpoint = "lca0503/speech-chatgpt-base-ar-v2-epoch10-wotrans"
nar_checkpoint = "lca0503/speech-chatgpt-base-nar-v2-epoch4-wotrans"
device = "cuda" if torch.cuda.is_available() else "cpu"
base_path = "/work/b0990106x/trl"
agent_input_dir = f"{base_path}/data-encodec"
audio_output_dir = f"{base_path}/output/chosen_rejected_audio"
ar_tokenizer = AutoTokenizer.from_pretrained(ar_checkpoint)
ar_model = BartForConditionalGeneration.from_pretrained(ar_checkpoint)
nar_tokenizer = AutoTokenizer.from_pretrained(nar_checkpoint)
nar_model = NARBartForConditionalGeneration.from_pretrained(nar_checkpoint)

if not os.path.exists(audio_output_dir):
    os.makedirs(audio_output_dir)

with open("dpo_data_all_v2.json") as f:
    dpo_data = json.load(f)

# Function to calculate reward
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
        return reward
    except Exception as e:
        print("Error:", e)
        print("get_reward function end ___________________________")
        return None


prompts = dpo_data['prompt']
chosen = dpo_data['chosen']
rejected = dpo_data['rejected']
args_predict = SimpleNamespace(output_path=f"{base_path}/output/chosen_rejected_audio/example.wav", seed=0, device=device)

data_len = len(prompts)
print("data_len_1: ", data_len)

chosen_reward = []
rejected_reward = []

# Data
# Assuming `pack_inputs_v2` and `ar_tokenizer` are already defined

observation_list = []
decode_obs_input_str = []
all_src_encodec_layers = []

all_src_encodec = []
all_instruction = []
all_tgt_encodec = []

all_tgt_encodec_layers = []
layer_len = 8

dataset = load_from_disk(agent_input_dir)
print("whole dataset len", len(dataset))

for i in range(layer_len):
    all_src_encodec_layers.append(dataset[f"src_encodec_{i}"])
    all_tgt_encodec_layers.append(dataset[f"tgt_encodec_{i}"])

for i in range(len(dataset)):
    src_encodec = []
    tgt_encodec = []
    for j in range(layer_len):
        src_encodec.append(all_src_encodec_layers[j][i])
        tgt_encodec.append(all_tgt_encodec_layers[j][i])
    all_src_encodec.append(src_encodec)
    all_tgt_encodec.append(tgt_encodec)
    all_instruction.append(dataset["instruction"][i])
    
    size_of_packed_input = (len(all_src_encodec[i][0]) + len(ar_tokenizer(all_instruction[i])["input_ids"][1:-1]) + 3)
    # print("size_of_packed_input:", size_of_packed_input)
    if size_of_packed_input <= 1024 or size_of_packed_input < 4:
        observation_list.append(
            {
                "input": "",
                "src_encodec": [all_src_encodec_layers[j][i] for j in range(layer_len)],
                "instruction": all_instruction[i],
                "tgt_encodec": [all_tgt_encodec_layers[j][i] for j in range(layer_len)],
            }
        )
    # else:
    #     print(f"Notice: Packed input size too large for processing: {size_of_packed_input} elements. Instruction: '{all_instruction[i]}'")
data_len_2 = len(observation_list)
print("data_len_2:", data_len_2)
# make this for loop tqdm
for i in tqdm(range(data_len)):
    chosen_data = chosen[i]
    rejected_data = rejected[i]
    single_src_encodec = observation_list[i]["src_encodec"]
    single_instruction = observation_list[i]["instruction"]

    chosen_data_id = ar_tokenizer(chosen_data)["input_ids"][1:-1]
    rejected_data_id = ar_tokenizer(rejected_data)["input_ids"][1:-1]
    if len(chosen_data_id) <= 1022 and len(rejected_data_id) <= 1022:
        try:
            temp1 = get_ar_prediction_v2(args_predict, chosen_data_id, nar_model, ar_tokenizer, nar_tokenizer, single_src_encodec, single_instruction, 0)
            good_reward = get_reward(args_predict.output_path)
        except Exception as e:
            print("Error:", e)
            print("chosen_data_id:", chosen_data_id)
            print("single_src_encodec:", single_src_encodec[0])
            print("single_instruction:", single_instruction)
            good_reward = None
            raise
        try:
            temp2 = get_ar_prediction_v2(args_predict, rejected_data_id , nar_model, ar_tokenizer, nar_tokenizer, single_src_encodec, single_instruction, 1)
            bad_reward = get_reward(args_predict.output_path)
        except Exception as e:
            print("Error:", e)
            print("rejected_data_id:", rejected_data_id)
            print("single_src_encodec:", single_src_encodec[0])
            print("single_instruction:", single_instruction)
            bad_reward = None
            raise

        chosen_reward.append(good_reward)
        rejected_reward.append(bad_reward)

filtered_rejected_reward = [r for r in rejected_reward if r is not None]
filtered_chosen_reward = [r for r in chosen_reward if r is not None]

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

rejected_metrics = calculate_metrics(filtered_rejected_reward)
chosen_metrics = calculate_metrics(filtered_chosen_reward)

# Save rejected_metrics and chosen_metrics into a json file
metrics = {
    "rejected_metrics": rejected_metrics,
    "chosen_metrics": chosen_metrics,
    "rejected_reward": rejected_reward,
    "chosen_reward": chosen_reward,
}

with open("metrics.json", "w") as outfile:
    json.dump(metrics, outfile, indent=4)