from vc.trainer_encodec_vc_inference import get_ar_prediction_batch, get_ar_prediction_v3
from transformers import BartForConditionalGeneration, AutoTokenizer
from vc.encodec_model.nar_bart_model import NARBartForConditionalGeneration
from dpo_eval import extract_data_from_json, get_reward_mos
from types import SimpleNamespace
import torch
import os
import time
import numpy as np

# Constants and settings
ar_checkpoint = "lca0503/speech-chatgpt-base-ar-v2-epoch10-wotrans"
nar_checkpoint = "lca0503/speech-chatgpt-base-nar-v2-epoch4-wotrans"
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load models and tokenizers
print("Loading models and tokenizers...")
nar_model = NARBartForConditionalGeneration.from_pretrained(nar_checkpoint)
ar_tokenizer = AutoTokenizer.from_pretrained(ar_checkpoint)
nar_tokenizer = AutoTokenizer.from_pretrained(nar_checkpoint)
ar_model = BartForConditionalGeneration.from_pretrained(ar_checkpoint, return_dict=True)
ar_model.to(device)
nar_model.to(device)

# Extract data
print("Extracting data...")
all_src_encodec, all_instruction, all_tgt_encodec = extract_data_from_json('dpo_data/src_encodec.json')
base_path = "/work/b0990106x/trl"
ts = "batch_inference"
args_predict = SimpleNamespace(output_path=f"{base_path}/output/{ts}/example.wav", seed=0, device=device)
agent_output_dir = os.path.join(base_path, "output", ts)  # Path of saving the generated audio for reward model to evaluate
os.makedirs(agent_output_dir, exist_ok=True)



testing = False  # Set to True to run a single test with a fixed data size
num_trials = 5  # Number of times to repeat the timing and reward collection
data_size = 2
max_data_size = 2

# all_src_encodec = all_src_encodec[:3]  # Take the first 2 examples
# all_instruction = all_instruction[:3]
# all_tgt_encodec = all_tgt_encodec[:3]

if not testing:
# Timing for the second block (2)
    while data_size <= max_data_size:   
        print(f"Testing with {data_size} examples...")  

        current_src_encodec = all_src_encodec[:data_size]
        current_instruction = all_instruction[:data_size]

        try:
            torch.cuda.empty_cache()
            start_time_2 = time.time()
            d, e, f, g = get_ar_prediction_batch(args_predict, ar_model, nar_model, ar_tokenizer, nar_tokenizer, current_src_encodec, current_instruction, episode_counter=3, temperature=1.0)
            end_time_2 = time.time()
            time_taken_2 = end_time_2 - start_time_2
            print(f"Time taken for (2): {time_taken_2:.4f} seconds")
            allocated_memory = torch.cuda.memory_allocated(device)
            reserved_memory = torch.cuda.memory_reserved(device)
            print(f"Time taken for {data_size} examples: {time_taken_2:.4f} seconds")
            print(f"GPU Memory Allocated: {allocated_memory / (1024 ** 3):.4f} GB")
            print(f"GPU Memory Reserved: {reserved_memory / (1024 ** 3):.4f} GB")
            data_size += 1  # Increase the data size for the next test
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Out of memory error at data size: {data_size}")
                torch.cuda.empty_cache()
            else:
                raise e
    
else:
    # Initialize lists to store timing and rewards
    times_1 = []
    times_2 = []
    rewards_1 = []
    rewards_2 = []

    current_src_encodec = all_src_encodec[:data_size]
    current_instruction = all_instruction[:data_size]

    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")

        audio_path_list = []
        
        # Timing for the first block (1)
        torch.cuda.empty_cache()  # Clear the cache before running the first block
        start_time_1 = time.time()
        for i in range(data_size):
            a, b, c = get_ar_prediction_v3(args_predict, ar_model, nar_model, ar_tokenizer, nar_tokenizer, current_src_encodec[i], current_instruction[i], episode_counter=i, temperature=1.0)
            audio_path_list.append(c)
        end_time_1 = time.time()
        time_taken_1 = end_time_1 - start_time_1
        times_1.append(time_taken_1)
        
        # Clear cache and reset models between tests
        torch.cuda.empty_cache()  # Clear CUDA cache to avoid overlap between tests
        ar_model = BartForConditionalGeneration.from_pretrained(ar_checkpoint, return_dict=True).to(device)
        nar_model = NARBartForConditionalGeneration.from_pretrained(nar_checkpoint).to(device)
        
        # Timing for the second block (2)
        start_time_2 = time.time()
        d, e, f, g = get_ar_prediction_batch(args_predict, ar_model, nar_model, ar_tokenizer, nar_tokenizer, current_src_encodec, current_instruction, episode_counter=data_size, temperature=1.0)
        audio_path_list.extend(g)
        end_time_2 = time.time()
        time_taken_2 = end_time_2 - start_time_2
        times_2.append(time_taken_2)
        
        # Evaluate rewards
        # Evaluate first block
        reward_trial_1 = []
        for i in range(data_size):
            reward_mos_1 = get_reward_mos(audio_path_list[i], base_path)
            reward_trial_1.append(reward_mos_1)
            reward_trial_1_filter_none = list(filter(None, reward_trial_1))
        rewards_1.append(np.mean(reward_trial_1))
        
        # Evaluate second block
        reward_trial_2 = []
        for j in range(data_size, 2 * data_size):
            reward_mos_2 = get_reward_mos(audio_path_list[j], base_path)
            reward_trial_2.append(reward_mos_2)
            reward_trial_2_filter_none = list(filter(None, reward_trial_2))
        rewards_2.append(np.mean(reward_trial_2))

        print(f"Time taken for (1): {time_taken_1:.4f} seconds")
        print(f"Time taken for (2): {time_taken_2:.4f} seconds")
        print(f"Reward for (1): {np.mean(reward_trial_1):.4f}")
        print(f"Reward for (2): {np.mean(reward_trial_2):.4f}")

    # Calculate average times and rewards
    avg_time_1 = np.mean(times_1)
    avg_time_2 = np.mean(times_2)
    avg_reward_1 = np.mean(rewards_1)
    avg_reward_2 = np.mean(rewards_2)

    print(f"\nAverage time taken for (1) over {num_trials} trials: {avg_time_1:.4f} seconds")
    print(f"Average time taken for (2) over {num_trials} trials: {avg_time_2:.4f} seconds")
    print(f"Average reward for (1) over {num_trials} trials: {avg_reward_1:.4f}")
    print(f"Average reward for (2) over {num_trials} trials: {avg_reward_2:.4f}")
