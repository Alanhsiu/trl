from vc.trainer_encodec_vc_inference import get_ar_prediction_batch, get_ar_prediction_v3
from transformers import BartForConditionalGeneration, AutoTokenizer
from vc.encodec_model.nar_bart_model import NARBartForConditionalGeneration
from dpo_eval import extract_data_from_json
from types import SimpleNamespace
import torch
import os

ar_checkpoint = "lca0503/speech-chatgpt-base-ar-v2-epoch10-wotrans"
nar_checkpoint = "lca0503/speech-chatgpt-base-nar-v2-epoch4-wotrans"
device = "cuda" if torch.cuda.is_available() else "cpu"

nar_model = NARBartForConditionalGeneration.from_pretrained(nar_checkpoint)
ar_tokenizer = AutoTokenizer.from_pretrained(ar_checkpoint)
nar_tokenizer = AutoTokenizer.from_pretrained(nar_checkpoint)
ar_tokenizer.pad_token = ar_tokenizer.eos_token
ar_model = BartForConditionalGeneration.from_pretrained(ar_checkpoint, return_dict=True)
ar_model.to(device)


all_src_encodec, all_instruction, all_tgt_encodec = extract_data_from_json('dpo_data/src_encodec.json')
base_path = "/work/b0990106x/trl"
ts = "audio_sample"

args_predict = SimpleNamespace(output_path=f"{base_path}/output/{ts}/example.wav", seed=0, device=device)

agent_output_dir = os.path.join(base_path, "output", ts) # Path of saving the generated audio for reward model to evaluate
os.makedirs(agent_output_dir, exist_ok=True)

# all_src_encodec = all_src_encodec[0] # it takes the first 2 examples
# all_instruction = all_instruction[0]
# all_tgt_encodec = all_tgt_encodec[0]
length_of_inference = 1

for i in range(length_of_inference):
    a, b, c = get_ar_prediction_v3(args_predict, ar_model, nar_model, ar_tokenizer, nar_tokenizer, all_src_encodec[i], all_instruction[i], episode_counter=i, temperature = 1.0)


