import gym
import logging
import random
import sys
import torch
from torch import autocast

import sys
sys.path.append("/work/b0990106x/TextRL/vc")
from vc.trainer_encodec_vc_inference import get_ar_prediction


class TextRLEnv(gym.Env):
    def __init__(
        self,
        model,
        tokenizer,
        nar_model,
        nar_tokenizer,
        args_predict,
        observation_input=[],
        max_length=1000,
        compare_sample=2,
        unfreeze_layer_from_past=0,
    ):
        try:
            tokvocab = tokenizer.get_vocab()
        except:
            tokvocab = tokenizer.vocab
            pass
        vocabs = list(dict(sorted(tokvocab.items(), key=lambda item: item[1])).keys())
        self.action_space = gym.spaces.Discrete(len(vocabs))
        self.actions = vocabs
        self.model = model
        self.tokenizer = tokenizer
        self.nar_model = nar_model
        self.nar_tokenizer = nar_tokenizer
        self.args_predict = args_predict
        self.observation_space = observation_input
        self.compare_sample = compare_sample
        self.target_table = {}
        self.episode_counter = -1
        self.unfreeze_layer_from_past = (
            1 if unfreeze_layer_from_past == 0 else unfreeze_layer_from_past
        )
        self.env_max_length = min(
            max(self.model.config.max_length, self.tokenizer.model_max_length),
            max_length,
        )
        print("model name: ", self.model.__class__.__name__)
        self.reset()

        self.gen_stop_toks = []
        logging.disable(sys.maxsize)
        if self.tokenizer.sep_token:
            self.gen_stop_toks.append(self.tokenizer.sep_token)
        if self.tokenizer.eos_token:
            self.gen_stop_toks.append(self.tokenizer.eos_token)
        logging.disable(logging.NOTSET)

    def step(
        self, action
    ):  # This is the step function that is called by the environment
        predicted, finish, predicted_str = self._predict(vocab_id=action)
        reward = self.get_reward(self.input_item, predicted, finish) 
        # self.input_item is the entire input of the current episode 
        # predicted is the predicted output of the current episode
        # finish is a boolean value that indicates whether the episode has finished
        self.predicted = predicted
        return (
            self._get_obs(predicted),
            reward,
            finish,
            {"predicted_str": predicted_str},
        )

    def get_reward(self, input_item, predicted_list, finish):
        reward = [0] * self.compare_sample
        return reward

    def gat_obs_input(self, input_item):
        return input_item["input"]

    @autocast("cuda")
    def reset(
        self, input_item=None
    ):  # reset is used to reset the environment to its initial state
        print("----------------------------- reset -----------------------------")
        self.predicted = [
            []
        ] * self.compare_sample  # if compare_sample is 2, then self.predicted = [[], []]
        self.predicted_end = [
            False
        ] * self.compare_sample  # if compare_sample is 2, then self.predicted_end = [False, False]
        self.input_item = {"input": ""}
        self.episode_counter += 1

        while True:
            if input_item is None:
                self.input_item = random.choice(self.observation_space)
            else:
                self.input_item = input_item

            # Reference here for the input_item (20240416)
            single_src_encodec = self.input_item["src_encodec"]
            single_instruction = self.input_item["instruction"]
            self.single_src_encodec = single_src_encodec
            self.single_instruction = single_instruction
            size_of_packed_input = (
                len(single_src_encodec[0])
                + len(self.tokenizer(single_instruction)["input_ids"][1:-1])
                + 3
            )
            # print("size_of_packed_input: ", size_of_packed_input)
            # print("single_instruction: ", single_instruction)

            if size_of_packed_input > 1024 or size_of_packed_input < 4:
                print(
                    f"Notice: Packed input size too large or too small for processing: {size_of_packed_input} elements. Instruction: '{single_instruction}'"
                )
                continue  # Continue to select a new random item

            break  # Break the loop if size is within limits
        if self.input_item["input"] == "":
            decode_ar = get_ar_prediction(
                self.args_predict,
                self.model,
                self.nar_model,
                self.tokenizer,
                self.nar_tokenizer,
                self.single_src_encodec,
                self.single_instruction,
                self.episode_counter,
            )
            decode_ar_str = self.tokenizer.convert_tokens_to_string(
                [f"v_tok_{u}" for u in decode_ar]
            )
            self.input_item["input"] = decode_ar_str
            print("Input is set to: ", self.input_item["input"])

        return self._get_obs(self.predicted)

    @autocast("cuda")
    def _get_obs(self, predicted=[]):
        with torch.inference_mode():
            obs_list = []
            # print("predicted: ", predicted)
            for p_text in predicted:
                # p_text_str = self.tokenizer.convert_tokens_to_string(p_text)
                # print("p_text_str: ", p_text_str)
                if (
                    len([k for k, v in self.model.named_parameters() if "decoder" in k])
                    > 0
                ):
                    # print("(case1) self.model.__class__.__name__: ", self.model.__class__.__name__)
                    feature_dict = self.tokenizer(
                        [self.gat_obs_input(self.input_item)],
                        return_tensors="pt",
                        return_token_type_ids=False,
                        add_special_tokens=True,
                    ).to(self.model.device)
                    if len(p_text) > 0:
                        decoder_input_ids = [
                            self.model.config.decoder_start_token_id
                        ] + self.tokenizer.convert_tokens_to_ids(p_text)
                        dec_input = torch.tensor([decoder_input_ids]).to(
                            self.model.device
                        )
                        feature_dict["decoder_input_ids"] = dec_input
                    else:
                        feature_dict["decoder_input_ids"] = torch.tensor(
                            [[self.model.config.decoder_start_token_id]]
                        ).to(self.model.device)
                    with torch.cuda.amp.autocast(enabled=False):
                        prediction = self.model(
                            **feature_dict, output_hidden_states=True
                        )
                    outputs = prediction.decoder_hidden_states[
                        -self.unfreeze_layer_from_past
                    ].squeeze(0)
                else:
                    print("error")
                obs_list.append(outputs.data[-1])
            return torch.stack(obs_list)

    def _predict(self, vocab_id):
        predicted_list = {}
        predicted_list_end = {}
        with torch.inference_mode():
            for i, (v_id, predicted, predicted_end) in enumerate(
                zip(vocab_id, self.predicted, self.predicted_end)
            ):
                predicted_list_end[i] = False
                if not predicted_end:
                    pred_word = self.actions[v_id]
                    if (
                        pred_word in self.gen_stop_toks
                        or len(pred_word) < 1
                        or len(predicted) > self.env_max_length
                    ):
                        predicted_list_end[i] = True
                        predicted_list[i] = [pred_word]
                    else:
                        predicted_list[i] = [pred_word]
                else:
                    predicted_list_end[i] = True
                    predicted_list[i] = [""]

            for i, (l, e) in enumerate(
                zip(predicted_list.values(), predicted_list_end.values())
            ):
                self.predicted[i] = self.predicted[i] + l
                self.predicted_end[i] = e

            return (
                self.predicted,
                all(self.predicted_end),
                [self.tokenizer.convert_tokens_to_string(i) for i in self.predicted],
            )