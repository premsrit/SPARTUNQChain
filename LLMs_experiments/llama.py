import pandas as pd
import transformers
import torch


from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

pre_prompt_org = ("You will be given story and question. Then, you will answer either only Yes or No "
                  "based on given story. Candidate answer: [Yes, No]")

pre_prompt = ("You will be given story and question. Then, you will answer either Yes or No "
              "with formal logical explanation based on given story. Candidate answer: [Yes, No]")

pre_prompt_COT = ("You will be given story and question. Then, you will answer either Yes or No "
                  "with explanation based on given story step by step. Candidate answer: [Yes, No]\n")

pre_prompt_relation = ("You will be given relation and question. "
                       "Then, you will answer either Yes or No based on given relations.")

pre_prompt_relation_COT = ("You will be given relation and question. "
                           "Then, you will answer only either Yes or No with explanation based on given relations. "
                           "If the answer is uncertain, you must answer No. Candidate answer: [Yes, No]")

pre_prompt_relation_COS = ("You will be given relation and question. "
                           "Then, you will answer only either Yes or No with explanation based on given relations. "
                           "If the answer is uncertain, you must answer No. Candidate answer: [Yes, No]")

step_game_prompt = ("You will be given story and question. Then, you will answer based on given story. Candidate answer: [left, right, above, below, lower-left, lower-right, upper-left, upper-right, overlap]. Only answer the final answer without explanation. If need, you can guess the answer\n")

def call_llm(messages, model, temperature=0.1, max_token=1024):

    prompt = model.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True)

    terminators = [
        model.tokenizer.eos_token_id,
        model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model(
        prompt,
        max_new_tokens=max_token,
        eos_token_id=terminators,
        temperature=0.000001,
        pad_token_id=model.tokenizer.eos_token_id
    )
    
    return outputs[0]["generated_text"][len(prompt):]


def setup_llm_call(dataset, prompt, model_id="meta-llama/Meta-Llama-3-8B-Instruct", 
                   save_file=None,
                   few_shot=(),
                   additional_prompt="",
                   save_columns=("story", "question", "label", "predict", "Reasoning_Steps"),
                   include_question=True,
                   debug=False):
    result_gpt = []

    model = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        )
    
    print(model_id, "debug:", debug)
    for story, question, label, reasoning_steps in tqdm(dataset):
        chat_msg = ([{"role": "system", "content": prompt + additional_prompt}]
                    + list(few_shot)
                    + [{"role": "user", "content": story + " " + (question if include_question else "")}])
        if debug:
            print(chat_msg)
            continue
        pred = call_llm(chat_msg, model)
        result_gpt.append([story, question, label, pred, reasoning_steps])

    if save_file:
        df = pd.DataFrame(result_gpt, columns=save_columns)
        df.to_csv("LLMs_experiments/llama_results/" + save_file + ".csv")


def model_selection(model_name):
    return "meta-llama/Meta-Llama-3-8B-Instruct"
