import os
import time

import openai
import json
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from dotenv import load_dotenv

load_dotenv()

pre_prompt_org = ("You are chat-GPT. You will be given story and question. Then, you will answer either only Yes or No "
                  "based on given story. Candidate answer: [Yes, No]")

pre_prompt = ("You are chat-GPT.You will be given story and question. Then, you will answer either Yes or No "
              "with formal logical explanation based on given story. Candidate answer: [Yes, No]")

pre_prompt_COT = ("You are chat-GPT. You will be given story and question. Then, you will answer either Yes or No "
                  "with explanation based on given story step by step. Candidate answer: [Yes, No]\n")

pre_prompt_relation = ("You are chat-GPT. You will be given relation and question. "
                       "Then, you will answer either Yes or No based on given relations.")

pre_prompt_relation_COT = ("You are chat-GPT. You will be given relation and question. "
                           "Then, you will answer only either Yes or No with explanation based on given relations. "
                           "If the answer is uncertain, you must answer No. Candidate answer: [Yes, No]")

pre_prompt_relation_COS = ("You are chat-GPT. You will be given relation and question. "
                           "Then, you will answer only either Yes or No with explanation based on given relations. "
                           "If the answer is uncertain, you must answer No. Candidate answer: [Yes, No]")

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORGANIZATION")  # HLR
)


def call_llm(message, model="gpt-3.5-turbo", temperature=0, max_token=1024, max_tried=10):
    chat_prompt = {
        "model": model,
        "messages": message,
        "temperature": temperature,
        "max_tokens": max_token
    }

    for _ in range(max_tried):
        try:
            respond = client.chat.completions.create(**chat_prompt)
            pred = respond.choices[0].message.content
            return pred
        except openai.BadRequestError as e:
            print(f"Invalid API request: {e}")
            return ""
        except openai.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
        except openai.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
        except openai.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
        except openai.APIError as e:
            # Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        except openai.AuthenticationError as e:
            print(f"OpenAI API authentication error: {e}")
            pass
        except:
            print("Other service error")
            pass
        time.sleep(12)

    return ""


def setup_llm_call(dataset, prompt, model_id="gpt-3.5-turbo", save_file=None,
                   few_shot=(),
                   additional_prompt="",
                   save_columns=("story", "question", "label", "predict", "Reasoning_Steps"),
                   include_question=True,
                   debug=False):
    result_gpt = []
    print(model_id, "debug:", debug)
    for story, question, label, reasoning_steps in tqdm(dataset):
        chat_msg = ([{"role": "system", "content": prompt + additional_prompt}]
                    + list(few_shot)
                    + [{"role": "user", "content": story + " " + (question if include_question else "")}])
        if debug:
            print(chat_msg)
            continue
        pred = call_llm(chat_msg, model=model_id)
        result_gpt.append([story, question, label, pred, reasoning_steps])

    if save_file:
        df = pd.DataFrame(result_gpt, columns=save_columns)
        df.to_csv("LLMs_experiments/gpt_results/" + save_file + ".csv")


def model_selection(model_name):
    if model_name == "GPT3-5":
        return "gpt-3.5-turbo"
    elif model_name == "GPT4-turbo":
        return "gpt-4-1106-preview"
    elif model_name == "GPT4":
        return "gpt-4-0613"
    return ""
