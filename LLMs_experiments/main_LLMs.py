import argparse
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from utilis_llm import dataset_selection, method_selection, eval, eval_RESQ, eval_stepgame
import sys
sys.path.append("/DataSet")

def main(args):
    dataset_name = args.dataset.lower()
    LLM_model = args.LLM.lower()

    used_COT_prompting = args.used_same_COT_prompt

    if LLM_model == "llama":
        from llama import setup_llm_call, model_selection, pre_prompt, pre_prompt_org, pre_prompt_COT, step_game_prompt
        save_result_dir = "LLMs_experiments/llama_results/"
    else:
        from gpt import setup_llm_call, model_selection, pre_prompt, pre_prompt_org, pre_prompt_COT
        save_result_dir = "LLMs_experiments/gpt_results/"

    all_prompt = [pre_prompt_org,  # Zero-shot
                  pre_prompt_org,  # Few-shot
                  pre_prompt_COT,  # Chain of thought(COT)
                  pre_prompt if not used_COT_prompting else pre_prompt_COT,  # Formal relation COT
                  pre_prompt if not used_COT_prompting else pre_prompt_COT,  # Formal relation COT 2
                  pre_prompt if not used_COT_prompting else pre_prompt_COT,  # Formal relation COT 3
                  pre_prompt_COT,  # Chain of Symbols
                  pre_prompt_COT  # COT modified
                  ]
        
    dataset = dataset_selection(dataset_name)
    few_shot_data = method_selection(dataset_name,
                                             args.method,
                                             used_same_human=args.used_org_human_ex,
                                             )
    prompt = all_prompt[args.method]
    all_method = ["zero-shot", "few-shot", "COT", "COT-tripet", "COT-tripet-no-Q", "COT-tripet-new", "COS", "COT-new"]
    if not args.eval:
        print("Running {:} model with {:} dataset on {:}".format(LLM_model, dataset_name, all_method[args.method]))
        print(model_selection(args.model))
        if args.stepgame:
            prompt = step_game_prompt
        setup_llm_call(dataset, prompt,
                      model_id=model_selection(args.model),
                      save_file=args.save_file,
                      few_shot=few_shot_data,
                      debug=False)
        print("Saving filename:", args.save_file)
        #eval(save_result_dir + args.save_file + ".csv", COT=args.method > 1)
        if dataset_name == "resq":
            eval_RESQ(save_result_dir + args.save_file + ".csv", COT_included=args.method > 1, DK="DK")
    else:
        if dataset_name == "stepgame":
            eval_stepgame(save_result_dir + args.save_file + ".csv")
            return
        eval(save_result_dir + args.save_file + ".csv", COT=args.method > 1)
        if dataset_name == "resq":
            eval_RESQ(save_result_dir + args.save_file + ".csv", COT_included=args.method > 1, DK="DK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running GPT api")
    parser.add_argument("--LLM", dest="LLM", type=str, default="GPT")
    parser.add_argument("--model", dest="model", type=str, default="GPT3-5")
    parser.add_argument("--dataset", dest="dataset", type=str, default="Human", help="Option: Human, ResQ")
    parser.add_argument("--method", dest="method", type=int, default=0,
                        help="Option, 0:Zero-shot, 1:Few-shot, 2:COT")
    parser.add_argument("--save_file", dest="save_file", type=str, default="gpt_run")
    parser.add_argument("--eval", dest="eval", type=bool, default=False)
    parser.add_argument("--used_org_human_ex", dest="used_org_human_ex", type=bool, default=True)
    parser.add_argument("--used_same_COT_prompt", dest="used_same_COT_prompt", type=bool, default=True)
    parser.add_argument("--stepgame", dest="stepgame", type=bool, default=False)
    args = parser.parse_args()
    main(args)
    # reasoning_step = {}
    # data = RESQ_reader("DataSet/ReSQ/train_resq.json")
    # for story_txt, question_txt, label, steps in data:
    #     reasoning_step[steps] = reasoning_step.get(steps, 0) + 1
    # print(reasoning_step)
    # reasoning_step = {}
    # data = RESQ_reader("DataSet/ReSQ/test_resq.json")
    # for story_txt, question_txt, label, steps in data:
    #     reasoning_step[steps] = reasoning_step.get(steps, 0) + 1
    # print(reasoning_step)
    # reasoning_step = {}
    # data = RESQ_reader("DataSet/ReSQ/dev_resq.json")
    # find_result(dataset="human")
    # eval("GPT_results/GPT3-5-turbo_human_extraction_COS.csv", COT=True)
    # eval("GPT_results/GPT3-5-turbo_human_extraction_COT2.csv", COT=True)
    # eval_RESQ("GPT_results/GPT3-5-turbo_ResQ_COT.csv", COT_included=True, DK="DK")
    # eval_RESQ("GPT_results/GPT3-5-turbo_ResQ_few_shots.csv", DK="DK")
    # eval_RESQ("GPT_results/GPT3-5-turbo_resq_extraction_result_few_shots.csv", DK="No")
    # show_all_results("Human", "No")
    # main()
    # extract_and_find_result(do_extraction=False, do_find_result=False)
    # eval("GPT_results/GPT3-5-turbo_human_extraction.csv")

    # dataset = SPARTQA_reader("DataSet/human_test.json")
    # setup_gpt_api(dataset, "gpt-3.5-turbo", "GPT3-5-turbo_human_test_eval", prompt=pre_prompt_relation)
    # eval("GPT_results/GPT3-5-turbo_human_test.csv")
    # eval("GPT_results/GPT-3-5-turbo_human_COT_new.csv", COT=True)
    # eval("GPT_results/GPT3-5-turbo_human_extraction.csv")
    # eval("GPT_results/GPT3-5-turbo_human_extraction_few_shot.csv")
    # eval("GPT_results/GPT3-5-turbo_human_extraction_COT.csv", COT=True)
    # eval("GPT_results/GPT3-5-turbo_human_extraction_COT_Rule.csv", COT=True)