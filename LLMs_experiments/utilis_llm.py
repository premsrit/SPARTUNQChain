import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import time

extraction_relation_prompt = "You will do your best to extract all spatial relations from given story."


def RESQ_reader(file, reasoning=None):
    with open(file) as json_file:
        data = json.load(json_file)
    dataset = []
    count = 0
    for story in data["data"]:
        story_txt = " ".join(story['story'])
        run_id = 0
        for question in story["questions"]:
            steps = question["step_of_reasoning"] if isinstance(question["step_of_reasoning"], int) else 0
            question_txt = question["question"]
            label = question["answer"][0]
            dataset.append([story_txt, question_txt, label, steps])
            run_id += 1
            count += 1

    return dataset


def SPARTQA_reader(file):
    with open(file) as json_file:
        data = json.load(json_file)
    dataset = []
    count = 0
    for story in data["data"]:
        story_txt = " ".join(story['story'])
        for question in story["questions"]:
            question_txt = question["question"]
            if question["q_type"] != "YN":
                continue
            label = question["answer"][0] if question["answer"][0] != "DK" else "No"
            dataset.append([story_txt, question_txt, label, -1])
            count += 1

    return dataset


def StepGame_reader(data_dir, steps=10):
    files = ["qa" + str(i + 1) + "_test.json" for i in range(steps)]

    dataset = []
    print(data_dir, files)
    for step, file in enumerate(files):
        import os
        print(os.path.abspath(f"{data_dir}/{file}"))
        with open(f"{data_dir}/{file}", 'r') as json_file:
            data = json.load(json_file)
        for story_ind in list(data):
            story = data[story_ind]
            story_txt = " ".join(story["story"])

            question_txt = story["question"]
            # Variable need
            candidates = ["left", "right", "above", "below", "lower-left",
                          "lower-right", "upper-left", "upper-right", "overlap"]
            label = story["label"]
            dataset.append([story_txt, question_txt, label, step + 1])

    return dataset


def read_relations_question(relation_file, question_file):
    with open(relation_file) as csv_file:
        relation_data = pd.read_csv(csv_file)
    with open(question_file) as json_file:
        data = json.load(json_file)
    count = 0
    relations = relation_data["result"].tolist()
    dataset = []
    for story in data["data"]:
        story_txt = " ".join(story['story'])
        for question in story["questions"]:
            question_txt = question["question"]
            if question["q_type"] != "YN":
                continue
            label = question["answer"][0] if question["answer"][0] != "DK" else "No"
            dataset.append([relations[count], question_txt, label, -1])
            count += 1
    if count != len(relations):
        print("Extraction Error !!")
        return []
    return dataset


def eval(file, COT=False):
    print("Evaluation of", file)
    df = pd.read_csv(file)
    check = {"Yes": 0, "No": 0, "DK": 0}
    labels = []
    pred_list = []
    total = {"all": 0, 0: 0, 1: 0, 2: 0}
    acc = {"all": 0, 0: 0, 1: 0, 2: 0}
    i = 0
    for ind, data in df.iterrows():
        total['all'] += 1
        if COT:
            all_pred: str = data["predict"]
            if all_pred.find("Answer: ") != -1:
                answer = all_pred[all_pred.rfind("Answer: "):]
            elif all_pred.find("answer is ") != -1:
                answer = all_pred[all_pred.rfind("answer is "):]
            else:
                answer = all_pred
        else:
            answer = data["predict"]
        # print(answer)
        pred = "Yes" if "yes" in answer.lower() else "No" if "no" in answer.lower() else "No"
        label = data["label"]
        # print(pred, label)
        check[label] += 1
        acc["all"] += 1 if label == pred else 0
        labels.append(label)
        pred_list.append(pred)
        i += 1
    print("acc", acc['all'] / total['all'] * 100)
    print(confusion_matrix(labels, pred_list, labels=["Yes", "No"]))


def eval_stepgame(file):
    def transform_answer(raw_answer):
        raw_answer_list = raw_answer.replace(",", "").split(" ")
        if len(raw_answer_list) < 2:
            return raw_answer

        real_answer = ""
        if "above" in raw_answer_list or "upper" in raw_answer_list:
            real_answer += "upper-"
        elif "below" in raw_answer_list or "lower" in raw_answer_list:
            real_answer += "lower-"

        if "right" in raw_answer_list:
            real_answer += "right"
        elif "left" in raw_answer_list:
            real_answer += "left"

        if len(real_answer) == 0:
            return raw_answer

        if real_answer[-1] == "-":
            real_answer = real_answer[:-1]

        return real_answer

    df = pd.read_csv(file)
    labels = []
    pred_list = []
    total = {"all": 0}
    acc = {"all": 0}
    i = 0
    for ind, data in df.iterrows():
        total['all'] += 1
        answer = str(data["predict"])
        # print(answer)
        pred = answer.lower()
        label = data["label"]
        pred_transform = transform_answer(pred)
        # print(pred, label)
        acc["all"] += 1 if (label == pred or pred_transform == label) else 0
        labels.append(label)
        pred_list.append(pred)
        i += 1
        reasoning_steps = data["Reasoning_Steps"]

        if reasoning_steps not in total:
            total[reasoning_steps] = 0
            acc[reasoning_steps] = 0

        acc[reasoning_steps] += 1 if (pred_transform == label) else 0
        total[reasoning_steps] += 1

    print("acc total:", acc['all'] / total['all'] * 100)
    for i in range(10):
        if i + 1 in total:
            print(f"acc step {i + 1}: {acc[i + 1] / total[i + 1] * 100} %")


def eval_RESQ(file, COT_included=False, DK="No", save_incorrect=False):
    df = pd.read_csv(file)
    acc = 0
    total = 0
    check = {"Yes": 0, "No": 0, "DK": 0}
    labels = []
    pred_list = []
    total = {"all": 0, 0: 0, 1: 0, 2: 0}
    acc = {"all": 0, 0: 0, 1: 0, 2: 0}
    # df2 = pd.read_csv("GPT_results/RESQ_chatGPT_all.csv")
    incorrect_sample = []
    steps = [(data["story"] + data["question"], data["Reasoning_Steps"]) for ind, data in df.iterrows()]
    for ind, data in df.iterrows():
        total['all'] += 1
        if COT_included:
            all_pred: str = data["predict"]
            if all_pred.find("Answer: ") != -1:
                answer = all_pred[all_pred.rfind("Answer: "):]
            elif all_pred.find("answer is ") != -1:
                answer = all_pred[all_pred.rfind("answer is "):]
            else:
                answer = all_pred
        else:
            answer = data["predict"]
        pred = "Yes" if "yes" in answer.lower() else "No" if "no" in answer.lower() else DK
        reason_steps = steps[int(ind)][1]
        total[reason_steps] += 1
        label = data["label"]
        # print("Question:", data["question"])
        # print("Label:", label)
        # print("-" * 10)
        check[label] += 1
        acc["all"] += 1 if data["label"] == pred else 0
        acc[reason_steps] += 1 if data["label"] == pred else 0
        if data["label"] != pred:
            incorrect_sample += [[data["story"], data["question"], reason_steps, label, pred, data["predict"]]]
        labels.append(label)
        pred_list.append(pred)
    # Yes_f1 = f1_score(labels, pred_list, labels=["Yes", "No"], pos_label="Yes")
    # No_f1 = f1_score(labels, pred_list, labels=["Yes", "No"], pos_label="No")
    print("acc", acc['all'] / total['all'] * 100)
    print(total[0] / total['all'], total[1] / total['all'], total[2] / total['all'])
    print("1-step acc", acc[1] / total[1] * 100)
    print("2-step acc", acc[2] / total[2] * 100)
    print("Null   acc", acc[0] / total[0] * 100)
    print(confusion_matrix(labels, pred_list, labels=["Yes", "No", "DK"]))
    if save_incorrect:
        pd.DataFrame(incorrect_sample, columns=["story", "question", "reason_steps", 'label', 'pred', 'COT']).to_csv(
            "GPT_results/GPT_incorrect_samples.csv")
    # print("Yes answer F1:", Yes_f1)
    # print("No answer F1:", No_f1)
    # print("Macro F1", (Yes_f1 + No_f1) / 2)


def dataset_selection(dataset_name):
    if dataset_name == "human":
        return SPARTQA_reader("DataSet/human_test.json")
    elif dataset_name == "resq":
        return RESQ_reader("DataSet/ReSQ/test_resq.json")
    elif dataset_name == "new_human":
        return SPARTQA_reader("DataSet/new_human_test.json")
    elif dataset_name == "stepgame":
        return StepGame_reader("DataSet/StepGame", steps=10)
    return None


def method_selection(dataset_name, method_number, used_same_human=True):
    human_few_shot = [
        {"role": "user", "content": "There exist a big red square, a big red triangle, a medium red square, "
                                    "and a small green circle in a block called A. The triangle and medium "
                                    "square are touching the bottom edge of the block. The big and medium "
                                    "square are touching the right edge of the block. And the circle is above "
                                    "the big square which is above the medium square. There is another block "
                                    "called B to the left of block A. A medium green square is touching the "
                                    "left edge of block B and is below a medium red square. The medium red "
                                    "square is above and to the left of a small green square. Also a medium "
                                    "red triangle is below and to the left of the small square." +
                                    " "
                                    + "Is the medium red triangle below the red square in block B?"},
        {"role": "assistant", "content": "Yes"},
        {"role": "user", "content": "There are three blocks A, B and C. Block A has a medium white triangle "
                                    "and a small red rectangle. There exists a medium white rectangle to the "
                                    "left of the triangle and to the right of the red rectangle. Aove block A "
                                    "there is block B. It has a small white oval touching the bottom edge of "
                                    "the block and a medium white triangle to the right of it. Below block A "
                                    "there is block C. Its has a small green triangle touching the top edge "
                                    "of the block. A medium white triangle is below a small green rectangle. "
                                    "Another small green triangle is below and to the right of the white "
                                    "triangle." +
                                    " "
                                    + "Are all rectangles above a white triangle?"},
        {"role": "assistant", "content": "Yes"},
        {"role": "user", "content": "There is a block called A. Inside A, there is a black triangle that is "
                                    "above and to the left of a black circle. There is another block called B "
                                    "which is to the right of A. In B, there are two yellow circles, "
                                    "a blue circle, and a blue square. The blue circle is touching the left "
                                    "edge of B. One of the yellow circles is to the right of the blue circle "
                                    "and to the left of the blue square. The blue square is touching the "
                                    "right edge of B and is above the other yellow circle, Which is also "
                                    "touching the right edge of B." +
                                    " "
                                    + "Does block B have all of the circles inside it?"},
        {"role": "assistant", "content": "No"},
        {"role": "user", "content": "There is a block called A. In A, there is a large blue square near and below "
                                    "a medium blue triangle. Far to the left of the medium blue triangle is a small"
                                    " yellow square. Far above the medium blue triangle is a small blue triangle. "
                                    "There is another block called B, which is to the right of A. In B, "
                                    "There is a small yellow circle touching the left edge of B. "
                                    "Far to the right of the circle is a large yellow square. "
                                    "Near and above the large square is a small yellow square. "
                                    "There is another block called C, which is to the right of B. "
                                    "In C, there is a small blue circle near and to the left of a "
                                    "large yellow triangle. The large yellow triangle is near and to the "
                                    "left of a large blue triangle. The large blue triangle is "
                                    "far above a large blue circle. The large blue circle is touching "
                                    "the bottom edge of C."
                                    " "
                                    "Are the large blue things in C to the left of the large blue thing in A?"},
        {"role": "assistant", "content": "No"}
    ]
    COT_human = [{"role": "user",
                  "content": "There exist a big red square, a big red triangle, a medium red square, "
                             "and a small green circle in a block called A. The triangle and medium "
                             "square are touching the bottom edge of the block. The big and medium "
                             "square are touching the right edge of the block. And the circle is above "
                             "the big square which is above the medium square. There is another block "
                             "called B to the left of block A. A medium green square is touching the "
                             "left edge of block B and is below a medium red square. The medium red "
                             "square is above and to the left of a small green square. Also a medium "
                             "red triangle is below and to the left of the small square." +
                             " "
                             + "Is the medium red triangle below the red square in block B?"},
                 {"role": "assistant",
                  "content": "From information provided:"
                             "The medium red square is above a small green square in B."
                             "Therefore, the small green square is below the medium red square in B\n"
                             "The medium red triangle is below the small square in B, "
                             "and the small green square is below the medium red square in B."
                             "Therefore, the medium red triangle is below the medium red square in B"
                             "Answer: Yes"},
                 {"role": "user",
                  "content": "There are three blocks A, B and C. Block A has a medium white triangle "
                             "and a small red rectangle. There exists a medium white rectangle to the "
                             "left of the triangle and to the right of the red rectangle. Aove block A "
                             "there is block B. It has a small white oval touching the bottom edge of "
                             "the block and a medium white triangle to the right of it. Below block A "
                             "there is block C. Its has a small green triangle touching the top edge "
                             "of the block. A medium white triangle is below a small green rectangle. "
                             "Another small green triangle is below and to the right of the white "
                             "triangle." +
                             " "
                             + "Are all rectangles above a white triangle?"},
                 {"role": "assistant",
                  "content": "From given information: "
                             "A medium white triangle is below a small green rectangle in C."
                             "Then, small green rectangle is above medium white triangle in C."
                             "A small red rectangle in block A, "
                             "a medium white triangle in C, and block C is below block A."
                             "Then, a medium white triangle is below a A small red rectangle"
                             "A medium white triangle is below a A small red rectangle"
                             "Therefore, a small red rectangle is above a medium white triangle."
                             "A medium white rectangle is in A,"
                             "a medium white triangle in C, and block C is below block A."
                             "Then, a medium white triangle is below a A medium white rectangle"
                             "A medium white triangle is below a A medium white rectangle"
                             "Therefore, a medium white rectangle is above a medium white triangle."
                             "A medium white rectangle is above a medium white triangle."
                             "Thus, all rectangles from given story are above a white triangle. "
                             "Answer: Yes"},
                 {"role": "user",
                  "content": "There is a block called A. Inside A, there is a black triangle that is "
                             "above and to the left of a black circle. There is another block called B "
                             "which is to the right of A. In B, there are two yellow circles, "
                             "a blue circle, and a blue square. The blue circle is touching the left "
                             "edge of B. One of the yellow circles is to the right of the blue circle "
                             "and to the left of the blue square. The blue square is touching the "
                             "right edge of B and is above the other yellow circle, Which is also "
                             "touching the right edge of B." +
                             " "
                             + "Does block B have all of the circles inside it?"},
                 {"role": "assistant",
                  "content": "From given relation, "
                             "A black triangle is above a black circle in A."
                             "Then, there is black circle in A."
                             "Therefore, not all circle are in B."
                             "Answer: No"},
                 {"role": "user",
                  "content": "There is a block called A. In A, there is a large blue square near and below "
                             "a medium blue triangle. Far to the left of the medium blue triangle is a small"
                             " yellow square. Far above the medium blue triangle is a small blue triangle. "
                             "There is another block called B, which is to the right of A. In B, "
                             "There is a small yellow circle touching the left edge of B. "
                             "Far to the right of the circle is a large yellow square. "
                             "Near and above the large square is a small yellow square. "
                             "There is another block called C, which is to the right of B. "
                             "In C, there is a small blue circle near and to the left of a "
                             "large yellow triangle. The large yellow triangle is near and to the "
                             "left of a large blue triangle. The large blue triangle is "
                             "far above a large blue circle. The large blue circle is touching "
                             "the bottom edge of C."
                             " "
                             "Are the large blue things in C to the left of the large blue thing in A?"},
                 {"role": "assistant", "content": "Based on the provided content. "
                                                  "Block B is to the right if block A."
                                                  "Block C is to the right of block B."
                                                  "Therefore, block C is to the right of block A."
                                                  "In block C, "
                                                  "the yellow triangle is near and to the left of large blue triangle. "
                                                  "Therefore, there is large blue triangle in C."
                                                  "In block A, "
                                                  "there is a large blue square near and below a medium blue triangle."
                                                  "Therefore, there is large blue square in A."
                                                  "Then, "
                                                  "large blue triangle in C is to the right of large blue square in A."
                                                  "So, large blue triangle in C "
                                                  "is not to the left of large blue square in A"
                                                  "Answer: No"}
                 ]

    COT_human_formal = [
        {"role": "user", "content": "There exist a big red square, a big red triangle, a medium red square, "
                                    "and a small green circle in a block called A. The triangle and medium "
                                    "square are touching the bottom edge of the block. The big and medium "
                                    "square are touching the right edge of the block. And the circle is above "
                                    "the big square which is above the medium square. There is another block "
                                    "called B to the left of block A. A medium green square is touching the "
                                    "left edge of block B and is below a medium red square. The medium red "
                                    "square is above and to the left of a small green square. Also a medium "
                                    "red triangle is below and to the left of the small square." +
                                    " "
                                    + "Is the medium red triangle below the red square in block B?"},
        {"role": "assistant", "content": "Question: Below(Medium red triangle in B, red square in B) \n"
                                         "Related relations from story: "
                                         "Below(Medium red triangle in B, red square in B) \n"
                                         "Steps: Below(Medium red triangle in B, red square in block B) \n"
                                         "Answer: Yes"},
        {"role": "user", "content": "There are three blocks A, B and C. Block A has a medium white triangle "
                                    "and a small red rectangle. There exists a medium white rectangle to the "
                                    "left of the triangle and to the right of the red rectangle. Aove block A "
                                    "there is block B. It has a small white oval touching the bottom edge of "
                                    "the block and a medium white triangle to the right of it. Below block A "
                                    "there is block C. Its has a small green triangle touching the top edge "
                                    "of the block. A medium white triangle is below a small green rectangle. "
                                    "Another small green triangle is below and to the right of the white "
                                    "triangle." +
                                    " "
                                    + "Are all rectangles above a white triangle?"},
        {"role": "assistant", "content": "Question: "
                                         "For all x where x is rectangles, Above(x, white triangle) \n"
                                         "Related relations from story: "
                                         "x are Small red rectangle in A, medium white rectangle in A, "
                                         "and small green rectangle in C \n"
                                         "Below(Medium white triangle in C, small green rectangle in C) \n"
                                         "Contain(C, Medium white triangle)\n"
                                         "Contain(A, Small red rectangle)\n"
                                         "Contain(A, Medium white rectangle)\n"
                                         "Below(C, A) \n"
                                         "Steps: Below(C, A) => Above(A, C)\n"
                                         "Contain(A, Small red rectangle)  + Above(A, C) "
                                         "+ Contain(C, Medium white triangle) "
                                         "=> Above(Small red rectangle, Medium White triangle)\n"
                                         "Contain(A, Medium white rectangle) + Above(A, C)"
                                         "+ Contain(C, Medium white triangle)"
                                         "=> Above(Medium white rectangle, Medium White triangle)\n"
                                         "Below(Medium white triangle in C, small green rectangle in C) => "
                                         "Above(small green rectangle, Medium white triangle)"
                                         "Then, all x are above a white white triangle in C. "
                                         "Answer: Yes"},
        {"role": "user", "content": "There is a block called A. Inside A, there is a black triangle that is "
                                    "above and to the left of a black circle. There is another block called B "
                                    "which is to the right of A. In B, there are two yellow circles, "
                                    "a blue circle, and a blue square. The blue circle is touching the left "
                                    "edge of B. One of the yellow circles is to the right of the blue circle "
                                    "and to the left of the blue square. The blue square is touching the "
                                    "right edge of B and is above the other yellow circle, Which is also "
                                    "touching the right edge of B." +
                                    " "
                                    + "Does block B have all of the circles inside it?"},
        {"role": "assistant", "content": "Question, For all x, Contain(B, x) where x is any circle\n"
                                         "Related relations from story: "
                                         "Contain(B, Yellow Circle 1), Contain(A, Black Circle), "
                                         "Contain(B, Blue Circle), Contain(B, Yellow Circle 2)\n"
                                         "Steps: Contain(A, Black Circle) => !Contain(B, Black Circle) \n"
                                         "!Contain(B, Black Circle) "
                                         "=> Some x, !Contain(B, x) where x is any circle \n"
                                         "Some x, !Contain(B, x) where x is any circle => "
                                         "!for all x, Contain(B, x) where x is any circle"
                                         "Answer: No"},
        {"role": "user",
         "content": "There is a block called A. In A, there is a large blue square near and below "
                    "a medium blue triangle. Far to the left of the medium blue triangle is a small"
                    " yellow square. Far above the medium blue triangle is a small blue triangle. "
                    "There is another block called B, which is to the right of A. In B, "
                    "There is a small yellow circle touching the left edge of B. "
                    "Far to the right of the circle is a large yellow square. "
                    "Near and above the large square is a small yellow square. "
                    "There is another block called C, which is to the right of B. "
                    "In C, there is a small blue circle near and to the left of a "
                    "large yellow triangle. The large yellow triangle is near and to the "
                    "left of a large blue triangle. The large blue triangle is "
                    "far above a large blue circle. The large blue circle is touching "
                    "the bottom edge of C."
                    " "
                    "Are the large blue things in C to the left of the large blue thing in A?"},
        {"role": "assistant", "content": "Question: Left(large blue thing in C, large blue thing in A)"
                                         "Related relation from story: "
                                         "Right(B, A), Right(C, B), "
                                         "Contain(C, large blue triangle), Contain(A, large blue square)"
                                         "Steps: Right(C, B) + Right(B, A) => Right(C, A)\n"
                                         "Contain(C, large blue triangle) + Right(C, A) "
                                         "+ Contain(A, large blue square) "
                                         "=> Right(large blue triangle in C, large blue square in A)"
                                         "Right(large blue triangle in C, large blue square in A) "
                                         "=> !Left(large blue thing in C, large blue thing in A)"
                                         "Answer: No"}
    ]

    COT_human_formal_2 = [
        {"role": "user", "content": "There exist a big red square, a big red triangle, a medium red square, "
                                    "and a small green circle in a block called A. The triangle and medium "
                                    "square are touching the bottom edge of the block. The big and medium "
                                    "square are touching the right edge of the block. And the circle is above "
                                    "the big square which is above the medium square. There is another block "
                                    "called B to the left of block A. A medium green square is touching the "
                                    "left edge of block B and is below a medium red square. The medium red "
                                    "square is above and to the left of a small green square. Also a medium "
                                    "red triangle is below and to the left of the small square." +
                                    " "
                                    + "Is the medium red triangle below the red square in block B?"},
        {"role": "assistant", "content": "Below(Medium red triangle in B, red square in block B) \n"
                                         "Answer: Yes"},
        {"role": "user", "content": "There are three blocks A, B and C. Block A has a medium white triangle "
                                    "and a small red rectangle. There exists a medium white rectangle to the "
                                    "left of the triangle and to the right of the red rectangle. Aove block A "
                                    "there is block B. It has a small white oval touching the bottom edge of "
                                    "the block and a medium white triangle to the right of it. Below block A "
                                    "there is block C. Its has a small green triangle touching the top edge "
                                    "of the block. A medium white triangle is below a small green rectangle. "
                                    "Another small green triangle is below and to the right of the white "
                                    "triangle." +
                                    " "
                                    + "Are all rectangles above a white triangle?"},
        {"role": "assistant", "content": "Below(C, A) => Above(A, C)\n"
                                         "Contain(A, Small red rectangle) + Above(A, C) + Contain(C, Medium "
                                         "white triangle)=>Above(Small red rectangle, Medium White triangle)\n"
                                         "Above(A, C) & Contain(A, SMedium white rectangle) "
                                         "& Contain(C, Medium white triangle)"
                                         "=>Above(Medium white rectangle, Medium White triangle)\n"
                                         "Below(Medium white triangle in C, small gree rectangle in C) =>"
                                         "Above(small gree rectangle, Medium white triangle)"
                                         "Answer: Yes"},
        {"role": "user", "content": "There is a block called A. Inside A, there is a black triangle that is "
                                    "above and to the left of a black circle. There is another block called B "
                                    "which is to the right of A. In B, there are two yellow circles, "
                                    "a blue circle, and a blue square. The blue circle is touching the left "
                                    "edge of B. One of the yellow circles is to the right of the blue circle "
                                    "and to the left of the blue square. The blue square is touching the "
                                    "right edge of B and is above the other yellow circle, Which is also "
                                    "touching the right edge of B." +
                                    " "
                                    + "Does block B have all of the circles inside it?"},
        {"role": "assistant", "content": "Contain(A, Black Circle) => ~Contain(B, Black Circle) \n"
                                         "~Contain(B, Black Circle) "
                                         "=> Some x, ~Contain(B, x) where x is Circle \n"
                                         "Answer: No"},
        {"role": "user",
         "content": "There is a block called A. In A, there is a large blue square near and below "
                    "a medium blue triangle. Far to the left of the medium blue triangle is a small"
                    " yellow square. Far above the medium blue triangle is a small blue triangle. "
                    "There is another block called B, which is to the right of A. In B, "
                    "There is a small yellow circle touching the left edge of B. "
                    "Far to the right of the circle is a large yellow square. "
                    "Near and above the large square is a small yellow square. "
                    "There is another block called C, which is to the right of B. "
                    "In C, there is a small blue circle near and to the left of a "
                    "large yellow triangle. The large yellow triangle is near and to the "
                    "left of a large blue triangle. The large blue triangle is "
                    "far above a large blue circle. The large blue circle is touching "
                    "the bottom edge of C."
                    " "
                    "Are the large blue things in C to the left of the large blue thing in A?"},
        {"role": "assistant", "content": "Question: Left(large blue thing in C, large blue thing in A)"
                                         "Related relation from story: "
                                         "Right(B, A), Right(C, B), "
                                         "Contain(C, large blue triangle), Contain(A, large blue square)"
                                         "Steps: Right(C, B) + Right(B, A) => Right(C, A)\n"
                                         "Contain(C, large blue triangle) + Right(C, A) "
                                         "+ Contain(A, large blue square) "
                                         "=> Right(large blue triangle in C, large blue square in A)"
                                         "Right(large blue triangle in C, large blue square in A) "
                                         "=> !Left(large blue thing in C, large blue thing in A)"
                                         "Answer: No"}
    ]
    COT_human_formal_3 = [
        {"role": "user", "content": "There exist a big red square, a big red triangle, a medium red square, "
                                    "and a small green circle in a block called A. The triangle and medium "
                                    "square are touching the bottom edge of the block. The big and medium "
                                    "square are touching the right edge of the block. And the circle is above "
                                    "the big square which is above the medium square. There is another block "
                                    "called B to the left of block A. A medium green square is touching the "
                                    "left edge of block B and is below a medium red square. The medium red "
                                    "square is above and to the left of a small green square. Also a medium "
                                    "red triangle is below and to the left of the small square." +
                                    " "
                                    + "Is the medium red triangle below the red square in block B?"},
        {"role": "assistant", "content": "From the story, the following information will "
                                         "help getting the answer of "
                                         "Below((medium, red, triangle, B), (?, red, square, B)):"
                                         "Below((medium, red, triangle, B), (?, red, square, B))"
                                         "Based on the information, "
                                         "Below((medium, red, triangle, B), (?, red, square, B)) "
                                         "== Below((medium, red, triangle, B), (?, red, square, B))"
                                         "Therefore, it is true."
                                         "Answer: Yes"},
        {"role": "user", "content": "There are three blocks A, B and C. Block A has a medium white triangle "
                                    "and a small red rectangle. There exists a medium white rectangle to the "
                                    "left of the triangle and to the right of the red rectangle. Aove block A "
                                    "there is block B. It has a small white oval touching the bottom edge of "
                                    "the block and a medium white triangle to the right of it. Below block A "
                                    "there is block C. Its has a small green triangle touching the top edge "
                                    "of the block. A medium white triangle is below a small green rectangle. "
                                    "Another small green triangle is below and to the right of the white "
                                    "triangle." +
                                    " "
                                    + "Are all rectangles above a white triangle?"},
        {"role": "assistant", "content": "Based on the given information, "
                                         "the following relation will help answer the question of "
                                         "Above((?,?,rectangle,?), (?, white, triangle,?)):"
                                         "(small, red, rectangle, A), (medium, white, rectangle, A),"
                                         "(small, green, rectangle, C), (medium, white, triangle, C),"
                                         "Below(C, A),"
                                         "Below((medium, white, triangle, C), (small, green, rectangle, C))\n"
                                         "Step: Below(C, A) => Above(A, C)\n"
                                         "(small, red, rectangle, A) + Above(A, C) + (medium, white, triangle, C) => "
                                         "Above((small, red, rectangle, A), (medium, white, triangle, C))\n"
                                         "(medium, white, rectangle, A) + Above(A, C) + (medium, white, triangle, C)=> "
                                         "Above((medium, white, rectangle, A), (medium, white, triangle, C))\n"
                                         "Below((medium, white, triangle, C), (small, green, rectangle, C)) =>"
                                         "Above((small, green, rectangle, C), (medium, white, triangle, C))"
                                         "Then, Above((?,?,rectangle,?), (?, white, triangle,?)) "
                                         "Answer: Yes"},
        {"role": "user", "content": "There is a block called A. Inside A, there is a black triangle that is "
                                    "above and to the left of a black circle. There is another block called B "
                                    "which is to the right of A. In B, there are two yellow circles, "
                                    "a blue circle, and a blue square. The blue circle is touching the left "
                                    "edge of B. One of the yellow circles is to the right of the blue circle "
                                    "and to the left of the blue square. The blue square is touching the "
                                    "right edge of B and is above the other yellow circle, Which is also "
                                    "touching the right edge of B." +
                                    " "
                                    + "Does block B have all of the circles inside it?"},
        {"role": "assistant", "content": "Based on the given information, "
                                         "the following relation will help answer the question of "
                                         "(?, ?, circle, B):"
                                         "(-, yellow, circle, B), (-, black, circle, A), "
                                         "(-, blue, circle1, B), (-, yellow, circlc2, B))"
                                         "Steps: (-, black, circle, A) => !(?,?,circle,B) \n"
                                         "Answer: No"},

        {"role": "user",
         "content": "There is a block called A. In A, there is a large blue square near and below "
                    "a medium blue triangle. Far to the left of the medium blue triangle is a small"
                    " yellow square. Far above the medium blue triangle is a small blue triangle. "
                    "There is another block called B, which is to the right of A. In B, "
                    "There is a small yellow circle touching the left edge of B. "
                    "Far to the right of the circle is a large yellow square. "
                    "Near and above the large square is a small yellow square. "
                    "There is another block called C, which is to the right of B. "
                    "In C, there is a small blue circle near and to the left of a "
                    "large yellow triangle. The large yellow triangle is near and to the "
                    "left of a large blue triangle. The large blue triangle is "
                    "far above a large blue circle. The large blue circle is touching "
                    "the bottom edge of C."
                    " "
                    "Are the large blue things in C to the left of the large blue thing in A?"},
        {"role": "assistant", "content": "Let analyze the following relation to answer the question of "
                                         "Left((large, blue, ?, C), (large, blue, ?, A)):"
                                         "Right(B, A), Right(C, B),"
                                         "(large, blue, triangle, C), (large, blue, square, A)\n"
                                         "Steps: Right(C, B) + Right(B, A) => Right(C, A)\n"
                                         "(large, blue, triangle, C) + Right(C, A) + (large, blue, square, A) => "
                                         "Right((large, blue, triangle, C), (large, blue, square, A))\n"
                                         "Right((large, blue, triangle, C), (large, blue, square, A)) => "
                                         "!Left((large, blue, triangle, C), (large, blue, square, A))\n"
                                         "!Left((large, blue, triangle, C), (large, blue, square, A)) => "
                                         "!Left((large, blue, ?, C), (large, blue, ?, A))"
                                         "Answer: No"}
    ]
    # ↑ (Up)
    # ↓ (Down)
    # < (Left)
    # > (Right)
    # ~ (near)
    # = (touching)
    # ∞ (far form)
    COS_human = [
        {"role": "user", "content": "There exist a big red square, a big red triangle, a medium red square, "
                                    "and a small green circle in a block called A. The triangle and medium "
                                    "square are touching the bottom edge of the block. The big and medium "
                                    "square are touching the right edge of the block. And the circle is above "
                                    "the big square which is above the medium square. There is another block "
                                    "called B to the left of block A. A medium green square is touching the "
                                    "left edge of block B and is below a medium red square. The medium red "
                                    "square is above and to the left of a small green square. Also a medium "
                                    "red triangle is below and to the left of the small square." +
                                    " "
                                    + "Is the medium red triangle below the red square in block B?"},
        {"role": "assistant", "content": "Let analyze the relations given from context to answer question."
                                         "B: 1. (medium, red, triangle) ↓, < (small, red, square)"
                                         "Based on the information, "
                                         "B: 1. (medium, red, triangle) ↓, < (small, red, square)"
                                         "2. (medium, red, triangle) ↓ (small, red, square)"
                                         "Answer: Yes"},
        {"role": "user", "content": "There are three blocks A, B and C. Block A has a medium white triangle "
                                    "and a small red rectangle. There exists a medium white rectangle to the "
                                    "left of the triangle and to the right of the red rectangle. Aove block A "
                                    "there is block B. It has a small white oval touching the bottom edge of "
                                    "the block and a medium white triangle to the right of it. Below block A "
                                    "there is block C. Its has a small green triangle touching the top edge "
                                    "of the block. A medium white triangle is below a small green rectangle. "
                                    "Another small green triangle is below and to the right of the white "
                                    "triangle." +
                                    " "
                                    + "Are all rectangles above a white triangle?"},
        {"role": "assistant", "content": "Let consider the related relations to answer the questions,"
                                         "A: 1. (small, red, rectangle) "
                                         "2. (medium, white, rectangle) "
                                         "C: 1. (small, green, rectangle) = edge "
                                         "2. (medium, white, triangle) ) ↓ (small, green, rectangle) "
                                         "C ↓ A"
                                         "Based on the information, "
                                         "let determined the relation of all rectangle and white triangles, "
                                         "1. C ↓ A. Therefore, A ↑ C "
                                         "2. A ↑ C, A: (small, red, rectangle), and C: (medium, white, triangle). "
                                         "Therefore, A: (small, red, rectangle) ↑ C: (medium, white, triangle) "
                                         "3. A ↑ C, A: (medium, white, rectangle), and C: (medium, white, triangle). "
                                         "Therefore, A: (medium, white, rectangle) ↑ C: (medium, white, triangle). "
                                         "4. C: (medium, white, triangle) ) ↓ (small, green, rectangle). Therefore, "
                                         "(small, green, rectangle) ↑ (medium, white, triangle)."
                                         "5. A: (small, red, rectangle) ↑ C: (medium, white, triangle), "
                                         "A: (medium, white, rectangle) ↑ C: (medium, white, triangle), and "
                                         "C: (small, green, rectangle) ↑ (medium, white, triangle), therefore,"
                                         "all rectangle above a white triangle"
                                         "Answer: Yes"},
        {"role": "user", "content": "There is a block called A. Inside A, there is a black triangle that is "
                                    "above and to the left of a black circle. There is another block called B "
                                    "which is to the right of A. In B, there are two yellow circles, "
                                    "a blue circle, and a blue square. The blue circle is touching the left "
                                    "edge of B. One of the yellow circles is to the right of the blue circle "
                                    "and to the left of the blue square. The blue square is touching the "
                                    "right edge of B and is above the other yellow circle, Which is also "
                                    "touching the right edge of B." +
                                    " "
                                    + "Does block B have all of the circles inside it?"},
        {"role": "assistant", "content": "Let consider the related relations to answer the questions,"
                                         "A: 1. (black, triangle) ↑< (black, circle)"
                                         "B: 1. (yellow, circle1) right (blue, circle)"
                                         "2. (blue, square) ↑ (yellow, circle2) "
                                         "3.(blue, circle) =left edge of B"
                                         "Based on the information, "
                                         "let determined the relation of all rectangle and white triangles, "
                                         "1. A: 1. (black, triangle) ↑< (black, circle). Therefore, "
                                         "A: (black, circle)"
                                         "2. A: (black, circle). Therefore, B doesn't have all circle"
                                         "Answer: No"},

        {"role": "user",
         "content": "There is a block called A. In A, there is a large blue square near and below "
                    "a medium blue triangle. Far to the left of the medium blue triangle is a small"
                    " yellow square. Far above the medium blue triangle is a small blue triangle. "
                    "There is another block called B, which is to the right of A. In B, "
                    "There is a small yellow circle touching the left edge of B. "
                    "Far to the right of the circle is a large yellow square. "
                    "Near and above the large square is a small yellow square. "
                    "There is another block called C, which is to the right of B. "
                    "In C, there is a small blue circle near and to the left of a "
                    "large yellow triangle. The large yellow triangle is near and to the "
                    "left of a large blue triangle. The large blue triangle is "
                    "far above a large blue circle. The large blue circle is touching "
                    "the bottom edge of C."
                    " "
                    "Are the large blue things in C to the left of the large blue thing in A?"},
        {"role": "assistant", "content": "Let consider the related relations to answer the questions,"
                                         "B > A,"
                                         "C > B"
                                         "A: 1. (large, blue square) ~↓ (medium, blue square)"
                                         "C: 1. (large, yellow, triangle)"
                                         "Based on the information, "
                                         "let determined the relation large blue things in C and the large blue thing "
                                         "in A,"
                                         "1. C > B and B > A. Therefore, C > A"
                                         "2. A: 1. (large, blue, square) ~↓ (medium, blue square). "
                                         "Therefore, A: (large, blue, square)"
                                         "3. C > A, A: (large, blue, square), and C: (large, yellow, triangle)."
                                         "Therefore, C: (large, yellow, triangle) > A: (large, blue, square)."
                                         "4. C: (large, yellow, triangle) > A: (large, blue, square)."
                                         "Therefore, C: (large, yellow, triangle) !< A: (large, blue, square)"
                                         "large blue things in C is not to the left of the large blue thing in A."
                                         "Answer: No"}
    ]

    COT_human_formal_cos_based = [
        {"role": "user", "content": "There exist a big red square, a big red triangle, a medium red square, "
                                    "and a small green circle in a block called A. The triangle and medium "
                                    "square are touching the bottom edge of the block. The big and medium "
                                    "square are touching the right edge of the block. And the circle is above "
                                    "the big square which is above the medium square. There is another block "
                                    "called B to the left of block A. A medium green square is touching the "
                                    "left edge of block B and is below a medium red square. The medium red "
                                    "square is above and to the left of a small green square. Also a medium "
                                    "red triangle is below and to the left of the small square." +
                                    " "
                                    + "Is the medium red triangle below the red square in block B?"},
        {"role": "assistant", "content": "Let analyze the relations given from context to answer question."
                                         "B: 1. (medium, red, triangle) below, left (small, red, square)"
                                         "Based on the information, "
                                         "B: 1. (medium, red, triangle) below, left  (small, red, square)"
                                         "2. (medium, red, triangle) below (small, red, square)"
                                         "Answer: Yes"},
        {"role": "user", "content": "There are three blocks A, B and C. Block A has a medium white triangle "
                                    "and a small red rectangle. There exists a medium white rectangle to the "
                                    "left of the triangle and to the right of the red rectangle. Aove block A "
                                    "there is block B. It has a small white oval touching the bottom edge of "
                                    "the block and a medium white triangle to the right of it. Below block A "
                                    "there is block C. Its has a small green triangle touching the top edge "
                                    "of the block. A medium white triangle is below a small green rectangle. "
                                    "Another small green triangle is below and to the right of the white "
                                    "triangle." +
                                    " "
                                    + "Are all rectangles above a white triangle?"},
        {"role": "assistant", "content": "Let consider the related relations to answer the questions,"
                                         "A: 1. (small, red, rectangle) "
                                         "2. (medium, white, rectangle) "
                                         "C: 1. (small, green, rectangle) touch edge of C"
                                         "2. (medium, white, triangle) ) below (small, green, rectangle) "
                                         "C blow A"
                                         "Based on the information, "
                                         "let determined the relation of all rectangle and white triangles, "
                                         "1. C below A. Therefore, A above C "
                                         "2. A above C, A: (small, red, rectangle), and C: (medium, white, triangle). "
                                         "Therefore, A: (small, red, rectangle) above C: (medium, white, triangle) "
                                         "3. A above C, A: (medium, white, rectangle), and C: (medium, white, triangle). "
                                         "Therefore, A: (medium, white, rectangle) above C: (medium, white, triangle). "
                                         "4. C: (medium, white, triangle) ) below (small, green, rectangle). Therefore, "
                                         "(small, green, rectangle) above (medium, white, triangle)."
                                         "5. A: (small, red, rectangle) above C: (medium, white, triangle), "
                                         "A: (medium, white, rectangle) above C: (medium, white, triangle), and "
                                         "C: (small, green, rectangle) above (medium, white, triangle), therefore,"
                                         "all rectangle above a white triangle"
                                         "Answer: Yes"},
        {"role": "user", "content": "There is a block called A. Inside A, there is a black triangle that is "
                                    "above and to the left of a black circle. There is another block called B "
                                    "which is to the right of A. In B, there are two yellow circles, "
                                    "a blue circle, and a blue square. The blue circle is touching the left "
                                    "edge of B. One of the yellow circles is to the right of the blue circle "
                                    "and to the left of the blue square. The blue square is touching the "
                                    "right edge of B and is above the other yellow circle, Which is also "
                                    "touching the right edge of B." +
                                    " "
                                    + "Does block B have all of the circles inside it?"},
        {"role": "assistant", "content": "Let consider the related relations to answer the questions,"
                                         "A: 1. (-, black, triangle) above, left (-, black circle)"
                                         "B: 1. (-, yellow, circle1) right (-, blue, circle)"
                                         "2. (-, blue, square) above (-, yellow, circle2) "
                                         "3.(-, blue, circle) touch left edge of B"
                                         "Based on the information, "
                                         "let determined the relation of all rectangle and white triangles, "
                                         "1. A: 1. (-, black, triangle) above, left (-, black circle). Therefore, "
                                         "A: (-, black circle)"
                                         "2. A: (-, black circle). Therefore, B doesn't have all circle"
                                         "Answer: No"},

        {"role": "user",
         "content": "There is a block called A. In A, there is a large blue square near and below "
                    "a medium blue triangle. Far to the left of the medium blue triangle is a small"
                    " yellow square. Far above the medium blue triangle is a small blue triangle. "
                    "There is another block called B, which is to the right of A. In B, "
                    "There is a small yellow circle touching the left edge of B. "
                    "Far to the right of the circle is a large yellow square. "
                    "Near and above the large square is a small yellow square. "
                    "There is another block called C, which is to the right of B. "
                    "In C, there is a small blue circle near and to the left of a "
                    "large yellow triangle. The large yellow triangle is near and to the "
                    "left of a large blue triangle. The large blue triangle is "
                    "far above a large blue circle. The large blue circle is touching "
                    "the bottom edge of C."
                    " "
                    "Are the large blue things in C to the left of the large blue thing in A?"},
        {"role": "assistant", "content": "Let consider the related relations to answer the questions,"
                                         "B right A,"
                                         "C right B"
                                         "A: 1. (large, blue square) near, below (medium, blue square)"
                                         "C: 1. (large, yellow, triangle)"
                                         "Based on the information, "
                                         "let determined the relation large blue things in C and the large blue thing "
                                         "in A,"
                                         "1. C right B and B right A. Therefore, C right A"
                                         "2. A: 1. (large, blue, square) near, below (medium, blue square). "
                                         "Therefore, A: (large, blue, square)"
                                         "3. C right A, A: (large, blue, square), and C: (large, yellow, triangle)."
                                         "Therefore, C: (large, yellow, triangle) right A: (large, blue, square)."
                                         "4. C: (large, yellow, triangle) right A: (large, blue, square)."
                                         "Therefore, C: (large, yellow, triangle) not left A: (large, blue, square)"
                                         "large blue things in C is not to the left of the large blue thing in A."
                                         "Answer: No"}
    ]

    human_example_list = [[], human_few_shot, COT_human,
                          COT_human_formal,
                          COT_human_formal_2,
                          COT_human_formal_3,
                          COS_human,
                          COT_human_formal_cos_based]

    RESQ_few_shot = [{"role": "user", "content": "Six pupils are sitting around a round table on wooden chairs , "
                                                 "each of them is holding a fruit ( tangerine ? ) and is wearing a "
                                                 "hat ( except for one kid ) ." + "pictures and comics on the wall "
                                                                                  "behind them ." + "car tyres in the"
                                                                                                    " background on "
                                                                                                    "the right ." +
                                                 " "
                                                 + "Is the Picture in front of the car tyres?"},
                     {"role": "assistant", "content": "No"},

                     {"role": "user",
                      "content": "Excited kids are standing at their desks in a classroom.some of them seem to be very "
                                 "happy and are laughing. there are pictures of trees on the wall at the back .two girls "
                                 "have an open book on their desk ." +
                                 " "
                                 + "Is the desks inside the classroom?"},
                     {"role": "assistant", "content": "Yes"},

                     {"role": "user",
                      "content": "many people are working on the bare brickwork of a house or are watching "
                                 "the workers from below ." + "many wires are sticking out of the bare "
                                                              "brickwork ." + "one person is climbing up "
                                                                              "a ladder ." + "many "
                                                                                             "pallets are "
                                                                                             "lying piled "
                                                                                             "up in front "
                                                                                             "of the "
                                                                                             "house ." +
                                 " "
                                 + "Is the ladder touching the house?"},
                     {"role": "assistant", "content": "Yes"},
                     {"role": "user",
                      "content": "a dark-skinned , dark-haired boy wearing dark blue dungarees and a light blue and "
                                 "pink pullover is standing in front of a brick-wall" + "Is the wall in front of the "
                                                                                        "dark blue thing?"},
                     {"role": "assistant", "content": "No"}
                     ]

    RESQ_COT = [{"role": "user", "content": "Six pupils are sitting around a round table on wooden chairs , "
                                            "each of them is holding a fruit ( tangerine ? ) and is wearing a "
                                            "hat ( except for one kid ) ." + "pictures and comics on the wall "
                                                                             "behind them ." + "car tyres in the"
                                                                                               " background on "
                                                                                               "the right ." +
                                            " "
                                            + "Is the Picture in front of the car tyres?"},
                {"role": "assistant", "content": "Pictures are on the wall. Car tyres are in the background on "
                                                 "the right. Therefore, Car tryes are in the same wall as "
                                                 "pictures. Then, pictures aren't in front of car tryes."
                                                 "Answer: No"},

                {"role": "user",
                 "content": "Excited kids are standing at their desks in a classroom.some of them seem to be very "
                            "happy and are laughing. there are pictures of trees on the wall at the back. Two girls "
                            "have an open book on their desk ." +
                            " "
                            + "Is the desks inside the classroom?"},
                {"role": "assistant", "content": "From story, excited kids are standing at their desks in a "
                                                 "classroom. Therefore, kids are standing in a class room. Then, "
                                                 "desks are inside the classroom since the kids standing at desks." +
                                                 " Answer: Yes"},

                {"role": "user",
                 "content": "many people are working on the bare brickwork of a house or are watching "
                            "the workers from below . many wires are sticking out of the bare brickwork .one person "
                            "is climbing up a ladder . many pallets are lying piled up in front of the house ." +
                            " "
                            + "Is the ladder touching the house?"},
                {"role": "assistant", "content": "From story, many people are working on the bare brickwork of a "
                                                 "house and one person is climbing up a ladder. That person is "
                                                 "working on the bare brickwork. Therefore, ladder must touch the "
                                                 "house for him to working on the bare brickwork of the house." +
                                                 "Answer: Yes"},

                {"role": "user",
                 "content": "a dark-skinned , dark-haired boy wearing dark blue dungarees and a light blue and "
                            "pink pullover is standing in front of a brick-wall." + " Is the wall in front of the "
                                                                                    "dark blue thing?"},
                {"role": "assistant", "content": "From story, a dark-skinned , dark-haired boy is wearing dark blue "
                                                 "dungarees. Then, the dark blue thing is with the boy. A boy is "
                                                 "standing in front of a brick-wall. Therefor, the boy is in front of "
                                                 "the wall. Since blue dark thing is with the boy, blue dark thing is "
                                                 "in front of the wall. Then, the wall is behind the dark blue thing."
                                                 + "Answer: No"},
                {"role": "user",
                 "content": "a dark-skinned , dark-haired boy wearing dark blue dungarees and a light blue and "
                            "pink pullover is standing in front of a brick-wall." + " Is the wall in front of the "
                                                                                    "red thing?"},
                {"role": "assistant", "content": "From story, there is no red thing in the story. Therefor, no"
                                                 + "Answer: No"},
                {"role": "user",
                 "content": "a dark-skinned , dark-haired boy wearing dark blue dungarees and a light blue and "
                            "pink pullover is standing in front of a brick-wall." + " Is the boy touch wall?"},
                {"role": "assistant", "content": "From story, the boy stands in front of the door. "
                                                 "However, the boy may not touch it or touch it. Since it is uncertain, the answer is no."
                                                 + "Answer: No"}
                ]

    RESQ_COT_formal = [{"role": "user", "content": "Six pupils are sitting around a round table on wooden chairs , "
                                                   "each of them is holding a fruit ( tangerine ? ) and is wearing a "
                                                   "hat ( except for one kid ) ." + "pictures and comics on the wall "
                                                                                    "behind them ." + "car tyres in the"
                                                                                                      " background on "
                                                                                                      "the right ." +
                                                   " "
                                                   + "Is the Picture in front of the car tyres?"},
                       {"role": "assistant", "content": "Story: Sit(Pupils, Wooden chairs), Around(Pupils, Round "
                                                        "Table), Hold(Pupils, Fruit), Wear(Pupils, Hat), On(Comics, "
                                                        "Wall), On(Picture, wall), On(Car tyres, Wall) \n"
                                                        "Question: Front(Picture, Car tyres) \n"
                                                        "Steps: On(Picture, wall) & On(Car tyres, wall) => "
                                                        "~Front(Picture, Car tyres) \n"
                                                        "Answer: No"},

                       {"role": "user",
                        "content": "Excited kids are standing at their desks in a classroom.some of them seem to be "
                                   "very"
                                   "happy and are laughing. there are pictures of trees on the wall at the back. Two "
                                   "girls"
                                   "have an open book on their desk ." +
                                   " "
                                   + "Is the desks inside the classroom?"},
                       {"role": "assistant", "content": "Story: Near(Kids, Desk), Inside(Kids, classroom), On(Tree "
                                                        "Picture, Wall), On(Book, Desk) \n"
                                                        "Question: Inside(Desks, classroom) \n"
                                                        "Steps: Near(Kids, Desks) & Inside(Kids, classroom) => "
                                                        "Inside(Desks, classroom) \n" +
                                                        " Answer: Yes"},

                       {"role": "user",
                        "content": "many people are working on the bare brickwork of a house or are watching "
                                   "the workers from below . many wires are sticking out of the bare brickwork .one "
                                   "person"
                                   "is climbing up a ladder . many pallets are lying piled up in front of the house ." +
                                   " "
                                   + "Is the ladder touching the house?"},
                       {"role": "assistant", "content": "Story: Watching(People, Worker), Below(People, Worker), "
                                                        "Stick(Wires, Bare Brickwork), Working(Person, Brickwork), "
                                                        "Climbing(Person, Ladder), Front(Pallets, House) \n"
                                                        "Question: Touch(Ladder, House) \n"
                                                        "Steps: Working(Person, Brickwork) & Climbing(Person, Ladder) "
                                                        "=> ClimbTo(Person, House) \n" +
                                                        "ClimbTo(Person, House) & Climbing(Person, Ladder) "
                                                        "=> Touch(Ladder, House) \n"
                                                        "Answer: Yes"},

                       {"role": "user",
                        "content": "a dark-skinned , dark-haired boy wearing dark blue dungarees and a light blue and "
                                   "pink pullover is standing in front of a brick-wall." + "Is the wall in front of the"
                                                                                           "dark blue thing?"},
                       {"role": "assistant",
                        "content": "Story: Wear(Boy, Dark blue dungarees), Wear(Boy, light blue and pink pullover), "
                                   "Front(Boy, Brick-wall) \n"
                                   "Question: Front(Brick-wall, Boy) \n"
                                   "Steps: Wear(Boy, Dark blue dungarees) => Contain(Boy, Dark blue dungarees) \n"
                                   "Contain(Boy, Dark blue dungarees) & Front(Boy, Brick-wall) => Front(Boy, "
                                   "Brick-wall)\n"
                                   "Front(Boy, Brick-wall) => ~Front(Brick-wall, Boy)\n"
                                   "Answer: No"}
                       ]

    RESQ_COT_formal_3 = [{"role": "user", "content": "Six pupils are sitting around a round table on wooden chairs , "
                                                     "each of them is holding a fruit ( tangerine ? ) and is wearing a "
                                                     "hat ( except for one kid ) ." + "pictures and comics on the wall "
                                                                                      "behind them ." + "car tyres in the"
                                                                                                        " background on "
                                                                                                        "the right ." +
                                                     " "
                                                     + "Is the Picture in front of the car tyres?"},
                         {"role": "assistant", "content": "Steps: On(Picture, wall) & On(Car tyres, wall) => "
                                                          "~Front(Picture, Car tyres) \n"
                                                          "Answer: No"},

                         {"role": "user",
                          "content": "Excited kids are standing at their desks in a classroom.some of them seem to be "
                                     "very"
                                     "happy and are laughing. there are pictures of trees on the wall at the back. Two "
                                     "girls"
                                     "have an open book on their desk ." +
                                     " "
                                     + "Is the desks inside the classroom?"},
                         {"role": "assistant", "content": "Steps: Near(Kids, Desks) & Inside(Kids, classroom) => "
                                                          "Inside(Desks, classroom) \n" +
                                                          " Answer: Yes"},

                         {"role": "user",
                          "content": "many people are working on the bare brickwork of a house or are watching "
                                     "the workers from below . many wires are sticking out of the bare brickwork .one "
                                     "person"
                                     "is climbing up a ladder . many pallets are lying piled up in front of the house ." +
                                     " "
                                     + "Is the ladder touching the house?"},
                         {"role": "assistant",
                          "content": "Steps: Working(Person, Brickwork) & Climbing(Person, Ladder) "
                                     "=> ClimbTo(Person, House) \n" +
                                     "ClimbTo(Person, House) & Climbing(Person, Ladder) "
                                     "=> Touch(Ladder, House) \n"
                                     "Answer: Yes"},

                         {"role": "user",
                          "content": "a dark-skinned , dark-haired boy wearing dark blue dungarees and a light blue and "
                                     "pink pullover is standing in front of a brick-wall." + "Is the wall in front of the"
                                                                                             "dark blue thing?"},
                         {"role": "assistant",
                          "content": "Steps: Wear(Boy, Dark blue dungarees) => Contain(Boy, Dark blue dungarees) \n"
                                     "Contain(Boy, Dark blue dungarees) & Front(Boy, Brick-wall) => Front(Boy, "
                                     "Brick-wall)\n"
                                     "Front(Boy, Brick-wall) => ~Front(Brick-wall, Boy)\n"
                                     "Answer: No"}
                         ]

    RESQ_example_list = [[], RESQ_few_shot, RESQ_COT, RESQ_COT_formal, RESQ_COT_formal_3]

    new_human_few_shots = [
        # Indirect No
        {"role": "user",
         "content": "There are three blocks of A, B, and C. Block A is to the left of block B. Block B is in the "
                    "middle. In block A, the little red triangle is near and to the left of the large red rectangle. "
                    "The large"
                    "red rectangle is above the midsize black triangle and the little black rectangle. In block B, "
                    "the midsize black triangle is to the right of the little black triangle. "
                    "The midsize green oval is below the midsize "
                    "black triangle and the little black triangle. The midsize green oval is touching the bottom "
                    "corner of block"
                    "B. In block C, the midsize black oval is above the midsize black rectangle. The midsize green "
                    "oval is below"
                    "the little red triangle. The midsize black oval is to the left of the little red triangle."
                    +
                    " "
                    + "Is the large red rectangle to the left of the little black triangle?"},
        {"role": "assistant",
         "content": "No"},
        # Indirect Yes
        {"role": "user",
         "content": "There are three blocks of A, B, and C. Block A is to the left of Block C. Block B is between "
                    "block A and block C. In block A, one large blue triangle is to the left of the midsize red "
                    "square and they"
                    "are touching the bottom of the block. The other large blue triangle is above the midsize red square and to "
                    "the left of the large white triangle. The midsize white circle is above the large blue "
                    "triangles. In block"
                    "B, the small white triangle is to the left of the small white circle. "
                    "The midsize red square is above the "
                    "midsize red circle. One of the small blue squares is above the midsize red square. The other small blue "
                    "square is above the midsize red circle and below the midsize red square. In block C, there is one large red "
                    "square."
                    +
                    " "
                    + "Is the midsize red square below the midsize white circle?"
         },
        {"role": "assistant",
         "content": "Yes"},
        # Direct yes
        {"role": "user",
         "content": "There are three blocks of A, B, and C. Block A is to the left of block B, and block B is to the left of block C. In block A, the midsize white oval is touching the bottom of the block. A midsize black triangle is above and to the left of the white oval. Another midsize black triangle and the little blue oval are above and to the right of the white oval. The big black triangle is above the midsize black triangles. The little blue rectangle is to the right and above the midsize black triangles. In block B, the midsize white oval is touching the corner of the block. The midsize blue oval is above the white oval. The little black rectangle is to the right of the blue oval. In block C, there is a little blue rectangle."
                    +
                    " "
                    + "Is the little blue oval to the right of the midsize white oval?"
         },
        {"role": "assistant",
         "content": "Yes"},
        # Direct no
        {"role": "user",
         "content": "There are three blocks of A, B, and C. Block A is to the left of block B, and block B is to the left of block C. In block A, one medium white circle is touching the bottom of the block. The medium green triangle is above the white circle. Another white circle is to the right and below the green triangle. The medium black rectangle is above the white circle. The little white triangle is above the medium black rectangle. The little black circle is to the left of the green triangle. The little black rectangle is above the little black circle. In block B, the large black rectangle is to the left of the large green triangle. The little green triangle is below large triangle and large rectangle. In block C, the large black circle and the little green triangle are touching the bottom of the block. The medium white triangle is above the black circle. The little green triangle is above the white triangle. The little green circle is above the little green triangle."
                    +
                    " "
                    + "In block A, is the little white triangle below the medium black rectangle?"
         },
        {"role": "assistant",
         "content": "No"},
    ]
    new_human_COT = [
        # Indirect No
        {"role": "user",
         "content": "There are three blocks of A, B, and C. Block A is to the left of block B. Block B is in the "
                    "middle. In block A, the little red triangle is near and to the left of the large red rectangle. "
                    "The large"
                    "red rectangle is above the midsize black triangle and the little black rectangle. In block B, "
                    "the midsize black triangle is to the right of the little black triangle. "
                    "The midsize green oval is below the midsize "
                    "black triangle and the little black triangle. The midsize green oval is touching the bottom "
                    "corner of block"
                    "B. In block C, the midsize black oval is above the midsize black rectangle. The midsize green "
                    "oval is below"
                    "the little red triangle. The midsize black oval is to the left of the little red triangle."
                    +
                    " "
                    + "Is the large red rectangle to the left of the little black triangle?"},
        {"role": "assistant",
         "content": "No"},
        # Indirect Yes
        {"role": "user",
         "content": "There are three blocks of A, B, and C. Block A is to the left of Block C. Block B is between "
                    "block A and block C. In block A, one large blue triangle is to the left of the midsize red "
                    "square and they"
                    "are touching the bottom of the block. The other large blue triangle is above the midsize red square and to "
                    "the left of the large white triangle. The midsize white circle is above the large blue "
                    "triangles. In block"
                    "B, the small white triangle is to the left of the small white circle. "
                    "The midsize red square is above the "
                    "midsize red circle. One of the small blue squares is above the midsize red square. The other small blue "
                    "square is above the midsize red circle and below the midsize red square. In block C, there is one large red "
                    "square."
                    +
                    " "
                    + "Is the midsize red square below the midsize white circle?"
         },
        {"role": "assistant",
         "content": "Yes"},
        # Direct yes
        {"role": "user",
         "content": "There are three blocks of A, B, and C. Block A is to the left of block B, and block B is to the left of block C. In block A, the midsize white oval is touching the bottom of the block. A midsize black triangle is above and to the left of the white oval. Another midsize black triangle and the little blue oval are above and to the right of the white oval. The big black triangle is above the midsize black triangles. The little blue rectangle is to the right and above the midsize black triangles. In block B, the midsize white oval is touching the corner of the block. The midsize blue oval is above the white oval. The little black rectangle is to the right of the blue oval. In block C, there is a little blue rectangle."
                    +
                    " "
                    + "Is the little blue oval to the right of the midsize white oval?"
         },
        {"role": "assistant",
         "content": "Yes"},
        # Direct no
        {"role": "user",
         "content": "There are three blocks of A, B, and C. Block A is to the left of block B, and block B is to the left of block C. In block A, one medium white circle is touching the bottom of the block. The medium green triangle is above the white circle. Another white circle is to the right and below the green triangle. The medium black rectangle is above the white circle. The little white triangle is above the medium black rectangle. The little black circle is to the left of the green triangle. The little black rectangle is above the little black circle. In block B, the large black rectangle is to the left of the large green triangle. The little green triangle is below large triangle and large rectangle. In block C, the large black circle and the little green triangle are touching the bottom of the block. The medium white triangle is above the black circle. The little green triangle is above the white triangle. The little green circle is above the little green triangle."
                    +
                    " "
                    + "In block A, is the little white triangle below the medium black rectangle?"
         },
        {"role": "assistant",
         "content": "No"},
    ]

    new_human_example_list = [[], new_human_few_shots, new_human_COT, new_human_few_shots]
    step_game_few_shots = [{"role": "user",
                            "content": "E and S are parallel, and S is below E."
                                       + " "
                                       + "What is the relation of the agent E to the agent S?"},
                           {"role": "assistant", "content": "above"},

                           {"role": "user",
                            "content": "U is directly north east of Y. "
                                       + "C is sitting at the lower left position to U. "
                                       + "H is at a 45 degree angle to Y, in the upper lefthand corner."
                                       + " "
                                       + "What is the relation of the agent C to the agent Y?"},
                           {"role": "assistant", "content": "overlap"},

                           {"role": "user",
                            "content": "J is north west of B"
                                       + " "
                                       + "What is relation of the agent B to the agent J?"},
                           {"role": "assistant", "content": "lower-right"},

                           {"role": "user",
                            "content": "B is on the top side to A."
                                       + " T is there and A is at the 10 position of a clock face."
                                       + " D is below T at 4 o'clock."
                                       + " B is to the left of G with a small gap between them."
                                       + " G is at P's 3 o'clock."
                                       + " "
                                       + "What is the relation of the agent T to the agent P?"},
                           {"role": "assistant", "content": "right"},

                           {"role": "user",
                            "content": "M is south east of Z"
                                       + " "
                                       + "What is relation of the agent Z to the agent M?"},
                           {"role": "assistant", "content": "upper-left"}]

    step_game_few_shot2 = [{'role': 'user',
                            'content': 'The object S is positioned directly above the object Q. What is the relation of the agent S to the agent Q?'},
                           {'role': 'assistant', 'content': 'above'},
                           {'role': 'user',
                            'content': 'E and L are vertical and E is below L. B is sitting at the upper left position to E. The object labeled C is positioned to the right of the object labeled B. What is the relation of the agent E to the agent L?'},
                           {'role': 'assistant', 'content': 'below'},
                           {'role': 'user',
                            'content': "Z is sitting at the upper left position to V. The object V is upper and slightly to the right of the object Y. Y is on the top side to J. J is diagonally right and above X. H is at the 9 o'clock position relative to X. What is the relation of the agent J to the agent Y?"},
                           {'role': 'assistant', 'content': 'below'},
                           {'role': 'user',
                            'content': 'I is on the top and R is at the bottom. What is the relation of the agent I to the agent R?'},
                           {'role': 'assistant', 'content': 'above'},
                           {'role': 'user',
                            'content': 'E is above C with a small gap between them. E is to the right of Y. What is the relation of the agent Y to the agent E?'},
                           {'role': 'assistant', 'content': 'left'},
                           {'role': 'user',
                            'content': 'D is placed in the right direction of N. N and U are next to each other with N on the left and U on the right. X and D are in a vertical line with X on top. X and W are parallel, and X is over W. What is the relation of the agent N to the agent W?'},
                           {'role': 'assistant', 'content': 'left'},
                           {'role': 'user',
                            'content': 'K is at the bottom of U vertically. Y is there and X is at the 10 position of a clock face. C is below and to the left of K. U is over there with Y above. F is over there and X is directly above it. What is the relation of the agent F to the agent U?'},
                           {'role': 'assistant', 'content': 'lower-left'},
                           {'role': 'user',
                            'content': 'G is over there and Z is on the top of it. Z is sitting at the upper right position to Y. What is the relation of the agent Y to the agent Z?'},
                           {'role': 'assistant', 'content': 'lower-left'},
                           {'role': 'user',
                            'content': 'H is placed in the left direction of M. The object M is lower and slightly to the right of the object N. What is the relation of the agent M to the agent N?'},
                           {'role': 'assistant', 'content': 'lower-right'},
                           {'role': 'user',
                            'content': "Y is at the 6 o'clock position relative to L. C is on the right side and below I. L is placed in the right direction of I. What is the relation of the agent I to the agent C?"},
                           {'role': 'assistant', 'content': 'upper-left'},
                           {'role': 'user',
                            'content': 'L is positioned in the top left corner of Z. The object N and H are there. The object H is above and slightly to the left of the object N. N is there and Z is at the 10 position of a clock face. What is the relation of the agent Z to the agent L?'},
                           {'role': 'assistant', 'content': 'lower-right'},
                           {'role': 'user',
                            'content': 'The object Z is upper and slightly to the right of the object P. Z and A are in a vertical line with A below Z. U is sitting in the left direction of A. P is above and to the left of M. What is the relation of the agent P to the agent U?'},
                           {'role': 'assistant', 'content': 'overlap'},
                           {'role': 'user',
                            'content': 'M is under Q. If P is the center of a clock face, F is located between 4 and 5. E is directly north west of K. The object E is positioned above and to the left of object Q. The object P is upper and slightly to the left of the object K. What is the relation of the agent K to the agent Q?'},
                           {'role': 'assistant', 'content': 'overlap'},
                           {'role': 'user',
                            'content': 'N is at the lower side of F. P is under M. Y is over N. M is positioned in the front right corner of L. The object labeled P is positioned to the left of the object labeled Y. What is the relation of the agent P to the agent L?'},
                           {'role': 'assistant', 'content': 'right'},
                           {'role': 'user',
                            'content': 'G is lower left to N. What is the relation of the agent N to the agent G?'},
                           {'role': 'assistant', 'content': 'upper-right'},
                           {'role': 'user',
                            'content': 'W is over there and K is directly above it. Q is to the right and above W at an angle of about 45 degrees. What is the relation of the agent Q to the agent K?'},
                           {'role': 'assistant', 'content': 'right'},
                           {'role': 'user',
                            'content': 'K is positioned left to G. H is to the bottom left of G. K is positioned in the lower right corner of F. X is diagonally to the bottom left of H. What is the relation of the agent F to the agent H?'},
                           {'role': 'assistant', 'content': 'upper-left'},
                           {'role': 'user',
                            'content': 'O is on the right and H is on the left. X and J are in a horizontal line with J on the right. H is at the bottom of X vertically. What is the relation of the agent J to the agent H?'},
                           {'role': 'assistant', 'content': 'upper-right'}]

    if dataset_name == "human":
        example_list = human_example_list
    elif dataset_name == "resq":
        example_list = RESQ_example_list
    elif dataset_name == "new_human":
        example_list = human_example_list if used_same_human else new_human_example_list
    elif dataset_name == "stepgame":
        example_list = [[], step_game_few_shot2]
    else:
        example_list = [[] * 3]

    # with open("FormalRuleText2.txt", 'r') as file:
    #     rule_formal = ""
    #     for line in file.readlines():
    #         rule_formal += line
    # additional_prompt = "Consider using the following rules to help with explanation:\n" + rule_formal

    return example_list[method_number]
