import torch
from domiknows.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor, JointSensor
from domiknows.sensor.pytorch.learners import ModuleLearner, LSTMLearner
from models import *
from utils import *
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from sklearn import preprocessing
from transformers import RobertaModel
import numpy as np


def program_declaration(cur_device, *, pmd=False, beta=0.5, sampling=False, sampleSize=1, dropout=False,
                        constraints=False, model='bert'):
    from graph import graph, story, question, answer_class, story_contain, \
        symmetric, s_quest1, s_quest2, reverse, r_quest1, r_quest2, \
        transitive, t_quest1, t_quest2, t_quest3, transitive_topo, tt_quest1, tt_quest2, tt_quest3, tt_quest4

    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")

    def str_to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def make_labels(label_list):
        labels = label_list.split("@@")
        label_nums = [0 if label == "Yes" else 1 if label == "No" else 2 for label in labels]
        return str_to_int_list(label_nums)

    def make_question(questions, stories, relations, q_ids, labels):
        num_labels = make_labels(labels)
        ids = str_to_int_list(q_ids.split("@@"))
        return torch.ones(len(questions.split("@@")), 1), questions.split("@@"), stories.split("@@"), \
            relations.split("@@"), ids, num_labels

    question[story_contain, "question", "story", "relation", "id", "label"] = \
        JointSensor(story["questions"], story["stories"], story["relations"],
                    story["question_ids"], story["labels"], forward=make_question, device=cur_device)

    def read_label(_, label):
        return label

    question[answer_class] = FunctionalSensor(story_contain, "label", forward=read_label, label=True, device=cur_device)
    print("Using the {:} as the baseline model".format(model))

    if model == "roberta":
        question["input_ids"] = JointSensor(story_contain, 'question', "story",
                                            forward=RoBERTaTokenizer(), device=cur_device)
        clf = MultipleClassYNRoberta.from_pretrained('roberta-base', device=cur_device, drp=dropout)
    elif model == "t5-adapter":
        question["input_ids"] = JointSensor(story_contain, 'question', "story",
                                            forward=T5Tokenizer('google/flan-t5-base'), device=cur_device)
        clf = MultipleClassYNT5('google/flan-t5-base', device=cur_device, adapter=True)
    elif model == "llama3":
        model_ids = "meta-llama/Meta-Llama-3-8B"
        tokenizer = Llama3Tokenizer(model_ids)
        
        question["input_ids"] = JointSensor(story_contain, 'question', "story",
                                            forward=tokenizer, device=cur_device)
        clf = MultipleClassYNLlama3(model_ids, tokenizer, device=cur_device, adapter=True)
    else:
        question["input_ids"] = JointSensor(story_contain, 'question', "story",
                                            forward=BERTTokenizer(), device=cur_device)
        clf = MultipleClassYN.from_pretrained('bert-base-uncased', device=cur_device, drp=dropout)

    question[answer_class] = ModuleLearner("input_ids", module=clf, device=cur_device)

    poi_list = [question, answer_class]

    # Including the constraints relation check
    if constraints:
        print("Include logical constraints")
        symmetric[s_quest1.reversed, s_quest2.reversed] = \
            CompositionCandidateSensor(
                relations=(s_quest1.reversed, s_quest2.reversed),
                forward=check_symmetric, device=cur_device)

        reverse[r_quest1.reversed, r_quest2.reversed] = \
            CompositionCandidateSensor(
                relations=(r_quest1.reversed, r_quest2.reversed),
                forward=check_reverse, device=cur_device)

        transitive[t_quest1.reversed, t_quest2.reversed, t_quest3.reversed] = \
            CompositionCandidateSensor(
                relations=(t_quest1.reversed, t_quest2.reversed, t_quest3.reversed),
                forward=check_transitive, device=cur_device)

        transitive_topo[tt_quest1.reversed, tt_quest2.reversed, tt_quest3.reversed, tt_quest4.reversed] = \
            CompositionCandidateSensor(
                relations=(tt_quest1.reversed, tt_quest2.reversed, tt_quest3.reversed, tt_quest4.reversed),
                forward=check_transitive_topo, device=cur_device)

        poi_list.extend([symmetric, reverse, transitive, transitive_topo])

    from domiknows.program.metric import PRF1Tracker, PRF1Tracker, DatanodeCMMetric, MacroAverageTracker, ValueTracker
    from domiknows.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss, BCEFocalLoss
    from domiknows.program import LearningBasedProgram, SolverPOIProgram
    from domiknows.program.lossprogram import SampleLossProgram, PrimalDualProgram
    from domiknows.program.model.pytorch import model_helper, PoiModel, SolverModel

    infer_list = ['local/argmax']  # ['ILP', 'local/argmax']
    if pmd:
        print("Using Primal Dual Program")
        program = PrimalDualProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    beta=beta,
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    device=cur_device)
    elif sampling:
        program = SampleLossProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    sample=True,
                                    sampleSize=sampleSize,
                                    sampleGlobalLoss=False,
                                    beta=1,
                                    device=cur_device)
    else:
        print("Using Base Program")
        program = SolverPOIProgram(graph,
                                   poi=poi_list,
                                   inferTypes=infer_list,
                                   loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                   metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                           'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                   device=cur_device)

    return program


def program_declaration_spartun_fr(device, *, pmd=False, beta=0.5, sampling=False, sampleSize=1, dropout=False,
                                   constraints=False, spartun=True, model="bert"):
    program = None
    from graph_spartun_rel import graph, story, story_contain, question, \
        left, right, above, below, behind, front, near, far, disconnected, touch, \
        overlap, coveredby, inside, cover, contain, inverse, inv_question1, inv_question2, \
        transitive, tran_quest1, tran_quest2, tran_quest3, tran_topo, tran_topo_quest1, \
        tran_topo_quest2, tran_topo_quest3, tran_topo_quest4

    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")
    all_labels = ["left", "right", "above", "below", "behind", "front", "near", "far", "dc", "ec", "po", "tpp", "ntpp",
                  "tppi", "ntppi"]

    def to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def make_labels(label_list):
        labels = label_list.split("@@")
        all_labels_list = [[] for _ in range(15)]
        for bits_label in labels:
            bits_label = int(bits_label)
            cur_label = 1
            for ind, label in enumerate(all_labels):
                all_labels_list[ind].append(1 if bits_label & cur_label else 0)
                cur_label *= 2

        # label_nums = [0 if label == "Yes" else 1 if label == "No" else 2 for label in labels]
        return [to_int_list(labels_list) for labels_list in all_labels_list]

    def make_question(questions, stories, relations, q_ids, labels):
        all_labels = make_labels(labels)
        ids = to_int_list(q_ids.split("@@"))
        left_list, right_list, above_list, below_list, behind_list, \
            front_list, near_list, far_list, dc_list, ec_list, po_list, \
            tpp_list, ntpp_list, tppi_list, ntppi_list = all_labels
        return torch.ones(len(questions.split("@@")), 1), questions.split("@@"), stories.split("@@"), \
            relations.split("@@"), ids, left_list, right_list, above_list, below_list, behind_list, \
            front_list, near_list, far_list, dc_list, ec_list, po_list, \
            tpp_list, ntpp_list, tppi_list, ntppi_list

    question[story_contain, "question", "story", "relation", "id", "left_label", "right_label",
    "above_label", "below_label", "behind_label", "front_label", "near_label", "far_label", "dc_label", "ec_label", "po_label",
    "tpp_label", "ntpp_label", "tppi_label", "ntppi_label"] = \
        JointSensor(story["questions"], story["stories"], story["relations"],
                    story["question_ids"], story["labels"], forward=make_question, device=device)

    def read_label(_, label):
        return label

    # Model
    if model == "t5-adapter":
        t5_model_id = "google/flan-t5-base"
        print("Using", t5_model_id)
        question["input_ids"] = JointSensor(story_contain, 'question', "story",
                                            forward=T5Tokenizer(t5_model_id), device=device)

        all_answers = [left, right, above, below, behind, front,
                       near, far, disconnected, touch, overlap, coveredby,
                       inside, cover, contain]
        expected_label = ["left", "right", "above", "below", "behind", "front",
                          "near", "far", "disconnected", "touch", "overlap", "covered by",
                          "inside", "cover", "contain"]

        clf1 = MultipleClassFRT5(t5_model_id, expected_label, device=device, adapter=True)
        question["hidden_layer"] = ModuleLearner("input_ids", module=clf1, device=device)
        question[left] = ModuleLearner("hidden_layer",
                                       module=ClassifyLabelT5(expected_label[0], map_index=clf1.map_label,
                                                              device=device),
                                       device=device)
        question[right] = ModuleLearner("hidden_layer",
                                        module=ClassifyLabelT5(expected_label[1], map_index=clf1.map_label,
                                                               device=device),
                                        device=device)
        question[above] = ModuleLearner("hidden_layer",
                                        module=ClassifyLabelT5(expected_label[2], map_index=clf1.map_label,
                                                               device=device),
                                        device=device)
        question[below] = ModuleLearner("hidden_layer",
                                        module=ClassifyLabelT5(expected_label[3], map_index=clf1.map_label,
                                                               device=device),
                                        device=device)
        question[behind] = ModuleLearner("hidden_layer",
                                         module=ClassifyLabelT5(expected_label[4], map_index=clf1.map_label,
                                                                device=device),
                                         device=device)
        question[front] = ModuleLearner("hidden_layer",
                                        module=ClassifyLabelT5(expected_label[5], map_index=clf1.map_label,
                                                               device=device),
                                        device=device)
        question[near] = ModuleLearner("hidden_layer",
                                       module=ClassifyLabelT5(expected_label[6], map_index=clf1.map_label,
                                                              device=device),
                                       device=device)
        question[far] = ModuleLearner("hidden_layer",
                                      module=ClassifyLabelT5(expected_label[7], map_index=clf1.map_label,
                                                             device=device),
                                      device=device)
        question[disconnected] = ModuleLearner("hidden_layer",
                                               module=ClassifyLabelT5(expected_label[8], map_index=clf1.map_label,
                                                                      device=device),
                                               device=device)
        question[touch] = ModuleLearner("hidden_layer",
                                        module=ClassifyLabelT5(expected_label[9], map_index=clf1.map_label,
                                                               device=device),
                                        device=device)
        question[overlap] = ModuleLearner("hidden_layer",
                                          module=ClassifyLabelT5(expected_label[10], map_index=clf1.map_label,
                                                                 device=device),
                                          device=device)
        question[coveredby] = ModuleLearner("hidden_layer",
                                            module=ClassifyLabelT5(expected_label[11], map_index=clf1.map_label,
                                                                   device=device),
                                            device=device)
        question[inside] = ModuleLearner("hidden_layer",
                                         module=ClassifyLabelT5(expected_label[12], map_index=clf1.map_label,
                                                                device=device),
                                         device=device)
        question[cover] = ModuleLearner("hidden_layer",
                                        module=ClassifyLabelT5(expected_label[13], map_index=clf1.map_label,
                                                               device=device),
                                        device=device)
        question[contain] = ModuleLearner("hidden_layer",
                                          module=ClassifyLabelT5(expected_label[14], map_index=clf1.map_label,
                                                                 device=device),
                                          device=device)
    else:
        print("Using BERT")
        question["input_ids"] = JointSensor(story_contain, 'question', "story",
                                            forward=BERTTokenizer(), device=device)
        clf1 = MultipleClassYN_Hidden.from_pretrained('bert-base-uncased', device=device, drp=dropout)
        question["hidden_layer"] = ModuleLearner("input_ids", module=clf1, device=device)
        all_answers = [left, right, above, below, behind, front,
                       near, far, disconnected, touch, overlap, coveredby,
                       inside, cover, contain]
        question[left] = ModuleLearner("hidden_layer",
                                       module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                       device=device)
        question[right] = ModuleLearner("hidden_layer",
                                        module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                        device=device)
        question[above] = ModuleLearner("hidden_layer",
                                        module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                        device=device)
        question[below] = ModuleLearner("hidden_layer",
                                        module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                        device=device)
        question[behind] = ModuleLearner("hidden_layer",
                                         module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                         device=device)
        question[front] = ModuleLearner("hidden_layer",
                                        module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                        device=device)
        question[near] = ModuleLearner("hidden_layer",
                                       module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                       device=device)
        question[far] = ModuleLearner("hidden_layer",
                                      module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                      device=device)
        question[disconnected] = ModuleLearner("hidden_layer",
                                               module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                               device=device)
        question[touch] = ModuleLearner("hidden_layer",
                                        module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                        device=device)
        question[overlap] = ModuleLearner("hidden_layer",
                                          module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                          device=device)
        question[coveredby] = ModuleLearner("hidden_layer",
                                            module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                            device=device)
        question[inside] = ModuleLearner("hidden_layer",
                                         module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                         device=device)
        question[cover] = ModuleLearner("hidden_layer",
                                        module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                        device=device)
        question[contain] = ModuleLearner("hidden_layer",
                                          module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                          device=device)

    # Reading label
    question[left] = FunctionalSensor(story_contain, "left_label", forward=read_label, label=True, device=device)
    question[right] = FunctionalSensor(story_contain, "right_label", forward=read_label, label=True, device=device)
    question[above] = FunctionalSensor(story_contain, "above_label", forward=read_label, label=True, device=device)
    question[below] = FunctionalSensor(story_contain, "below_label", forward=read_label, label=True, device=device)
    question[behind] = FunctionalSensor(story_contain, "behind_label", forward=read_label, label=True, device=device)
    question[front] = FunctionalSensor(story_contain, "front_label", forward=read_label, label=True, device=device)
    question[near] = FunctionalSensor(story_contain, "near_label", forward=read_label, label=True, device=device)
    question[far] = FunctionalSensor(story_contain, "far_label", forward=read_label, label=True, device=device)
    question[disconnected] = FunctionalSensor(story_contain, "dc_label", forward=read_label, label=True, device=device)
    question[touch] = FunctionalSensor(story_contain, "ec_label", forward=read_label, label=True, device=device)
    question[overlap] = FunctionalSensor(story_contain, "po_label", forward=read_label, label=True, device=device)
    question[coveredby] = FunctionalSensor(story_contain, "tpp_label", forward=read_label, label=True, device=device)
    question[inside] = FunctionalSensor(story_contain, "ntpp_label", forward=read_label, label=True, device=device)
    question[cover] = FunctionalSensor(story_contain, "tppi_label", forward=read_label, label=True, device=device)
    question[contain] = FunctionalSensor(story_contain, "ntppi_label", forward=read_label, label=True, device=device)

    # question[left] = ModuleLearner("hidden_layer",
    #                                module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
    #                                device=device)
    # question[right] = ModuleLearner("hidden_layer",
    #                                 module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
    #                                 device=device)
    # question[above] = ModuleLearner("hidden_layer",
    #                                 module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
    #                                 device=device)
    # question[below] = ModuleLearner("hidden_layer",
    #                                 module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
    #                                 device=device)
    # question[behind] = ModuleLearner("hidden_layer",
    #                                  module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
    #                                  device=device)
    # question[front] = ModuleLearner("hidden_layer",
    #                                 module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
    #                                 device=device)
    # question[near] = ModuleLearner("hidden_layer",
    #                                module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
    #                                device=device)
    # question[far] = ModuleLearner("hidden_layer",
    #                               module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
    #                               device=device)
    # question[disconnected] = ModuleLearner("hidden_layer",
    #                                        module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
    #                                        device=device)
    # question[touch] = ModuleLearner("hidden_layer",
    #                                 module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device)
    # question[overlap] = ModuleLearner("hidden_layer",
    #                                   module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device)
    # question[coveredby] = ModuleLearner("hidden_layer",
    #                                     module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
    #                                     device=device)
    # question[inside] = ModuleLearner("hidden_layer",
    #                                  module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
    #                                  device=device)
    #
    # question[cover] = ModuleLearner("hidden_layer",
    #                                 module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
    #                                 device=device)
    #
    # question[contain] = ModuleLearner("hidden_layer",
    #                                   module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
    #                                   device=device)

    poi_list = [question, left, right, above, below, behind, front, near, far,
                disconnected, touch, overlap, coveredby, inside, cover, contain]

    if constraints:
        print("Included constraints")
        inverse[inv_question1.reversed, inv_question2.reversed] = \
            CompositionCandidateSensor(
                relations=(inv_question1.reversed, inv_question2.reversed),
                forward=check_symmetric, device=device)

        transitive[tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed] = \
            CompositionCandidateSensor(
                relations=(tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed),
                forward=check_transitive, device=device)

        tran_topo[tran_topo_quest1.reversed, tran_topo_quest2.reversed,
        tran_topo_quest3.reversed, tran_topo_quest4.reversed] = \
            CompositionCandidateSensor(
                relations=(tran_topo_quest1.reversed, tran_topo_quest2.reversed
                           , tran_topo_quest3.reversed, tran_topo_quest4.reversed),
                forward=check_transitive_topo, device=device)
        poi_list.extend([inverse, transitive, tran_topo])

    from domiknows.program.metric import PRF1Tracker, PRF1Tracker, DatanodeCMMetric, MacroAverageTracker, ValueTracker
    from domiknows.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss, BCEFocalLoss
    from domiknows.program import LearningBasedProgram, SolverPOIProgram
    from domiknows.program.lossprogram import SampleLossProgram, PrimalDualProgram
    from domiknows.program.model.pytorch import model_helper, PoiModel, SolverModel

    infer_list = ['ILP', 'local/argmax']  # ['ILP', 'local/argmax']
    if pmd:
        print("Using PMD program")
        program = PrimalDualProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    beta=beta,
                                    metric={
                                        'ILP': PRF1Tracker(DatanodeCMMetric()),
                                        'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    device=device)
    elif sampling:
        program = SampleLossProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={
                                        'ILP': PRF1Tracker(DatanodeCMMetric()),
                                        'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    sample=True,
                                    sampleSize=sampleSize,
                                    sampleGlobalLoss=False,
                                    beta=1,
                                    device=device)
    else:
        print("Using Base program")
        program = SolverPOIProgram(graph,
                                   poi=poi_list,
                                   inferTypes=infer_list,
                                   loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                   metric={
                                       'ILP': PRF1Tracker(DatanodeCMMetric()),
                                       'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                   device=device)

    return program


def program_declaration_StepGame(device, *, pmd=False, beta=0.5, sampling=False, sampleSize=1, dropout=False,
                                 constraints=False, spartun=True):
    program = None
    from graph_stepgame import graph, story, story_contain, question, \
        left, right, above, below, lower_left, lower_right, upper_left, upper_right, overlap
    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")
    all_labels = ["left", "right", "above", "below", "lower-left",
                  "lower-right", "upper-left", "upper-right", "overlap"]

    def to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def make_labels(label_list):
        labels = label_list.split("@@")
        all_labels_list = [[] for _ in range(9)]
        for cur_label in labels:
            for ind, label in enumerate(all_labels):
                all_labels_list[ind].append(1 if label == cur_label else 0)
        # label_nums = [0 if label == "Yes" else 1 if label == "No" else 2 for label in labels]
        return [to_int_list(labels_list) for labels_list in all_labels_list]

    def make_question(questions, stories, relations, q_ids, labels):
        all_labels = make_labels(labels)
        ids = to_int_list(q_ids.split("@@"))
        left_list, right_list, above_list, below_list, lower_left_list, \
            lower_right_list, upper_left_list, upper_right_list, over_lap_list = all_labels
        return torch.ones(len(questions.split("@@")), 1), questions.split("@@"), stories.split("@@"), \
            relations.split("@@"), ids, left_list, right_list, above_list, below_list, lower_left_list, \
            lower_right_list, upper_left_list, upper_right_list, over_lap_list

    question[story_contain, "question", "story", "relation", "id", "left_label", "right_label",
    "above_label", "below_label", "lower_left_label", "lower_right_label", "upper_left_label", "upper_right_label", "overlap_label"] = \
        JointSensor(story["questions"], story["stories"], story["relations"],
                    story["question_ids"], story["labels"], forward=make_question, device=device)

    def read_label(_, label):
        return label

    # question[answer_class] =
    # FunctionalSensor(story_contain, "label", forward=read_label, label=True, device=cur_device)
    # Replace with all classes

    question["input_ids"] = JointSensor(story_contain, 'question', "story",
                                        forward=BERTTokenizer(), device=device)

    clf1 = MultipleClassYN_Hidden.from_pretrained('bert-base-uncased', device=device, drp=dropout)

    question["hidden_layer"] = ModuleLearner("input_ids", module=clf1, device=device)

    question[left] = ModuleLearner("hidden_layer",
                                   module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                   device=device)
    question[left] = FunctionalSensor(story_contain, "left_label", forward=read_label, label=True, device=device)

    question[right] = ModuleLearner("hidden_layer",
                                    module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                    device=device)
    question[right] = FunctionalSensor(story_contain, "right_label", forward=read_label, label=True, device=device)

    question[above] = ModuleLearner("hidden_layer",
                                    module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                    device=device)
    question[above] = FunctionalSensor(story_contain, "above_label", forward=read_label, label=True, device=device)

    question[below] = ModuleLearner("hidden_layer",
                                    module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                    device=device)
    question[below] = FunctionalSensor(story_contain, "below_label", forward=read_label, label=True, device=device)

    question[lower_left] = ModuleLearner("hidden_layer",
                                         module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                         device=device)
    question[lower_left] = FunctionalSensor(story_contain, "lower_left_label", forward=read_label, label=True,
                                            device=device)

    question[lower_right] = ModuleLearner("hidden_layer",
                                          module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                          device=device)
    question[lower_right] = FunctionalSensor(story_contain, "lower_right_label", forward=read_label, label=True,
                                             device=device)

    question[upper_left] = ModuleLearner("hidden_layer",
                                         module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                         device=device)
    question[upper_left] = FunctionalSensor(story_contain, "upper_left_label", forward=read_label, label=True,
                                            device=device)

    question[upper_right] = ModuleLearner("hidden_layer",
                                          module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                          device=device)
    question[upper_right] = FunctionalSensor(story_contain, "upper_right_label", forward=read_label, label=True,
                                             device=device)

    question[overlap] = ModuleLearner("hidden_layer",
                                      module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                      device=device)
    question[overlap] = FunctionalSensor(story_contain, "overlap_label", forward=read_label, label=True, device=device)

    poi_list = [question, left, right, above, below, lower_left, lower_right, upper_left, upper_right,
                overlap]

    from domiknows.program.metric import PRF1Tracker, PRF1Tracker, DatanodeCMMetric, MacroAverageTracker, ValueTracker
    from domiknows.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss, BCEFocalLoss
    from domiknows.program import LearningBasedProgram, SolverPOIProgram
    from domiknows.program.lossprogram import SampleLossProgram, PrimalDualProgram
    from domiknows.program.model.pytorch import model_helper, PoiModel, SolverModel

    infer_list = ['local/argmax']  # ['ILP', 'local/argmax']
    if pmd:
        program = PrimalDualProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    beta=beta,
                                    metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    device=device)
    elif sampling:
        program = SampleLossProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    sample=True,
                                    sampleSize=sampleSize,
                                    sampleGlobalLoss=False,
                                    beta=1,
                                    device=device)
    else:
        program = SolverPOIProgram(graph,
                                   poi=poi_list,
                                   inferTypes=infer_list,
                                   loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                   metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                   device=device)

    return program


def program_declaration_spartun_fr_T5(device, *, pmd=False, beta=0.5, sampling=False, sampleSize=1, dropout=False,
                                      constraints=False, spartun=True):
    from graph_spartun_rel import graph, story, story_contain, question, \
        left, right, above, below, behind, front, near, far, disconnected, touch, \
        overlap, coveredby, inside, cover, contain, inverse, inv_question1, inv_question2, \
        transitive, tran_quest1, tran_quest2, tran_quest3, tran_topo, tran_topo_quest1, \
        tran_topo_quest2, tran_topo_quest3, tran_topo_quest4, output_for_loss

    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")
    all_labels = ["left", "right", "above", "below", "behind", "front",
                  "near", "far", "disconnected", "touch", "overlap", "covered by",
                  "inside", "cover", "contain"]
    map_label_index = {text: i for i, text in enumerate(all_labels)}

    def to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def to_float_list(x):
        return torch.Tensor([float(i) for i in x])

    def make_labels(label_list):
        labels = label_list.split("@@")
        text_label = ["" for _ in range(len(labels))]

        for ind, bits_label in enumerate(labels):
            bits_label = int(bits_label)
            cur_bit = 1
            for label in all_labels:
                if bits_label & cur_bit:
                    text_label[ind] += label if text_label[ind] == "" else (", " + label)
                cur_bit *= 2
        # label_nums = [0 if label == "Yes" else 1 if label == "No" else 2 for label in labels]
        return text_label

    def make_question(questions, stories, relations, q_ids, labels):
        text_label = make_labels(labels)
        ids = to_int_list(q_ids.split("@@"))

        return torch.ones(len(questions.split("@@")), 1), questions.split("@@"), stories.split("@@"), relations.split(
            "@@"), ids, text_label

    question[story_contain, "question", "story", "relation", "id", "text_labels"] = \
        JointSensor(story["questions"], story["stories"], story["relations"],
                    story["question_ids"], story["labels"], forward=make_question, device=device)

    T5_model = T5WithLora("google/flan-t5-base", device=device, adapter=True)
    # defined loss based on the model
    LossT5 = T5LossFunction(T5_model=T5_model)
    t5_outTokenizer = T5TokenizerOutput('google/flan-t5-base')
    t5_inTokenizer = T5TokenizerInput('google/flan-t5-base')
    question[output_for_loss] = JointSensor(story_contain, 'question', "story",
                                            forward=t5_inTokenizer, device=device)

    question["input_ids"] = JointSensor(story_contain, 'question', "story", True,
                                        forward=t5_inTokenizer, device=device)

    question[output_for_loss] = FunctionalSensor(story_contain,
                                                 'text_labels',
                                                 forward=t5_outTokenizer,
                                                 label=True,
                                                 device=device)

    all_answers = [left, right, above, below, behind, front,
                   near, far, disconnected, touch, overlap, coveredby,
                   inside, cover, contain]

    question["output_encoder"] = ModuleLearner(story_contain, "input_ids", module=T5_model, device=device)
    question["output_decoder"] = FunctionalSensor(story_contain, "output_encoder",
                                                  forward=T5TokenizerDecoder('google/flan-t5-base'), device=device)

    def read_decoder(_, decoder_list):
        text_label = [[0] * 15 for _ in range(len(decoder_list))]
        for ind, text_decode in enumerate(decoder_list):
            text_decode = text_decode.replace("and", "")
            all_relations = text_decode.strip().split(", ")
            for relation in all_relations:
                relation = relation.strip()
                if relation not in map_label_index:
                    continue
                text_label[ind][map_label_index[relation]] = 1
        list_tensor = [to_float_list(labels_list) for labels_list in text_label]
        return torch.stack(list_tensor)

    def read_label(_, relation_list, index):
        label = relation_list[:, index].reshape((-1, 1))
        label = torch.concat((torch.ones_like(label) - label, label), dim=-1)
        return label

    question["output_relations"] = FunctionalSensor(story_contain, "output_decoder", forward=read_decoder,
                                                    device=device)

    question[left] = FunctionalSensor(story_contain, "output_relations", 0, forward=read_label, device=device)
    question[right] = FunctionalSensor(story_contain, "output_relations", 1, forward=read_label, device=device)
    question[above] = FunctionalSensor(story_contain, "output_relations", 2, forward=read_label, device=device)
    question[below] = FunctionalSensor(story_contain, "output_relations", 3, forward=read_label, device=device)
    question[behind] = FunctionalSensor(story_contain, "output_relations", 4, forward=read_label, device=device)
    question[front] = FunctionalSensor(story_contain, "output_relations", 5, forward=read_label, device=device)
    question[near] = FunctionalSensor(story_contain, "output_relations", 6, forward=read_label, device=device)
    question[far] = FunctionalSensor(story_contain, "output_relations", 7, forward=read_label, device=device)
    question[disconnected] = FunctionalSensor(story_contain, "output_relations", 8, forward=read_label, device=device)
    question[touch] = FunctionalSensor(story_contain, "output_relations", 9, forward=read_label, device=device)
    question[overlap] = FunctionalSensor(story_contain, "output_relations", 10, forward=read_label, device=device)
    question[coveredby] = FunctionalSensor(story_contain, "output_relations", 11, forward=read_label, device=device)
    question[inside] = FunctionalSensor(story_contain, "output_relations", 12, forward=read_label, device=device)
    question[cover] = FunctionalSensor(story_contain, "output_relations", 13, forward=read_label, device=device)
    question[contain] = FunctionalSensor(story_contain, "output_relations", 14, forward=read_label, device=device)

    poi_list = [question, left, right, above, below, behind, front, near, far,
                disconnected, touch, overlap, coveredby, inside, cover, contain, output_for_loss]

    if constraints:
        inverse[inv_question1.reversed, inv_question2.reversed] = \
            CompositionCandidateSensor(
                relations=(inv_question1.reversed, inv_question2.reversed),
                forward=check_symmetric, device=device)

        transitive[tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed] = \
            CompositionCandidateSensor(
                relations=(tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed),
                forward=check_transitive, device=device)

        tran_topo[tran_topo_quest1.reversed, tran_topo_quest2.reversed,
        tran_topo_quest3.reversed, tran_topo_quest4.reversed] = \
            CompositionCandidateSensor(
                relations=(tran_topo_quest1.reversed, tran_topo_quest2.reversed
                           , tran_topo_quest3.reversed, tran_topo_quest4.reversed),
                forward=check_transitive_topo, device=device)
        poi_list.extend([inverse, transitive, tran_topo])

    from domiknows.program.metric import PRF1Tracker, PRF1Tracker, DatanodeCMMetric, MacroAverageTracker, ValueTracker
    from domiknows.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss, BCEFocalLoss
    from domiknows.program import LearningBasedProgram, SolverPOIProgram
    from domiknows.program.lossprogram import SampleLossProgram, PrimalDualProgram
    from domiknows.program.model.pytorch import model_helper, PoiModel, SolverModel

    infer_list = ['local/argmax']  # ['ILP', 'local/argmax']
    if pmd:
        print("Using PMD program")
        program = PrimalDualProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=ValueTracker(LossT5),
                                    beta=beta,
                                    device=device)
    elif sampling:
        program = SampleLossProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=ValueTracker(LossT5),
                                    sample=True,
                                    sampleSize=sampleSize,
                                    sampleGlobalLoss=False,
                                    beta=1,
                                    device=device)
    else:
        print("Using Base program")
        program = SolverPOIProgram(graph,
                                   poi=poi_list,
                                   inferTypes=infer_list,
                                   loss=ValueTracker(LossT5),
                                   device=device)

    return program


def program_declaration_spartun_fr_T5_v2(device, *, pmd=False, beta=0.5, sampling=False, sampleSize=1, dropout=False,
                                         constraints=False, spartun=True):
    from graph_spartun_rel import graph, story, story_contain, question, \
        left, right, above, below, behind, front, near, far, disconnected, touch, \
        overlap, coveredby, inside, cover, contain, inverse, inv_question1, inv_question2, \
        transitive, tran_quest1, tran_quest2, tran_quest3, tran_topo, tran_topo_quest1, \
        tran_topo_quest2, tran_topo_quest3, tran_topo_quest4, output_for_loss

    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")
    all_labels = ["left", "right", "above", "below", "behind", "front",
                  "near", "far", "disconnected", "touch", "overlap", "covered by",
                  "inside", "cover", "contain"]
    map_label_index = {text: i for i, text in enumerate(all_labels)}

    def to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def to_float_list(x):
        return torch.Tensor([float(i) for i in x])

    def make_labels(label_list):
        labels = label_list.split("@@")
        text_label = ["" for _ in range(len(labels))]

        for ind, bits_label in enumerate(labels):
            bits_label = int(bits_label)
            cur_bit = 1
            for label in all_labels:
                text_label[ind] += label + ":" + ("yes" if bits_label & cur_bit else "no") + " "
                cur_bit *= 2
        # label_nums = [0 if label == "Yes" else 1 if label == "No" else 2 for label in labels]
        # print(text_label)
        return text_label

    def make_question(questions, stories, relations, q_ids, labels):
        text_label = make_labels(labels)
        ids = to_int_list(q_ids.split("@@"))

        return torch.ones(len(questions.split("@@")), 1), questions.split("@@"), stories.split("@@"), relations.split(
            "@@"), ids, text_label

    question[story_contain, "question", "story", "relation", "id", "text_labels"] = \
        JointSensor(story["questions"], story["stories"], story["relations"],
                    story["question_ids"], story["labels"], forward=make_question, device=device)

    T5_model = T5WithLora("google/flan-t5-base", device=device, adapter=True)
    # defined loss based on the model
    LossT5 = T5LossFunction(T5_model=T5_model)
    t5_outTokenizer = T5TokenizerOutput('google/flan-t5-base')
    t5_inTokenizer = T5TokenizerInput('google/flan-t5-base')
    question[output_for_loss] = JointSensor(story_contain, 'question', "story",
                                            forward=t5_inTokenizer, device=device)

    question["input_ids"] = JointSensor(story_contain, 'question', "story", True,
                                        forward=t5_inTokenizer, device=device)

    question[output_for_loss] = FunctionalSensor(story_contain,
                                                 'text_labels',
                                                 forward=t5_outTokenizer,
                                                 label=True,
                                                 device=device)

    all_answers = [left, right, above, below, behind, front,
                   near, far, disconnected, touch, overlap, coveredby,
                   inside, cover, contain]

    question["output_encoder"] = ModuleLearner(story_contain, "input_ids", module=T5_model, device=device)
    question["output_decoder"] = FunctionalSensor(story_contain, "output_encoder",
                                                  forward=T5TokenizerDecoder('google/flan-t5-base'), device=device)

    def read_decoder(_, decoder_list):
        text_label = [[0] * 15 for _ in range(len(decoder_list))]
        for ind, text_decode in enumerate(decoder_list):
            all_relations = text_decode.strip()
            for label in all_labels:
                if all_relations.find(label + ":" + "yes"):  # This is may be wrong
                    text_label[ind][map_label_index[label]] = 1
        list_tensor = [to_float_list(labels_list) for labels_list in text_label]
        return torch.stack(list_tensor)

    def read_label(_, relation_list, index):
        label = relation_list[:, index].reshape((-1, 1))
        label = torch.concat((torch.ones_like(label) - label, label), dim=-1)
        return label

    question["output_relations"] = FunctionalSensor(story_contain, "output_decoder", forward=read_decoder,
                                                    device=device)

    question[left] = FunctionalSensor(story_contain, "output_relations", 0, forward=read_label, device=device)
    question[right] = FunctionalSensor(story_contain, "output_relations", 1, forward=read_label, device=device)
    question[above] = FunctionalSensor(story_contain, "output_relations", 2, forward=read_label, device=device)
    question[below] = FunctionalSensor(story_contain, "output_relations", 3, forward=read_label, device=device)
    question[behind] = FunctionalSensor(story_contain, "output_relations", 4, forward=read_label, device=device)
    question[front] = FunctionalSensor(story_contain, "output_relations", 5, forward=read_label, device=device)
    question[near] = FunctionalSensor(story_contain, "output_relations", 6, forward=read_label, device=device)
    question[far] = FunctionalSensor(story_contain, "output_relations", 7, forward=read_label, device=device)
    question[disconnected] = FunctionalSensor(story_contain, "output_relations", 8, forward=read_label, device=device)
    question[touch] = FunctionalSensor(story_contain, "output_relations", 9, forward=read_label, device=device)
    question[overlap] = FunctionalSensor(story_contain, "output_relations", 10, forward=read_label, device=device)
    question[coveredby] = FunctionalSensor(story_contain, "output_relations", 11, forward=read_label, device=device)
    question[inside] = FunctionalSensor(story_contain, "output_relations", 12, forward=read_label, device=device)
    question[cover] = FunctionalSensor(story_contain, "output_relations", 13, forward=read_label, device=device)
    question[contain] = FunctionalSensor(story_contain, "output_relations", 14, forward=read_label, device=device)

    poi_list = [question, left, right, above, below, behind, front, near, far,
                disconnected, touch, overlap, coveredby, inside, cover, contain, output_for_loss]

    if constraints:
        inverse[inv_question1.reversed, inv_question2.reversed] = \
            CompositionCandidateSensor(
                relations=(inv_question1.reversed, inv_question2.reversed),
                forward=check_symmetric, device=device)

        transitive[tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed] = \
            CompositionCandidateSensor(
                relations=(tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed),
                forward=check_transitive, device=device)

        tran_topo[tran_topo_quest1.reversed, tran_topo_quest2.reversed,
        tran_topo_quest3.reversed, tran_topo_quest4.reversed] = \
            CompositionCandidateSensor(
                relations=(tran_topo_quest1.reversed, tran_topo_quest2.reversed
                           , tran_topo_quest3.reversed, tran_topo_quest4.reversed),
                forward=check_transitive_topo, device=device)
        poi_list.extend([inverse, transitive, tran_topo])

    from domiknows.program.metric import PRF1Tracker, PRF1Tracker, DatanodeCMMetric, MacroAverageTracker, ValueTracker
    from domiknows.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss, BCEFocalLoss
    from domiknows.program import LearningBasedProgram, SolverPOIProgram
    from domiknows.program.lossprogram import SampleLossProgram, PrimalDualProgram
    from domiknows.program.model.pytorch import model_helper, PoiModel, SolverModel

    infer_list = ['local/argmax']  # ['ILP', 'local/argmax']
    if pmd:
        print("Using PMD program")
        program = PrimalDualProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=ValueTracker(LossT5),
                                    beta=beta,
                                    device=device)
    elif sampling:
        program = SampleLossProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=ValueTracker(LossT5),
                                    sample=True,
                                    sampleSize=sampleSize,
                                    sampleGlobalLoss=False,
                                    beta=1,
                                    device=device)
    else:
        print("Using Base program")
        program = SolverPOIProgram(graph,
                                   poi=poi_list,
                                   inferTypes=infer_list,
                                   loss=ValueTracker(LossT5),
                                   device=device)

    return program


def program_declaration_spartun_fr_T5_v3(device, *, pmd=False, beta=0.5, sampling=False, sampleSize=1, dropout=False,
                                         constraints=False, spartun=True):
    program = None
    from graph_spartun_rel import graph, story, story_contain, question, \
        left, right, above, below, behind, front, near, far, disconnected, touch, \
        overlap, coveredby, inside, cover, contain, inverse, inv_question1, inv_question2, \
        transitive, tran_quest1, tran_quest2, tran_quest3, tran_topo, tran_topo_quest1, \
        tran_topo_quest2, tran_topo_quest3, tran_topo_quest4

    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")
    all_labels = ["left", "right", "above", "below", "behind", "front", "near", "far", "dc", "ec", "po", "tpp", "ntpp",
                  "tppi", "ntppi"]

    def to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def make_labels(label_list):
        labels = label_list.split("@@")
        all_labels_list = [[] for _ in range(15)]
        for bits_label in labels:
            bits_label = int(bits_label)
            cur_label = 1
            for ind, label in enumerate(all_labels):
                all_labels_list[ind].append(1 if bits_label & cur_label else 0)
                cur_label *= 2

        # label_nums = [0 if label == "Yes" else 1 if label == "No" else 2 for label in labels]
        return [to_int_list(labels_list) for labels_list in all_labels_list]

    def make_text_labels(label_list):
        labels = label_list.split("@@")
        text_label = ["" for _ in range(len(labels))]

        for ind, bits_label in enumerate(labels):
            bits_label = int(bits_label)
            cur_bit = 1
            for label in all_labels:
                if bits_label & cur_bit:
                    text_label[ind] += label if text_label[ind] == "" else (", " + label)
                cur_bit *= 2
        # label_nums = [0 if label == "Yes" else 1 if label == "No" else 2 for label in labels]
        return text_label

    def make_question(questions, stories, relations, q_ids, labels):
        all_labels = make_labels(labels)
        text_labels = make_text_labels(labels)
        ids = to_int_list(q_ids.split("@@"))
        left_list, right_list, above_list, below_list, behind_list, \
            front_list, near_list, far_list, dc_list, ec_list, po_list, \
            tpp_list, ntpp_list, tppi_list, ntppi_list = all_labels
        return torch.ones(len(questions.split("@@")), 1), questions.split("@@"), stories.split("@@"), \
            relations.split("@@"), ids, left_list, right_list, above_list, below_list, behind_list, \
            front_list, near_list, far_list, dc_list, ec_list, po_list, \
            tpp_list, ntpp_list, tppi_list, ntppi_list, text_labels

    question[story_contain, "question", "story", "relation", "id", "left_label", "right_label",
    "above_label", "below_label", "behind_label", "front_label", "near_label", "far_label", "dc_label", "ec_label", "po_label",
    "tpp_label", "ntpp_label", "tppi_label", "ntppi_label", "text_label"] = \
        JointSensor(story["questions"], story["stories"], story["relations"],
                    story["question_ids"], story["labels"], forward=make_question, device=device)

    def read_label(_, label):
        return label

    print("Using T5")

    t5_outtokenizer = T5TokenizerOutput('google/flan-t5-base')
    t5_inTokenizer = T5TokenizerInput('google/flan-t5-base')

    t5_normal_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    label_text = ["left", "right", "above", "below", "behind", "front",
                  "near", "far", "disconnected", "touch", "overlap", "covered by",
                  "inside", "cover", "contain"]
    
    t5_model = T5WithLoraGenerativeCLF("google/flan-t5-base",
                                       label=label_text,
                                       tokenizer=t5_normal_tokenizer,
                                       device=device,
                                       adapter=True)

    question["input_ids"] = JointSensor(story_contain, 'question', "story", True,
                                        forward=t5_inTokenizer, device=device)

    question["label_input_ids"] = JointSensor(story_contain, "text_label", True,
                                              forward=t5_outtokenizer, device=device)


    question["hidden_layer"] = ModuleLearner(story_contain, "input_ids", "label_input_ids", module=t5_model, device=device)

    all_answers = [left, right, above, below, behind, front,
                   near, far, disconnected, touch, overlap, coveredby,
                   inside, cover, contain]
    hidden_layers = 2
    question[left] = ModuleLearner("hidden_layer",
                                   module=
                                   ClassifyLayer2(t5_model.hidden_size, hidden_layer=hidden_layers, device=device, drp=dropout),
                                   device=device)
    question[right] = ModuleLearner("hidden_layer",
                                    module=
                                    ClassifyLayer2(t5_model.hidden_size, hidden_layer=hidden_layers, device=device, drp=dropout),
                                    device=device)
    question[above] = ModuleLearner("hidden_layer",
                                    module=
                                    ClassifyLayer2(t5_model.hidden_size, hidden_layer=hidden_layers, device=device, drp=dropout),
                                    device=device)
    question[below] = ModuleLearner("hidden_layer",
                                    module=
                                    ClassifyLayer2(t5_model.hidden_size, hidden_layer=hidden_layers, device=device, drp=dropout),
                                    device=device)
    question[behind] = ModuleLearner("hidden_layer",
                                     module=
                                     ClassifyLayer2(t5_model.hidden_size, hidden_layer=hidden_layers, device=device, drp=dropout),
                                     device=device)
    question[front] = ModuleLearner("hidden_layer",
                                    module=
                                    ClassifyLayer2(t5_model.hidden_size, hidden_layer=hidden_layers, device=device, drp=dropout),
                                    device=device)
    question[near] = ModuleLearner("hidden_layer",
                                   module=
                                   ClassifyLayer2(t5_model.hidden_size, hidden_layer=hidden_layers, device=device, drp=dropout),
                                   device=device)
    question[far] = ModuleLearner("hidden_layer",
                                  module=
                                  ClassifyLayer2(t5_model.hidden_size, hidden_layer=hidden_layers, device=device, drp=dropout),
                                  device=device)
    question[disconnected] = ModuleLearner("hidden_layer",
                                           module=
                                           ClassifyLayer2(t5_model.hidden_size, hidden_layer=hidden_layers, device=device, drp=dropout),
                                           device=device)
    question[touch] = ModuleLearner("hidden_layer",
                                    module=
                                    ClassifyLayer2(t5_model.hidden_size, hidden_layer=hidden_layers, device=device, drp=dropout),
                                    device=device)
    question[overlap] = ModuleLearner("hidden_layer",
                                      module=
                                      ClassifyLayer2(t5_model.hidden_size, hidden_layer=hidden_layers, device=device, drp=dropout),
                                      device=device)
    question[coveredby] = ModuleLearner("hidden_layer",
                                        module=
                                        ClassifyLayer2(t5_model.hidden_size, hidden_layer=hidden_layers, device=device, drp=dropout),
                                        device=device)
    question[inside] = ModuleLearner("hidden_layer",
                                     module=ClassifyLayer2(t5_model.hidden_size, hidden_layer=hidden_layers, device=device, drp=dropout),
                                     device=device)
    question[cover] = ModuleLearner("hidden_layer",
                                    module=ClassifyLayer2(t5_model.hidden_size, hidden_layer=hidden_layers, device=device, drp=dropout),
                                    device=device)
    question[contain] = ModuleLearner("hidden_layer",
                                      module=ClassifyLayer2(t5_model.hidden_size, hidden_layer=hidden_layers, device=device, drp=dropout),
                                      device=device)

    # Reading label
    question[left] = FunctionalSensor(story_contain, "left_label", forward=read_label, label=True, device=device)
    question[right] = FunctionalSensor(story_contain, "right_label", forward=read_label, label=True, device=device)
    question[above] = FunctionalSensor(story_contain, "above_label", forward=read_label, label=True, device=device)
    question[below] = FunctionalSensor(story_contain, "below_label", forward=read_label, label=True, device=device)
    question[behind] = FunctionalSensor(story_contain, "behind_label", forward=read_label, label=True, device=device)
    question[front] = FunctionalSensor(story_contain, "front_label", forward=read_label, label=True, device=device)
    question[near] = FunctionalSensor(story_contain, "near_label", forward=read_label, label=True, device=device)
    question[far] = FunctionalSensor(story_contain, "far_label", forward=read_label, label=True, device=device)
    question[disconnected] = FunctionalSensor(story_contain, "dc_label", forward=read_label, label=True, device=device)
    question[touch] = FunctionalSensor(story_contain, "ec_label", forward=read_label, label=True, device=device)
    question[overlap] = FunctionalSensor(story_contain, "po_label", forward=read_label, label=True, device=device)
    question[coveredby] = FunctionalSensor(story_contain, "tpp_label", forward=read_label, label=True, device=device)
    question[inside] = FunctionalSensor(story_contain, "ntpp_label", forward=read_label, label=True, device=device)
    question[cover] = FunctionalSensor(story_contain, "tppi_label", forward=read_label, label=True, device=device)
    question[contain] = FunctionalSensor(story_contain, "ntppi_label", forward=read_label, label=True, device=device)

    poi_list = [question, left, right, above, below, behind, front, near, far,
                disconnected, touch, overlap, coveredby, inside, cover, contain]

    if constraints:
        print("Included constraints")
        inverse[inv_question1.reversed, inv_question2.reversed] = \
            CompositionCandidateSensor(
                relations=(inv_question1.reversed, inv_question2.reversed),
                forward=check_symmetric, device=device)

        transitive[tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed] = \
            CompositionCandidateSensor(
                relations=(tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed),
                forward=check_transitive, device=device)

        tran_topo[tran_topo_quest1.reversed, tran_topo_quest2.reversed,
        tran_topo_quest3.reversed, tran_topo_quest4.reversed] = \
            CompositionCandidateSensor(
                relations=(tran_topo_quest1.reversed, tran_topo_quest2.reversed
                           , tran_topo_quest3.reversed, tran_topo_quest4.reversed),
                forward=check_transitive_topo, device=device)
        poi_list.extend([inverse, transitive, tran_topo])

    from domiknows.program.metric import PRF1Tracker, PRF1Tracker, DatanodeCMMetric, MacroAverageTracker, ValueTracker
    from domiknows.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss, BCEFocalLoss
    from domiknows.program import LearningBasedProgram, SolverPOIProgram
    from domiknows.program.lossprogram import SampleLossProgram, PrimalDualProgram
    from domiknows.program.model.pytorch import model_helper, PoiModel, SolverModel

    infer_list = ['ILP', 'local/argmax']  # ['ILP', 'local/argmax']
    if pmd:
        print("Using PMD program")
        program = PrimalDualProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    beta=beta,
                                    metric={
                                        'ILP': PRF1Tracker(DatanodeCMMetric()),
                                        'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    device=device)
    elif sampling:
        program = SampleLossProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={
                                        'ILP': PRF1Tracker(DatanodeCMMetric()),
                                        'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    sample=True,
                                    sampleSize=sampleSize,
                                    sampleGlobalLoss=False,
                                    beta=1,
                                    device=device)
    else:
        print("Using Base program")
        program = SolverPOIProgram(graph,
                                   poi=poi_list,
                                   inferTypes=infer_list,
                                   loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                   metric={
                                       'ILP': PRF1Tracker(DatanodeCMMetric()),
                                       'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                   device=device)

    return program


def program_declaration_StepGame_T5(device, *, pmd=False, beta=0.5, sampling=False, sampleSize=1, dropout=False,
                                    constraints=False, spartun=True):
    from graph_stepgame import graph, story, story_contain, question, \
        left, right, above, below, lower_left, lower_right, upper_left, upper_right, overlap, output_for_loss

    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")

    all_labels = ["left", "right", "above", "below", "lower-left",
                  "lower-right", "upper-left", "upper-right", "overlap"]

    map_label_index = {text: i for i, text in enumerate(all_labels)}

    def to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def to_float_list(x):
        return torch.Tensor([float(i) for i in x])

    def make_labels(label_list):
        labels = label_list.split("@@")

        # label_nums = [0 if label == "Yes" else 1 if label == "No" else 2 for label in labels]
        return labels

    def make_question(questions, stories, relations, q_ids, labels):
        text_label = make_labels(labels)
        ids = to_int_list(q_ids.split("@@"))

        return torch.ones(len(questions.split("@@")), 1), questions.split("@@"), stories.split("@@"), relations.split(
            "@@"), ids, text_label

    question[story_contain, "question", "story", "relation", "id", "text_labels"] = \
        JointSensor(story["questions"], story["stories"], story["relations"],
                    story["question_ids"], story["labels"], forward=make_question, device=device)

    T5_model = T5WithLora("google/flan-t5-base", device=device, adapter=True)
    # defined loss based on the model
    LossT5 = T5LossFunction(T5_model=T5_model)
    t5_outTokenizer = T5TokenizerOutput('google/flan-t5-base')
    t5_inTokenizer = T5TokenizerInput('google/flan-t5-base')
    question[output_for_loss] = JointSensor(story_contain, 'question', "story",
                                            forward=t5_inTokenizer, device=device)

    question["input_ids"] = JointSensor(story_contain, 'question', "story", True,
                                        forward=t5_inTokenizer, device=device)

    question[output_for_loss] = FunctionalSensor(story_contain,
                                                 'text_labels',
                                                 forward=t5_outTokenizer,
                                                 label=True,
                                                 device=device)

    all_answers = [left, right, above, below, lower_left, lower_right, upper_left, upper_right, overlap]

    question["output_encoder"] = ModuleLearner(story_contain, "input_ids", module=T5_model, device=device)
    question["output_decoder"] = FunctionalSensor(story_contain, "output_encoder",
                                                  forward=T5TokenizerDecoder('google/flan-t5-base'), device=device)

    def read_decoder(_, decoder_list):
        text_label = [[0] * 15 for _ in range(len(decoder_list))]
        for ind, text_decode in enumerate(decoder_list):
            text_decode = text_decode.replace("and", "")
            all_relations = text_decode.strip().split(", ")
            for relation in all_relations:
                relation = relation.strip()
                if relation not in map_label_index:
                    continue
                text_label[ind][map_label_index[relation]] = 1
        list_tensor = [to_float_list(labels_list) for labels_list in text_label]
        return torch.stack(list_tensor)

    def read_label(_, relation_list, index):
        label = relation_list[:, index].reshape((-1, 1))
        label = torch.concat((torch.ones_like(label) - label, label), dim=-1)
        return label

    question["output_relations"] = FunctionalSensor(story_contain, "output_decoder", forward=read_decoder,
                                                    device=device)

    question[left] = FunctionalSensor(story_contain, "output_relations", 0, forward=read_label, device=device)
    question[right] = FunctionalSensor(story_contain, "output_relations", 1, forward=read_label, device=device)
    question[above] = FunctionalSensor(story_contain, "output_relations", 2, forward=read_label, device=device)
    question[below] = FunctionalSensor(story_contain, "output_relations", 3, forward=read_label, device=device)
    question[lower_left] = FunctionalSensor(story_contain, "output_relations", 4, forward=read_label, device=device)
    question[lower_right] = FunctionalSensor(story_contain, "output_relations", 5, forward=read_label, device=device)
    question[upper_left] = FunctionalSensor(story_contain, "output_relations", 6, forward=read_label, device=device)
    question[upper_right] = FunctionalSensor(story_contain, "output_relations", 7, forward=read_label, device=device)
    question[overlap] = FunctionalSensor(story_contain, "output_relations", 8, forward=read_label, device=device)

    poi_list = [question, left, right, above, below, lower_left,
                lower_right, upper_left, upper_right, overlap, output_for_loss]

    from domiknows.program.metric import PRF1Tracker, DatanodeCMMetric, ValueTracker
    from domiknows.program import SolverPOIProgram
    from domiknows.program.lossprogram import SampleLossProgram, PrimalDualProgram
    from domiknows.program.model.pytorch import SolverModel

    infer_list = ['local/argmax']  # ['ILP', 'local/argmax']
    if pmd:
        print("Using PMD program")
        program = PrimalDualProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=ValueTracker(LossT5),
                                    beta=beta,
                                    device=device)
    elif sampling:
        program = SampleLossProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=ValueTracker(LossT5),
                                    sample=True,
                                    sampleSize=sampleSize,
                                    sampleGlobalLoss=False,
                                    beta=1,
                                    device=device)
    else:
        print("Using Base program")
        program = SolverPOIProgram(graph,
                                   poi=poi_list,
                                   inferTypes=infer_list,
                                   loss=ValueTracker(LossT5),
                                   device=device)

    return program
