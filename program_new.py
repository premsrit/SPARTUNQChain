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
