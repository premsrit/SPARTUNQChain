from domiknows.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor, JointSensor
from domiknows.sensor.pytorch.learners import ModuleLearner, LSTMLearner
from models import *
from utils import *
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor


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
