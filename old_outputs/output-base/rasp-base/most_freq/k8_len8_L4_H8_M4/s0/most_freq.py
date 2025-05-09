import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/rasp/most_freq/k8_len8_L4_H8_M4/s0/most_freq_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 3

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 5
        elif q_position in {1, 5, 6}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 6

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 2
        elif q_position in {2, 3}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 6

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"2", "0", "3", "5", "1"}:
            return k_token == "0"
        elif q_token in {"4", "<s>"}:
            return k_token == "4"

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 1}:
            return token == "4"
        elif position in {2, 3, 4, 5}:
            return token == "<s>"
        elif position in {6, 7}:
            return token == "<pad>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 3, 4, 5, 6}:
            return token == "1"
        elif position in {1}:
            return token == "3"
        elif position in {2}:
            return token == "<s>"
        elif position in {7}:
            return token == "<pad>"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 3, 4, 5, 6, 7}:
            return token == "3"
        elif position in {1}:
            return token == "5"
        elif position in {2}:
            return token == "<s>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 3, 4, 5, 6}:
            return token == "2"
        elif position in {1}:
            return token == "5"
        elif position in {2, 7}:
            return token == "<s>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, position):
        key = (attn_0_0_output, position)
        if key in {
            ("0", 6),
            ("1", 4),
            ("1", 5),
            ("1", 6),
            ("2", 3),
            ("2", 4),
            ("2", 5),
            ("2", 6),
            ("3", 4),
            ("3", 5),
            ("3", 6),
            ("4", 6),
            ("5", 3),
            ("5", 4),
            ("5", 5),
            ("5", 6),
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 6),
        }:
            return 3
        elif key in {
            ("0", 1),
            ("1", 1),
            ("1", 2),
            ("2", 1),
            ("2", 2),
            ("3", 1),
            ("3", 2),
            ("4", 1),
            ("5", 1),
            ("5", 2),
            ("<s>", 1),
            ("<s>", 2),
        }:
            return 0
        elif key in {("0", 2)}:
            return 7
        return 2

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position, attn_0_2_output):
        key = (position, attn_0_2_output)
        if key in {
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "5"),
            (6, "<s>"),
            (7, "0"),
            (7, "1"),
            (7, "2"),
            (7, "3"),
            (7, "4"),
            (7, "5"),
            (7, "<s>"),
        }:
            return 7
        elif key in {
            (0, "0"),
            (0, "1"),
            (0, "2"),
            (0, "4"),
            (0, "5"),
            (0, "<s>"),
            (1, "2"),
            (1, "4"),
            (2, "4"),
        }:
            return 3
        elif key in {(1, "0"), (1, "1"), (1, "3"), (1, "<s>")}:
            return 0
        elif key in {(0, "3")}:
            return 6
        elif key in {(1, "5")}:
            return 4
        return 2

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(positions, attn_0_2_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_1_output):
        key = (num_attn_0_2_output, num_attn_0_1_output)
        if key in {(2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (6, 1), (7, 0), (7, 1)}:
            return 3
        elif key in {(0, 0), (1, 0)}:
            return 4
        return 5

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {0}:
            return 1
        elif key in {1}:
            return 0
        return 3

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3, 4, 5, 6}:
            return k_position == 0
        elif q_position in {7}:
            return k_position == 4

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_3_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3, 7}:
            return k_position == 0
        elif q_position in {4, 5, 6}:
            return k_position == 7

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_3_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(position, mlp_0_0_output):
        if position in {0, 1}:
            return mlp_0_0_output == 3
        elif position in {2, 4, 7}:
            return mlp_0_0_output == 2
        elif position in {3}:
            return mlp_0_0_output == 5
        elif position in {5}:
            return mlp_0_0_output == 7
        elif position in {6}:
            return mlp_0_0_output == 0

    attn_1_2_pattern = select_closest(mlp_0_0_outputs, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_3_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1, 5}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3, 7}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 4

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_2_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_0_output, token):
        if attn_0_0_output in {"<s>", "0", "3", "5", "1", "4"}:
            return token == "2"
        elif attn_0_0_output in {"2"}:
            return token == "<s>"

    num_attn_1_0_pattern = select(tokens, attn_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 1, 6, 7}:
            return token == "4"
        elif mlp_0_0_output in {2, 3}:
            return token == "<s>"
        elif mlp_0_0_output in {4}:
            return token == "0"
        elif mlp_0_0_output in {5}:
            return token == "5"

    num_attn_1_1_pattern = select(tokens, mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, attn_0_2_output):
        if position in {0, 1}:
            return attn_0_2_output == "1"
        elif position in {2, 3, 4, 5, 6}:
            return attn_0_2_output == "<s>"
        elif position in {7}:
            return attn_0_2_output == "<pad>"

    num_attn_1_2_pattern = select(attn_0_2_outputs, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, attn_0_2_output):
        if position in {0}:
            return attn_0_2_output == "3"
        elif position in {1}:
            return attn_0_2_output == "2"
        elif position in {2, 3, 5}:
            return attn_0_2_output == "<s>"
        elif position in {4, 6, 7}:
            return attn_0_2_output == "<pad>"

    num_attn_1_3_pattern = select(attn_0_2_outputs, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_3_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(position, attn_1_1_output):
        key = (position, attn_1_1_output)
        if key in {
            (0, "0"),
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (0, "<s>"),
            (1, "5"),
            (4, "5"),
            (5, "5"),
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "5"),
            (6, "<s>"),
            (7, "0"),
            (7, "1"),
            (7, "2"),
            (7, "4"),
            (7, "5"),
            (7, "<s>"),
        }:
            return 7
        elif key in {(2, "0"), (2, "5"), (3, "5")}:
            return 2
        elif key in {(1, "1"), (1, "2"), (1, "3"), (1, "<s>")}:
            return 5
        elif key in {(1, "0")}:
            return 6
        elif key in {(1, "4")}:
            return 4
        return 3

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(positions, attn_1_1_outputs)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position, attn_1_2_output):
        key = (position, attn_1_2_output)
        if key in {
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (0, "<s>"),
            (6, "<s>"),
            (7, "1"),
            (7, "2"),
            (7, "3"),
            (7, "4"),
            (7, "5"),
            (7, "<s>"),
        }:
            return 2
        elif key in {
            (0, "0"),
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "<s>"),
            (2, "3"),
        }:
            return 5
        elif key in {
            (1, "5"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "4"),
            (2, "5"),
            (2, "<s>"),
            (7, "0"),
        }:
            return 1
        elif key in {(3, "0"), (4, "0"), (5, "0"), (6, "0")}:
            return 4
        return 6

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(positions, attn_1_2_outputs)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_1_output, num_attn_1_1_output):
        key = (num_attn_0_1_output, num_attn_1_1_output)
        if key in {
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (3, 15),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (5, 11),
            (5, 12),
            (5, 13),
            (5, 14),
            (5, 15),
            (6, 14),
            (6, 15),
        }:
            return 5
        elif key in {
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
        }:
            return 4
        elif key in {(0, 0)}:
            return 2
        return 1

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_3_output):
        key = num_attn_0_3_output
        if key in {0}:
            return 5
        return 7

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 7}:
            return k_position == 0
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5, 6}:
            return k_position == 7

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, tokens)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(mlp_1_1_output, mlp_0_0_output):
        if mlp_1_1_output in {0, 2, 3, 7}:
            return mlp_0_0_output == 5
        elif mlp_1_1_output in {1, 6}:
            return mlp_0_0_output == 3
        elif mlp_1_1_output in {4}:
            return mlp_0_0_output == 0
        elif mlp_1_1_output in {5}:
            return mlp_0_0_output == 6

    attn_2_1_pattern = select_closest(mlp_0_0_outputs, mlp_1_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(position, attn_1_3_output):
        if position in {0}:
            return attn_1_3_output == "<s>"
        elif position in {1}:
            return attn_1_3_output == "5"
        elif position in {2, 3, 4, 5, 6, 7}:
            return attn_1_3_output == "4"

    attn_2_2_pattern = select_closest(attn_1_3_outputs, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(mlp_0_1_output, mlp_1_0_output):
        if mlp_0_1_output in {0}:
            return mlp_1_0_output == 3
        elif mlp_0_1_output in {1, 2, 3, 4, 5, 6, 7}:
            return mlp_1_0_output == 6

    attn_2_3_pattern = select_closest(mlp_1_0_outputs, mlp_0_1_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, mlp_1_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_0_output, token):
        if attn_0_0_output in {"5", "0"}:
            return token == "2"
        elif attn_0_0_output in {"1", "4", "2", "3"}:
            return token == "5"
        elif attn_0_0_output in {"<s>"}:
            return token == "0"

    num_attn_2_0_pattern = select(tokens, attn_0_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, ones)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, token):
        if position in {0}:
            return token == "4"
        elif position in {1}:
            return token == "0"
        elif position in {2, 7}:
            return token == "<s>"
        elif position in {3, 4, 5, 6}:
            return token == "1"

    num_attn_2_1_pattern = select(tokens, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, ones)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1}:
            return token == "3"
        elif position in {2, 7}:
            return token == "<pad>"
        elif position in {3, 4, 5, 6}:
            return token == "5"

    num_attn_2_2_pattern = select(tokens, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, ones)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_0_output, token):
        if attn_0_0_output in {"<s>", "2", "0", "3", "5", "4"}:
            return token == "1"
        elif attn_0_0_output in {"1"}:
            return token == "<s>"

    num_attn_2_3_pattern = select(tokens, attn_0_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, ones)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(position):
        key = position
        if key in {0, 4}:
            return 5
        elif key in {1}:
            return 1
        elif key in {2}:
            return 2
        elif key in {3}:
            return 3
        return 0

    mlp_2_0_outputs = [mlp_2_0(k0) for k0 in positions]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_0_output, attn_2_2_output):
        key = (attn_1_0_output, attn_2_2_output)
        if key in {
            ("0", "4"),
            ("2", "4"),
            ("3", "4"),
            ("4", "4"),
            ("4", "<s>"),
            ("5", "4"),
        }:
            return 6
        elif key in {
            ("1", "4"),
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
            ("4", "3"),
            ("4", "5"),
            ("<s>", "4"),
        }:
            return 2
        return 0

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_2_2_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_2_output):
        key = num_attn_2_2_output
        if key in {0}:
            return 1
        return 3

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_2_2_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {0}:
            return 6
        return 0

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # attn_3_0 ####################################################
    def predicate_3_0(q_token, k_token):
        if q_token in {"2", "0"}:
            return k_token == "5"
        elif q_token in {"1", "3", "5"}:
            return k_token == "<s>"
        elif q_token in {"4"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == "3"

    attn_3_0_pattern = select_closest(tokens, tokens, predicate_3_0)
    attn_3_0_outputs = aggregate(attn_3_0_pattern, mlp_1_0_outputs)
    attn_3_0_output_scores = classifier_weights.loc[
        [("attn_3_0_outputs", str(v)) for v in attn_3_0_outputs]
    ]

    # attn_3_1 ####################################################
    def predicate_3_1(mlp_2_0_output, token):
        if mlp_2_0_output in {0, 4, 5, 6, 7}:
            return token == "3"
        elif mlp_2_0_output in {1}:
            return token == "0"
        elif mlp_2_0_output in {2, 3}:
            return token == "<s>"

    attn_3_1_pattern = select_closest(tokens, mlp_2_0_outputs, predicate_3_1)
    attn_3_1_outputs = aggregate(attn_3_1_pattern, mlp_0_1_outputs)
    attn_3_1_output_scores = classifier_weights.loc[
        [("attn_3_1_outputs", str(v)) for v in attn_3_1_outputs]
    ]

    # attn_3_2 ####################################################
    def predicate_3_2(position, mlp_2_0_output):
        if position in {0, 5, 6, 7}:
            return mlp_2_0_output == 6
        elif position in {1}:
            return mlp_2_0_output == 5
        elif position in {2, 3, 4}:
            return mlp_2_0_output == 4

    attn_3_2_pattern = select_closest(mlp_2_0_outputs, positions, predicate_3_2)
    attn_3_2_outputs = aggregate(attn_3_2_pattern, num_mlp_0_1_outputs)
    attn_3_2_output_scores = classifier_weights.loc[
        [("attn_3_2_outputs", str(v)) for v in attn_3_2_outputs]
    ]

    # attn_3_3 ####################################################
    def predicate_3_3(token, mlp_1_0_output):
        if token in {"0"}:
            return mlp_1_0_output == 6
        elif token in {"1", "<s>", "2", "3"}:
            return mlp_1_0_output == 4
        elif token in {"4"}:
            return mlp_1_0_output == 2
        elif token in {"5"}:
            return mlp_1_0_output == 5

    attn_3_3_pattern = select_closest(mlp_1_0_outputs, tokens, predicate_3_3)
    attn_3_3_outputs = aggregate(attn_3_3_pattern, mlp_1_0_outputs)
    attn_3_3_output_scores = classifier_weights.loc[
        [("attn_3_3_outputs", str(v)) for v in attn_3_3_outputs]
    ]

    # num_attn_3_0 ####################################################
    def num_predicate_3_0(attn_0_3_output, attn_1_3_output):
        if attn_0_3_output in {"1", "<s>", "0"}:
            return attn_1_3_output == "1"
        elif attn_0_3_output in {"4", "2", "3", "5"}:
            return attn_1_3_output == "<s>"

    num_attn_3_0_pattern = select(attn_1_3_outputs, attn_0_3_outputs, num_predicate_3_0)
    num_attn_3_0_outputs = aggregate_sum(num_attn_3_0_pattern, num_attn_1_2_outputs)
    num_attn_3_0_output_scores = classifier_weights.loc[
        [("num_attn_3_0_outputs", "_") for v in num_attn_3_0_outputs]
    ].mul(num_attn_3_0_outputs, axis=0)

    # num_attn_3_1 ####################################################
    def num_predicate_3_1(attn_0_0_output, token):
        if attn_0_0_output in {"1", "<s>", "5", "0"}:
            return token == "<s>"
        elif attn_0_0_output in {"2"}:
            return token == "2"
        elif attn_0_0_output in {"3"}:
            return token == "3"
        elif attn_0_0_output in {"4"}:
            return token == "4"

    num_attn_3_1_pattern = select(tokens, attn_0_0_outputs, num_predicate_3_1)
    num_attn_3_1_outputs = aggregate_sum(num_attn_3_1_pattern, num_attn_0_0_outputs)
    num_attn_3_1_output_scores = classifier_weights.loc[
        [("num_attn_3_1_outputs", "_") for v in num_attn_3_1_outputs]
    ].mul(num_attn_3_1_outputs, axis=0)

    # num_attn_3_2 ####################################################
    def num_predicate_3_2(mlp_0_0_output, token):
        if mlp_0_0_output in {0}:
            return token == "0"
        elif mlp_0_0_output in {1, 2, 3, 4, 5}:
            return token == "4"
        elif mlp_0_0_output in {6}:
            return token == "<s>"
        elif mlp_0_0_output in {7}:
            return token == "<pad>"

    num_attn_3_2_pattern = select(tokens, mlp_0_0_outputs, num_predicate_3_2)
    num_attn_3_2_outputs = aggregate_sum(num_attn_3_2_pattern, num_attn_1_1_outputs)
    num_attn_3_2_output_scores = classifier_weights.loc[
        [("num_attn_3_2_outputs", "_") for v in num_attn_3_2_outputs]
    ].mul(num_attn_3_2_outputs, axis=0)

    # num_attn_3_3 ####################################################
    def num_predicate_3_3(attn_0_0_output, token):
        if attn_0_0_output in {"0"}:
            return token == "0"
        elif attn_0_0_output in {"<s>", "2", "3", "1", "4"}:
            return token == "5"
        elif attn_0_0_output in {"5"}:
            return token == "<s>"

    num_attn_3_3_pattern = select(tokens, attn_0_0_outputs, num_predicate_3_3)
    num_attn_3_3_outputs = aggregate_sum(num_attn_3_3_pattern, ones)
    num_attn_3_3_output_scores = classifier_weights.loc[
        [("num_attn_3_3_outputs", "_") for v in num_attn_3_3_outputs]
    ].mul(num_attn_3_3_outputs, axis=0)

    # mlp_3_0 #####################################################
    def mlp_3_0(mlp_2_0_output, attn_2_2_output):
        key = (mlp_2_0_output, attn_2_2_output)
        if key in {
            (0, "0"),
            (0, "4"),
            (2, "2"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "4"),
            (3, "5"),
            (5, "0"),
            (5, "4"),
            (6, "2"),
        }:
            return 4
        elif key in {
            (2, "0"),
            (2, "5"),
            (6, "0"),
            (6, "5"),
            (7, "0"),
            (7, "1"),
            (7, "5"),
        }:
            return 0
        elif key in {(1, "0"), (1, "1"), (1, "2"), (1, "3"), (1, "4"), (1, "<s>")}:
            return 5
        elif key in {(2, "4"), (7, "4")}:
            return 2
        elif key in {(1, "5"), (2, "3"), (4, "3"), (6, "3")}:
            return 6
        elif key in {(2, "1"), (6, "1"), (6, "4")}:
            return 3
        elif key in {(4, "0"), (4, "4")}:
            return 7
        return 1

    mlp_3_0_outputs = [
        mlp_3_0(k0, k1) for k0, k1 in zip(mlp_2_0_outputs, attn_2_2_outputs)
    ]
    mlp_3_0_output_scores = classifier_weights.loc[
        [("mlp_3_0_outputs", str(v)) for v in mlp_3_0_outputs]
    ]

    # mlp_3_1 #####################################################
    def mlp_3_1(mlp_2_0_output, attn_2_2_output):
        key = (mlp_2_0_output, attn_2_2_output)
        if key in {
            (0, "4"),
            (1, "0"),
            (1, "1"),
            (1, "3"),
            (1, "5"),
            (1, "<s>"),
            (2, "4"),
            (3, "4"),
            (4, "4"),
            (5, "4"),
            (6, "4"),
            (7, "0"),
            (7, "1"),
            (7, "3"),
            (7, "4"),
            (7, "5"),
            (7, "<s>"),
        }:
            return 2
        elif key in {
            (0, "3"),
            (1, "2"),
            (2, "0"),
            (2, "1"),
            (2, "3"),
            (2, "5"),
            (2, "<s>"),
            (3, "0"),
            (3, "1"),
            (3, "3"),
            (3, "5"),
            (4, "3"),
            (5, "3"),
        }:
            return 6
        elif key in {(1, "4")}:
            return 3
        return 0

    mlp_3_1_outputs = [
        mlp_3_1(k0, k1) for k0, k1 in zip(mlp_2_0_outputs, attn_2_2_outputs)
    ]
    mlp_3_1_output_scores = classifier_weights.loc[
        [("mlp_3_1_outputs", str(v)) for v in mlp_3_1_outputs]
    ]

    # num_mlp_3_0 #################################################
    def num_mlp_3_0(num_attn_2_2_output, num_attn_1_2_output):
        key = (num_attn_2_2_output, num_attn_1_2_output)
        if key in {
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (0, 20),
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 28),
            (0, 29),
            (0, 30),
            (0, 31),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (1, 30),
            (1, 31),
        }:
            return 2
        elif key in {(0, 0), (0, 1), (1, 0)}:
            return 3
        return 4

    num_mlp_3_0_outputs = [
        num_mlp_3_0(k0, k1)
        for k0, k1 in zip(num_attn_2_2_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_3_0_output_scores = classifier_weights.loc[
        [("num_mlp_3_0_outputs", str(v)) for v in num_mlp_3_0_outputs]
    ]

    # num_mlp_3_1 #################################################
    def num_mlp_3_1(num_attn_1_3_output):
        key = num_attn_1_3_output
        if key in {0}:
            return 2
        return 4

    num_mlp_3_1_outputs = [num_mlp_3_1(k0) for k0 in num_attn_1_3_outputs]
    num_mlp_3_1_output_scores = classifier_weights.loc[
        [("num_mlp_3_1_outputs", str(v)) for v in num_mlp_3_1_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                attn_3_0_output_scores,
                attn_3_1_output_scores,
                attn_3_2_output_scores,
                attn_3_3_output_scores,
                mlp_3_0_output_scores,
                mlp_3_1_output_scores,
                num_mlp_3_0_output_scores,
                num_mlp_3_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
                num_attn_3_0_output_scores,
                num_attn_3_1_output_scores,
                num_attn_3_2_output_scores,
                num_attn_3_3_output_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(run(["<s>", "1", "5", "1", "2", "0", "3"]))
