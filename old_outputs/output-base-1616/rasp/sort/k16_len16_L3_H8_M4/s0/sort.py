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
        "output-base-1616/rasp/sort/k16_len16_L3_H8_M4/s0/sort_weights.csv",
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
    def predicate_0_0(position, token):
        if position in {0, 6, 7, 9, 10, 11, 12, 13, 14}:
            return token == "9"
        elif position in {1, 2, 3, 4, 5, 8}:
            return token == "3"
        elif position in {15}:
            return token == "12"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 1
        elif q_position in {1, 2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0}:
            return token == "12"
        elif position in {1}:
            return token == "1"
        elif position in {2}:
            return token == "0"
        elif position in {3, 4, 5}:
            return token == "2"
        elif position in {6, 7, 8, 9, 10, 11, 12}:
            return token == "8"
        elif position in {13}:
            return token == "9"
        elif position in {14}:
            return token == "4"
        elif position in {15}:
            return token == "3"

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4, 14}:
            return k_position == 5
        elif q_position in {8, 5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {12, 13, 15}:
            return k_position == 8

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 7, 12, 14, 15}:
            return token == "10"
        elif position in {1, 2, 4, 13}:
            return token == "1"
        elif position in {3}:
            return token == "0"
        elif position in {5}:
            return token == "11"
        elif position in {6}:
            return token == "12"
        elif position in {8, 9, 10, 11}:
            return token == "</s>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 12}:
            return token == "11"
        elif position in {1, 9}:
            return token == "</s>"
        elif position in {2, 3}:
            return token == "1"
        elif position in {11, 4}:
            return token == "0"
        elif position in {13, 5, 15}:
            return token == "10"
        elif position in {10, 6}:
            return token == "2"
        elif position in {8, 14, 7}:
            return token == "12"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 7
        elif q_position in {1, 15}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {5, 6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 0

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 11, 4}:
            return token == "10"
        elif position in {1, 2, 3, 12}:
            return token == "0"
        elif position in {5}:
            return token == "12"
        elif position in {6, 7}:
            return token == "11"
        elif position in {8, 9, 15}:
            return token == "1"
        elif position in {10}:
            return token == "<s>"
        elif position in {13}:
            return token == "</s>"
        elif position in {14}:
            return token == "2"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {1, 2, 3}:
            return 10
        elif key in {4, 5}:
            return 11
        elif key in {6}:
            return 6
        return 14

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {0, 5, 6, 7, 8}:
            return 7
        elif key in {1, 2, 15}:
            return 13
        elif key in {3, 4}:
            return 4
        return 5

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_3_output):
        key = (num_attn_0_1_output, num_attn_0_3_output)
        if key in {
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (4, 0),
            (5, 0),
        }:
            return 3
        elif key in {(0, 0), (0, 1), (1, 0)}:
            return 14
        return 1

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_2_output):
        key = (num_attn_0_1_output, num_attn_0_2_output)
        if key in {
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 0),
            (2, 1),
            (2, 2),
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
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
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
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 8),
            (5, 9),
            (5, 10),
            (5, 11),
            (5, 12),
            (5, 13),
            (5, 14),
            (5, 15),
            (6, 6),
            (6, 7),
            (6, 8),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            (7, 8),
            (7, 9),
            (7, 10),
            (7, 11),
            (7, 12),
            (7, 13),
            (7, 14),
            (7, 15),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 15),
            (9, 12),
            (9, 13),
            (9, 14),
            (9, 15),
            (10, 14),
            (10, 15),
        }:
            return 3
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
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (2, 14),
            (2, 15),
        }:
            return 7
        elif key in {(0, 0)}:
            return 14
        return 1

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {0, 1, 3, 5, 6, 7, 8, 11, 12, 14}:
            return k_mlp_0_0_output == 10
        elif q_mlp_0_0_output in {2}:
            return k_mlp_0_0_output == 4
        elif q_mlp_0_0_output in {4}:
            return k_mlp_0_0_output == 14
        elif q_mlp_0_0_output in {9, 15}:
            return k_mlp_0_0_output == 5
        elif q_mlp_0_0_output in {10}:
            return k_mlp_0_0_output == 6
        elif q_mlp_0_0_output in {13}:
            return k_mlp_0_0_output == 2

    attn_1_0_pattern = select_closest(mlp_0_0_outputs, mlp_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(mlp_0_1_output, mlp_0_0_output):
        if mlp_0_1_output in {0, 15}:
            return mlp_0_0_output == 9
        elif mlp_0_1_output in {1, 12}:
            return mlp_0_0_output == 2
        elif mlp_0_1_output in {2, 13}:
            return mlp_0_0_output == 3
        elif mlp_0_1_output in {10, 3, 5}:
            return mlp_0_0_output == 6
        elif mlp_0_1_output in {9, 4}:
            return mlp_0_0_output == 8
        elif mlp_0_1_output in {8, 11, 6, 7}:
            return mlp_0_0_output == 10
        elif mlp_0_1_output in {14}:
            return mlp_0_0_output == 11

    attn_1_1_pattern = select_closest(mlp_0_0_outputs, mlp_0_1_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_3_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 1, 3, 7}:
            return token == "12"
        elif mlp_0_1_output in {2, 13, 6}:
            return token == "6"
        elif mlp_0_1_output in {11, 4}:
            return token == "3"
        elif mlp_0_1_output in {12, 5}:
            return token == "7"
        elif mlp_0_1_output in {8}:
            return token == "8"
        elif mlp_0_1_output in {9, 15}:
            return token == "4"
        elif mlp_0_1_output in {10}:
            return token == "1"
        elif mlp_0_1_output in {14}:
            return token == "9"

    attn_1_2_pattern = select_closest(tokens, mlp_0_1_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 1, 2, 4}:
            return k_position == 3
        elif q_position in {10, 3}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 10
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {8, 13, 7}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {11, 14}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 7
        elif q_position in {15}:
            return k_position == 5

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, mlp_0_1_output):
        if position in {0, 1, 5, 6, 7, 10, 11}:
            return mlp_0_1_output == 4
        elif position in {2, 3, 4, 15}:
            return mlp_0_1_output == 7
        elif position in {8, 14}:
            return mlp_0_1_output == 2
        elif position in {9}:
            return mlp_0_1_output == 5
        elif position in {12, 13}:
            return mlp_0_1_output == 14

    num_attn_1_0_pattern = select(mlp_0_1_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_0_output, mlp_0_1_output):
        if mlp_0_0_output in {0, 1, 2, 3, 4, 6, 9, 11, 12, 15}:
            return mlp_0_1_output == 5
        elif mlp_0_0_output in {8, 5}:
            return mlp_0_1_output == 2
        elif mlp_0_0_output in {7}:
            return mlp_0_1_output == 4
        elif mlp_0_0_output in {10}:
            return mlp_0_1_output == 7
        elif mlp_0_0_output in {13}:
            return mlp_0_1_output == 13
        elif mlp_0_0_output in {14}:
            return mlp_0_1_output == 1

    num_attn_1_1_pattern = select(mlp_0_1_outputs, mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_0_output, token):
        if attn_0_0_output in {
            "2",
            "11",
            "1",
            "6",
            "4",
            "</s>",
            "8",
            "9",
            "3",
            "5",
            "0",
            "10",
            "7",
            "12",
            "<s>",
        }:
            return token == "11"

    num_attn_1_2_pattern = select(tokens, attn_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_mlp_0_1_output, k_mlp_0_1_output):
        if q_mlp_0_1_output in {0, 1, 2, 3, 4, 10, 11, 12, 13}:
            return k_mlp_0_1_output == 7
        elif q_mlp_0_1_output in {5}:
            return k_mlp_0_1_output == 10
        elif q_mlp_0_1_output in {6, 7, 8, 9, 15}:
            return k_mlp_0_1_output == 5
        elif q_mlp_0_1_output in {14}:
            return k_mlp_0_1_output == 3

    num_attn_1_3_pattern = select(mlp_0_1_outputs, mlp_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_3_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_0_output, position):
        key = (attn_1_0_output, position)
        if key in {
            ("0", 10),
            ("0", 12),
            ("0", 13),
            ("1", 10),
            ("1", 12),
            ("1", 13),
            ("10", 10),
            ("11", 7),
            ("11", 8),
            ("11", 9),
            ("11", 10),
            ("11", 12),
            ("11", 13),
            ("12", 10),
            ("2", 2),
            ("2", 3),
            ("2", 8),
            ("2", 9),
            ("3", 3),
            ("3", 10),
            ("3", 12),
            ("3", 13),
            ("4", 3),
            ("4", 7),
            ("4", 9),
            ("4", 10),
            ("5", 3),
            ("5", 9),
            ("6", 3),
            ("6", 7),
            ("6", 9),
            ("6", 10),
            ("8", 3),
            ("8", 4),
            ("8", 5),
            ("8", 6),
            ("8", 7),
            ("8", 8),
            ("8", 9),
            ("8", 10),
            ("8", 12),
            ("8", 13),
            ("8", 15),
            ("9", 3),
            ("9", 4),
            ("9", 6),
            ("9", 7),
            ("9", 9),
            ("9", 10),
            ("9", 12),
            ("9", 13),
            ("9", 15),
            ("</s>", 3),
            ("</s>", 9),
            ("</s>", 10),
            ("<s>", 10),
            ("<s>", 12),
            ("<s>", 13),
        }:
            return 14
        elif key in {
            ("0", 0),
            ("1", 0),
            ("10", 0),
            ("10", 12),
            ("10", 13),
            ("11", 0),
            ("12", 0),
            ("12", 6),
            ("12", 12),
            ("12", 13),
            ("2", 0),
            ("2", 6),
            ("2", 10),
            ("2", 12),
            ("2", 13),
            ("2", 14),
            ("3", 0),
            ("4", 0),
            ("4", 12),
            ("4", 13),
            ("5", 0),
            ("5", 7),
            ("5", 8),
            ("5", 10),
            ("5", 12),
            ("5", 13),
            ("6", 0),
            ("6", 12),
            ("6", 13),
            ("7", 0),
            ("7", 10),
            ("7", 12),
            ("7", 13),
            ("8", 0),
            ("8", 14),
            ("9", 0),
            ("9", 14),
            ("</s>", 0),
            ("</s>", 12),
            ("</s>", 13),
            ("<s>", 0),
        }:
            return 3
        elif key in {
            ("0", 2),
            ("0", 3),
            ("1", 2),
            ("1", 8),
            ("10", 2),
            ("10", 3),
            ("10", 4),
            ("10", 7),
            ("10", 8),
            ("11", 2),
            ("12", 2),
            ("12", 7),
            ("12", 8),
            ("3", 2),
            ("3", 8),
            ("4", 2),
            ("4", 8),
            ("5", 2),
            ("6", 2),
            ("6", 8),
            ("7", 2),
            ("7", 7),
            ("7", 8),
            ("8", 2),
            ("9", 2),
            ("9", 5),
            ("9", 8),
            ("</s>", 2),
            ("</s>", 8),
            ("<s>", 2),
            ("<s>", 8),
        }:
            return 9
        elif key in {
            ("0", 11),
            ("1", 14),
            ("10", 14),
            ("11", 14),
            ("12", 5),
            ("12", 14),
            ("3", 14),
            ("4", 11),
            ("4", 14),
            ("5", 11),
            ("5", 14),
            ("6", 5),
            ("6", 14),
            ("7", 3),
            ("7", 4),
            ("7", 5),
            ("7", 6),
            ("7", 9),
            ("7", 11),
            ("7", 14),
            ("7", 15),
            ("</s>", 14),
            ("<s>", 14),
        }:
            return 8
        elif key in {
            ("1", 11),
            ("10", 11),
            ("11", 11),
            ("12", 11),
            ("2", 1),
            ("2", 11),
            ("3", 11),
            ("6", 11),
            ("8", 11),
            ("9", 11),
            ("</s>", 11),
            ("<s>", 11),
        }:
            return 10
        elif key in {
            ("3", 1),
            ("4", 1),
            ("5", 1),
            ("6", 1),
            ("7", 1),
            ("8", 1),
            ("9", 1),
            ("</s>", 1),
            ("<s>", 1),
        }:
            return 11
        elif key in {("0", 1), ("10", 1), ("11", 1), ("12", 1)}:
            return 0
        elif key in {("1", 3), ("10", 6)}:
            return 1
        elif key in {("1", 1)}:
            return 2
        return 4

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_0_outputs, positions)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position, attn_1_3_output):
        key = (position, attn_1_3_output)
        if key in {
            (3, "0"),
            (3, "1"),
            (3, "10"),
            (3, "11"),
            (3, "12"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "5"),
            (3, "6"),
            (3, "7"),
            (3, "8"),
            (3, "9"),
            (3, "</s>"),
            (3, "<s>"),
            (4, "0"),
            (4, "1"),
            (4, "10"),
            (4, "11"),
            (4, "12"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "6"),
            (4, "7"),
            (4, "8"),
            (4, "9"),
            (4, "</s>"),
            (4, "<s>"),
            (5, "0"),
            (5, "1"),
            (5, "10"),
            (5, "11"),
            (5, "12"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "5"),
            (5, "6"),
            (5, "7"),
            (5, "8"),
            (5, "9"),
            (5, "</s>"),
            (5, "<s>"),
            (6, "10"),
        }:
            return 5
        elif key in {
            (6, "0"),
            (6, "1"),
            (6, "11"),
            (6, "12"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "5"),
            (6, "6"),
            (6, "7"),
            (6, "8"),
            (6, "</s>"),
            (6, "<s>"),
            (7, "0"),
            (7, "1"),
            (7, "11"),
            (7, "12"),
            (7, "2"),
            (7, "3"),
            (7, "4"),
            (7, "5"),
            (7, "6"),
            (7, "7"),
            (7, "8"),
            (7, "</s>"),
            (7, "<s>"),
            (8, "0"),
            (8, "1"),
            (8, "11"),
            (8, "12"),
            (8, "2"),
            (8, "3"),
            (8, "4"),
            (8, "5"),
            (8, "6"),
            (8, "7"),
            (8, "8"),
            (8, "</s>"),
            (8, "<s>"),
            (14, "1"),
            (14, "12"),
            (14, "2"),
            (14, "3"),
            (14, "8"),
        }:
            return 7
        elif key in {
            (1, "1"),
            (1, "10"),
            (2, "0"),
            (2, "1"),
            (2, "10"),
            (2, "11"),
            (2, "12"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "5"),
            (2, "6"),
            (2, "7"),
            (2, "8"),
            (2, "9"),
            (2, "</s>"),
            (2, "<s>"),
        }:
            return 3
        elif key in {
            (6, "9"),
            (7, "10"),
            (7, "9"),
            (8, "10"),
            (8, "9"),
            (10, "10"),
            (10, "2"),
            (10, "8"),
            (11, "10"),
            (12, "10"),
            (12, "2"),
            (12, "8"),
            (14, "10"),
            (15, "10"),
        }:
            return 0
        elif key in {
            (1, "0"),
            (1, "12"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "5"),
            (1, "6"),
            (1, "7"),
            (1, "8"),
            (1, "9"),
            (1, "</s>"),
            (1, "<s>"),
        }:
            return 11
        elif key in {(1, "11")}:
            return 8
        return 4

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(positions, attn_1_3_outputs)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {0}:
            return 13
        return 3

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_3_output, num_attn_1_1_output):
        key = (num_attn_1_3_output, num_attn_1_1_output)
        if key in {
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
            (4, 2),
            (5, 0),
            (5, 1),
            (5, 2),
            (6, 0),
            (6, 1),
            (6, 2),
            (7, 0),
            (7, 1),
            (7, 2),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (25, 0),
            (25, 1),
            (25, 2),
            (26, 1),
            (26, 2),
            (27, 2),
        }:
            return 14
        elif key in {
            (26, 0),
            (27, 0),
            (27, 1),
            (28, 0),
            (28, 1),
            (28, 2),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
        }:
            return 2
        elif key in {(0, 0)}:
            return 12
        return 3

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"0", "7", "4"}:
            return k_token == "1"
        elif q_token in {"1", "3", "5", "10", "2"}:
            return k_token == "0"
        elif q_token in {"11"}:
            return k_token == "10"
        elif q_token in {"8", "12"}:
            return k_token == "11"
        elif q_token in {"6"}:
            return k_token == "3"
        elif q_token in {"9"}:
            return k_token == "7"
        elif q_token in {"</s>"}:
            return k_token == "9"
        elif q_token in {"<s>"}:
            return k_token == "<pad>"

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(mlp_1_1_output, token):
        if mlp_1_1_output in {0, 4, 7, 8, 10, 12}:
            return token == "<s>"
        elif mlp_1_1_output in {1}:
            return token == "10"
        elif mlp_1_1_output in {2, 13}:
            return token == "11"
        elif mlp_1_1_output in {11, 3, 5}:
            return token == "</s>"
        elif mlp_1_1_output in {6}:
            return token == "3"
        elif mlp_1_1_output in {9}:
            return token == "4"
        elif mlp_1_1_output in {14}:
            return token == "9"
        elif mlp_1_1_output in {15}:
            return token == "2"

    attn_2_1_pattern = select_closest(tokens, mlp_1_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"0", "2"}:
            return k_token == "4"
        elif q_token in {"1"}:
            return k_token == "11"
        elif q_token in {"10"}:
            return k_token == "8"
        elif q_token in {"11"}:
            return k_token == "3"
        elif q_token in {"12"}:
            return k_token == "5"
        elif q_token in {"3"}:
            return k_token == "6"
        elif q_token in {"6", "4"}:
            return k_token == "12"
        elif q_token in {"5"}:
            return k_token == "1"
        elif q_token in {"7", "<s>"}:
            return k_token == "2"
        elif q_token in {"8", "9"}:
            return k_token == "0"
        elif q_token in {"</s>"}:
            return k_token == "<s>"

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 3}:
            return token == "<pad>"
        elif mlp_0_0_output in {1, 4}:
            return token == "<s>"
        elif mlp_0_0_output in {2, 12}:
            return token == "10"
        elif mlp_0_0_output in {5}:
            return token == "9"
        elif mlp_0_0_output in {10, 11, 6, 14}:
            return token == "</s>"
        elif mlp_0_0_output in {7}:
            return token == "3"
        elif mlp_0_0_output in {8}:
            return token == "2"
        elif mlp_0_0_output in {9}:
            return token == "4"
        elif mlp_0_0_output in {13}:
            return token == "0"
        elif mlp_0_0_output in {15}:
            return token == "12"

    attn_2_3_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 1, 6, 7, 15}:
            return token == "0"
        elif mlp_0_0_output in {2, 3, 4, 8, 11}:
            return token == "1"
        elif mlp_0_0_output in {5, 14}:
            return token == "5"
        elif mlp_0_0_output in {9, 10, 12, 13}:
            return token == "10"

    num_attn_2_0_pattern = select(tokens, mlp_0_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, ones)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, num_mlp_0_0_output):
        if position in {0, 10, 5, 7}:
            return num_mlp_0_0_output == 1
        elif position in {1}:
            return num_mlp_0_0_output == 14
        elif position in {2, 3, 4}:
            return num_mlp_0_0_output == 3
        elif position in {9, 6, 15}:
            return num_mlp_0_0_output == 4
        elif position in {8, 11}:
            return num_mlp_0_0_output == 5
        elif position in {12, 13}:
            return num_mlp_0_0_output == 6
        elif position in {14}:
            return num_mlp_0_0_output == 11

    num_attn_2_1_pattern = select(num_mlp_0_0_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, num_mlp_0_0_output):
        if position in {0, 3, 4, 5, 15}:
            return num_mlp_0_0_output == 3
        elif position in {1, 2}:
            return num_mlp_0_0_output == 14
        elif position in {8, 9, 6}:
            return num_mlp_0_0_output == 1
        elif position in {7}:
            return num_mlp_0_0_output == 4
        elif position in {10, 11}:
            return num_mlp_0_0_output == 5
        elif position in {12}:
            return num_mlp_0_0_output == 6
        elif position in {13, 14}:
            return num_mlp_0_0_output == 0

    num_attn_2_2_pattern = select(num_mlp_0_0_outputs, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_0_0_output, attn_0_3_output):
        if mlp_0_0_output in {0, 15, 14, 7}:
            return attn_0_3_output == "4"
        elif mlp_0_0_output in {8, 1}:
            return attn_0_3_output == "1"
        elif mlp_0_0_output in {2, 3, 4, 9, 10, 11, 12, 13}:
            return attn_0_3_output == "0"
        elif mlp_0_0_output in {5}:
            return attn_0_3_output == "3"
        elif mlp_0_0_output in {6}:
            return attn_0_3_output == "10"

    num_attn_2_3_pattern = select(attn_0_3_outputs, mlp_0_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_0_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_2_output, position):
        key = (attn_2_2_output, position)
        if key in {
            ("1", 8),
            ("10", 2),
            ("10", 3),
            ("10", 4),
            ("10", 12),
            ("11", 3),
            ("11", 4),
            ("11", 9),
            ("11", 10),
            ("11", 12),
            ("12", 3),
            ("12", 4),
            ("12", 13),
            ("2", 3),
            ("2", 4),
            ("2", 6),
            ("2", 9),
            ("2", 10),
            ("4", 8),
            ("4", 9),
            ("4", 10),
            ("4", 11),
            ("4", 12),
            ("4", 13),
            ("5", 3),
            ("5", 8),
            ("5", 9),
            ("5", 10),
            ("5", 11),
            ("5", 12),
            ("5", 13),
            ("6", 3),
            ("6", 4),
            ("6", 5),
            ("6", 9),
            ("6", 10),
            ("6", 11),
            ("6", 12),
            ("6", 13),
            ("6", 14),
            ("7", 9),
            ("7", 11),
            ("7", 12),
            ("7", 13),
            ("8", 5),
            ("8", 9),
            ("8", 13),
            ("8", 14),
            ("9", 5),
            ("9", 8),
            ("9", 9),
            ("9", 11),
            ("9", 13),
            ("9", 14),
            ("</s>", 1),
            ("</s>", 9),
            ("</s>", 11),
        }:
            return 4
        elif key in {
            ("0", 6),
            ("0", 7),
            ("0", 8),
            ("0", 13),
            ("0", 14),
            ("0", 15),
            ("1", 4),
            ("1", 9),
            ("1", 12),
            ("1", 14),
            ("10", 5),
            ("10", 6),
            ("10", 9),
            ("10", 10),
            ("10", 11),
            ("10", 14),
            ("10", 15),
            ("11", 2),
            ("11", 15),
            ("12", 2),
            ("12", 15),
            ("2", 2),
            ("2", 7),
            ("2", 15),
            ("3", 7),
            ("3", 9),
            ("3", 15),
            ("4", 2),
            ("4", 15),
            ("5", 2),
            ("5", 15),
            ("6", 2),
            ("7", 15),
            ("</s>", 15),
        }:
            return 14
        elif key in {
            ("0", 2),
            ("0", 3),
            ("1", 2),
            ("1", 3),
            ("12", 1),
            ("2", 1),
            ("3", 1),
            ("4", 1),
            ("5", 1),
            ("6", 1),
            ("7", 1),
            ("7", 2),
            ("7", 3),
            ("8", 1),
            ("8", 2),
            ("8", 3),
            ("9", 1),
            ("9", 2),
            ("9", 3),
            ("</s>", 2),
            ("</s>", 3),
            ("<s>", 1),
        }:
            return 0
        elif key in {
            ("0", 5),
            ("1", 5),
            ("1", 6),
            ("1", 7),
            ("1", 10),
            ("1", 11),
            ("1", 13),
            ("1", 15),
            ("4", 5),
            ("5", 5),
            ("7", 5),
            ("</s>", 5),
            ("</s>", 6),
            ("</s>", 13),
            ("</s>", 14),
        }:
            return 11
        elif key in {("0", 1), ("1", 1)}:
            return 8
        elif key in {("11", 1)}:
            return 3
        elif key in {("10", 1)}:
            return 15
        return 9

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_2_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(position, attn_2_1_output):
        key = (position, attn_2_1_output)
        if key in {
            (0, "11"),
            (0, "12"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (0, "6"),
            (0, "7"),
            (0, "<s>"),
            (6, "11"),
            (6, "12"),
            (6, "2"),
            (6, "6"),
            (6, "7"),
            (8, "1"),
            (8, "12"),
            (8, "3"),
            (8, "4"),
            (8, "5"),
            (8, "6"),
            (8, "7"),
            (8, "<s>"),
            (9, "11"),
            (9, "2"),
            (9, "4"),
            (9, "5"),
            (9, "6"),
            (9, "7"),
            (10, "2"),
            (10, "4"),
            (10, "6"),
            (11, "0"),
            (11, "4"),
            (11, "5"),
            (11, "6"),
            (12, "11"),
            (12, "4"),
            (12, "5"),
            (12, "6"),
            (12, "7"),
            (13, "3"),
            (13, "4"),
            (13, "5"),
            (13, "6"),
            (13, "7"),
            (13, "8"),
            (13, "9"),
            (14, "11"),
            (14, "12"),
            (14, "2"),
            (14, "3"),
            (14, "4"),
            (14, "5"),
            (14, "6"),
            (14, "7"),
            (14, "8"),
            (14, "9"),
            (14, "</s>"),
            (14, "<s>"),
            (15, "11"),
            (15, "12"),
        }:
            return 7
        elif key in {
            (0, "1"),
            (0, "10"),
            (0, "8"),
            (2, "1"),
            (2, "10"),
            (2, "11"),
            (2, "3"),
            (2, "4"),
            (2, "5"),
            (2, "6"),
            (2, "7"),
            (2, "8"),
            (2, "9"),
            (2, "</s>"),
            (2, "<s>"),
            (3, "0"),
            (3, "10"),
            (3, "11"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "5"),
            (3, "6"),
            (3, "7"),
            (3, "8"),
            (3, "9"),
            (3, "</s>"),
            (3, "<s>"),
            (4, "10"),
            (4, "11"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "6"),
            (4, "7"),
            (4, "8"),
            (4, "9"),
            (4, "</s>"),
            (4, "<s>"),
            (6, "1"),
            (6, "3"),
            (6, "4"),
            (6, "5"),
            (6, "8"),
            (6, "9"),
            (6, "</s>"),
            (6, "<s>"),
            (7, "1"),
            (14, "10"),
            (15, "0"),
            (15, "1"),
            (15, "10"),
            (15, "2"),
            (15, "3"),
            (15, "4"),
            (15, "5"),
            (15, "6"),
            (15, "7"),
            (15, "8"),
            (15, "9"),
            (15, "</s>"),
            (15, "<s>"),
        }:
            return 13
        elif key in {
            (2, "0"),
            (2, "12"),
            (2, "2"),
            (3, "1"),
            (3, "12"),
            (4, "0"),
            (4, "1"),
            (5, "0"),
            (5, "</s>"),
            (7, "0"),
            (7, "2"),
            (7, "3"),
            (7, "7"),
            (7, "8"),
            (7, "9"),
            (7, "</s>"),
            (8, "0"),
            (8, "</s>"),
            (12, "0"),
        }:
            return 14
        elif key in {
            (1, "0"),
            (1, "5"),
            (1, "6"),
            (1, "7"),
            (1, "8"),
            (1, "9"),
            (1, "</s>"),
            (1, "<s>"),
            (12, "1"),
            (13, "0"),
            (13, "1"),
            (13, "10"),
            (13, "12"),
            (13, "2"),
            (13, "</s>"),
            (13, "<s>"),
            (14, "0"),
            (14, "1"),
        }:
            return 15
        elif key in {(1, "11"), (1, "12"), (1, "2"), (1, "3"), (1, "4")}:
            return 5
        elif key in {(4, "12"), (6, "10"), (11, "11")}:
            return 12
        elif key in {(1, "1"), (1, "10")}:
            return 9
        return 0

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(positions, attn_2_1_outputs)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {0}:
            return 4
        return 10

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_1_output, num_attn_1_3_output):
        key = (num_attn_1_1_output, num_attn_1_3_output)
        if key in {(1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4)}:
            return 8
        elif key in {(0, 1), (0, 2)}:
            return 15
        elif key in {(1, 0), (1, 1)}:
            return 12
        elif key in {(2, 0), (3, 0)}:
            return 7
        elif key in {(0, 0)}:
            return 0
        elif key in {(0, 3)}:
            return 9
        elif key in {(0, 4)}:
            return 10
        return 3

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
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


print(run(["<s>", "3", "3", "2", "4", "</s>"]))
