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
        "output-base-1616/rasp/double_hist/k16_len16_L3_H4_M2/s0/double_hist_weights.csv",
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
            return k_position == 5
        elif q_position in {1, 4, 5, 6}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {8, 3}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 11
        elif q_position in {9, 10}:
            return k_position == 10
        elif q_position in {11, 13, 14, 15}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 12

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2, 3, 4, 6}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 7
        elif q_position in {8}:
            return k_position == 8
        elif q_position in {9}:
            return k_position == 9
        elif q_position in {10, 11}:
            return k_position == 11
        elif q_position in {12, 13, 14, 15}:
            return k_position == 13

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"3", "1"}:
            return k_token == "12"
        elif q_token in {"10"}:
            return k_token == "10"
        elif q_token in {"11"}:
            return k_token == "9"
        elif q_token in {"12"}:
            return k_token == "8"
        elif q_token in {"13"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "6"
        elif q_token in {"6", "9"}:
            return k_token == "13"
        elif q_token in {"7"}:
            return k_token == "11"
        elif q_token in {"8"}:
            return k_token == "7"
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, num_var0_embeddings)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
        if q_token in {"2", "4", "0"}:
            return k_token == "<s>"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"<s>", "10"}:
            return k_token == "<pad>"
        elif q_token in {"11"}:
            return k_token == "11"
        elif q_token in {"12"}:
            return k_token == "12"
        elif q_token in {"13"}:
            return k_token == "13"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"6"}:
            return k_token == "6"
        elif q_token in {"7"}:
            return k_token == "7"
        elif q_token in {"8"}:
            return k_token == "8"
        elif q_token in {"9"}:
            return k_token == "9"

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, num_var0_embeddings)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_1_output):
        key = (attn_0_0_output, attn_0_1_output)
        if key in {
            (0, 9),
            (0, 10),
            (0, 11),
            (1, 9),
            (1, 11),
            (2, 11),
            (3, 11),
            (4, 9),
            (4, 10),
            (4, 11),
            (5, 9),
            (5, 10),
            (5, 11),
            (6, 9),
            (6, 10),
            (6, 11),
            (7, 11),
            (8, 11),
            (9, 0),
            (9, 1),
            (9, 5),
            (9, 11),
            (10, 0),
            (10, 1),
            (10, 5),
            (10, 11),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (11, 7),
            (11, 8),
            (11, 9),
            (11, 10),
            (11, 11),
        }:
            return 9
        elif key in {
            (0, 13),
            (0, 14),
            (1, 10),
            (1, 13),
            (1, 14),
            (2, 10),
            (2, 13),
            (3, 10),
            (3, 13),
            (7, 9),
            (7, 10),
            (8, 10),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 6),
            (9, 7),
            (9, 8),
            (9, 9),
            (9, 10),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 6),
            (10, 7),
            (10, 8),
            (10, 9),
            (10, 10),
            (13, 0),
            (13, 3),
            (13, 13),
            (13, 14),
            (14, 3),
            (14, 13),
            (14, 14),
        }:
            return 8
        elif key in {
            (0, 15),
            (1, 15),
            (2, 12),
            (2, 15),
            (3, 15),
            (4, 15),
            (5, 15),
            (6, 15),
            (7, 15),
            (8, 15),
            (9, 15),
            (10, 15),
            (11, 15),
            (12, 2),
            (12, 15),
            (13, 15),
            (14, 15),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (15, 7),
            (15, 8),
            (15, 9),
            (15, 10),
            (15, 11),
            (15, 12),
            (15, 13),
            (15, 14),
            (15, 15),
        }:
            return 13
        elif key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
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
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
        }:
            return 4
        elif key in {
            (0, 6),
            (0, 7),
            (1, 6),
            (1, 7),
            (3, 6),
            (3, 7),
            (4, 6),
            (4, 7),
            (5, 6),
            (5, 7),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 6),
            (6, 7),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (7, 7),
        }:
            return 7
        elif key in {
            (0, 8),
            (1, 8),
            (3, 8),
            (4, 8),
            (5, 8),
            (6, 4),
            (6, 5),
            (6, 8),
            (7, 8),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
        }:
            return 6
        elif key in {
            (0, 5),
            (1, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
        }:
            return 12
        elif key in {(2, 6), (2, 7), (2, 8), (2, 9), (3, 9), (8, 9)}:
            return 5
        return 10

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        if key in {
            (0, 0),
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
        }:
            return 14
        return 9

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 2}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 1
        elif q_position in {5, 7}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 8
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 10
        elif q_position in {15}:
            return k_position == 0

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, num_mlp_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 5, 6, 7, 9, 10, 11}:
            return k_position == 8
        elif q_position in {1, 2, 3, 4}:
            return k_position == 7
        elif q_position in {8, 12, 13, 14, 15}:
            return k_position == 9

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(q_attn_0_0_output, k_attn_0_0_output):
        if q_attn_0_0_output in {0, 1, 2, 3, 4}:
            return k_attn_0_0_output == 5
        elif q_attn_0_0_output in {5, 6, 7}:
            return k_attn_0_0_output == 6
        elif q_attn_0_0_output in {8, 14}:
            return k_attn_0_0_output == 7
        elif q_attn_0_0_output in {9}:
            return k_attn_0_0_output == 9
        elif q_attn_0_0_output in {10}:
            return k_attn_0_0_output == 10
        elif q_attn_0_0_output in {11, 12, 13}:
            return k_attn_0_0_output == 11
        elif q_attn_0_0_output in {15}:
            return k_attn_0_0_output == 8

    num_attn_1_0_pattern = select(attn_0_0_outputs, attn_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"<s>", "1", "12", "11", "9", "7", "8", "3"}:
            return k_token == "<s>"
        elif q_token in {"10"}:
            return k_token == "10"
        elif q_token in {"13", "5", "6"}:
            return k_token == "<pad>"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"4"}:
            return k_token == "4"

    num_attn_1_1_pattern = select(tokens, tokens, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_1_output, attn_0_1_output):
        key = (attn_1_1_output, attn_0_1_output)
        if key in {
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (2, 7),
            (2, 8),
            (2, 9),
            (3, 7),
            (3, 9),
            (4, 7),
            (4, 9),
            (5, 7),
            (5, 9),
            (6, 7),
            (7, 0),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (7, 7),
            (7, 9),
            (8, 9),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 5),
            (9, 6),
            (9, 7),
            (9, 8),
            (9, 9),
            (14, 9),
        }:
            return 8
        elif key in {
            (0, 4),
            (1, 4),
            (2, 4),
            (2, 6),
            (3, 4),
            (3, 6),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 6),
            (5, 6),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6),
            (14, 4),
            (14, 6),
        }:
            return 0
        elif key in {
            (0, 10),
            (3, 10),
            (4, 10),
            (6, 9),
            (6, 10),
            (7, 10),
            (8, 10),
            (9, 10),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (10, 9),
            (10, 10),
        }:
            return 7
        elif key in {
            (3, 8),
            (4, 8),
            (5, 8),
            (6, 8),
            (7, 1),
            (7, 8),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
            (14, 7),
        }:
            return 10
        elif key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 14),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 14),
        }:
            return 3
        elif key in {
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (14, 5),
        }:
            return 11
        elif key in {(0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3), (14, 3)}:
            return 6
        return 9

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_0_1_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_1_output, num_attn_1_1_output):
        key = (num_attn_0_1_output, num_attn_1_1_output)
        if key in {
            (0, 0),
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
        }:
            return 14
        return 1

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 9
        elif q_position in {1, 12, 4, 5}:
            return k_position == 3
        elif q_position in {2, 3, 6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8, 9, 15}:
            return k_position == 1
        elif q_position in {10}:
            return k_position == 7
        elif q_position in {11}:
            return k_position == 2
        elif q_position in {13}:
            return k_position == 11
        elif q_position in {14}:
            return k_position == 8

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 10
        elif q_position in {1, 2, 3, 4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {9, 6}:
            return k_position == 3
        elif q_position in {7}:
            return k_position == 1
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {10, 11, 12, 13}:
            return k_position == 9
        elif q_position in {14, 15}:
            return k_position == 2

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, num_mlp_1_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_1_output, attn_1_0_output):
        if attn_1_1_output in {0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14}:
            return attn_1_0_output == 9
        elif attn_1_1_output in {2, 4, 12}:
            return attn_1_0_output == 5
        elif attn_1_1_output in {15}:
            return attn_1_0_output == 0

    num_attn_2_0_pattern = select(attn_1_0_outputs, attn_1_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(q_token, k_token):
        if q_token in {"2", "4", "0", "10"}:
            return k_token == "<pad>"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"11"}:
            return k_token == "11"
        elif q_token in {"12"}:
            return k_token == "12"
        elif q_token in {"13"}:
            return k_token == "13"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"6"}:
            return k_token == "6"
        elif q_token in {"7"}:
            return k_token == "7"
        elif q_token in {"8"}:
            return k_token == "8"
        elif q_token in {"9"}:
            return k_token == "9"
        elif q_token in {"<s>"}:
            return k_token == "10"

    num_attn_2_1_pattern = select(tokens, tokens, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_var0_embeddings)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_0_output, attn_1_1_output):
        key = (attn_2_0_output, attn_1_1_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 14),
            (2, 0),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 7),
            (2, 14),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 12),
            (3, 14),
            (3, 15),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 14),
            (6, 2),
            (7, 2),
            (7, 4),
            (7, 5),
            (11, 0),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 14),
            (12, 0),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 14),
            (14, 0),
            (14, 2),
            (14, 3),
            (14, 4),
            (14, 5),
        }:
            return 3
        elif key in {
            (1, 12),
            (1, 15),
            (5, 1),
            (5, 8),
            (5, 9),
            (5, 10),
            (5, 11),
            (5, 13),
            (5, 14),
            (5, 15),
            (8, 15),
            (9, 12),
            (9, 15),
            (10, 15),
            (11, 1),
            (11, 15),
            (13, 15),
            (14, 15),
            (15, 1),
            (15, 8),
            (15, 9),
            (15, 10),
            (15, 11),
            (15, 13),
            (15, 14),
            (15, 15),
        }:
            return 11
        elif key in {
            (0, 15),
            (1, 3),
            (1, 4),
            (4, 15),
            (5, 3),
            (9, 2),
            (9, 3),
            (9, 4),
            (12, 1),
            (12, 15),
            (15, 3),
        }:
            return 15
        elif key in {
            (1, 5),
            (5, 5),
            (5, 6),
            (5, 7),
            (6, 15),
            (7, 15),
            (9, 5),
            (15, 5),
            (15, 6),
            (15, 7),
        }:
            return 4
        elif key in {
            (1, 2),
            (1, 6),
            (1, 7),
            (1, 14),
            (6, 1),
            (9, 6),
            (9, 7),
            (9, 14),
            (14, 7),
        }:
            return 10
        elif key in {(2, 15), (5, 0), (5, 2), (5, 4), (15, 0), (15, 2), (15, 4)}:
            return 1
        return 2

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_0_outputs, attn_1_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_0_output, num_attn_2_1_output):
        key = (num_attn_2_0_output, num_attn_2_1_output)
        if key in {
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (8, 1),
            (9, 1),
            (10, 1),
            (11, 1),
            (12, 1),
            (13, 2),
            (14, 2),
            (15, 2),
            (16, 2),
            (17, 2),
            (18, 2),
        }:
            return 4
        elif key in {(7, 0), (8, 0), (9, 0), (10, 0), (11, 0)}:
            return 11
        elif key in {(3, 0), (4, 0), (5, 0), (6, 0)}:
            return 13
        elif key in {(1, 0), (2, 0)}:
            return 3
        elif key in {(0, 0)}:
            return 9
        return 12

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                mlp_0_0_output_scores,
                num_mlp_0_0_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                mlp_1_0_output_scores,
                num_mlp_1_0_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                mlp_2_0_output_scores,
                num_mlp_2_0_output_scores,
                num_var0_embedding_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
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


print(run(["<s>", "1", "8", "1", "2", "9", "0", "5", "8", "2", "11", "10", "9"]))
