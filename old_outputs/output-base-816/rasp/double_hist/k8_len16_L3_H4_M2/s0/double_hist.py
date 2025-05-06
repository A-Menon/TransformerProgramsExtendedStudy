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
        "output-base-816/rasp/double_hist/k8_len16_L3_H4_M2/s0/double_hist_weights.csv",
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
    def predicate_0_0(q_token, k_token):
        if q_token in {"0", "3", "2", "5"}:
            return k_token == "5"
        elif q_token in {"4", "1"}:
            return k_token == "2"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"0", "3", "2", "5"}:
            return k_token == "2"
        elif q_token in {"4", "1"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
        if q_token in {"0", "3", "4"}:
            return k_token == ""
        elif q_token in {"1"}:
            return k_token == "<pad>"
        elif q_token in {"2", "<s>"}:
            return k_token == "2"
        elif q_token in {"5"}:
            return k_token == "5"

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, num_var0_embeddings)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2", "<s>"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, num_var0_embeddings)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, attn_0_0_output):
        key = (attn_0_1_output, attn_0_0_output)
        if key in {
            ("0", "0"),
            ("0", "5"),
            ("0", "<s>"),
            ("1", "4"),
            ("1", "5"),
            ("1", "<s>"),
            ("2", "<s>"),
            ("3", "0"),
            ("3", "5"),
            ("3", "<s>"),
            ("4", "0"),
            ("4", "4"),
            ("4", "5"),
            ("4", "<s>"),
            ("5", "0"),
            ("5", "3"),
            ("5", "4"),
            ("5", "5"),
            ("<s>", "5"),
        }:
            return 2
        elif key in {
            ("0", "1"),
            ("0", "3"),
            ("0", "4"),
            ("1", "1"),
            ("1", "3"),
            ("3", "1"),
            ("3", "3"),
            ("3", "4"),
            ("4", "1"),
            ("4", "3"),
            ("5", "1"),
            ("<s>", "2"),
        }:
            return 8
        elif key in {
            ("1", "2"),
            ("2", "1"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "5"),
            ("3", "2"),
            ("4", "2"),
            ("5", "2"),
            ("5", "<s>"),
        }:
            return 13
        elif key in {("0", "2"), ("1", "0"), ("2", "0")}:
            return 14
        return 12

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_0_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_1_output):
        key = (num_attn_0_0_output, num_attn_0_1_output)
        if key in {(0, 0), (0, 1), (0, 2)}:
            return 2
        elif key in {(0, 3), (1, 0), (1, 1)}:
            return 4
        return 8

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(num_mlp_0_0_output, token):
        if num_mlp_0_0_output in {0, 5}:
            return token == "4"
        elif num_mlp_0_0_output in {1, 2}:
            return token == "0"
        elif num_mlp_0_0_output in {3}:
            return token == "2"
        elif num_mlp_0_0_output in {4}:
            return token == "3"
        elif num_mlp_0_0_output in {8, 6, 15}:
            return token == "5"
        elif num_mlp_0_0_output in {7, 9, 10, 11, 12, 13, 14}:
            return token == ""

    attn_1_0_pattern = select_closest(tokens, num_mlp_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 1, 2, 3, 4, 5, 8, 13}:
            return k_position == 6
        elif q_position in {12, 6}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 5
        elif q_position in {9, 14}:
            return k_position == 3
        elif q_position in {10, 11}:
            return k_position == 2
        elif q_position in {15}:
            return k_position == 10

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2", "<s>"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"

    num_attn_1_0_pattern = select(tokens, tokens, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_var0_embeddings)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    num_attn_1_1_pattern = select(tokens, tokens, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_var0_embeddings)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(num_mlp_0_0_output, attn_1_1_output):
        key = (num_mlp_0_0_output, attn_1_1_output)
        if key in {
            (3, 6),
            (3, 8),
            (3, 13),
            (3, 14),
            (3, 15),
            (4, 0),
            (4, 1),
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
            (5, 13),
            (5, 15),
            (6, 13),
            (6, 15),
            (7, 11),
            (7, 14),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (11, 7),
            (11, 8),
            (11, 9),
            (11, 11),
            (11, 12),
            (11, 13),
            (11, 14),
            (11, 15),
            (14, 13),
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
            (6, 9),
            (7, 0),
            (7, 6),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 15),
        }:
            return 9
        elif key in {
            (1, 14),
            (3, 0),
            (5, 8),
            (5, 14),
            (6, 14),
            (7, 13),
            (10, 0),
            (10, 6),
            (10, 13),
            (10, 14),
            (12, 0),
            (12, 13),
            (12, 14),
            (14, 0),
            (14, 1),
            (14, 6),
            (14, 8),
            (14, 14),
        }:
            return 8
        elif key in {
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
            (2, 14),
            (2, 15),
            (7, 5),
        }:
            return 4
        elif key in {
            (1, 1),
            (1, 13),
            (3, 1),
            (3, 4),
            (3, 5),
            (3, 9),
            (9, 0),
            (9, 1),
            (9, 13),
        }:
            return 1
        elif key in {(5, 9)}:
            return 6
        return 10

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, attn_1_1_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_0_output, num_attn_1_1_output):
        key = (num_attn_0_0_output, num_attn_1_1_output)
        if key in {
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
            (1, 16),
            (1, 17),
            (2, 0),
            (2, 1),
        }:
            return 7
        elif key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
        }:
            return 4
        return 9

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_num_mlp_1_0_output, k_num_mlp_1_0_output):
        if q_num_mlp_1_0_output in {0, 9}:
            return k_num_mlp_1_0_output == 0
        elif q_num_mlp_1_0_output in {1, 2, 3, 5, 6, 8, 10, 14}:
            return k_num_mlp_1_0_output == 11
        elif q_num_mlp_1_0_output in {4, 12, 7}:
            return k_num_mlp_1_0_output == 9
        elif q_num_mlp_1_0_output in {11, 15}:
            return k_num_mlp_1_0_output == 15
        elif q_num_mlp_1_0_output in {13}:
            return k_num_mlp_1_0_output == 6

    attn_2_0_pattern = select_closest(
        num_mlp_1_0_outputs, num_mlp_1_0_outputs, predicate_2_0
    )
    attn_2_0_outputs = aggregate(attn_2_0_pattern, num_mlp_1_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0, 2, 3, 15}:
            return k_position == 11
        elif q_position in {1}:
            return k_position == 10
        elif q_position in {4, 5, 6}:
            return k_position == 9
        elif q_position in {7, 8, 10, 11, 12}:
            return k_position == 14
        elif q_position in {9, 13}:
            return k_position == 2
        elif q_position in {14}:
            return k_position == 5

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, num_mlp_1_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(num_mlp_1_0_output, mlp_1_0_output):
        if num_mlp_1_0_output in {0, 6}:
            return mlp_1_0_output == 2
        elif num_mlp_1_0_output in {1, 2, 13}:
            return mlp_1_0_output == 7
        elif num_mlp_1_0_output in {8, 3}:
            return mlp_1_0_output == 3
        elif num_mlp_1_0_output in {4}:
            return mlp_1_0_output == 4
        elif num_mlp_1_0_output in {9, 5}:
            return mlp_1_0_output == 5
        elif num_mlp_1_0_output in {7, 10, 11, 14, 15}:
            return mlp_1_0_output == 13
        elif num_mlp_1_0_output in {12}:
            return mlp_1_0_output == 12

    num_attn_2_0_pattern = select(
        mlp_1_0_outputs, num_mlp_1_0_outputs, num_predicate_2_0
    )
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(num_mlp_1_0_output, mlp_1_0_output):
        if num_mlp_1_0_output in {0, 9}:
            return mlp_1_0_output == 9
        elif num_mlp_1_0_output in {8, 1, 10, 13}:
            return mlp_1_0_output == 0
        elif num_mlp_1_0_output in {2}:
            return mlp_1_0_output == 5
        elif num_mlp_1_0_output in {3}:
            return mlp_1_0_output == 3
        elif num_mlp_1_0_output in {4}:
            return mlp_1_0_output == 13
        elif num_mlp_1_0_output in {5, 14}:
            return mlp_1_0_output == 8
        elif num_mlp_1_0_output in {12, 6}:
            return mlp_1_0_output == 6
        elif num_mlp_1_0_output in {15, 7}:
            return mlp_1_0_output == 4
        elif num_mlp_1_0_output in {11}:
            return mlp_1_0_output == 7

    num_attn_2_1_pattern = select(
        mlp_1_0_outputs, num_mlp_1_0_outputs, num_predicate_2_1
    )
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_0_output, num_mlp_1_0_output):
        key = (attn_2_0_output, num_mlp_1_0_output)
        if key in {
            (0, 1),
            (0, 4),
            (0, 5),
            (0, 6),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 8),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 14),
            (2, 1),
            (2, 5),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 8),
            (3, 10),
            (3, 11),
            (3, 12),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 8),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (5, 1),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 8),
            (5, 10),
            (5, 12),
            (5, 13),
            (5, 14),
            (6, 1),
            (6, 5),
            (6, 12),
            (8, 1),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 12),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (12, 8),
            (12, 12),
            (12, 14),
            (13, 1),
            (14, 1),
            (14, 12),
        }:
            return 0
        elif key in {
            (2, 2),
            (2, 3),
            (2, 6),
            (2, 8),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 14),
            (2, 15),
            (5, 11),
            (6, 8),
            (6, 10),
            (6, 11),
            (8, 6),
            (8, 8),
            (8, 10),
            (8, 11),
            (8, 13),
            (9, 6),
            (9, 10),
            (12, 10),
            (12, 11),
            (12, 13),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
            (13, 8),
            (13, 10),
            (13, 12),
            (13, 13),
            (13, 14),
            (14, 3),
            (14, 4),
            (14, 5),
            (14, 6),
            (14, 8),
            (14, 10),
            (14, 11),
            (14, 13),
            (14, 14),
            (15, 3),
            (15, 6),
        }:
            return 7
        elif key in {
            (1, 13),
            (2, 4),
            (2, 13),
            (3, 13),
            (3, 14),
            (3, 15),
            (5, 2),
            (5, 15),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 6),
            (6, 13),
            (6, 14),
            (6, 15),
            (15, 4),
        }:
            return 8
        elif key in {
            (3, 7),
            (5, 7),
            (6, 7),
            (7, 5),
            (7, 6),
            (7, 7),
            (7, 15),
            (10, 7),
            (11, 7),
            (13, 7),
            (14, 7),
            (15, 7),
        }:
            return 14
        elif key in {(1, 7), (1, 15), (2, 7), (4, 7), (8, 7), (8, 15), (12, 7)}:
            return 15
        return 9

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_0_outputs, num_mlp_1_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_1_output, num_attn_2_0_output):
        key = (num_attn_2_1_output, num_attn_2_0_output)
        if key in {
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
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
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
            (1, 32),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 36),
            (1, 37),
            (1, 38),
            (1, 39),
            (1, 40),
            (1, 41),
            (1, 42),
            (1, 43),
            (1, 44),
            (1, 45),
            (1, 46),
            (1, 47),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
            (2, 17),
            (2, 18),
            (2, 19),
            (2, 20),
            (2, 21),
            (2, 22),
            (2, 23),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (2, 30),
            (2, 31),
            (2, 32),
            (2, 33),
            (2, 34),
            (2, 35),
            (2, 36),
            (2, 37),
            (2, 38),
            (2, 39),
            (2, 40),
            (2, 41),
            (2, 42),
            (2, 43),
            (2, 44),
            (2, 45),
            (2, 46),
            (2, 47),
            (3, 14),
            (3, 15),
            (3, 16),
            (3, 17),
            (3, 18),
            (3, 19),
            (3, 20),
            (3, 21),
            (3, 22),
            (3, 23),
            (3, 24),
            (3, 25),
            (3, 26),
            (3, 27),
            (3, 28),
            (3, 29),
            (3, 30),
            (3, 31),
            (3, 32),
            (3, 33),
            (3, 34),
            (3, 35),
            (3, 36),
            (3, 37),
            (3, 38),
            (3, 39),
            (3, 40),
            (3, 41),
            (3, 42),
            (3, 43),
            (3, 44),
            (3, 45),
            (3, 46),
            (3, 47),
            (4, 18),
            (4, 19),
            (4, 20),
            (4, 21),
            (4, 22),
            (4, 23),
            (4, 24),
            (4, 25),
            (4, 26),
            (4, 27),
            (4, 28),
            (4, 29),
            (4, 30),
            (4, 31),
            (4, 32),
            (4, 33),
            (4, 34),
            (4, 35),
            (4, 36),
            (4, 37),
            (4, 38),
            (4, 39),
            (4, 40),
            (4, 41),
            (4, 42),
            (4, 43),
            (4, 44),
            (4, 45),
            (4, 46),
            (4, 47),
            (5, 22),
            (5, 23),
            (5, 24),
            (5, 25),
            (5, 26),
            (5, 27),
            (5, 28),
            (5, 29),
            (5, 30),
            (5, 31),
            (5, 32),
            (5, 33),
            (5, 34),
            (5, 35),
            (5, 36),
            (5, 37),
            (5, 38),
            (5, 39),
            (5, 40),
            (5, 41),
            (5, 42),
            (5, 43),
            (5, 44),
            (5, 45),
            (5, 46),
            (5, 47),
            (6, 25),
            (6, 26),
            (6, 27),
            (6, 28),
            (6, 29),
            (6, 30),
            (6, 31),
            (6, 32),
            (6, 33),
            (6, 34),
            (6, 35),
            (6, 36),
            (6, 37),
            (6, 38),
            (6, 39),
            (6, 40),
            (6, 41),
            (6, 42),
            (6, 43),
            (6, 44),
            (6, 45),
            (6, 46),
            (6, 47),
            (7, 29),
            (7, 30),
            (7, 31),
            (7, 32),
            (7, 33),
            (7, 34),
            (7, 35),
            (7, 36),
            (7, 37),
            (7, 38),
            (7, 39),
            (7, 40),
            (7, 41),
            (7, 42),
            (7, 43),
            (7, 44),
            (7, 45),
            (7, 46),
            (7, 47),
            (8, 33),
            (8, 34),
            (8, 35),
            (8, 36),
            (8, 37),
            (8, 38),
            (8, 39),
            (8, 40),
            (8, 41),
            (8, 42),
            (8, 43),
            (8, 44),
            (8, 45),
            (8, 46),
            (8, 47),
            (9, 37),
            (9, 38),
            (9, 39),
            (9, 40),
            (9, 41),
            (9, 42),
            (9, 43),
            (9, 44),
            (9, 45),
            (9, 46),
            (9, 47),
            (10, 41),
            (10, 42),
            (10, 43),
            (10, 44),
            (10, 45),
            (10, 46),
            (10, 47),
            (11, 45),
            (11, 46),
            (11, 47),
        }:
            return 5
        elif key in {
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (6, 1),
            (7, 0),
            (7, 1),
            (8, 0),
            (8, 1),
            (9, 0),
            (9, 1),
            (10, 0),
            (10, 1),
            (11, 0),
            (11, 1),
            (12, 0),
            (12, 1),
            (13, 0),
            (13, 1),
            (14, 0),
            (14, 1),
            (15, 0),
            (15, 1),
            (16, 0),
            (16, 1),
            (16, 2),
            (17, 0),
            (17, 1),
            (17, 2),
            (18, 0),
            (18, 1),
            (18, 2),
            (19, 0),
            (19, 1),
            (19, 2),
            (20, 0),
            (20, 1),
            (20, 2),
            (21, 0),
            (21, 1),
            (21, 2),
            (22, 0),
            (22, 1),
            (22, 2),
            (23, 0),
            (23, 1),
            (23, 2),
            (24, 0),
            (24, 1),
            (24, 2),
            (25, 0),
            (25, 1),
            (25, 2),
            (26, 0),
            (26, 1),
            (26, 2),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
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
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (43, 4),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (44, 4),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (45, 4),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (47, 0),
            (47, 1),
            (47, 2),
            (47, 3),
            (47, 4),
            (47, 5),
        }:
            return 0
        elif key in {(0, 0), (1, 0), (2, 0)}:
            return 2
        return 9

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_2_0_outputs)
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


print(run(["<s>", "0", "2", "5", "2", "4", "3", "5", "4", "5", "4", "5", "0", "5"]))
