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
        "output-base-816/rasp/reverse/k8_len16_L3_H8_M2/s0/reverse_weights.csv",
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
        if q_position in {0, 9, 3}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 13
        elif q_position in {2}:
            return k_position == 11
        elif q_position in {8, 4}:
            return k_position == 7
        elif q_position in {10, 5}:
            return k_position == 5
        elif q_position in {12, 6}:
            return k_position == 3
        elif q_position in {14, 7}:
            return k_position == 1
        elif q_position in {11}:
            return k_position == 4
        elif q_position in {13}:
            return k_position == 2
        elif q_position in {15}:
            return k_position == 9

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 10}:
            return k_position == 4
        elif q_position in {1, 5}:
            return k_position == 10
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3, 15}:
            return k_position == 9
        elif q_position in {13, 4, 12}:
            return k_position == 1
        elif q_position in {9, 6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 15
        elif q_position in {11}:
            return k_position == 3
        elif q_position in {14}:
            return k_position == 7

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 8, 9}:
            return k_position == 14
        elif q_position in {1, 15}:
            return k_position == 9
        elif q_position in {2, 7}:
            return k_position == 12
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {5, 6}:
            return k_position == 8
        elif q_position in {10, 11, 12, 13, 14}:
            return k_position == 15

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 8, 4}:
            return k_position == 6
        elif q_position in {1, 3}:
            return k_position == 8
        elif q_position in {2}:
            return k_position == 10
        elif q_position in {12, 5}:
            return k_position == 3
        elif q_position in {13, 6}:
            return k_position == 2
        elif q_position in {9, 10, 7}:
            return k_position == 5
        elif q_position in {11}:
            return k_position == 4
        elif q_position in {14}:
            return k_position == 1
        elif q_position in {15}:
            return k_position == 9

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"0", "3"}:
            return position == 9
        elif token in {"1", "4"}:
            return position == 6
        elif token in {"2"}:
            return position == 10
        elif token in {"</s>"}:
            return position == 11
        elif token in {"<s>"}:
            return position == 0

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"1", "0"}:
            return position == 0
        elif token in {"2", "3", "</s>", "4"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 7

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"0", "4", "2"}:
            return position == 6
        elif token in {"1"}:
            return position == 9
        elif token in {"3"}:
            return position == 7
        elif token in {"<s>", "</s>"}:
            return position == 15

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"1", "0"}:
            return position == 0
        elif token in {"2", "3"}:
            return position == 9
        elif token in {"4"}:
            return position == 8
        elif token in {"</s>"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 7

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, attn_0_3_output):
        key = (attn_0_2_output, attn_0_3_output)
        if key in {("0", "1"), ("1", "1"), ("2", "1"), ("3", "1"), ("4", "1")}:
            return 8
        elif key in {
            ("0", "0"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("1", "0"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("2", "0"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("3", "0"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("4", "0"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "<s>"),
        }:
            return 1
        elif key in {
            ("0", "<s>"),
            ("1", "<s>"),
            ("3", "<s>"),
            ("4", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
        }:
            return 13
        elif key in {("1", "</s>"), ("</s>", "1")}:
            return 5
        elif key in {("</s>", "</s>"), ("</s>", "<s>"), ("<s>", "</s>")}:
            return 9
        elif key in {("4", "</s>"), ("</s>", "4")}:
            return 14
        elif key in {("2", "</s>"), ("</s>", "2")}:
            return 0
        elif key in {("3", "</s>"), ("</s>", "3")}:
            return 7
        elif key in {("2", "<s>")}:
            return 11
        return 15

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_3_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(one):
        key = one
        return 0

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in ones]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 1, 2}:
            return k_position == 13
        elif q_position in {3, 15}:
            return k_position == 11
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {5}:
            return k_position == 9
        elif q_position in {6, 8, 9, 13, 14}:
            return k_position == 1
        elif q_position in {10, 12, 7}:
            return k_position == 2
        elif q_position in {11}:
            return k_position == 3

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 8, 7}:
            return k_position == 14
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {9, 5}:
            return k_position == 0
        elif q_position in {13, 6}:
            return k_position == 15
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11, 14}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 1

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 10}:
            return k_position == 4
        elif q_position in {1, 3}:
            return k_position == 12
        elif q_position in {2}:
            return k_position == 8
        elif q_position in {4, 15}:
            return k_position == 11
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 15
        elif q_position in {9, 7}:
            return k_position == 3
        elif q_position in {8, 11}:
            return k_position == 2
        elif q_position in {12, 13, 14}:
            return k_position == 1

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 14
        elif q_position in {2}:
            return k_position == 12
        elif q_position in {3, 5}:
            return k_position == 7
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 3
        elif q_position in {9, 11, 13}:
            return k_position == 2
        elif q_position in {10, 12, 14}:
            return k_position == 1
        elif q_position in {15}:
            return k_position == 11

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_0_output, position):
        if attn_0_0_output in {"0"}:
            return position == 15
        elif attn_0_0_output in {"1", "4"}:
            return position == 11
        elif attn_0_0_output in {"2"}:
            return position == 10
        elif attn_0_0_output in {"3"}:
            return position == 7
        elif attn_0_0_output in {"<s>", "</s>"}:
            return position == 2

    num_attn_1_0_pattern = select(positions, attn_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_1_output, mlp_0_0_output):
        if attn_0_1_output in {"2", "</s>", "0", "3", "4"}:
            return mlp_0_0_output == 5
        elif attn_0_1_output in {"1"}:
            return mlp_0_0_output == 11
        elif attn_0_1_output in {"<s>"}:
            return mlp_0_0_output == 15

    num_attn_1_1_pattern = select(mlp_0_0_outputs, attn_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_0_output, position):
        if attn_0_0_output in {"2", "1", "0", "3", "4"}:
            return position == 1
        elif attn_0_0_output in {"<s>", "</s>"}:
            return position == 3

    num_attn_1_2_pattern = select(positions, attn_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_attn_0_0_output, k_attn_0_0_output):
        if q_attn_0_0_output in {"0", "3", "2"}:
            return k_attn_0_0_output == "<s>"
        elif q_attn_0_0_output in {"1", "<s>", "4", "</s>"}:
            return k_attn_0_0_output == "1"

    num_attn_1_3_pattern = select(attn_0_0_outputs, attn_0_0_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_1_output, attn_0_0_output):
        key = (attn_0_1_output, attn_0_0_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("2", "0"),
            ("2", "1"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
        }:
            return 9
        elif key in {("1", "<s>")}:
            return 15
        elif key in {("3", "<s>"), ("</s>", "3"), ("</s>", "<s>"), ("<s>", "<s>")}:
            return 6
        elif key in {("0", "<s>")}:
            return 0
        elif key in {("</s>", "4")}:
            return 4
        elif key in {("2", "<s>")}:
            return 5
        elif key in {("4", "<s>")}:
            return 2
        elif key in {("</s>", "2")}:
            return 11
        elif key in {("</s>", "0")}:
            return 14
        return 8

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_0_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_2_output, num_attn_0_3_output):
        key = (num_attn_1_2_output, num_attn_0_3_output)
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
            (1, 2),
            (1, 3),
            (1, 4),
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
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
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
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (4, 16),
            (4, 17),
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
            (5, 16),
            (5, 17),
            (5, 18),
            (5, 19),
            (5, 20),
            (5, 21),
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
            (6, 19),
            (6, 20),
            (6, 21),
            (6, 22),
            (6, 23),
            (6, 24),
            (6, 25),
            (6, 26),
            (6, 27),
            (6, 28),
            (6, 29),
            (6, 30),
            (6, 31),
            (7, 23),
            (7, 24),
            (7, 25),
            (7, 26),
            (7, 27),
            (7, 28),
            (7, 29),
            (7, 30),
            (7, 31),
            (8, 26),
            (8, 27),
            (8, 28),
            (8, 29),
            (8, 30),
            (8, 31),
            (9, 30),
            (9, 31),
        }:
            return 3
        elif key in {(0, 0)}:
            return 4
        return 15

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0, 11, 12, 15}:
            return k_position == 14
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 9
        elif q_position in {3}:
            return k_position == 15
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {5, 6}:
            return k_position == 11
        elif q_position in {8, 7}:
            return k_position == 12
        elif q_position in {9, 10, 13}:
            return k_position == 13
        elif q_position in {14}:
            return k_position == 0

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, token):
        if position in {0, 9, 11, 13, 15}:
            return token == "</s>"
        elif position in {1, 4, 5, 7, 14}:
            return token == ""
        elif position in {2, 10}:
            return token == "3"
        elif position in {3, 12}:
            return token == "0"
        elif position in {6}:
            return token == "4"
        elif position in {8}:
            return token == "<s>"

    attn_2_1_pattern = select_closest(tokens, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {0, 12}:
            return k_position == 14
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 7
        elif q_position in {4, 5}:
            return k_position == 10
        elif q_position in {6}:
            return k_position == 2
        elif q_position in {8, 7}:
            return k_position == 11
        elif q_position in {9, 11}:
            return k_position == 13
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {13, 15}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 0

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_2_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0, 15}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 8
        elif q_position in {5, 14}:
            return k_position == 1
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {11, 7}:
            return k_position == 11
        elif q_position in {8, 9}:
            return k_position == 12
        elif q_position in {10, 13}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 14

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_0_output, mlp_0_0_output):
        if attn_1_0_output in {"2", "0", "<s>", "3", "4"}:
            return mlp_0_0_output == 15
        elif attn_1_0_output in {"1"}:
            return mlp_0_0_output == 9
        elif attn_1_0_output in {"</s>"}:
            return mlp_0_0_output == 12

    num_attn_2_0_pattern = select(mlp_0_0_outputs, attn_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(q_position, k_position):
        if q_position in {0, 6, 15}:
            return k_position == 8
        elif q_position in {1, 4, 9, 10, 12, 13}:
            return k_position == 6
        elif q_position in {2, 3, 5, 11, 14}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 2
        elif q_position in {8}:
            return k_position == 9

    num_attn_2_1_pattern = select(positions, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_1_output, position):
        if attn_1_1_output in {"0", "4"}:
            return position == 11
        elif attn_1_1_output in {"1"}:
            return position == 6
        elif attn_1_1_output in {"2", "3"}:
            return position == 7
        elif attn_1_1_output in {"<s>", "</s>"}:
            return position == 9

    num_attn_2_2_pattern = select(positions, attn_1_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_0_output, position):
        if attn_1_0_output in {"0"}:
            return position == 10
        elif attn_1_0_output in {"1", "4"}:
            return position == 14
        elif attn_1_0_output in {"2", "3"}:
            return position == 11
        elif attn_1_0_output in {"</s>"}:
            return position == 4
        elif attn_1_0_output in {"<s>"}:
            return position == 3

    num_attn_2_3_pattern = select(positions, attn_1_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_0_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_0_output, attn_1_1_output):
        key = (attn_0_0_output, attn_1_1_output)
        if key in {("<s>", "4"), ("<s>", "</s>"), ("<s>", "<s>")}:
            return 11
        elif key in {
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
        }:
            return 4
        elif key in {("<s>", "0"), ("<s>", "1")}:
            return 7
        elif key in {("<s>", "2")}:
            return 14
        return 8

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_1_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_1_output, num_attn_1_2_output):
        key = (num_attn_2_1_output, num_attn_1_2_output)
        if key in {(0, 0)}:
            return 13
        return 4

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_1_2_outputs)
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
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                mlp_0_0_output_scores,
                num_mlp_0_0_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                num_mlp_1_0_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                num_mlp_2_0_output_scores,
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


print(run(["<s>", "0", "4", "3", "3", "0", "1", "1", "4", "2", "</s>"]))
