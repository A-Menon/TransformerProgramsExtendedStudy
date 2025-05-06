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
        "output-base-816/rasp/most_freq/k8_len16_L4_H8_M4/s0/most_freq_weights.csv",
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
        if q_position in {0, 3}:
            return k_position == 2
        elif q_position in {1, 2, 6}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {11, 5}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8, 14}:
            return k_position == 5
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {15}:
            return k_position == 13

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 12}:
            return k_position == 3
        elif q_position in {1, 15}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4, 14}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8, 13}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 15
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 9

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 10}:
            return k_position == 4
        elif q_position in {1, 11}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {8, 3}:
            return k_position == 1
        elif q_position in {4, 6, 14}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {9}:
            return k_position == 12
        elif q_position in {12, 13}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 13

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 9, 5}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 7
        elif q_position in {2, 6}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 4
        elif q_position in {8, 10}:
            return k_position == 12
        elif q_position in {11, 13}:
            return k_position == 9
        elif q_position in {12}:
            return k_position == 10
        elif q_position in {14, 15}:
            return k_position == 11

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 3}:
            return token == "<s>"
        elif position in {1}:
            return token == "2"
        elif position in {2, 7, 8, 11, 12, 13, 14, 15}:
            return token == ""
        elif position in {4, 5, 6}:
            return token == "3"
        elif position in {9, 10}:
            return token == "<pad>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 5, 9, 10, 11, 12, 14}:
            return token == "<s>"
        elif position in {1, 7}:
            return token == "4"
        elif position in {2}:
            return token == "<pad>"
        elif position in {3, 4}:
            return token == ""
        elif position in {13, 6, 15}:
            return token == "2"
        elif position in {8}:
            return token == "1"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 5, 6, 7, 8}:
            return token == "5"
        elif position in {1}:
            return token == "0"
        elif position in {2, 10, 11, 12, 13, 14, 15}:
            return token == ""
        elif position in {3, 4}:
            return token == "<s>"
        elif position in {9}:
            return token == "<pad>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15}:
            return token == "<s>"
        elif position in {1}:
            return token == "3"
        elif position in {2, 3, 14}:
            return token == ""

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position, attn_0_0_output):
        key = (position, attn_0_0_output)
        if key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "4"),
            (1, "5"),
            (1, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "5"),
            (2, "<s>"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "5"),
            (3, "<s>"),
            (11, "3"),
        }:
            return 13
        elif key in {
            (0, "0"),
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (0, "<s>"),
        }:
            return 5
        elif key in {
            (5, "0"),
            (5, "1"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "5"),
            (5, "<s>"),
        }:
            return 7
        elif key in {
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "<s>"),
        }:
            return 0
        elif key in {(1, "3")}:
            return 6
        elif key in {(8, "3")}:
            return 15
        return 9

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(positions, attn_0_0_outputs)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_0_output, position):
        key = (attn_0_0_output, position)
        if key in {
            ("0", 0),
            ("0", 3),
            ("0", 4),
            ("0", 7),
            ("1", 0),
            ("1", 3),
            ("1", 4),
            ("1", 7),
            ("2", 0),
            ("2", 3),
            ("2", 4),
            ("2", 7),
            ("3", 3),
            ("3", 4),
            ("3", 7),
            ("4", 0),
            ("4", 3),
            ("4", 4),
            ("4", 7),
            ("4", 13),
            ("5", 0),
            ("5", 3),
            ("5", 4),
            ("5", 7),
            ("<s>", 0),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 7),
        }:
            return 8
        elif key in {
            ("0", 6),
            ("0", 8),
            ("0", 12),
            ("0", 13),
            ("1", 6),
            ("1", 8),
            ("1", 12),
            ("1", 13),
            ("2", 6),
            ("2", 8),
            ("2", 12),
            ("2", 13),
            ("3", 6),
            ("3", 8),
            ("3", 12),
            ("3", 13),
            ("4", 6),
            ("4", 8),
            ("4", 12),
            ("5", 6),
            ("5", 8),
            ("5", 12),
            ("5", 13),
            ("<s>", 6),
            ("<s>", 8),
            ("<s>", 12),
            ("<s>", 13),
        }:
            return 11
        elif key in {
            ("0", 5),
            ("1", 5),
            ("2", 5),
            ("3", 5),
            ("4", 5),
            ("5", 5),
            ("<s>", 5),
        }:
            return 10
        elif key in {("0", 2), ("1", 2), ("2", 2), ("3", 2), ("4", 2), ("<s>", 2)}:
            return 7
        elif key in {("1", 1), ("2", 1), ("3", 0), ("3", 1)}:
            return 6
        elif key in {("5", 1), ("<s>", 1)}:
            return 15
        elif key in {("4", 1), ("5", 2)}:
            return 1
        elif key in {("0", 1)}:
            return 12
        return 0

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_0_outputs, positions)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_3_output):
        key = num_attn_0_3_output
        if key in {0}:
            return 1
        return 9

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {0}:
            return 6
        elif key in {1}:
            return 0
        return 9

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, num_mlp_0_0_output):
        if position in {0, 2, 4, 8, 11}:
            return num_mlp_0_0_output == 9
        elif position in {1, 9}:
            return num_mlp_0_0_output == 1
        elif position in {10, 3}:
            return num_mlp_0_0_output == 13
        elif position in {5}:
            return num_mlp_0_0_output == 14
        elif position in {6}:
            return num_mlp_0_0_output == 0
        elif position in {13, 14, 7}:
            return num_mlp_0_0_output == 2
        elif position in {12}:
            return num_mlp_0_0_output == 6
        elif position in {15}:
            return num_mlp_0_0_output == 10

    attn_1_0_pattern = select_closest(num_mlp_0_0_outputs, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 14}:
            return position == 15
        elif mlp_0_0_output in {1}:
            return position == 13
        elif mlp_0_0_output in {2}:
            return position == 9
        elif mlp_0_0_output in {3}:
            return position == 3
        elif mlp_0_0_output in {13, 4, 12}:
            return position == 2
        elif mlp_0_0_output in {10, 5}:
            return position == 5
        elif mlp_0_0_output in {9, 6}:
            return position == 6
        elif mlp_0_0_output in {7}:
            return position == 7
        elif mlp_0_0_output in {8}:
            return position == 8
        elif mlp_0_0_output in {11}:
            return position == 14
        elif mlp_0_0_output in {15}:
            return position == 1

    attn_1_1_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_1_output, num_mlp_0_1_output):
        if mlp_0_1_output in {0, 1, 9, 14, 15}:
            return num_mlp_0_1_output == 0
        elif mlp_0_1_output in {2, 3, 5, 6, 11, 13}:
            return num_mlp_0_1_output == 6
        elif mlp_0_1_output in {4}:
            return num_mlp_0_1_output == 11
        elif mlp_0_1_output in {7}:
            return num_mlp_0_1_output == 9
        elif mlp_0_1_output in {8}:
            return num_mlp_0_1_output == 8
        elif mlp_0_1_output in {10}:
            return num_mlp_0_1_output == 10
        elif mlp_0_1_output in {12}:
            return num_mlp_0_1_output == 12

    attn_1_2_pattern = select_closest(
        num_mlp_0_1_outputs, mlp_0_1_outputs, predicate_1_2
    )
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, mlp_0_1_output):
        if token in {"0"}:
            return mlp_0_1_output == 12
        elif token in {"1"}:
            return mlp_0_1_output == 0
        elif token in {"2", "4"}:
            return mlp_0_1_output == 1
        elif token in {"3"}:
            return mlp_0_1_output == 11
        elif token in {"<s>", "5"}:
            return mlp_0_1_output == 15

    attn_1_3_pattern = select_closest(mlp_0_1_outputs, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, token):
        if position in {0, 3}:
            return token == "<s>"
        elif position in {1, 2, 4}:
            return token == "0"
        elif position in {5}:
            return token == "1"
        elif position in {6}:
            return token == "2"
        elif position in {8, 9, 14, 7}:
            return token == ""
        elif position in {10, 11, 12, 13, 15}:
            return token == "<pad>"

    num_attn_1_0_pattern = select(tokens, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_0_output, attn_0_0_output):
        if mlp_0_0_output in {0, 7, 8, 9, 10, 14, 15}:
            return attn_0_0_output == "5"
        elif mlp_0_0_output in {1}:
            return attn_0_0_output == "2"
        elif mlp_0_0_output in {2, 3, 12, 6}:
            return attn_0_0_output == ""
        elif mlp_0_0_output in {4}:
            return attn_0_0_output == "3"
        elif mlp_0_0_output in {13, 11, 5}:
            return attn_0_0_output == "<s>"

    num_attn_1_1_pattern = select(attn_0_0_outputs, mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {0, 2, 3, 5}:
            return token == "<s>"
        elif position in {1, 4}:
            return token == "5"
        elif position in {6}:
            return token == "2"
        elif position in {7}:
            return token == "<pad>"
        elif position in {8, 9, 10, 11, 12, 13, 14, 15}:
            return token == ""

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0, 2, 3}:
            return token == "<s>"
        elif position in {1}:
            return token == "2"
        elif position in {4, 5}:
            return token == "5"
        elif position in {6, 7, 8, 9, 10, 11, 12, 13, 14, 15}:
            return token == ""

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(position, attn_1_0_output):
        key = (position, attn_1_0_output)
        if key in {
            (3, "0"),
            (5, "0"),
            (5, "1"),
            (5, "2"),
            (5, "4"),
            (5, "5"),
            (5, "<s>"),
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "4"),
            (6, "5"),
            (6, "<s>"),
            (7, "0"),
            (7, "1"),
            (7, "2"),
            (7, "4"),
            (7, "5"),
            (7, "<s>"),
            (8, "0"),
            (8, "1"),
            (8, "2"),
            (8, "4"),
            (8, "5"),
            (8, "<s>"),
        }:
            return 3
        elif key in {
            (0, "3"),
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "<s>"),
            (5, "3"),
            (6, "3"),
            (7, "3"),
            (15, "0"),
            (15, "2"),
            (15, "3"),
            (15, "4"),
            (15, "5"),
            (15, "<s>"),
        }:
            return 0
        elif key in {(8, "3"), (9, "3")}:
            return 4
        elif key in {
            (1, "5"),
            (2, "0"),
            (2, "5"),
            (9, "0"),
            (9, "1"),
            (9, "2"),
            (9, "4"),
            (9, "5"),
            (9, "<s>"),
        }:
            return 2
        elif key in {(1, "0")}:
            return 9
        elif key in {(1, "3"), (2, "1"), (2, "3"), (2, "4"), (3, "1"), (3, "3")}:
            return 10
        elif key in {(0, "0"), (0, "1"), (0, "2"), (0, "4"), (0, "5"), (0, "<s>")}:
            return 8
        elif key in {(1, "1"), (1, "2"), (1, "4"), (1, "<s>"), (2, "2"), (2, "<s>")}:
            return 15
        elif key in {(3, "2"), (3, "4"), (3, "<s>")}:
            return 11
        elif key in {(3, "5")}:
            return 14
        return 5

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(positions, attn_1_0_outputs)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position, attn_1_2_output):
        key = (position, attn_1_2_output)
        if key in {
            (4, "0"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "<s>"),
            (9, "0"),
            (9, "2"),
            (9, "3"),
            (9, "4"),
            (9, "5"),
            (9, "<s>"),
            (10, "0"),
            (10, "3"),
            (10, "4"),
            (10, "<s>"),
            (12, "0"),
            (12, "2"),
            (12, "3"),
            (12, "4"),
            (12, "5"),
            (12, "<s>"),
            (13, "0"),
            (13, "2"),
            (13, "3"),
            (13, "4"),
            (13, "5"),
            (13, "<s>"),
            (14, "3"),
            (14, "4"),
            (14, "5"),
            (14, "<s>"),
        }:
            return 7
        elif key in {
            (0, "0"),
            (0, "1"),
            (0, "3"),
            (0, "4"),
            (0, "<s>"),
            (1, "0"),
            (1, "1"),
            (1, "3"),
            (1, "4"),
            (1, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "5"),
            (2, "<s>"),
        }:
            return 1
        elif key in {
            (7, "0"),
            (7, "1"),
            (7, "2"),
            (7, "3"),
            (7, "4"),
            (7, "5"),
            (7, "<s>"),
            (11, "4"),
            (15, "3"),
            (15, "4"),
        }:
            return 6
        elif key in {(3, "3"), (3, "4"), (3, "<s>"), (5, "<s>")}:
            return 11
        elif key in {(0, "2"), (0, "5"), (1, "2"), (1, "5"), (14, "2")}:
            return 12
        elif key in {(8, "<s>"), (11, "3"), (11, "<s>"), (15, "<s>")}:
            return 0
        return 15

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(positions, attn_1_2_outputs)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        if key in {0}:
            return 0
        elif key in {1}:
            return 5
        return 15

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_0_output, num_attn_0_1_output):
        key = (num_attn_1_0_output, num_attn_0_1_output)
        if key in {
            (4, 2),
            (5, 2),
            (5, 3),
            (6, 2),
            (6, 3),
            (6, 4),
            (7, 3),
            (7, 4),
            (8, 3),
            (8, 4),
            (8, 5),
            (9, 3),
            (9, 4),
            (9, 5),
            (9, 6),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (11, 7),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (12, 7),
            (12, 8),
            (13, 4),
            (13, 5),
            (13, 6),
            (13, 7),
            (13, 8),
            (13, 9),
            (14, 4),
            (14, 5),
            (14, 6),
            (14, 7),
            (14, 8),
            (14, 9),
            (14, 10),
            (15, 4),
            (15, 5),
            (15, 6),
            (15, 7),
            (15, 8),
            (15, 9),
            (15, 10),
            (15, 11),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 7),
            (16, 8),
            (16, 9),
            (16, 10),
            (16, 11),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (17, 7),
            (17, 8),
            (17, 9),
            (17, 10),
            (17, 11),
            (17, 12),
            (18, 3),
            (18, 5),
            (18, 6),
            (18, 7),
            (18, 8),
            (18, 9),
            (18, 10),
            (18, 11),
            (18, 12),
            (18, 13),
            (19, 3),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 8),
            (19, 9),
            (19, 10),
            (19, 11),
            (19, 12),
            (19, 13),
            (19, 14),
            (20, 3),
            (20, 4),
            (20, 5),
            (20, 6),
            (20, 7),
            (20, 8),
            (20, 9),
            (20, 10),
            (20, 11),
            (20, 12),
            (20, 13),
            (20, 14),
            (21, 4),
            (21, 5),
            (21, 6),
            (21, 7),
            (21, 8),
            (21, 9),
            (21, 10),
            (21, 11),
            (21, 12),
            (21, 13),
            (21, 14),
            (21, 15),
            (22, 4),
            (22, 5),
            (22, 6),
            (22, 7),
            (22, 8),
            (22, 9),
            (22, 10),
            (22, 11),
            (22, 12),
            (22, 13),
            (22, 14),
            (22, 15),
            (22, 16),
            (23, 4),
            (23, 5),
            (23, 6),
            (23, 7),
            (23, 8),
            (23, 9),
            (23, 10),
            (23, 11),
            (23, 12),
            (23, 13),
            (23, 14),
            (23, 15),
            (23, 16),
            (23, 17),
            (24, 4),
            (24, 5),
            (24, 6),
            (24, 7),
            (24, 8),
            (24, 9),
            (24, 10),
            (24, 11),
            (24, 12),
            (24, 13),
            (24, 14),
            (24, 15),
            (24, 16),
            (24, 17),
            (25, 4),
            (25, 5),
            (25, 6),
            (25, 7),
            (25, 8),
            (25, 9),
            (25, 10),
            (25, 11),
            (25, 12),
            (25, 13),
            (25, 14),
            (25, 15),
            (25, 16),
            (25, 17),
            (25, 18),
            (26, 4),
            (26, 5),
            (26, 6),
            (26, 7),
            (26, 8),
            (26, 9),
            (26, 10),
            (26, 11),
            (26, 12),
            (26, 13),
            (26, 14),
            (26, 15),
            (26, 16),
            (26, 17),
            (26, 18),
            (26, 19),
            (27, 5),
            (27, 6),
            (27, 7),
            (27, 8),
            (27, 9),
            (27, 10),
            (27, 11),
            (27, 12),
            (27, 13),
            (27, 14),
            (27, 15),
            (27, 16),
            (27, 17),
            (27, 18),
            (27, 19),
            (27, 20),
            (28, 5),
            (28, 6),
            (28, 7),
            (28, 8),
            (28, 9),
            (28, 10),
            (28, 11),
            (28, 12),
            (28, 13),
            (28, 14),
            (28, 15),
            (28, 16),
            (28, 17),
            (28, 18),
            (28, 19),
            (28, 20),
            (29, 5),
            (29, 6),
            (29, 7),
            (29, 8),
            (29, 9),
            (29, 10),
            (29, 11),
            (29, 12),
            (29, 13),
            (29, 14),
            (29, 15),
            (29, 16),
            (29, 17),
            (29, 18),
            (29, 19),
            (29, 20),
            (29, 21),
            (30, 5),
            (30, 6),
            (30, 7),
            (30, 8),
            (30, 9),
            (30, 10),
            (30, 11),
            (30, 12),
            (30, 13),
            (30, 14),
            (30, 15),
            (30, 16),
            (30, 17),
            (30, 18),
            (30, 19),
            (30, 20),
            (30, 21),
            (30, 22),
            (31, 5),
            (31, 6),
            (31, 7),
            (31, 8),
            (31, 9),
            (31, 10),
            (31, 11),
            (31, 12),
            (31, 13),
            (31, 14),
            (31, 15),
            (31, 16),
            (31, 17),
            (31, 18),
            (31, 19),
            (31, 20),
            (31, 21),
            (31, 22),
            (31, 23),
        }:
            return 2
        elif key in {
            (2, 2),
            (3, 2),
            (3, 3),
            (4, 3),
            (4, 4),
            (5, 4),
            (5, 5),
            (6, 5),
            (6, 6),
            (6, 7),
            (7, 5),
            (7, 6),
            (7, 7),
            (7, 8),
            (8, 6),
            (8, 7),
            (8, 8),
            (8, 9),
            (9, 7),
            (9, 8),
            (9, 9),
            (9, 10),
            (10, 8),
            (10, 9),
            (10, 10),
            (10, 11),
            (11, 8),
            (11, 9),
            (11, 10),
            (11, 11),
            (11, 12),
            (11, 13),
            (12, 9),
            (12, 10),
            (12, 11),
            (12, 12),
            (12, 13),
            (12, 14),
            (13, 10),
            (13, 11),
            (13, 12),
            (13, 13),
            (13, 14),
            (13, 15),
            (14, 11),
            (14, 12),
            (14, 13),
            (14, 14),
            (14, 15),
            (14, 16),
            (15, 12),
            (15, 13),
            (15, 14),
            (15, 15),
            (15, 16),
            (15, 17),
            (16, 12),
            (16, 13),
            (16, 14),
            (16, 15),
            (16, 16),
            (16, 17),
            (16, 18),
            (17, 13),
            (17, 14),
            (17, 15),
            (17, 16),
            (17, 17),
            (17, 18),
            (17, 19),
            (18, 14),
            (18, 15),
            (18, 16),
            (18, 17),
            (18, 18),
            (18, 19),
            (18, 20),
            (18, 21),
            (19, 15),
            (19, 16),
            (19, 17),
            (19, 18),
            (19, 19),
            (19, 20),
            (19, 21),
            (19, 22),
            (20, 15),
            (20, 16),
            (20, 17),
            (20, 18),
            (20, 19),
            (20, 20),
            (20, 21),
            (20, 22),
            (20, 23),
            (21, 16),
            (21, 17),
            (21, 18),
            (21, 19),
            (21, 20),
            (21, 21),
            (21, 22),
            (21, 23),
            (21, 24),
            (22, 17),
            (22, 18),
            (22, 19),
            (22, 20),
            (22, 21),
            (22, 22),
            (22, 23),
            (22, 24),
            (22, 25),
            (23, 18),
            (23, 19),
            (23, 20),
            (23, 21),
            (23, 22),
            (23, 23),
            (23, 24),
            (23, 25),
            (23, 26),
            (24, 18),
            (24, 19),
            (24, 20),
            (24, 21),
            (24, 22),
            (24, 23),
            (24, 24),
            (24, 25),
            (24, 26),
            (24, 27),
            (25, 19),
            (25, 20),
            (25, 21),
            (25, 22),
            (25, 23),
            (25, 24),
            (25, 25),
            (25, 26),
            (25, 27),
            (25, 28),
            (25, 29),
            (26, 20),
            (26, 21),
            (26, 22),
            (26, 23),
            (26, 24),
            (26, 25),
            (26, 26),
            (26, 27),
            (26, 28),
            (26, 29),
            (26, 30),
            (27, 21),
            (27, 22),
            (27, 23),
            (27, 24),
            (27, 25),
            (27, 26),
            (27, 27),
            (27, 28),
            (27, 29),
            (27, 30),
            (27, 31),
            (28, 21),
            (28, 22),
            (28, 23),
            (28, 24),
            (28, 25),
            (28, 26),
            (28, 27),
            (28, 28),
            (28, 29),
            (28, 30),
            (28, 31),
            (29, 22),
            (29, 23),
            (29, 24),
            (29, 25),
            (29, 26),
            (29, 27),
            (29, 28),
            (29, 29),
            (29, 30),
            (29, 31),
            (30, 23),
            (30, 24),
            (30, 25),
            (30, 26),
            (30, 27),
            (30, 28),
            (30, 29),
            (30, 30),
            (30, 31),
            (31, 24),
            (31, 25),
            (31, 26),
            (31, 27),
            (31, 28),
            (31, 29),
            (31, 30),
            (31, 31),
        }:
            return 4
        elif key in {
            (5, 0),
            (6, 0),
            (7, 0),
            (8, 0),
            (9, 0),
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
            (15, 2),
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
            (25, 3),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
        }:
            return 5
        elif key in {
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (7, 2),
            (8, 1),
            (8, 2),
            (9, 1),
            (9, 2),
            (10, 2),
            (11, 2),
            (12, 2),
            (13, 2),
            (13, 3),
            (14, 2),
            (14, 3),
            (15, 3),
            (16, 3),
            (18, 4),
            (19, 4),
        }:
            return 11
        elif key in {(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)}:
            return 12
        elif key in {(0, 0), (1, 0)}:
            return 13
        elif key in {(2, 0), (3, 0)}:
            return 10
        elif key in {(0, 1)}:
            return 15
        elif key in {(1, 1)}:
            return 0
        elif key in {(2, 1)}:
            return 7
        elif key in {(4, 0)}:
            return 9
        return 14

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(mlp_1_1_output, attn_1_1_output):
        if mlp_1_1_output in {0, 2, 3, 4, 5, 8}:
            return attn_1_1_output == ""
        elif mlp_1_1_output in {1, 14, 6, 7}:
            return attn_1_1_output == "2"
        elif mlp_1_1_output in {9}:
            return attn_1_1_output == "0"
        elif mlp_1_1_output in {10}:
            return attn_1_1_output == "3"
        elif mlp_1_1_output in {11, 13}:
            return attn_1_1_output == "1"
        elif mlp_1_1_output in {12}:
            return attn_1_1_output == "5"
        elif mlp_1_1_output in {15}:
            return attn_1_1_output == "4"

    attn_2_0_pattern = select_closest(attn_1_1_outputs, mlp_1_1_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {0, 1, 2}:
            return k_mlp_0_0_output == 9
        elif q_mlp_0_0_output in {3, 7}:
            return k_mlp_0_0_output == 7
        elif q_mlp_0_0_output in {4, 5}:
            return k_mlp_0_0_output == 5
        elif q_mlp_0_0_output in {6}:
            return k_mlp_0_0_output == 15
        elif q_mlp_0_0_output in {8, 11}:
            return k_mlp_0_0_output == 13
        elif q_mlp_0_0_output in {9, 14}:
            return k_mlp_0_0_output == 3
        elif q_mlp_0_0_output in {10}:
            return k_mlp_0_0_output == 4
        elif q_mlp_0_0_output in {12, 13}:
            return k_mlp_0_0_output == 6
        elif q_mlp_0_0_output in {15}:
            return k_mlp_0_0_output == 8

    attn_2_1_pattern = select_closest(mlp_0_0_outputs, mlp_0_0_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(position, token):
        if position in {0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15}:
            return token == ""
        elif position in {4, 5}:
            return token == "3"
        elif position in {6}:
            return token == "<pad>"

    attn_2_2_pattern = select_closest(tokens, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "4"
        elif q_token in {"1", "4", "3", "2", "5"}:
            return k_token == "<s>"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_3_pattern = select_closest(tokens, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, mlp_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_1_1_output, token):
        if mlp_1_1_output in {0, 2, 3, 6, 9, 11, 13}:
            return token == "<s>"
        elif mlp_1_1_output in {1, 12, 7}:
            return token == "1"
        elif mlp_1_1_output in {4}:
            return token == "4"
        elif mlp_1_1_output in {5, 14}:
            return token == "5"
        elif mlp_1_1_output in {8, 15}:
            return token == ""
        elif mlp_1_1_output in {10}:
            return token == "<pad>"

    num_attn_2_0_pattern = select(tokens, mlp_1_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, ones)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, attn_0_0_output):
        if position in {0, 2, 3}:
            return attn_0_0_output == "<s>"
        elif position in {1}:
            return attn_0_0_output == "5"
        elif position in {4, 5, 6}:
            return attn_0_0_output == "4"
        elif position in {7, 8, 9, 10, 11, 12, 13, 14, 15}:
            return attn_0_0_output == ""

    num_attn_2_1_pattern = select(attn_0_0_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, ones)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, attn_0_0_output):
        if position in {0, 2, 3}:
            return attn_0_0_output == "<s>"
        elif position in {1}:
            return attn_0_0_output == "3"
        elif position in {4, 5, 6}:
            return attn_0_0_output == "2"
        elif position in {7, 8, 9, 10, 11, 12, 13, 14, 15}:
            return attn_0_0_output == ""

    num_attn_2_2_pattern = select(attn_0_0_outputs, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, ones)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(position, token):
        if position in {0, 7, 8, 9, 10, 11, 12, 13, 14, 15}:
            return token == ""
        elif position in {1}:
            return token == "5"
        elif position in {2, 3}:
            return token == "<s>"
        elif position in {4, 5, 6}:
            return token == "0"

    num_attn_2_3_pattern = select(tokens, positions, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, ones)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_1_output, attn_2_0_output):
        key = (attn_2_1_output, attn_2_0_output)
        if key in {
            ("0", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("2", "2"),
            ("3", "2"),
            ("3", "3"),
            ("4", "4"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "4"),
        }:
            return 10
        elif key in {("2", "1"), ("3", "1"), ("3", "4"), ("4", "2"), ("4", "3")}:
            return 6
        elif key in {
            ("0", "1"),
            ("0", "3"),
            ("0", "4"),
            ("0", "5"),
            ("0", "<s>"),
            ("1", "0"),
            ("1", "5"),
            ("1", "<s>"),
            ("3", "<s>"),
            ("4", "5"),
            ("4", "<s>"),
            ("5", "0"),
            ("5", "3"),
            ("5", "5"),
            ("5", "<s>"),
            ("<s>", "0"),
            ("<s>", "3"),
            ("<s>", "5"),
            ("<s>", "<s>"),
        }:
            return 13
        elif key in {("3", "0"), ("4", "0"), ("4", "1")}:
            return 0
        elif key in {("2", "5"), ("2", "<s>"), ("3", "5")}:
            return 2
        elif key in {("0", "2"), ("2", "0"), ("5", "1")}:
            return 3
        return 14

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_2_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(position, attn_2_0_output):
        key = (position, attn_2_0_output)
        if key in {
            (0, "0"),
            (0, "1"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (0, "<s>"),
            (1, "1"),
            (1, "3"),
            (1, "5"),
            (1, "<s>"),
            (2, "1"),
            (2, "3"),
            (2, "4"),
            (2, "5"),
            (2, "<s>"),
            (3, "4"),
            (4, "1"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "<s>"),
            (9, "0"),
            (9, "1"),
            (9, "3"),
            (9, "4"),
            (9, "5"),
            (9, "<s>"),
            (13, "0"),
            (13, "1"),
            (13, "3"),
            (13, "4"),
            (13, "5"),
            (13, "<s>"),
            (14, "0"),
            (14, "1"),
            (14, "3"),
            (14, "4"),
        }:
            return 13
        elif key in {
            (5, "3"),
            (5, "4"),
            (5, "5"),
            (5, "<s>"),
            (6, "4"),
            (7, "0"),
            (7, "1"),
            (7, "2"),
            (7, "3"),
            (7, "4"),
            (7, "5"),
            (7, "<s>"),
            (11, "2"),
            (15, "0"),
            (15, "1"),
            (15, "3"),
            (15, "4"),
            (15, "5"),
            (15, "<s>"),
        }:
            return 14
        elif key in {
            (0, "2"),
            (2, "2"),
            (3, "2"),
            (4, "0"),
            (4, "2"),
            (13, "2"),
            (14, "2"),
            (15, "2"),
        }:
            return 3
        elif key in {
            (1, "0"),
            (1, "2"),
            (2, "0"),
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "5"),
            (6, "<s>"),
            (9, "2"),
            (10, "2"),
        }:
            return 9
        elif key in {
            (1, "4"),
            (11, "0"),
            (11, "1"),
            (11, "3"),
            (11, "5"),
            (11, "<s>"),
            (14, "5"),
            (14, "<s>"),
        }:
            return 11
        elif key in {(3, "0"), (3, "1"), (3, "3"), (3, "5"), (3, "<s>")}:
            return 10
        elif key in {(10, "0"), (10, "1"), (10, "3"), (10, "5"), (10, "<s>")}:
            return 2
        elif key in {(5, "1"), (11, "4")}:
            return 12
        return 5

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(positions, attn_2_0_outputs)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_3_output):
        key = num_attn_2_3_output
        if key in {0}:
            return 12
        return 8

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_2_3_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_3_output):
        key = num_attn_2_3_output
        if key in {1, 2}:
            return 15
        return 8

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_2_3_outputs]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # attn_3_0 ####################################################
    def predicate_3_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "2"
        elif q_token in {"1", "4"}:
            return k_token == "3"
        elif q_token in {"2", "3"}:
            return k_token == "5"
        elif q_token in {"5"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_3_0_pattern = select_closest(tokens, tokens, predicate_3_0)
    attn_3_0_outputs = aggregate(attn_3_0_pattern, mlp_2_0_outputs)
    attn_3_0_output_scores = classifier_weights.loc[
        [("attn_3_0_outputs", str(v)) for v in attn_3_0_outputs]
    ]

    # attn_3_1 ####################################################
    def predicate_3_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 0
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 9
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {5, 6}:
            return k_position == 8
        elif q_position in {7, 8, 9, 12, 13, 14, 15}:
            return k_position == 14
        elif q_position in {10, 11}:
            return k_position == 13

    attn_3_1_pattern = select_closest(positions, positions, predicate_3_1)
    attn_3_1_outputs = aggregate(attn_3_1_pattern, mlp_1_0_outputs)
    attn_3_1_output_scores = classifier_weights.loc[
        [("attn_3_1_outputs", str(v)) for v in attn_3_1_outputs]
    ]

    # attn_3_2 ####################################################
    def predicate_3_2(q_token, k_token):
        if q_token in {"2", "0", "3"}:
            return k_token == "4"
        elif q_token in {"1", "5", "4"}:
            return k_token == "2"
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    attn_3_2_pattern = select_closest(tokens, tokens, predicate_3_2)
    attn_3_2_outputs = aggregate(attn_3_2_pattern, mlp_2_0_outputs)
    attn_3_2_output_scores = classifier_weights.loc[
        [("attn_3_2_outputs", str(v)) for v in attn_3_2_outputs]
    ]

    # attn_3_3 ####################################################
    def predicate_3_3(mlp_0_1_output, mlp_2_1_output):
        if mlp_0_1_output in {0, 11}:
            return mlp_2_1_output == 14
        elif mlp_0_1_output in {1, 2}:
            return mlp_2_1_output == 8
        elif mlp_0_1_output in {3, 4, 5, 6, 7, 8, 9, 13, 14}:
            return mlp_2_1_output == 7
        elif mlp_0_1_output in {10}:
            return mlp_2_1_output == 3
        elif mlp_0_1_output in {12}:
            return mlp_2_1_output == 5
        elif mlp_0_1_output in {15}:
            return mlp_2_1_output == 2

    attn_3_3_pattern = select_closest(mlp_2_1_outputs, mlp_0_1_outputs, predicate_3_3)
    attn_3_3_outputs = aggregate(attn_3_3_pattern, mlp_2_1_outputs)
    attn_3_3_output_scores = classifier_weights.loc[
        [("attn_3_3_outputs", str(v)) for v in attn_3_3_outputs]
    ]

    # num_attn_3_0 ####################################################
    def num_predicate_3_0(position, token):
        if position in {0, 3}:
            return token == "<s>"
        elif position in {1}:
            return token == "4"
        elif position in {2, 4, 6, 9, 11, 15}:
            return token == "1"
        elif position in {8, 12, 5, 14}:
            return token == ""
        elif position in {7}:
            return token == "5"
        elif position in {10, 13}:
            return token == "<pad>"

    num_attn_3_0_pattern = select(tokens, positions, num_predicate_3_0)
    num_attn_3_0_outputs = aggregate_sum(num_attn_3_0_pattern, ones)
    num_attn_3_0_output_scores = classifier_weights.loc[
        [("num_attn_3_0_outputs", "_") for v in num_attn_3_0_outputs]
    ].mul(num_attn_3_0_outputs, axis=0)

    # num_attn_3_1 ####################################################
    def num_predicate_3_1(position, attn_0_0_output):
        if position in {0}:
            return attn_0_0_output == "0"
        elif position in {1}:
            return attn_0_0_output == "5"
        elif position in {2, 3}:
            return attn_0_0_output == "<s>"
        elif position in {4, 5, 6, 7, 10, 12, 13, 14}:
            return attn_0_0_output == "3"
        elif position in {8, 9}:
            return attn_0_0_output == "4"
        elif position in {11, 15}:
            return attn_0_0_output == "1"

    num_attn_3_1_pattern = select(attn_0_0_outputs, positions, num_predicate_3_1)
    num_attn_3_1_outputs = aggregate_sum(num_attn_3_1_pattern, num_attn_0_3_outputs)
    num_attn_3_1_output_scores = classifier_weights.loc[
        [("num_attn_3_1_outputs", "_") for v in num_attn_3_1_outputs]
    ].mul(num_attn_3_1_outputs, axis=0)

    # num_attn_3_2 ####################################################
    def num_predicate_3_2(position, mlp_1_1_output):
        if position in {0, 1, 2, 3}:
            return mlp_1_1_output == 7
        elif position in {4}:
            return mlp_1_1_output == 3
        elif position in {5, 6}:
            return mlp_1_1_output == 11
        elif position in {7, 8, 9, 10, 12, 13, 14}:
            return mlp_1_1_output == 8
        elif position in {11}:
            return mlp_1_1_output == 10
        elif position in {15}:
            return mlp_1_1_output == 2

    num_attn_3_2_pattern = select(mlp_1_1_outputs, positions, num_predicate_3_2)
    num_attn_3_2_outputs = aggregate_sum(num_attn_3_2_pattern, ones)
    num_attn_3_2_output_scores = classifier_weights.loc[
        [("num_attn_3_2_outputs", "_") for v in num_attn_3_2_outputs]
    ].mul(num_attn_3_2_outputs, axis=0)

    # num_attn_3_3 ####################################################
    def num_predicate_3_3(q_attn_1_0_output, k_attn_1_0_output):
        if q_attn_1_0_output in {"<s>", "4", "3", "2", "0", "5"}:
            return k_attn_1_0_output == "1"
        elif q_attn_1_0_output in {"1"}:
            return k_attn_1_0_output == "3"

    num_attn_3_3_pattern = select(attn_1_0_outputs, attn_1_0_outputs, num_predicate_3_3)
    num_attn_3_3_outputs = aggregate_sum(num_attn_3_3_pattern, num_attn_0_2_outputs)
    num_attn_3_3_output_scores = classifier_weights.loc[
        [("num_attn_3_3_outputs", "_") for v in num_attn_3_3_outputs]
    ].mul(num_attn_3_3_outputs, axis=0)

    # mlp_3_0 #####################################################
    def mlp_3_0(num_mlp_2_1_output, attn_2_3_output):
        key = (num_mlp_2_1_output, attn_2_3_output)
        return 11

    mlp_3_0_outputs = [
        mlp_3_0(k0, k1) for k0, k1 in zip(num_mlp_2_1_outputs, attn_2_3_outputs)
    ]
    mlp_3_0_output_scores = classifier_weights.loc[
        [("mlp_3_0_outputs", str(v)) for v in mlp_3_0_outputs]
    ]

    # mlp_3_1 #####################################################
    def mlp_3_1(attn_2_0_output, attn_2_1_output):
        key = (attn_2_0_output, attn_2_1_output)
        if key in {
            ("0", "1"),
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "5"),
            ("1", "<s>"),
            ("2", "1"),
            ("3", "1"),
            ("5", "1"),
            ("<s>", "1"),
        }:
            return 9
        elif key in {
            ("0", "4"),
            ("2", "4"),
            ("3", "4"),
            ("4", "0"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "5"),
            ("4", "<s>"),
            ("5", "4"),
            ("<s>", "4"),
        }:
            return 12
        elif key in {("1", "4"), ("4", "1")}:
            return 8
        return 10

    mlp_3_1_outputs = [
        mlp_3_1(k0, k1) for k0, k1 in zip(attn_2_0_outputs, attn_2_1_outputs)
    ]
    mlp_3_1_output_scores = classifier_weights.loc[
        [("mlp_3_1_outputs", str(v)) for v in mlp_3_1_outputs]
    ]

    # num_mlp_3_0 #################################################
    def num_mlp_3_0(num_attn_1_3_output):
        key = num_attn_1_3_output
        if key in {1, 2}:
            return 10
        elif key in {0}:
            return 13
        return 9

    num_mlp_3_0_outputs = [num_mlp_3_0(k0) for k0 in num_attn_1_3_outputs]
    num_mlp_3_0_output_scores = classifier_weights.loc[
        [("num_mlp_3_0_outputs", str(v)) for v in num_mlp_3_0_outputs]
    ]

    # num_mlp_3_1 #################################################
    def num_mlp_3_1(num_attn_1_3_output, num_attn_2_3_output):
        key = (num_attn_1_3_output, num_attn_2_3_output)
        if key in {
            (1, 2),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 8),
            (5, 9),
            (5, 10),
            (6, 5),
            (6, 6),
            (6, 7),
            (6, 8),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 12),
            (7, 6),
            (7, 7),
            (7, 8),
            (7, 9),
            (7, 10),
            (7, 11),
            (7, 12),
            (7, 13),
            (7, 14),
            (8, 7),
            (8, 8),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 15),
            (8, 16),
            (9, 8),
            (9, 9),
            (9, 10),
            (9, 11),
            (9, 12),
            (9, 13),
            (9, 14),
            (9, 15),
            (9, 16),
            (9, 17),
            (9, 18),
            (10, 9),
            (10, 10),
            (10, 11),
            (10, 12),
            (10, 13),
            (10, 14),
            (10, 15),
            (10, 16),
            (10, 17),
            (10, 18),
            (10, 19),
            (10, 20),
            (11, 10),
            (11, 11),
            (11, 12),
            (11, 13),
            (11, 14),
            (11, 15),
            (11, 16),
            (11, 17),
            (11, 18),
            (11, 19),
            (11, 20),
            (11, 21),
            (11, 22),
            (12, 11),
            (12, 12),
            (12, 13),
            (12, 14),
            (12, 15),
            (12, 16),
            (12, 17),
            (12, 18),
            (12, 19),
            (12, 20),
            (12, 21),
            (12, 22),
            (12, 23),
            (12, 24),
            (13, 11),
            (13, 12),
            (13, 13),
            (13, 14),
            (13, 15),
            (13, 16),
            (13, 17),
            (13, 18),
            (13, 19),
            (13, 20),
            (13, 21),
            (13, 22),
            (13, 23),
            (13, 24),
            (13, 25),
            (13, 26),
            (14, 12),
            (14, 13),
            (14, 14),
            (14, 15),
            (14, 16),
            (14, 17),
            (14, 18),
            (14, 19),
            (14, 20),
            (14, 21),
            (14, 22),
            (14, 23),
            (14, 24),
            (14, 25),
            (14, 26),
            (14, 27),
            (14, 28),
            (15, 13),
            (15, 14),
            (15, 15),
            (15, 16),
            (15, 17),
            (15, 18),
            (15, 19),
            (15, 20),
            (15, 21),
            (15, 22),
            (15, 23),
            (15, 24),
            (15, 25),
            (15, 26),
            (15, 27),
            (15, 28),
            (15, 29),
            (15, 30),
            (16, 14),
            (16, 15),
            (16, 16),
            (16, 17),
            (16, 18),
            (16, 19),
            (16, 20),
            (16, 21),
            (16, 22),
            (16, 23),
            (16, 24),
            (16, 25),
            (16, 26),
            (16, 27),
            (16, 28),
            (16, 29),
            (16, 30),
            (16, 31),
            (16, 32),
            (17, 15),
            (17, 16),
            (17, 17),
            (17, 18),
            (17, 19),
            (17, 20),
            (17, 21),
            (17, 22),
            (17, 23),
            (17, 24),
            (17, 25),
            (17, 26),
            (17, 27),
            (17, 28),
            (17, 29),
            (17, 30),
            (17, 31),
            (17, 32),
            (17, 33),
            (17, 34),
            (18, 16),
            (18, 17),
            (18, 18),
            (18, 19),
            (18, 20),
            (18, 21),
            (18, 22),
            (18, 23),
            (18, 24),
            (18, 25),
            (18, 26),
            (18, 27),
            (18, 28),
            (18, 29),
            (18, 30),
            (18, 31),
            (18, 32),
            (18, 33),
            (18, 34),
            (18, 35),
            (18, 36),
            (19, 17),
            (19, 18),
            (19, 19),
            (19, 20),
            (19, 21),
            (19, 22),
            (19, 23),
            (19, 24),
            (19, 25),
            (19, 26),
            (19, 27),
            (19, 28),
            (19, 29),
            (19, 30),
            (19, 31),
            (19, 32),
            (19, 33),
            (19, 34),
            (19, 35),
            (19, 36),
            (19, 37),
            (19, 38),
            (20, 18),
            (20, 19),
            (20, 20),
            (20, 21),
            (20, 22),
            (20, 23),
            (20, 24),
            (20, 25),
            (20, 26),
            (20, 27),
            (20, 28),
            (20, 29),
            (20, 30),
            (20, 31),
            (20, 32),
            (20, 33),
            (20, 34),
            (20, 35),
            (20, 36),
            (20, 37),
            (20, 38),
            (20, 39),
            (20, 40),
            (21, 18),
            (21, 19),
            (21, 20),
            (21, 21),
            (21, 22),
            (21, 23),
            (21, 24),
            (21, 25),
            (21, 26),
            (21, 27),
            (21, 28),
            (21, 29),
            (21, 30),
            (21, 31),
            (21, 32),
            (21, 33),
            (21, 34),
            (21, 35),
            (21, 36),
            (21, 37),
            (21, 38),
            (21, 39),
            (21, 40),
            (21, 41),
            (21, 42),
            (22, 19),
            (22, 20),
            (22, 21),
            (22, 22),
            (22, 23),
            (22, 24),
            (22, 25),
            (22, 26),
            (22, 27),
            (22, 28),
            (22, 29),
            (22, 30),
            (22, 31),
            (22, 32),
            (22, 33),
            (22, 34),
            (22, 35),
            (22, 36),
            (22, 37),
            (22, 38),
            (22, 39),
            (22, 40),
            (22, 41),
            (22, 42),
            (22, 43),
            (22, 44),
            (23, 20),
            (23, 21),
            (23, 22),
            (23, 23),
            (23, 24),
            (23, 25),
            (23, 26),
            (23, 27),
            (23, 28),
            (23, 29),
            (23, 30),
            (23, 31),
            (23, 32),
            (23, 33),
            (23, 34),
            (23, 35),
            (23, 36),
            (23, 37),
            (23, 38),
            (23, 39),
            (23, 40),
            (23, 41),
            (23, 42),
            (23, 43),
            (23, 44),
            (23, 45),
            (23, 46),
            (24, 21),
            (24, 22),
            (24, 23),
            (24, 24),
            (24, 25),
            (24, 26),
            (24, 27),
            (24, 28),
            (24, 29),
            (24, 30),
            (24, 31),
            (24, 32),
            (24, 33),
            (24, 34),
            (24, 35),
            (24, 36),
            (24, 37),
            (24, 38),
            (24, 39),
            (24, 40),
            (24, 41),
            (24, 42),
            (24, 43),
            (24, 44),
            (24, 45),
            (24, 46),
            (24, 47),
            (24, 48),
            (25, 22),
            (25, 23),
            (25, 24),
            (25, 25),
            (25, 26),
            (25, 27),
            (25, 28),
            (25, 29),
            (25, 30),
            (25, 31),
            (25, 32),
            (25, 33),
            (25, 34),
            (25, 35),
            (25, 36),
            (25, 37),
            (25, 38),
            (25, 39),
            (25, 40),
            (25, 41),
            (25, 42),
            (25, 43),
            (25, 44),
            (25, 45),
            (25, 46),
            (25, 47),
            (25, 48),
            (25, 49),
            (25, 50),
            (26, 23),
            (26, 24),
            (26, 25),
            (26, 26),
            (26, 27),
            (26, 28),
            (26, 29),
            (26, 30),
            (26, 31),
            (26, 32),
            (26, 33),
            (26, 34),
            (26, 35),
            (26, 36),
            (26, 37),
            (26, 38),
            (26, 39),
            (26, 40),
            (26, 41),
            (26, 42),
            (26, 43),
            (26, 44),
            (26, 45),
            (26, 46),
            (26, 47),
            (26, 48),
            (26, 49),
            (26, 50),
            (26, 51),
            (27, 24),
            (27, 25),
            (27, 26),
            (27, 27),
            (27, 28),
            (27, 29),
            (27, 30),
            (27, 31),
            (27, 32),
            (27, 33),
            (27, 34),
            (27, 35),
            (27, 36),
            (27, 37),
            (27, 38),
            (27, 39),
            (27, 40),
            (27, 41),
            (27, 42),
            (27, 43),
            (27, 44),
            (27, 45),
            (27, 46),
            (27, 47),
            (27, 48),
            (27, 49),
            (27, 50),
            (27, 51),
            (27, 52),
            (27, 53),
            (28, 25),
            (28, 26),
            (28, 27),
            (28, 28),
            (28, 29),
            (28, 30),
            (28, 31),
            (28, 32),
            (28, 33),
            (28, 34),
            (28, 35),
            (28, 36),
            (28, 37),
            (28, 38),
            (28, 39),
            (28, 40),
            (28, 41),
            (28, 42),
            (28, 43),
            (28, 44),
            (28, 45),
            (28, 46),
            (28, 47),
            (28, 48),
            (28, 49),
            (28, 50),
            (28, 51),
            (28, 52),
            (28, 53),
            (28, 54),
            (28, 55),
            (29, 26),
            (29, 27),
            (29, 28),
            (29, 29),
            (29, 30),
            (29, 31),
            (29, 32),
            (29, 33),
            (29, 34),
            (29, 35),
            (29, 36),
            (29, 37),
            (29, 38),
            (29, 39),
            (29, 40),
            (29, 41),
            (29, 42),
            (29, 43),
            (29, 44),
            (29, 45),
            (29, 46),
            (29, 47),
            (29, 48),
            (29, 49),
            (29, 50),
            (29, 51),
            (29, 52),
            (29, 53),
            (29, 54),
            (29, 55),
            (29, 56),
            (29, 57),
            (30, 26),
            (30, 27),
            (30, 28),
            (30, 29),
            (30, 30),
            (30, 31),
            (30, 32),
            (30, 33),
            (30, 34),
            (30, 35),
            (30, 36),
            (30, 37),
            (30, 38),
            (30, 39),
            (30, 40),
            (30, 41),
            (30, 42),
            (30, 43),
            (30, 44),
            (30, 45),
            (30, 46),
            (30, 47),
            (30, 48),
            (30, 49),
            (30, 50),
            (30, 51),
            (30, 52),
            (30, 53),
            (30, 54),
            (30, 55),
            (30, 56),
            (30, 57),
            (30, 58),
            (30, 59),
            (31, 27),
            (31, 28),
            (31, 29),
            (31, 30),
            (31, 31),
            (31, 32),
            (31, 33),
            (31, 34),
            (31, 35),
            (31, 36),
            (31, 37),
            (31, 38),
            (31, 39),
            (31, 40),
            (31, 41),
            (31, 42),
            (31, 43),
            (31, 44),
            (31, 45),
            (31, 46),
            (31, 47),
            (31, 48),
            (31, 49),
            (31, 50),
            (31, 51),
            (31, 52),
            (31, 53),
            (31, 54),
            (31, 55),
            (31, 56),
            (31, 57),
            (31, 58),
            (31, 59),
            (31, 60),
            (31, 61),
            (32, 28),
            (32, 29),
            (32, 30),
            (32, 31),
            (32, 32),
            (32, 33),
            (32, 34),
            (32, 35),
            (32, 36),
            (32, 37),
            (32, 38),
            (32, 39),
            (32, 40),
            (32, 41),
            (32, 42),
            (32, 43),
            (32, 44),
            (32, 45),
            (32, 46),
            (32, 47),
            (32, 48),
            (32, 49),
            (32, 50),
            (32, 51),
            (32, 52),
            (32, 53),
            (32, 54),
            (32, 55),
            (32, 56),
            (32, 57),
            (32, 58),
            (32, 59),
            (32, 60),
            (32, 61),
            (32, 62),
            (32, 63),
            (33, 29),
            (33, 30),
            (33, 31),
            (33, 32),
            (33, 33),
            (33, 34),
            (33, 35),
            (33, 36),
            (33, 37),
            (33, 38),
            (33, 39),
            (33, 40),
            (33, 41),
            (33, 42),
            (33, 43),
            (33, 44),
            (33, 45),
            (33, 46),
            (33, 47),
            (33, 48),
            (33, 49),
            (33, 50),
            (33, 51),
            (33, 52),
            (33, 53),
            (33, 54),
            (33, 55),
            (33, 56),
            (33, 57),
            (33, 58),
            (33, 59),
            (33, 60),
            (33, 61),
            (33, 62),
            (33, 63),
            (34, 30),
            (34, 31),
            (34, 32),
            (34, 33),
            (34, 34),
            (34, 35),
            (34, 36),
            (34, 37),
            (34, 38),
            (34, 39),
            (34, 40),
            (34, 41),
            (34, 42),
            (34, 43),
            (34, 44),
            (34, 45),
            (34, 46),
            (34, 47),
            (34, 48),
            (34, 49),
            (34, 50),
            (34, 51),
            (34, 52),
            (34, 53),
            (34, 54),
            (34, 55),
            (34, 56),
            (34, 57),
            (34, 58),
            (34, 59),
            (34, 60),
            (34, 61),
            (34, 62),
            (34, 63),
            (35, 31),
            (35, 32),
            (35, 33),
            (35, 34),
            (35, 35),
            (35, 36),
            (35, 37),
            (35, 38),
            (35, 39),
            (35, 40),
            (35, 41),
            (35, 42),
            (35, 43),
            (35, 44),
            (35, 45),
            (35, 46),
            (35, 47),
            (35, 48),
            (35, 49),
            (35, 50),
            (35, 51),
            (35, 52),
            (35, 53),
            (35, 54),
            (35, 55),
            (35, 56),
            (35, 57),
            (35, 58),
            (35, 59),
            (35, 60),
            (35, 61),
            (35, 62),
            (35, 63),
            (36, 32),
            (36, 33),
            (36, 34),
            (36, 35),
            (36, 36),
            (36, 37),
            (36, 38),
            (36, 39),
            (36, 40),
            (36, 41),
            (36, 42),
            (36, 43),
            (36, 44),
            (36, 45),
            (36, 46),
            (36, 47),
            (36, 48),
            (36, 49),
            (36, 50),
            (36, 51),
            (36, 52),
            (36, 53),
            (36, 54),
            (36, 55),
            (36, 56),
            (36, 57),
            (36, 58),
            (36, 59),
            (36, 60),
            (36, 61),
            (36, 62),
            (36, 63),
            (37, 33),
            (37, 34),
            (37, 35),
            (37, 36),
            (37, 37),
            (37, 38),
            (37, 39),
            (37, 40),
            (37, 41),
            (37, 42),
            (37, 43),
            (37, 44),
            (37, 45),
            (37, 46),
            (37, 47),
            (37, 48),
            (37, 49),
            (37, 50),
            (37, 51),
            (37, 52),
            (37, 53),
            (37, 54),
            (37, 55),
            (37, 56),
            (37, 57),
            (37, 58),
            (37, 59),
            (37, 60),
            (37, 61),
            (37, 62),
            (37, 63),
            (38, 34),
            (38, 35),
            (38, 36),
            (38, 37),
            (38, 38),
            (38, 39),
            (38, 40),
            (38, 41),
            (38, 42),
            (38, 43),
            (38, 44),
            (38, 45),
            (38, 46),
            (38, 47),
            (38, 48),
            (38, 49),
            (38, 50),
            (38, 51),
            (38, 52),
            (38, 53),
            (38, 54),
            (38, 55),
            (38, 56),
            (38, 57),
            (38, 58),
            (38, 59),
            (38, 60),
            (38, 61),
            (38, 62),
            (38, 63),
            (39, 34),
            (39, 35),
            (39, 36),
            (39, 37),
            (39, 38),
            (39, 39),
            (39, 40),
            (39, 41),
            (39, 42),
            (39, 43),
            (39, 44),
            (39, 45),
            (39, 46),
            (39, 47),
            (39, 48),
            (39, 49),
            (39, 50),
            (39, 51),
            (39, 52),
            (39, 53),
            (39, 54),
            (39, 55),
            (39, 56),
            (39, 57),
            (39, 58),
            (39, 59),
            (39, 60),
            (39, 61),
            (39, 62),
            (39, 63),
            (40, 35),
            (40, 36),
            (40, 37),
            (40, 38),
            (40, 39),
            (40, 40),
            (40, 41),
            (40, 42),
            (40, 43),
            (40, 44),
            (40, 45),
            (40, 46),
            (40, 47),
            (40, 48),
            (40, 49),
            (40, 50),
            (40, 51),
            (40, 52),
            (40, 53),
            (40, 54),
            (40, 55),
            (40, 56),
            (40, 57),
            (40, 58),
            (40, 59),
            (40, 60),
            (40, 61),
            (40, 62),
            (40, 63),
            (41, 36),
            (41, 37),
            (41, 38),
            (41, 39),
            (41, 40),
            (41, 41),
            (41, 42),
            (41, 43),
            (41, 44),
            (41, 45),
            (41, 46),
            (41, 47),
            (41, 48),
            (41, 49),
            (41, 50),
            (41, 51),
            (41, 52),
            (41, 53),
            (41, 54),
            (41, 55),
            (41, 56),
            (41, 57),
            (41, 58),
            (41, 59),
            (41, 60),
            (41, 61),
            (41, 62),
            (41, 63),
            (42, 37),
            (42, 38),
            (42, 39),
            (42, 40),
            (42, 41),
            (42, 42),
            (42, 43),
            (42, 44),
            (42, 45),
            (42, 46),
            (42, 47),
            (42, 48),
            (42, 49),
            (42, 50),
            (42, 51),
            (42, 52),
            (42, 53),
            (42, 54),
            (42, 55),
            (42, 56),
            (42, 57),
            (42, 58),
            (42, 59),
            (42, 60),
            (42, 61),
            (42, 62),
            (42, 63),
            (43, 38),
            (43, 39),
            (43, 40),
            (43, 41),
            (43, 42),
            (43, 43),
            (43, 44),
            (43, 45),
            (43, 46),
            (43, 47),
            (43, 48),
            (43, 49),
            (43, 50),
            (43, 51),
            (43, 52),
            (43, 53),
            (43, 54),
            (43, 55),
            (43, 56),
            (43, 57),
            (43, 58),
            (43, 59),
            (43, 60),
            (43, 61),
            (43, 62),
            (43, 63),
            (44, 39),
            (44, 40),
            (44, 41),
            (44, 42),
            (44, 43),
            (44, 44),
            (44, 45),
            (44, 46),
            (44, 47),
            (44, 48),
            (44, 49),
            (44, 50),
            (44, 51),
            (44, 52),
            (44, 53),
            (44, 54),
            (44, 55),
            (44, 56),
            (44, 57),
            (44, 58),
            (44, 59),
            (44, 60),
            (44, 61),
            (44, 62),
            (44, 63),
            (45, 40),
            (45, 41),
            (45, 42),
            (45, 43),
            (45, 44),
            (45, 45),
            (45, 46),
            (45, 47),
            (45, 48),
            (45, 49),
            (45, 50),
            (45, 51),
            (45, 52),
            (45, 53),
            (45, 54),
            (45, 55),
            (45, 56),
            (45, 57),
            (45, 58),
            (45, 59),
            (45, 60),
            (45, 61),
            (45, 62),
            (45, 63),
            (46, 41),
            (46, 42),
            (46, 43),
            (46, 44),
            (46, 45),
            (46, 46),
            (46, 47),
            (46, 48),
            (46, 49),
            (46, 50),
            (46, 51),
            (46, 52),
            (46, 53),
            (46, 54),
            (46, 55),
            (46, 56),
            (46, 57),
            (46, 58),
            (46, 59),
            (46, 60),
            (46, 61),
            (46, 62),
            (46, 63),
            (47, 41),
            (47, 42),
            (47, 43),
            (47, 44),
            (47, 45),
            (47, 46),
            (47, 47),
            (47, 48),
            (47, 49),
            (47, 50),
            (47, 51),
            (47, 52),
            (47, 53),
            (47, 54),
            (47, 55),
            (47, 56),
            (47, 57),
            (47, 58),
            (47, 59),
            (47, 60),
            (47, 61),
            (47, 62),
            (47, 63),
            (48, 42),
            (48, 43),
            (48, 44),
            (48, 45),
            (48, 46),
            (48, 47),
            (48, 48),
            (48, 49),
            (48, 50),
            (48, 51),
            (48, 52),
            (48, 53),
            (48, 54),
            (48, 55),
            (48, 56),
            (48, 57),
            (48, 58),
            (48, 59),
            (48, 60),
            (48, 61),
            (48, 62),
            (48, 63),
            (49, 43),
            (49, 44),
            (49, 45),
            (49, 46),
            (49, 47),
            (49, 48),
            (49, 49),
            (49, 50),
            (49, 51),
            (49, 52),
            (49, 53),
            (49, 54),
            (49, 55),
            (49, 56),
            (49, 57),
            (49, 58),
            (49, 59),
            (49, 60),
            (49, 61),
            (49, 62),
            (49, 63),
            (50, 44),
            (50, 45),
            (50, 46),
            (50, 47),
            (50, 48),
            (50, 49),
            (50, 50),
            (50, 51),
            (50, 52),
            (50, 53),
            (50, 54),
            (50, 55),
            (50, 56),
            (50, 57),
            (50, 58),
            (50, 59),
            (50, 60),
            (50, 61),
            (50, 62),
            (50, 63),
            (51, 45),
            (51, 46),
            (51, 47),
            (51, 48),
            (51, 49),
            (51, 50),
            (51, 51),
            (51, 52),
            (51, 53),
            (51, 54),
            (51, 55),
            (51, 56),
            (51, 57),
            (51, 58),
            (51, 59),
            (51, 60),
            (51, 61),
            (51, 62),
            (51, 63),
            (52, 46),
            (52, 47),
            (52, 48),
            (52, 49),
            (52, 50),
            (52, 51),
            (52, 52),
            (52, 53),
            (52, 54),
            (52, 55),
            (52, 56),
            (52, 57),
            (52, 58),
            (52, 59),
            (52, 60),
            (52, 61),
            (52, 62),
            (52, 63),
            (53, 47),
            (53, 48),
            (53, 49),
            (53, 50),
            (53, 51),
            (53, 52),
            (53, 53),
            (53, 54),
            (53, 55),
            (53, 56),
            (53, 57),
            (53, 58),
            (53, 59),
            (53, 60),
            (53, 61),
            (53, 62),
            (53, 63),
            (54, 48),
            (54, 49),
            (54, 50),
            (54, 51),
            (54, 52),
            (54, 53),
            (54, 54),
            (54, 55),
            (54, 56),
            (54, 57),
            (54, 58),
            (54, 59),
            (54, 60),
            (54, 61),
            (54, 62),
            (54, 63),
            (55, 49),
            (55, 50),
            (55, 51),
            (55, 52),
            (55, 53),
            (55, 54),
            (55, 55),
            (55, 56),
            (55, 57),
            (55, 58),
            (55, 59),
            (55, 60),
            (55, 61),
            (55, 62),
            (55, 63),
            (56, 49),
            (56, 50),
            (56, 51),
            (56, 52),
            (56, 53),
            (56, 54),
            (56, 55),
            (56, 56),
            (56, 57),
            (56, 58),
            (56, 59),
            (56, 60),
            (56, 61),
            (56, 62),
            (56, 63),
            (57, 50),
            (57, 51),
            (57, 52),
            (57, 53),
            (57, 54),
            (57, 55),
            (57, 56),
            (57, 57),
            (57, 58),
            (57, 59),
            (57, 60),
            (57, 61),
            (57, 62),
            (57, 63),
            (58, 51),
            (58, 52),
            (58, 53),
            (58, 54),
            (58, 55),
            (58, 56),
            (58, 57),
            (58, 58),
            (58, 59),
            (58, 60),
            (58, 61),
            (58, 62),
            (58, 63),
            (59, 52),
            (59, 53),
            (59, 54),
            (59, 55),
            (59, 56),
            (59, 57),
            (59, 58),
            (59, 59),
            (59, 60),
            (59, 61),
            (59, 62),
            (59, 63),
            (60, 53),
            (60, 54),
            (60, 55),
            (60, 56),
            (60, 57),
            (60, 58),
            (60, 59),
            (60, 60),
            (60, 61),
            (60, 62),
            (60, 63),
            (61, 54),
            (61, 55),
            (61, 56),
            (61, 57),
            (61, 58),
            (61, 59),
            (61, 60),
            (61, 61),
            (61, 62),
            (61, 63),
            (62, 55),
            (62, 56),
            (62, 57),
            (62, 58),
            (62, 59),
            (62, 60),
            (62, 61),
            (62, 62),
            (62, 63),
            (63, 56),
            (63, 57),
            (63, 58),
            (63, 59),
            (63, 60),
            (63, 61),
            (63, 62),
            (63, 63),
        }:
            return 15
        elif key in {(0, 0)}:
            return 5
        elif key in {(0, 1)}:
            return 10
        elif key in {(1, 0)}:
            return 14
        elif key in {(1, 1)}:
            return 12
        return 0

    num_mlp_3_1_outputs = [
        num_mlp_3_1(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_2_3_outputs)
    ]
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


print(run(["<s>", "0", "2", "5", "2", "4", "3", "5", "4", "5", "4", "5", "0", "5"]))
