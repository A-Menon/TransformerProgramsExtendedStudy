import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys[: i + 1]) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys[: i + 1]] for i, q in enumerate(queries)]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/rasp/dyck2/k16_len16_L3_H4_M4/s0/dyck2_weights.csv",
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
        if q_position in {0, 14}:
            return k_position == 13
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
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
        elif q_position in {15}:
            return k_position == 14

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 2, 15}:
            return k_position == 14
        elif q_position in {1, 4, 14}:
            return k_position == 1
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

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"{", "("}:
            return position == 6
        elif token in {")", "}"}:
            return position == 0
        elif token in {"<s>"}:
            return position == 2

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
        if q_token in {")", "("}:
            return k_token == ")"
        elif q_token in {"{", "}", "<s>"}:
            return k_token == "}"

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, token):
        key = (attn_0_1_output, token)
        if key in {("(", ")"), ("{", "}")}:
            return 7
        elif key in {("(", "}"), (")", "<s>"), (")", "}"), ("}", "<s>"), ("}", "}")}:
            return 5
        elif key in {(")", ")"), ("{", ")"), ("}", ")")}:
            return 2
        elif key in {("<s>", ")"), ("<s>", "<s>"), ("<s>", "}")}:
            return 8
        elif key in {(")", "("), (")", "{"), ("}", "("), ("}", "{")}:
            return 0
        return 3

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, tokens)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_0_output, token):
        key = (attn_0_0_output, token)
        if key in {("(", ")"), (")", "("), (")", "{"), ("}", "("), ("}", "{")}:
            return 12
        elif key in {("(", "}"), ("<s>", ")"), ("<s>", "}"), ("{", ")")}:
            return 11
        elif key in {
            (")", ")"),
            (")", "<s>"),
            (")", "}"),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "}"),
        }:
            return 2
        elif key in {("{", "}")}:
            return 7
        return 3

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_0_outputs, tokens)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output):
        key = num_attn_0_1_output
        return 14

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(one):
        key = one
        return 3

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in ones]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 14}:
            return k_position == 13
        elif q_position in {1, 15}:
            return k_position == 1
        elif q_position in {2, 13}:
            return k_position == 2
        elif q_position in {3, 4, 5}:
            return k_position == 3
        elif q_position in {8, 11, 6, 7}:
            return k_position == 5
        elif q_position in {9}:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {12}:
            return k_position == 11

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 15}:
            return k_position == 13
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {3, 4}:
            return k_position == 3
        elif q_position in {5, 6}:
            return k_position == 5
        elif q_position in {8, 10, 14, 7}:
            return k_position == 7
        elif q_position in {9, 11}:
            return k_position == 9
        elif q_position in {12, 13}:
            return k_position == 11

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(num_mlp_0_1_output, mlp_0_0_output):
        if num_mlp_0_1_output in {0, 1, 4, 5, 7, 8, 9, 10, 12, 14, 15}:
            return mlp_0_0_output == 2
        elif num_mlp_0_1_output in {2, 3, 6, 11, 13}:
            return mlp_0_0_output == 5

    num_attn_1_0_pattern = select(
        mlp_0_0_outputs, num_mlp_0_1_outputs, num_predicate_1_0
    )
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(token, mlp_0_0_output):
        if token in {"{", "("}:
            return mlp_0_0_output == 7
        elif token in {")", "}"}:
            return mlp_0_0_output == 3
        elif token in {"<s>"}:
            return mlp_0_0_output == 8

    num_attn_1_1_pattern = select(mlp_0_0_outputs, tokens, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_1_output, position):
        key = (attn_1_1_output, position)
        if key in {
            (0, 9),
            (0, 11),
            (0, 13),
            (1, 0),
            (1, 3),
            (1, 9),
            (1, 11),
            (1, 13),
            (1, 15),
            (2, 3),
            (2, 9),
            (2, 11),
            (2, 13),
            (2, 15),
            (3, 0),
            (3, 3),
            (3, 9),
            (3, 11),
            (3, 13),
            (4, 3),
            (4, 9),
            (4, 11),
            (4, 13),
            (4, 15),
            (5, 3),
            (5, 9),
            (5, 11),
            (5, 13),
            (6, 3),
            (6, 9),
            (6, 11),
            (6, 13),
            (6, 15),
            (7, 3),
            (7, 9),
            (7, 11),
            (7, 13),
            (8, 3),
            (8, 9),
            (8, 11),
            (8, 13),
            (8, 15),
            (9, 0),
            (9, 3),
            (9, 5),
            (9, 9),
            (9, 11),
            (9, 13),
            (9, 15),
            (10, 3),
            (10, 9),
            (10, 11),
            (10, 13),
            (11, 9),
            (11, 11),
            (12, 9),
            (12, 11),
            (12, 13),
            (13, 9),
            (13, 11),
            (13, 13),
            (14, 9),
            (14, 11),
            (14, 13),
            (15, 3),
            (15, 9),
            (15, 11),
            (15, 13),
        }:
            return 3
        elif key in {
            (0, 1),
            (0, 6),
            (1, 1),
            (2, 1),
            (3, 7),
            (4, 1),
            (4, 6),
            (4, 8),
            (5, 1),
            (6, 12),
            (8, 1),
            (8, 8),
            (9, 1),
            (9, 12),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (11, 7),
            (11, 8),
            (11, 10),
            (11, 12),
            (11, 13),
            (11, 14),
            (11, 15),
            (12, 1),
            (12, 6),
            (13, 1),
            (13, 7),
            (14, 1),
            (14, 6),
            (15, 7),
        }:
            return 13
        elif key in {
            (2, 6),
            (2, 8),
            (5, 6),
            (5, 8),
            (6, 0),
            (6, 2),
            (6, 4),
            (6, 5),
            (6, 6),
            (6, 7),
            (6, 8),
            (8, 6),
            (10, 0),
            (10, 6),
            (10, 7),
            (10, 8),
            (14, 7),
            (15, 6),
        }:
            return 8
        elif key in {
            (2, 4),
            (2, 10),
            (2, 12),
            (2, 14),
            (3, 2),
            (3, 5),
            (3, 10),
            (3, 12),
            (3, 14),
            (10, 4),
            (15, 4),
        }:
            return 2
        elif key in {
            (0, 3),
            (1, 4),
            (3, 1),
            (3, 4),
            (3, 6),
            (7, 4),
            (9, 4),
            (12, 3),
            (13, 3),
            (13, 4),
            (14, 3),
        }:
            return 12
        elif key in {(0, 7), (1, 7), (4, 7), (7, 7), (7, 15), (9, 7), (12, 7)}:
            return 15
        elif key in {(1, 6), (7, 1), (7, 6), (9, 6), (13, 6)}:
            return 4
        elif key in {(7, 8), (7, 12), (12, 8), (14, 8)}:
            return 0
        elif key in {(4, 4), (7, 10), (7, 14)}:
            return 7
        elif key in {(3, 8)}:
            return 6
        return 5

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, positions)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_0_output, attn_1_1_output):
        key = (attn_0_0_output, attn_1_1_output)
        if key in {
            (")", 0),
            (")", 7),
            (")", 12),
            (")", 14),
            ("<s>", 7),
            ("}", 0),
            ("}", 4),
            ("}", 7),
            ("}", 8),
            ("}", 12),
            ("}", 13),
        }:
            return 14
        elif key in {
            ("(", 1),
            ("(", 2),
            ("(", 4),
            ("(", 5),
            ("(", 6),
            ("(", 10),
            ("(", 15),
            ("<s>", 1),
            ("{", 1),
            ("{", 2),
            ("{", 5),
            ("{", 6),
            ("{", 10),
            ("{", 15),
        }:
            return 7
        elif key in {
            ("(", 0),
            ("(", 7),
            ("(", 8),
            ("(", 12),
            ("(", 13),
            ("<s>", 0),
            ("<s>", 4),
            ("<s>", 12),
            ("{", 0),
            ("{", 4),
            ("{", 7),
            ("{", 8),
            ("{", 12),
            ("{", 13),
            ("{", 14),
        }:
            return 3
        elif key in {
            ("(", 3),
            (")", 1),
            (")", 3),
            ("<s>", 3),
            ("{", 3),
            ("}", 1),
            ("}", 3),
        }:
            return 8
        elif key in {("(", 14), ("<s>", 9), ("<s>", 13)}:
            return 4
        elif key in {("{", 9)}:
            return 12
        elif key in {(")", 9), ("}", 9), ("}", 14)}:
            return 1
        elif key in {(")", 4), (")", 13)}:
            return 15
        elif key in {("(", 9)}:
            return 13
        return 5

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_1_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output):
        key = num_attn_1_0_output
        if key in {0}:
            return 4
        return 12

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_1_0_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_1_output, num_attn_1_1_output):
        key = (num_attn_0_1_output, num_attn_1_1_output)
        if key in {
            (2, 0),
            (3, 0),
            (4, 0),
            (4, 1),
            (5, 0),
            (5, 1),
            (5, 2),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 5),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (12, 7),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
            (13, 7),
            (13, 8),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 4),
            (14, 5),
            (14, 6),
            (14, 7),
            (14, 8),
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
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 7),
            (16, 8),
            (16, 9),
            (16, 10),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (17, 7),
            (17, 8),
            (17, 9),
            (17, 10),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (18, 7),
            (18, 8),
            (18, 9),
            (18, 10),
            (18, 11),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 8),
            (19, 9),
            (19, 10),
            (19, 11),
            (20, 0),
            (20, 1),
            (20, 2),
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
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (21, 5),
            (21, 6),
            (21, 7),
            (21, 8),
            (21, 9),
            (21, 10),
            (21, 11),
            (21, 12),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
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
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
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
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
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
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
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
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
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
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
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
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
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
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
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
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
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
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
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
        }:
            return 4
        elif key in {(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4)}:
            return 7
        elif key in {(3, 1), (4, 2), (5, 3), (7, 4), (8, 5), (11, 7)}:
            return 11
        elif key in {(0, 0), (1, 0), (2, 1), (2, 2), (3, 2)}:
            return 12
        elif key in {(4, 3), (5, 4), (6, 4), (7, 5), (9, 6)}:
            return 9
        elif key in {(2, 3), (3, 3), (3, 4), (4, 4)}:
            return 5
        elif key in {(0, 1), (1, 2)}:
            return 10
        elif key in {(1, 1)}:
            return 8
        return 6

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_1_1_output, position):
        if attn_1_1_output in {0}:
            return position == 2
        elif attn_1_1_output in {1, 2, 3, 5, 6, 12}:
            return position == 5
        elif attn_1_1_output in {4}:
            return position == 3
        elif attn_1_1_output in {8, 7}:
            return position == 11
        elif attn_1_1_output in {9, 14}:
            return position == 13
        elif attn_1_1_output in {10}:
            return position == 7
        elif attn_1_1_output in {11}:
            return position == 1
        elif attn_1_1_output in {13}:
            return position == 4
        elif attn_1_1_output in {15}:
            return position == 6

    attn_2_0_pattern = select_closest(positions, attn_1_1_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 3}:
            return k_position == 2
        elif q_position in {9, 7}:
            return k_position == 3

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_1_output, mlp_0_1_output):
        if attn_1_1_output in {0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13}:
            return mlp_0_1_output == 11
        elif attn_1_1_output in {8, 14, 15}:
            return mlp_0_1_output == 5
        elif attn_1_1_output in {9, 10}:
            return mlp_0_1_output == 15

    num_attn_2_0_pattern = select(mlp_0_1_outputs, attn_1_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, mlp_1_1_output):
        if attn_1_1_output in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}:
            return mlp_1_1_output == 5

    num_attn_2_1_pattern = select(mlp_1_1_outputs, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_1_1_output, attn_2_1_output):
        key = (attn_1_1_output, attn_2_1_output)
        if key in {
            (0, 1),
            (0, 9),
            (0, 11),
            (0, 13),
            (1, 0),
            (1, 1),
            (1, 3),
            (1, 9),
            (1, 11),
            (1, 13),
            (3, 1),
            (3, 4),
            (3, 9),
            (3, 11),
            (4, 1),
            (4, 9),
            (4, 11),
            (6, 11),
            (7, 1),
            (7, 3),
            (7, 9),
            (7, 11),
            (7, 13),
            (8, 1),
            (8, 11),
            (9, 1),
            (9, 9),
            (9, 11),
            (9, 13),
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
            (11, 12),
            (11, 13),
            (11, 14),
            (12, 1),
            (12, 3),
            (12, 9),
            (12, 11),
            (12, 13),
            (13, 1),
            (13, 3),
            (13, 9),
            (13, 11),
            (14, 1),
            (14, 11),
        }:
            return 3
        elif key in {
            (0, 5),
            (0, 6),
            (1, 5),
            (1, 14),
            (3, 2),
            (3, 5),
            (3, 6),
            (3, 15),
            (4, 5),
            (5, 0),
            (5, 1),
            (5, 3),
            (5, 5),
            (5, 6),
            (5, 8),
            (5, 9),
            (5, 11),
            (5, 13),
            (5, 14),
            (7, 5),
            (7, 6),
            (8, 5),
            (9, 5),
            (9, 6),
            (12, 5),
            (12, 6),
            (13, 5),
            (13, 14),
            (14, 5),
        }:
            return 13
        elif key in {
            (0, 0),
            (0, 7),
            (0, 12),
            (4, 0),
            (4, 7),
            (4, 12),
            (7, 0),
            (7, 7),
            (7, 12),
            (9, 0),
            (9, 7),
            (9, 12),
            (12, 0),
            (12, 7),
            (12, 12),
            (13, 0),
            (13, 7),
            (13, 12),
        }:
            return 9
        elif key in {
            (0, 3),
            (1, 4),
            (1, 10),
            (1, 12),
            (3, 3),
            (4, 3),
            (4, 13),
            (8, 3),
            (8, 9),
            (9, 3),
            (9, 10),
            (13, 4),
            (14, 3),
        }:
            return 4
        elif key in {
            (0, 8),
            (1, 7),
            (1, 8),
            (3, 0),
            (3, 7),
            (3, 8),
            (3, 12),
            (3, 13),
            (3, 14),
            (7, 8),
            (9, 8),
            (12, 8),
            (13, 8),
        }:
            return 10
        elif key in {
            (1, 6),
            (2, 1),
            (2, 3),
            (3, 10),
            (6, 1),
            (6, 3),
            (10, 3),
            (13, 6),
            (15, 3),
        }:
            return 0
        elif key in {(8, 13), (13, 13), (14, 13)}:
            return 12
        return 11

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_2_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_1_output):
        key = attn_2_1_output
        if key in {0, 8, 12, 14}:
            return 3
        elif key in {2, 5, 11}:
            return 8
        elif key in {6, 15}:
            return 6
        elif key in {7, 13}:
            return 13
        return 10

    mlp_2_1_outputs = [mlp_2_1(k0) for k0 in attn_2_1_outputs]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_0_output):
        key = num_attn_2_0_output
        if key in {0}:
            return 4
        return 14

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_2_0_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_0_output, num_attn_2_0_output):
        key = (num_attn_1_0_output, num_attn_2_0_output)
        if key in {
            (1, 0),
            (2, 0),
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
            (11, 2),
            (12, 0),
            (12, 1),
            (12, 2),
            (13, 0),
            (13, 1),
            (13, 2),
            (14, 0),
            (14, 1),
            (14, 2),
            (15, 0),
            (15, 1),
            (15, 2),
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
            (22, 4),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (24, 4),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (25, 4),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (26, 4),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (27, 5),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (28, 5),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
            (29, 5),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (30, 5),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
            (31, 5),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 4),
            (32, 5),
            (32, 6),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (33, 5),
            (33, 6),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 4),
            (35, 5),
            (35, 6),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (36, 5),
            (36, 6),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 5),
            (37, 6),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (38, 5),
            (38, 6),
            (38, 7),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (40, 5),
            (40, 6),
            (40, 7),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (41, 6),
            (41, 7),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (42, 5),
            (42, 6),
            (42, 7),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (43, 4),
            (43, 5),
            (43, 6),
            (43, 7),
            (43, 8),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (44, 4),
            (44, 5),
            (44, 6),
            (44, 7),
            (44, 8),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (45, 4),
            (45, 5),
            (45, 6),
            (45, 7),
            (45, 8),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (46, 5),
            (46, 6),
            (46, 7),
            (46, 8),
            (47, 0),
            (47, 1),
            (47, 2),
            (47, 3),
            (47, 4),
            (47, 5),
            (47, 6),
            (47, 7),
            (47, 8),
        }:
            return 12
        elif key in {(0, 0)}:
            return 1
        return 14

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_2_0_outputs)
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
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
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


print(
    run(
        [
            "<s>",
            ")",
            ")",
            "}",
            "}",
            "(",
            "{",
            "}",
            "}",
            "{",
            "{",
            ")",
            "{",
            "{",
            ")",
            ")",
        ]
    )
)
