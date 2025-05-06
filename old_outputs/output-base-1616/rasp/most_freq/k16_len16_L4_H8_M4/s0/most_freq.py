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
        "output-base-1616/rasp/most_freq/k16_len16_L4_H8_M4/s0/most_freq_weights.csv",
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
        if q_position in {0, 12}:
            return k_position == 9
        elif q_position in {1, 4}:
            return k_position == 2
        elif q_position in {2, 13}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {9, 14, 7}:
            return k_position == 10
        elif q_position in {8}:
            return k_position == 12
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 14

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1, 4}:
            return k_position == 3
        elif q_position in {2, 12}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6, 7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {10, 11}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 6
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 7

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 5}:
            return k_position == 3
        elif q_position in {1, 4}:
            return k_position == 5
        elif q_position in {2, 11, 7}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {13, 6}:
            return k_position == 9
        elif q_position in {8, 14}:
            return k_position == 11
        elif q_position in {9, 12}:
            return k_position == 13
        elif q_position in {10}:
            return k_position == 14
        elif q_position in {15}:
            return k_position == 12

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 9
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9, 10}:
            return k_position == 11
        elif q_position in {11, 12}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 13

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"9", "4", "12", "6", "0"}:
            return position == 0
        elif token in {"1", "10"}:
            return position == 5
        elif token in {"11"}:
            return position == 2
        elif token in {"2", "13", "7"}:
            return position == 7
        elif token in {"5", "3"}:
            return position == 8
        elif token in {"8"}:
            return position == 1
        elif token in {"<s>"}:
            return position == 4

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"0"}:
            return position == 6
        elif token in {"1", "9", "3", "2", "13", "7"}:
            return position == 0
        elif token in {"<s>", "10"}:
            return position == 5
        elif token in {"11", "6"}:
            return position == 1
        elif token in {"5", "12"}:
            return position == 7
        elif token in {"8", "4"}:
            return position == 4

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"3", "4", "8", "13", "0"}:
            return position == 0
        elif token in {"1"}:
            return position == 1
        elif token in {"<s>", "10"}:
            return position == 6
        elif token in {"11", "9", "5", "12", "6"}:
            return position == 7
        elif token in {"2"}:
            return position == 2
        elif token in {"7"}:
            return position == 9

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"5", "13", "1", "0"}:
            return position == 0
        elif token in {"<s>", "12", "10"}:
            return position == 4
        elif token in {"11", "4"}:
            return position == 7
        elif token in {"8", "2"}:
            return position == 8
        elif token in {"3"}:
            return position == 2
        elif token in {"6"}:
            return position == 1
        elif token in {"7"}:
            return position == 5
        elif token in {"9"}:
            return position == 6

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position, attn_0_0_output):
        key = (position, attn_0_0_output)
        if key in {
            (2, "0"),
            (2, "1"),
            (2, "10"),
            (2, "12"),
            (2, "13"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "6"),
            (2, "7"),
            (2, "9"),
            (2, "<s>"),
            (3, "0"),
            (3, "1"),
            (3, "10"),
            (3, "11"),
            (3, "12"),
            (3, "13"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "6"),
            (3, "7"),
            (3, "8"),
            (3, "9"),
            (3, "<s>"),
            (4, "0"),
            (4, "1"),
            (4, "10"),
            (4, "12"),
            (4, "13"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "6"),
            (4, "7"),
            (4, "9"),
            (4, "<s>"),
            (5, "0"),
            (5, "1"),
            (5, "10"),
            (5, "12"),
            (5, "13"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "5"),
            (5, "6"),
            (5, "7"),
            (5, "8"),
            (5, "9"),
            (5, "<s>"),
            (6, "0"),
            (6, "1"),
            (6, "10"),
            (6, "12"),
            (6, "13"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "5"),
            (6, "6"),
            (6, "7"),
            (6, "8"),
            (6, "9"),
            (6, "<s>"),
        }:
            return 2
        elif key in {
            (7, "11"),
            (7, "2"),
            (7, "4"),
            (7, "7"),
            (8, "0"),
            (8, "1"),
            (8, "10"),
            (8, "11"),
            (8, "12"),
            (8, "13"),
            (8, "2"),
            (8, "3"),
            (8, "4"),
            (8, "5"),
            (8, "6"),
            (8, "7"),
            (8, "9"),
            (8, "<s>"),
        }:
            return 5
        elif key in {
            (0, "0"),
            (0, "1"),
            (0, "10"),
            (0, "11"),
            (0, "12"),
            (0, "13"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (0, "6"),
            (0, "7"),
            (0, "8"),
            (0, "9"),
            (0, "<s>"),
        }:
            return 9
        elif key in {
            (9, "0"),
            (9, "1"),
            (9, "10"),
            (9, "11"),
            (9, "12"),
            (9, "13"),
            (9, "2"),
            (9, "3"),
            (9, "4"),
            (9, "5"),
            (9, "6"),
            (9, "7"),
            (9, "8"),
            (9, "9"),
            (9, "<s>"),
        }:
            return 12
        elif key in {
            (7, "0"),
            (7, "1"),
            (7, "10"),
            (7, "12"),
            (7, "13"),
            (7, "3"),
            (7, "5"),
            (7, "6"),
            (7, "9"),
            (7, "<s>"),
        }:
            return 8
        elif key in {(1, "1"), (1, "12"), (1, "3"), (1, "6"), (1, "9")}:
            return 0
        elif key in {(2, "11"), (4, "11"), (4, "2"), (5, "11"), (6, "11")}:
            return 15
        elif key in {(1, "0"), (1, "10"), (1, "11")}:
            return 6
        elif key in {(1, "4"), (1, "5"), (1, "<s>")}:
            return 14
        elif key in {(2, "8"), (4, "8")}:
            return 3
        elif key in {(1, "2"), (1, "8")}:
            return 4
        elif key in {(2, "5"), (3, "5")}:
            return 10
        elif key in {(7, "8"), (8, "8")}:
            return 13
        elif key in {(1, "7")}:
            return 1
        elif key in {(1, "13")}:
            return 7
        return 11

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(positions, attn_0_0_outputs)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(token, position):
        key = (token, position)
        if key in {
            ("10", 2),
            ("10", 3),
            ("10", 4),
            ("11", 2),
            ("11", 3),
            ("11", 4),
            ("12", 2),
            ("12", 3),
            ("12", 4),
            ("13", 2),
            ("13", 3),
            ("2", 2),
            ("2", 3),
            ("3", 2),
            ("3", 3),
            ("4", 2),
            ("4", 3),
            ("5", 2),
            ("5", 3),
            ("6", 2),
            ("6", 3),
            ("7", 3),
            ("8", 2),
            ("8", 3),
            ("9", 2),
            ("9", 3),
            ("9", 4),
            ("<s>", 2),
            ("<s>", 3),
        }:
            return 2
        elif key in {
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("1", 4),
            ("13", 4),
            ("2", 4),
            ("3", 4),
            ("4", 4),
            ("5", 4),
            ("6", 4),
            ("7", 4),
            ("8", 4),
            ("<s>", 4),
        }:
            return 0
        elif key in {("1", 2), ("1", 3), ("10", 1)}:
            return 4
        elif key in {("12", 1), ("8", 1), ("9", 1)}:
            return 5
        elif key in {("1", 1), ("11", 1), ("<s>", 1)}:
            return 9
        elif key in {("13", 1), ("5", 1)}:
            return 6
        elif key in {("4", 1), ("7", 1)}:
            return 8
        elif key in {("0", 1), ("3", 1)}:
            return 10
        elif key in {("6", 1)}:
            return 7
        elif key in {("2", 1)}:
            return 11
        elif key in {("7", 2)}:
            return 13
        return 1

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(tokens, positions)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {0}:
            return 0
        return 5

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output, num_attn_0_2_output):
        key = (num_attn_0_3_output, num_attn_0_2_output)
        if key in {(0, 0)}:
            return 15
        return 9

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 15}:
            return k_position == 3
        elif q_position in {1, 12, 6}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 0
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5, 14}:
            return k_position == 4
        elif q_position in {11, 13, 7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 15
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 14

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 8
        elif q_position in {3}:
            return k_position == 11
        elif q_position in {4, 14}:
            return k_position == 5
        elif q_position in {5, 15}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9, 10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {12, 13}:
            return k_position == 15

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 9
        elif q_position in {2, 6}:
            return k_position == 4
        elif q_position in {3, 4}:
            return k_position == 3
        elif q_position in {11, 5}:
            return k_position == 0
        elif q_position in {8}:
            return k_position == 11
        elif q_position in {9}:
            return k_position == 14
        elif q_position in {10}:
            return k_position == 8
        elif q_position in {12, 14}:
            return k_position == 10
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 13

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_2_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10, 13}:
            return k_position == 13
        elif q_position in {11, 12}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 14
        elif q_position in {15}:
            return k_position == 15

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(q_token, k_token):
        if q_token in {"7", "9", "3", "12", "0"}:
            return k_token == "<pad>"
        elif q_token in {"1", "8", "5", "2", "6"}:
            return k_token == "<s>"
        elif q_token in {"10"}:
            return k_token == "10"
        elif q_token in {"11"}:
            return k_token == "11"
        elif q_token in {"13"}:
            return k_token == "13"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == "0"

    num_attn_1_0_pattern = select(tokens, tokens, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_3_output, token):
        if attn_0_3_output in {"0"}:
            return token == "0"
        elif attn_0_3_output in {"11", "1", "9", "5", "12", "2", "7"}:
            return token == "<s>"
        elif attn_0_3_output in {"10"}:
            return token == "10"
        elif attn_0_3_output in {"13", "4"}:
            return token == "<pad>"
        elif attn_0_3_output in {"3"}:
            return token == "3"
        elif attn_0_3_output in {"6"}:
            return token == "6"
        elif attn_0_3_output in {"8"}:
            return token == "8"
        elif attn_0_3_output in {"<s>"}:
            return token == "5"

    num_attn_1_1_pattern = select(tokens, attn_0_3_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_3_output, token):
        if attn_0_3_output in {"0"}:
            return token == "0"
        elif attn_0_3_output in {"1", "9", "4", "8", "5", "12", "2", "13", "7", "10"}:
            return token == "<s>"
        elif attn_0_3_output in {"11"}:
            return token == "11"
        elif attn_0_3_output in {"3"}:
            return token == "3"
        elif attn_0_3_output in {"6"}:
            return token == "6"
        elif attn_0_3_output in {"<s>"}:
            return token == "10"

    num_attn_1_2_pattern = select(tokens, attn_0_3_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_token, k_token):
        if q_token in {"11", "<s>", "8", "12", "2", "0", "10"}:
            return k_token == "<s>"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"13"}:
            return k_token == "13"
        elif q_token in {"6", "3", "7", "4"}:
            return k_token == "<pad>"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"9"}:
            return k_token == "9"

    num_attn_1_3_pattern = select(tokens, tokens, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(position, attn_1_2_output):
        key = (position, attn_1_2_output)
        if key in {
            (4, "13"),
            (4, "3"),
            (5, "13"),
            (5, "3"),
            (6, "13"),
            (6, "3"),
            (7, "13"),
            (7, "3"),
            (8, "13"),
            (8, "3"),
            (9, "13"),
            (9, "3"),
            (10, "13"),
            (10, "3"),
            (11, "13"),
            (11, "3"),
        }:
            return 7
        elif key in {
            (1, "11"),
            (5, "5"),
            (5, "8"),
            (6, "5"),
            (6, "8"),
            (7, "5"),
            (7, "8"),
            (8, "5"),
            (8, "8"),
            (9, "5"),
            (9, "8"),
            (10, "5"),
            (10, "8"),
            (11, "5"),
            (11, "8"),
        }:
            return 8
        elif key in {
            (0, "11"),
            (2, "11"),
            (3, "11"),
            (4, "11"),
            (5, "11"),
            (6, "11"),
            (7, "11"),
            (8, "11"),
            (9, "11"),
            (10, "11"),
            (11, "0"),
            (11, "11"),
        }:
            return 5
        elif key in {
            (5, "6"),
            (6, "6"),
            (7, "6"),
            (8, "6"),
            (9, "6"),
            (10, "6"),
            (11, "6"),
        }:
            return 6
        elif key in {(1, "13"), (1, "3"), (1, "4")}:
            return 9
        elif key in {(1, "5"), (1, "6")}:
            return 4
        elif key in {(1, "7")}:
            return 1
        elif key in {(2, "8")}:
            return 2
        elif key in {(1, "8")}:
            return 3
        elif key in {(1, "2")}:
            return 11
        elif key in {(1, "12")}:
            return 13
        return 10

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(positions, attn_1_2_outputs)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_3_output, attn_1_2_output):
        key = (attn_1_3_output, attn_1_2_output)
        if key in {
            ("0", "4"),
            ("0", "9"),
            ("10", "0"),
            ("10", "10"),
            ("10", "12"),
            ("10", "2"),
            ("10", "3"),
            ("10", "4"),
            ("10", "5"),
            ("10", "7"),
            ("10", "9"),
            ("10", "<s>"),
            ("12", "10"),
            ("12", "12"),
            ("12", "9"),
            ("2", "4"),
            ("4", "4"),
            ("4", "9"),
            ("7", "4"),
            ("7", "7"),
            ("8", "10"),
            ("8", "8"),
            ("9", "4"),
            ("9", "9"),
            ("<s>", "0"),
            ("<s>", "10"),
            ("<s>", "12"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "7"),
            ("<s>", "9"),
            ("<s>", "<s>"),
        }:
            return 9
        elif key in {
            ("0", "1"),
            ("1", "0"),
            ("1", "10"),
            ("1", "11"),
            ("1", "12"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "5"),
            ("1", "7"),
            ("1", "8"),
            ("1", "9"),
            ("1", "<s>"),
            ("10", "1"),
            ("11", "1"),
            ("12", "1"),
            ("12", "3"),
            ("12", "8"),
            ("2", "1"),
            ("3", "1"),
            ("3", "8"),
            ("3", "9"),
            ("5", "1"),
            ("5", "9"),
            ("8", "1"),
            ("9", "1"),
        }:
            return 0
        elif key in {
            ("0", "6"),
            ("1", "6"),
            ("10", "6"),
            ("11", "6"),
            ("12", "4"),
            ("12", "6"),
            ("12", "<s>"),
            ("2", "6"),
            ("3", "4"),
            ("3", "6"),
            ("3", "<s>"),
            ("4", "1"),
            ("4", "6"),
            ("4", "<s>"),
            ("5", "6"),
            ("6", "0"),
            ("6", "1"),
            ("6", "3"),
            ("6", "4"),
            ("6", "6"),
            ("6", "8"),
            ("6", "9"),
            ("6", "<s>"),
            ("7", "1"),
            ("7", "6"),
            ("8", "6"),
            ("9", "6"),
            ("<s>", "6"),
        }:
            return 12
        elif key in {
            ("0", "13"),
            ("1", "13"),
            ("11", "13"),
            ("12", "13"),
            ("13", "0"),
            ("13", "1"),
            ("13", "10"),
            ("13", "11"),
            ("13", "12"),
            ("13", "13"),
            ("13", "2"),
            ("13", "3"),
            ("13", "4"),
            ("13", "5"),
            ("13", "6"),
            ("13", "7"),
            ("13", "8"),
            ("13", "9"),
            ("13", "<s>"),
            ("2", "13"),
            ("3", "13"),
            ("4", "13"),
            ("5", "13"),
            ("6", "13"),
            ("7", "13"),
            ("8", "13"),
            ("9", "13"),
        }:
            return 5
        elif key in {
            ("4", "2"),
            ("4", "7"),
            ("6", "10"),
            ("6", "11"),
            ("6", "12"),
            ("6", "2"),
            ("6", "5"),
            ("6", "7"),
        }:
            return 4
        elif key in {("<s>", "1"), ("<s>", "13")}:
            return 3
        elif key in {("10", "13")}:
            return 15
        return 6

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_2_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_3_output):
        key = num_attn_0_3_output
        return 4

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_0_output, num_attn_0_3_output):
        key = (num_attn_1_0_output, num_attn_0_3_output)
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
            (5, 7),
            (5, 8),
            (5, 9),
            (5, 10),
            (5, 11),
            (5, 12),
            (5, 13),
            (5, 14),
            (5, 15),
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
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            (6, 16),
            (6, 17),
            (6, 18),
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
            (7, 10),
            (7, 11),
            (7, 12),
            (7, 13),
            (7, 14),
            (7, 15),
            (7, 16),
            (7, 17),
            (7, 18),
            (7, 19),
            (7, 20),
            (7, 21),
            (7, 22),
            (7, 23),
            (7, 24),
            (7, 25),
            (7, 26),
            (7, 27),
            (7, 28),
            (7, 29),
            (7, 30),
            (7, 31),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 15),
            (8, 16),
            (8, 17),
            (8, 18),
            (8, 19),
            (8, 20),
            (8, 21),
            (8, 22),
            (8, 23),
            (8, 24),
            (8, 25),
            (8, 26),
            (8, 27),
            (8, 28),
            (8, 29),
            (8, 30),
            (8, 31),
            (9, 14),
            (9, 15),
            (9, 16),
            (9, 17),
            (9, 18),
            (9, 19),
            (9, 20),
            (9, 21),
            (9, 22),
            (9, 23),
            (9, 24),
            (9, 25),
            (9, 26),
            (9, 27),
            (9, 28),
            (9, 29),
            (9, 30),
            (9, 31),
            (10, 15),
            (10, 16),
            (10, 17),
            (10, 18),
            (10, 19),
            (10, 20),
            (10, 21),
            (10, 22),
            (10, 23),
            (10, 24),
            (10, 25),
            (10, 26),
            (10, 27),
            (10, 28),
            (10, 29),
            (10, 30),
            (10, 31),
            (11, 17),
            (11, 18),
            (11, 19),
            (11, 20),
            (11, 21),
            (11, 22),
            (11, 23),
            (11, 24),
            (11, 25),
            (11, 26),
            (11, 27),
            (11, 28),
            (11, 29),
            (11, 30),
            (11, 31),
            (12, 18),
            (12, 19),
            (12, 20),
            (12, 21),
            (12, 22),
            (12, 23),
            (12, 24),
            (12, 25),
            (12, 26),
            (12, 27),
            (12, 28),
            (12, 29),
            (12, 30),
            (12, 31),
            (13, 20),
            (13, 21),
            (13, 22),
            (13, 23),
            (13, 24),
            (13, 25),
            (13, 26),
            (13, 27),
            (13, 28),
            (13, 29),
            (13, 30),
            (13, 31),
            (14, 21),
            (14, 22),
            (14, 23),
            (14, 24),
            (14, 25),
            (14, 26),
            (14, 27),
            (14, 28),
            (14, 29),
            (14, 30),
            (14, 31),
            (15, 23),
            (15, 24),
            (15, 25),
            (15, 26),
            (15, 27),
            (15, 28),
            (15, 29),
            (15, 30),
            (15, 31),
            (16, 25),
            (16, 26),
            (16, 27),
            (16, 28),
            (16, 29),
            (16, 30),
            (16, 31),
            (17, 26),
            (17, 27),
            (17, 28),
            (17, 29),
            (17, 30),
            (17, 31),
            (18, 28),
            (18, 29),
            (18, 30),
            (18, 31),
            (19, 29),
            (19, 30),
            (19, 31),
            (20, 31),
        }:
            return 1
        elif key in {
            (2, 0),
            (3, 0),
            (4, 0),
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
            (16, 0),
            (16, 1),
            (17, 0),
            (17, 1),
            (18, 0),
            (18, 1),
            (19, 0),
            (19, 1),
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
            return 14
        elif key in {(0, 0)}:
            return 3
        elif key in {(1, 0)}:
            return 0
        return 6

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(position, attn_1_0_output):
        if position in {0, 1, 14, 15}:
            return attn_1_0_output == "0"
        elif position in {2, 3, 4}:
            return attn_1_0_output == "<s>"
        elif position in {5}:
            return attn_1_0_output == "5"
        elif position in {11, 6, 7}:
            return attn_1_0_output == "7"
        elif position in {8, 10}:
            return attn_1_0_output == "2"
        elif position in {9}:
            return attn_1_0_output == "12"
        elif position in {12, 13}:
            return attn_1_0_output == "9"

    attn_2_0_pattern = select_closest(attn_1_0_outputs, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, token):
        if position in {0, 2}:
            return token == "0"
        elif position in {1}:
            return token == "2"
        elif position in {3, 4}:
            return token == "10"
        elif position in {12, 5}:
            return token == "7"
        elif position in {6}:
            return token == "5"
        elif position in {8, 11, 7}:
            return token == "8"
        elif position in {9, 10}:
            return token == "6"
        elif position in {13}:
            return token == "13"
        elif position in {14, 15}:
            return token == "<s>"

    attn_2_1_pattern = select_closest(tokens, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_0_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {0, 8}:
            return k_position == 12
        elif q_position in {1}:
            return k_position == 7
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5, 6}:
            return k_position == 9
        elif q_position in {7}:
            return k_position == 11
        elif q_position in {9, 10}:
            return k_position == 14
        elif q_position in {11, 12, 13}:
            return k_position == 0
        elif q_position in {14, 15}:
            return k_position == 1

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_0_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 13
        elif q_position in {1, 3}:
            return k_position == 2
        elif q_position in {2, 4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {15, 6, 14}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10, 11, 12}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 6

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, mlp_1_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(position, token):
        if position in {0}:
            return token == "13"
        elif position in {1, 2}:
            return token == "11"
        elif position in {3, 4}:
            return token == "<s>"
        elif position in {5, 6, 7, 8, 9, 10, 11}:
            return token == "1"
        elif position in {12, 13, 14, 15}:
            return token == "<pad>"

    num_attn_2_0_pattern = select(tokens, positions, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, token):
        if position in {0, 1}:
            return token == "1"
        elif position in {2, 3, 4}:
            return token == "<s>"
        elif position in {5, 6, 7, 8, 9, 11}:
            return token == "10"
        elif position in {10, 12, 13, 14, 15}:
            return token == "<pad>"

    num_attn_2_1_pattern = select(tokens, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(q_token, k_token):
        if q_token in {"1", "9", "4", "0", "10"}:
            return k_token == "<pad>"
        elif q_token in {"11", "2"}:
            return k_token == "<s>"
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
        elif q_token in {"<s>"}:
            return k_token == "11"

    num_attn_2_2_pattern = select(tokens, tokens, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 1}:
            return token == "0"
        elif mlp_0_1_output in {2, 5, 10, 11, 14}:
            return token == "<pad>"
        elif mlp_0_1_output in {3, 4, 6, 8, 9, 13}:
            return token == "<s>"
        elif mlp_0_1_output in {7}:
            return token == "6"
        elif mlp_0_1_output in {12}:
            return token == "7"
        elif mlp_0_1_output in {15}:
            return token == "12"

    num_attn_2_3_pattern = select(tokens, mlp_0_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_0_output, position):
        key = (attn_2_0_output, position)
        if key in {
            ("0", 4),
            ("0", 5),
            ("0", 6),
            ("0", 9),
            ("0", 10),
            ("0", 12),
            ("10", 6),
            ("10", 8),
            ("10", 9),
            ("10", 10),
            ("10", 11),
            ("10", 12),
            ("12", 4),
            ("12", 5),
            ("12", 6),
            ("12", 7),
            ("12", 8),
            ("12", 9),
            ("12", 10),
            ("12", 11),
            ("12", 12),
            ("13", 4),
            ("13", 5),
            ("13", 6),
            ("13", 7),
            ("13", 8),
            ("13", 9),
            ("13", 10),
            ("13", 11),
            ("13", 12),
            ("3", 1),
            ("4", 4),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("4", 9),
            ("4", 10),
            ("4", 11),
            ("4", 12),
            ("5", 4),
            ("5", 5),
            ("5", 6),
            ("5", 7),
            ("5", 8),
            ("5", 9),
            ("5", 10),
            ("5", 11),
            ("5", 12),
            ("7", 4),
            ("7", 5),
            ("7", 6),
            ("7", 7),
            ("7", 8),
            ("7", 9),
            ("7", 10),
            ("7", 11),
            ("7", 12),
            ("9", 12),
        }:
            return 15
        elif key in {
            ("0", 7),
            ("0", 8),
            ("1", 4),
            ("1", 5),
            ("1", 6),
            ("1", 7),
            ("1", 8),
            ("1", 9),
            ("1", 10),
            ("1", 11),
            ("1", 12),
            ("10", 4),
            ("10", 5),
            ("10", 7),
            ("11", 4),
            ("11", 5),
            ("11", 6),
            ("11", 7),
            ("11", 8),
            ("11", 9),
            ("11", 10),
            ("11", 11),
            ("11", 12),
            ("12", 2),
            ("3", 0),
            ("3", 2),
            ("3", 3),
            ("4", 2),
            ("4", 8),
            ("5", 2),
            ("8", 4),
            ("8", 5),
            ("8", 6),
            ("8", 7),
            ("8", 8),
            ("8", 9),
            ("8", 10),
            ("8", 11),
            ("8", 12),
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 11),
            ("<s>", 12),
        }:
            return 0
        elif key in {
            ("0", 13),
            ("0", 14),
            ("0", 15),
            ("1", 13),
            ("1", 14),
            ("1", 15),
            ("10", 13),
            ("11", 13),
            ("12", 13),
            ("13", 1),
            ("13", 13),
            ("13", 14),
            ("4", 13),
            ("4", 14),
            ("5", 13),
            ("5", 14),
            ("5", 15),
            ("6", 1),
            ("6", 15),
            ("7", 13),
            ("7", 14),
            ("7", 15),
            ("8", 1),
            ("8", 13),
            ("8", 14),
            ("8", 15),
            ("9", 13),
            ("9", 14),
            ("<s>", 13),
        }:
            return 7
        elif key in {
            ("2", 4),
            ("2", 5),
            ("2", 6),
            ("2", 7),
            ("2", 8),
            ("2", 9),
            ("2", 10),
            ("2", 11),
            ("2", 12),
        }:
            return 10
        elif key in {
            ("9", 4),
            ("9", 5),
            ("9", 6),
            ("9", 7),
            ("9", 8),
            ("9", 9),
            ("9", 10),
            ("9", 11),
        }:
            return 4
        elif key in {
            ("6", 0),
            ("6", 2),
            ("6", 3),
            ("6", 4),
            ("6", 5),
            ("6", 6),
            ("6", 13),
            ("6", 14),
        }:
            return 11
        elif key in {("6", 7), ("6", 8), ("6", 9), ("6", 10), ("6", 11), ("6", 12)}:
            return 14
        elif key in {("2", 0), ("2", 3), ("2", 13), ("2", 14), ("2", 15)}:
            return 13
        elif key in {("10", 1), ("11", 1), ("4", 1), ("5", 1)}:
            return 2
        elif key in {("12", 1), ("12", 14), ("12", 15), ("7", 1)}:
            return 12
        elif key in {("0", 1), ("1", 1), ("9", 1)}:
            return 3
        return 5

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_0_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_0_output, attn_1_3_output):
        key = (attn_2_0_output, attn_1_3_output)
        if key in {
            ("0", "0"),
            ("1", "1"),
            ("1", "10"),
            ("1", "13"),
            ("1", "7"),
            ("10", "1"),
            ("10", "10"),
            ("11", "1"),
            ("11", "13"),
            ("12", "12"),
            ("13", "1"),
            ("13", "10"),
            ("13", "13"),
            ("13", "2"),
            ("13", "6"),
            ("13", "7"),
            ("13", "8"),
            ("13", "<s>"),
            ("2", "1"),
            ("3", "1"),
            ("3", "3"),
            ("4", "0"),
            ("4", "1"),
            ("4", "12"),
            ("4", "4"),
            ("5", "5"),
            ("6", "1"),
            ("6", "10"),
            ("6", "11"),
            ("6", "13"),
            ("6", "6"),
            ("6", "8"),
            ("7", "1"),
            ("7", "4"),
            ("7", "7"),
            ("8", "1"),
            ("8", "13"),
            ("8", "6"),
            ("8", "8"),
            ("9", "9"),
            ("<s>", "1"),
            ("<s>", "10"),
        }:
            return 7
        elif key in {
            ("1", "12"),
            ("1", "4"),
            ("1", "9"),
            ("10", "12"),
            ("10", "4"),
            ("10", "9"),
            ("11", "9"),
            ("12", "9"),
            ("13", "12"),
            ("13", "4"),
            ("13", "9"),
            ("2", "12"),
            ("2", "4"),
            ("2", "9"),
            ("3", "10"),
            ("3", "11"),
            ("3", "12"),
            ("3", "13"),
            ("3", "2"),
            ("3", "4"),
            ("3", "6"),
            ("3", "7"),
            ("3", "8"),
            ("3", "9"),
            ("3", "<s>"),
            ("4", "9"),
            ("6", "12"),
            ("6", "4"),
            ("6", "9"),
            ("7", "12"),
            ("7", "9"),
            ("8", "12"),
            ("8", "4"),
            ("8", "9"),
            ("9", "1"),
            ("9", "10"),
            ("9", "12"),
            ("9", "13"),
            ("9", "2"),
            ("9", "4"),
            ("9", "6"),
            ("9", "7"),
            ("<s>", "12"),
            ("<s>", "4"),
            ("<s>", "9"),
        }:
            return 0
        elif key in {
            ("0", "1"),
            ("0", "10"),
            ("0", "11"),
            ("0", "13"),
            ("0", "2"),
            ("0", "6"),
            ("0", "8"),
            ("1", "0"),
            ("10", "0"),
            ("11", "0"),
            ("12", "0"),
            ("13", "0"),
            ("2", "0"),
            ("6", "0"),
            ("6", "7"),
            ("7", "0"),
            ("8", "0"),
            ("<s>", "0"),
        }:
            return 4
        elif key in {
            ("0", "3"),
            ("0", "<s>"),
            ("1", "3"),
            ("10", "3"),
            ("11", "3"),
            ("12", "3"),
            ("13", "3"),
            ("2", "3"),
            ("4", "3"),
            ("5", "3"),
            ("6", "3"),
            ("7", "3"),
            ("8", "3"),
            ("<s>", "3"),
        }:
            return 10
        elif key in {("3", "5"), ("9", "11"), ("9", "3"), ("9", "5"), ("9", "8")}:
            return 2
        elif key in {("0", "12"), ("0", "4"), ("0", "7"), ("0", "9"), ("3", "0")}:
            return 8
        elif key in {("5", "0"), ("9", "0"), ("9", "<s>")}:
            return 5
        elif key in {("0", "5")}:
            return 6
        return 14

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_0_outputs, attn_1_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output, num_attn_2_1_output):
        key = (num_attn_1_2_output, num_attn_2_1_output)
        return 10

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_2_output, num_attn_0_0_output):
        key = (num_attn_2_2_output, num_attn_0_0_output)
        if key in {(0, 0)}:
            return 13
        elif key in {(1, 0)}:
            return 9
        return 5

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_2_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # attn_3_0 ####################################################
    def predicate_3_0(position, mlp_0_1_output):
        if position in {0, 3, 15}:
            return mlp_0_1_output == 12
        elif position in {1}:
            return mlp_0_1_output == 10
        elif position in {2}:
            return mlp_0_1_output == 14
        elif position in {11, 4, 12}:
            return mlp_0_1_output == 1
        elif position in {5, 6, 7, 8, 9, 10}:
            return mlp_0_1_output == 15
        elif position in {13, 14}:
            return mlp_0_1_output == 6

    attn_3_0_pattern = select_closest(mlp_0_1_outputs, positions, predicate_3_0)
    attn_3_0_outputs = aggregate(attn_3_0_pattern, mlp_2_1_outputs)
    attn_3_0_output_scores = classifier_weights.loc[
        [("attn_3_0_outputs", str(v)) for v in attn_3_0_outputs]
    ]

    # attn_3_1 ####################################################
    def predicate_3_1(q_token, k_token):
        if q_token in {"9", "0"}:
            return k_token == "11"
        elif q_token in {"6", "1"}:
            return k_token == "9"
        elif q_token in {"11", "8", "4", "10"}:
            return k_token == "3"
        elif q_token in {"12"}:
            return k_token == "5"
        elif q_token in {"13", "<s>"}:
            return k_token == "0"
        elif q_token in {"2", "7"}:
            return k_token == "13"
        elif q_token in {"3"}:
            return k_token == "1"
        elif q_token in {"5"}:
            return k_token == "7"

    attn_3_1_pattern = select_closest(tokens, tokens, predicate_3_1)
    attn_3_1_outputs = aggregate(attn_3_1_pattern, mlp_2_1_outputs)
    attn_3_1_output_scores = classifier_weights.loc[
        [("attn_3_1_outputs", str(v)) for v in attn_3_1_outputs]
    ]

    # attn_3_2 ####################################################
    def predicate_3_2(position, mlp_1_1_output):
        if position in {0, 3, 4, 6, 7, 8, 9, 10, 11}:
            return mlp_1_1_output == 5
        elif position in {1}:
            return mlp_1_1_output == 4
        elif position in {2, 13}:
            return mlp_1_1_output == 15
        elif position in {5}:
            return mlp_1_1_output == 8
        elif position in {12}:
            return mlp_1_1_output == 1
        elif position in {14, 15}:
            return mlp_1_1_output == 9

    attn_3_2_pattern = select_closest(mlp_1_1_outputs, positions, predicate_3_2)
    attn_3_2_outputs = aggregate(attn_3_2_pattern, mlp_1_1_outputs)
    attn_3_2_output_scores = classifier_weights.loc[
        [("attn_3_2_outputs", str(v)) for v in attn_3_2_outputs]
    ]

    # attn_3_3 ####################################################
    def predicate_3_3(q_token, k_token):
        if q_token in {"6", "2", "13", "0", "7"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "13"
        elif q_token in {"3", "4", "10"}:
            return k_token == "9"
        elif q_token in {"11", "8"}:
            return k_token == "0"
        elif q_token in {"5", "12"}:
            return k_token == "2"
        elif q_token in {"9"}:
            return k_token == "6"
        elif q_token in {"<s>"}:
            return k_token == "10"

    attn_3_3_pattern = select_closest(tokens, tokens, predicate_3_3)
    attn_3_3_outputs = aggregate(attn_3_3_pattern, mlp_2_1_outputs)
    attn_3_3_output_scores = classifier_weights.loc[
        [("attn_3_3_outputs", str(v)) for v in attn_3_3_outputs]
    ]

    # num_attn_3_0 ####################################################
    def num_predicate_3_0(attn_1_3_output, token):
        if attn_1_3_output in {"1", "9", "3", "4", "12", "6", "2", "13", "0", "10"}:
            return token == "<s>"
        elif attn_1_3_output in {"11"}:
            return token == "11"
        elif attn_1_3_output in {"5"}:
            return token == "5"
        elif attn_1_3_output in {"7"}:
            return token == "<pad>"
        elif attn_1_3_output in {"8"}:
            return token == "8"
        elif attn_1_3_output in {"<s>"}:
            return token == "10"

    num_attn_3_0_pattern = select(tokens, attn_1_3_outputs, num_predicate_3_0)
    num_attn_3_0_outputs = aggregate_sum(num_attn_3_0_pattern, ones)
    num_attn_3_0_output_scores = classifier_weights.loc[
        [("num_attn_3_0_outputs", "_") for v in num_attn_3_0_outputs]
    ].mul(num_attn_3_0_outputs, axis=0)

    # num_attn_3_1 ####################################################
    def num_predicate_3_1(attn_1_3_output, token):
        if attn_1_3_output in {"0"}:
            return token == "<s>"
        elif attn_1_3_output in {
            "1",
            "3",
            "4",
            "8",
            "5",
            "<s>",
            "6",
            "2",
            "13",
            "7",
            "10",
        }:
            return token == "0"
        elif attn_1_3_output in {"11"}:
            return token == "<pad>"
        elif attn_1_3_output in {"12"}:
            return token == "12"
        elif attn_1_3_output in {"9"}:
            return token == "9"

    num_attn_3_1_pattern = select(tokens, attn_1_3_outputs, num_predicate_3_1)
    num_attn_3_1_outputs = aggregate_sum(num_attn_3_1_pattern, num_attn_0_0_outputs)
    num_attn_3_1_output_scores = classifier_weights.loc[
        [("num_attn_3_1_outputs", "_") for v in num_attn_3_1_outputs]
    ].mul(num_attn_3_1_outputs, axis=0)

    # num_attn_3_2 ####################################################
    def num_predicate_3_2(attn_0_3_output, token):
        if attn_0_3_output in {"9", "3", "5", "13", "0"}:
            return token == "<pad>"
        elif attn_0_3_output in {"1"}:
            return token == "1"
        elif attn_0_3_output in {"10"}:
            return token == "6"
        elif attn_0_3_output in {"11", "4", "8", "2", "6", "7"}:
            return token == "<s>"
        elif attn_0_3_output in {"12"}:
            return token == "12"
        elif attn_0_3_output in {"<s>"}:
            return token == "9"

    num_attn_3_2_pattern = select(tokens, attn_0_3_outputs, num_predicate_3_2)
    num_attn_3_2_outputs = aggregate_sum(num_attn_3_2_pattern, num_attn_1_3_outputs)
    num_attn_3_2_output_scores = classifier_weights.loc[
        [("num_attn_3_2_outputs", "_") for v in num_attn_3_2_outputs]
    ].mul(num_attn_3_2_outputs, axis=0)

    # num_attn_3_3 ####################################################
    def num_predicate_3_3(attn_1_3_output, token):
        if attn_1_3_output in {"7", "3", "8", "5", "12", "2", "0"}:
            return token == "<s>"
        elif attn_1_3_output in {"6", "1", "10"}:
            return token == "<pad>"
        elif attn_1_3_output in {"11"}:
            return token == "11"
        elif attn_1_3_output in {"13"}:
            return token == "13"
        elif attn_1_3_output in {"4"}:
            return token == "4"
        elif attn_1_3_output in {"9"}:
            return token == "9"
        elif attn_1_3_output in {"<s>"}:
            return token == "10"

    num_attn_3_3_pattern = select(tokens, attn_1_3_outputs, num_predicate_3_3)
    num_attn_3_3_outputs = aggregate_sum(num_attn_3_3_pattern, ones)
    num_attn_3_3_output_scores = classifier_weights.loc[
        [("num_attn_3_3_outputs", "_") for v in num_attn_3_3_outputs]
    ].mul(num_attn_3_3_outputs, axis=0)

    # mlp_3_0 #####################################################
    def mlp_3_0(attn_2_0_output, attn_1_3_output):
        key = (attn_2_0_output, attn_1_3_output)
        if key in {
            ("0", "2"),
            ("1", "2"),
            ("10", "2"),
            ("11", "2"),
            ("13", "2"),
            ("2", "0"),
            ("2", "1"),
            ("2", "10"),
            ("2", "11"),
            ("2", "13"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "5"),
            ("2", "6"),
            ("2", "7"),
            ("2", "8"),
            ("2", "9"),
            ("2", "<s>"),
            ("3", "2"),
            ("4", "2"),
            ("5", "2"),
            ("6", "2"),
            ("7", "2"),
            ("8", "2"),
            ("9", "2"),
            ("<s>", "2"),
        }:
            return 13
        elif key in {
            ("1", "0"),
            ("1", "12"),
            ("1", "9"),
            ("10", "12"),
            ("11", "12"),
            ("12", "0"),
            ("12", "1"),
            ("12", "10"),
            ("12", "11"),
            ("12", "13"),
            ("12", "3"),
            ("12", "4"),
            ("12", "5"),
            ("12", "6"),
            ("12", "7"),
            ("12", "8"),
            ("12", "9"),
            ("12", "<s>"),
            ("13", "12"),
            ("3", "12"),
            ("3", "9"),
            ("4", "12"),
            ("7", "12"),
            ("8", "12"),
            ("9", "12"),
        }:
            return 12
        elif key in {
            ("0", "12"),
            ("12", "12"),
            ("5", "12"),
            ("6", "12"),
            ("<s>", "12"),
        }:
            return 3
        elif key in {("2", "12")}:
            return 10
        elif key in {("12", "2")}:
            return 11
        return 1

    mlp_3_0_outputs = [
        mlp_3_0(k0, k1) for k0, k1 in zip(attn_2_0_outputs, attn_1_3_outputs)
    ]
    mlp_3_0_output_scores = classifier_weights.loc[
        [("mlp_3_0_outputs", str(v)) for v in mlp_3_0_outputs]
    ]

    # mlp_3_1 #####################################################
    def mlp_3_1(position, attn_2_0_output):
        key = (position, attn_2_0_output)
        if key in {
            (2, "4"),
            (3, "4"),
            (5, "0"),
            (5, "13"),
            (5, "3"),
            (5, "6"),
            (5, "<s>"),
            (6, "0"),
            (6, "10"),
            (6, "13"),
            (6, "3"),
            (6, "6"),
            (6, "<s>"),
            (7, "0"),
            (7, "13"),
            (7, "3"),
            (7, "6"),
            (7, "<s>"),
            (8, "0"),
            (8, "13"),
            (8, "3"),
            (8, "6"),
            (8, "<s>"),
            (9, "0"),
            (9, "13"),
            (9, "3"),
            (10, "0"),
            (10, "13"),
            (10, "3"),
            (10, "<s>"),
            (11, "0"),
            (11, "13"),
            (11, "3"),
            (11, "6"),
            (11, "<s>"),
            (12, "0"),
            (12, "10"),
            (12, "11"),
            (12, "12"),
            (12, "13"),
            (12, "2"),
            (12, "3"),
            (12, "5"),
            (12, "6"),
            (12, "7"),
            (12, "8"),
            (12, "9"),
            (12, "<s>"),
            (15, "4"),
        }:
            return 9
        elif key in {
            (3, "7"),
            (3, "9"),
            (4, "9"),
            (5, "10"),
            (5, "11"),
            (5, "12"),
            (5, "8"),
            (5, "9"),
            (6, "8"),
            (6, "9"),
            (7, "9"),
            (8, "10"),
            (8, "9"),
            (9, "1"),
            (9, "10"),
            (9, "11"),
            (9, "12"),
            (9, "8"),
            (9, "9"),
            (10, "1"),
            (10, "10"),
            (10, "11"),
            (10, "8"),
            (10, "9"),
            (11, "9"),
            (12, "1"),
            (13, "7"),
        }:
            return 15
        elif key in {
            (1, "0"),
            (1, "1"),
            (1, "10"),
            (1, "11"),
            (1, "12"),
            (1, "13"),
            (1, "2"),
            (1, "3"),
            (1, "5"),
            (1, "6"),
            (1, "8"),
            (1, "9"),
            (1, "<s>"),
            (13, "4"),
        }:
            return 11
        elif key in {
            (0, "7"),
            (0, "<s>"),
            (4, "7"),
            (5, "7"),
            (6, "7"),
            (7, "7"),
            (8, "7"),
            (9, "7"),
            (10, "7"),
            (11, "7"),
            (12, "4"),
            (14, "7"),
            (15, "7"),
        }:
            return 0
        elif key in {
            (0, "4"),
            (4, "4"),
            (5, "4"),
            (6, "4"),
            (7, "4"),
            (8, "4"),
            (10, "4"),
            (11, "4"),
            (14, "4"),
        }:
            return 10
        elif key in {(1, "7"), (2, "7"), (2, "9"), (9, "4")}:
            return 1
        elif key in {(1, "4"), (3, "3"), (4, "3")}:
            return 6
        elif key in {(9, "<s>")}:
            return 4
        return 5

    mlp_3_1_outputs = [mlp_3_1(k0, k1) for k0, k1 in zip(positions, attn_2_0_outputs)]
    mlp_3_1_output_scores = classifier_weights.loc[
        [("mlp_3_1_outputs", str(v)) for v in mlp_3_1_outputs]
    ]

    # num_mlp_3_0 #################################################
    def num_mlp_3_0(num_attn_2_0_output):
        key = num_attn_2_0_output
        if key in {0}:
            return 0
        elif key in {1}:
            return 9
        return 7

    num_mlp_3_0_outputs = [num_mlp_3_0(k0) for k0 in num_attn_2_0_outputs]
    num_mlp_3_0_output_scores = classifier_weights.loc[
        [("num_mlp_3_0_outputs", str(v)) for v in num_mlp_3_0_outputs]
    ]

    # num_mlp_3_1 #################################################
    def num_mlp_3_1(num_attn_2_1_output, num_attn_2_0_output):
        key = (num_attn_2_1_output, num_attn_2_0_output)
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
            (0, 48),
            (0, 49),
            (0, 50),
            (0, 51),
            (0, 52),
            (0, 53),
            (0, 54),
            (0, 55),
            (0, 56),
            (0, 57),
            (0, 58),
            (0, 59),
            (0, 60),
            (0, 61),
            (0, 62),
            (0, 63),
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
            (1, 48),
            (1, 49),
            (1, 50),
            (1, 51),
            (1, 52),
            (1, 53),
            (1, 54),
            (1, 55),
            (1, 56),
            (1, 57),
            (1, 58),
            (1, 59),
            (1, 60),
            (1, 61),
            (1, 62),
            (1, 63),
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
            (2, 48),
            (2, 49),
            (2, 50),
            (2, 51),
            (2, 52),
            (2, 53),
            (2, 54),
            (2, 55),
            (2, 56),
            (2, 57),
            (2, 58),
            (2, 59),
            (2, 60),
            (2, 61),
            (2, 62),
            (2, 63),
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
            (3, 48),
            (3, 49),
            (3, 50),
            (3, 51),
            (3, 52),
            (3, 53),
            (3, 54),
            (3, 55),
            (3, 56),
            (3, 57),
            (3, 58),
            (3, 59),
            (3, 60),
            (3, 61),
            (3, 62),
            (3, 63),
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
            (4, 48),
            (4, 49),
            (4, 50),
            (4, 51),
            (4, 52),
            (4, 53),
            (4, 54),
            (4, 55),
            (4, 56),
            (4, 57),
            (4, 58),
            (4, 59),
            (4, 60),
            (4, 61),
            (4, 62),
            (4, 63),
            (5, 42),
            (5, 43),
            (5, 44),
            (5, 45),
            (5, 46),
            (5, 47),
            (5, 48),
            (5, 49),
            (5, 50),
            (5, 51),
            (5, 52),
            (5, 53),
            (5, 54),
            (5, 55),
            (5, 56),
            (5, 57),
            (5, 58),
            (5, 59),
            (5, 60),
            (5, 61),
            (5, 62),
            (5, 63),
            (6, 52),
            (6, 53),
            (6, 54),
            (6, 55),
            (6, 56),
            (6, 57),
            (6, 58),
            (6, 59),
            (6, 60),
            (6, 61),
            (6, 62),
            (6, 63),
            (7, 62),
            (7, 63),
        }:
            return 11
        elif key in {
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
        }:
            return 6
        elif key in {(0, 0)}:
            return 2
        elif key in {(0, 1)}:
            return 14
        return 4

    num_mlp_3_1_outputs = [
        num_mlp_3_1(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_2_0_outputs)
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


print(run(["<s>", "1", "8", "1", "2", "9", "0", "5", "8", "2", "11", "10", "9"]))
