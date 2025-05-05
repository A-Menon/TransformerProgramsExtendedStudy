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
        "output-base-1616/rasp/reverse/k16_len16_L3_H8_M2/s0/reverse_weights.csv",
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
        if q_token in {"5", "6", "0", "9", "10", "11", "8", "4", "2", "3", "12", "7"}:
            return k_token == "1"
        elif q_token in {"1"}:
            return k_token == "3"
        elif q_token in {"</s>"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == "8"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 9
        elif q_position in {1, 5}:
            return k_position == 6
        elif q_position in {8, 2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 11
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {9}:
            return k_position == 3
        elif q_position in {10, 11, 12, 14, 15}:
            return k_position == 1
        elif q_position in {13}:
            return k_position == 15

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 8}:
            return k_position == 7
        elif q_position in {1}:
            return k_position == 10
        elif q_position in {2}:
            return k_position == 11
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {10, 4, 13, 7}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 3
        elif q_position in {9, 11, 12, 14, 15}:
            return k_position == 1

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {13}:
            return k_position == 2

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"6", "0", "1", "10", "2", "12"}:
            return position == 6
        elif token in {"11", "3", "9", "5"}:
            return position == 7
        elif token in {"4"}:
            return position == 5
        elif token in {"7"}:
            return position == 15
        elif token in {"8"}:
            return position == 8
        elif token in {"<s>", "</s>"}:
            return position == 2

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"10", "4", "0", "2"}:
            return position == 5
        elif token in {"1"}:
            return position == 2
        elif token in {"5", "6", "9", "11", "3", "12", "7"}:
            return position == 6
        elif token in {"8"}:
            return position == 7
        elif token in {"<s>", "</s>"}:
            return position == 4

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"6", "0", "4"}:
            return position == 5
        elif token in {"</s>", "1", "8", "2"}:
            return position == 4
        elif token in {"5", "9", "10", "11", "7"}:
            return position == 6
        elif token in {"3", "12"}:
            return position == 3
        elif token in {"<s>"}:
            return position == 2

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"6", "0", "9", "10", "</s>", "2", "3", "7"}:
            return position == 5
        elif token in {"1"}:
            return position == 4
        elif token in {"5", "11", "8", "4", "12"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 2

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_3_output, attn_0_1_output):
        key = (attn_0_3_output, attn_0_1_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "10"),
            ("0", "11"),
            ("0", "12"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "5"),
            ("0", "6"),
            ("0", "7"),
            ("0", "8"),
            ("0", "9"),
            ("0", "<s>"),
            ("11", "0"),
            ("11", "1"),
            ("11", "10"),
            ("11", "11"),
            ("11", "12"),
            ("11", "2"),
            ("11", "3"),
            ("11", "4"),
            ("11", "5"),
            ("11", "6"),
            ("11", "7"),
            ("11", "8"),
            ("11", "9"),
            ("11", "<s>"),
            ("12", "0"),
            ("12", "1"),
            ("12", "10"),
            ("12", "11"),
            ("12", "12"),
            ("12", "2"),
            ("12", "3"),
            ("12", "4"),
            ("12", "5"),
            ("12", "6"),
            ("12", "7"),
            ("12", "8"),
            ("12", "9"),
            ("12", "<s>"),
            ("6", "0"),
            ("6", "1"),
            ("6", "10"),
            ("6", "11"),
            ("6", "12"),
            ("6", "2"),
            ("6", "3"),
            ("6", "4"),
            ("6", "5"),
            ("6", "6"),
            ("6", "7"),
            ("6", "8"),
            ("6", "9"),
            ("6", "<s>"),
            ("8", "0"),
            ("8", "1"),
            ("8", "10"),
            ("8", "11"),
            ("8", "12"),
            ("8", "2"),
            ("8", "3"),
            ("8", "4"),
            ("8", "5"),
            ("8", "6"),
            ("8", "7"),
            ("8", "8"),
            ("8", "9"),
            ("8", "<s>"),
        }:
            return 13
        elif key in {
            ("2", "0"),
            ("2", "1"),
            ("2", "10"),
            ("2", "11"),
            ("2", "12"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "5"),
            ("2", "6"),
            ("2", "7"),
            ("2", "8"),
            ("2", "9"),
            ("2", "<s>"),
            ("3", "0"),
            ("3", "1"),
            ("3", "10"),
            ("3", "11"),
            ("3", "12"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "5"),
            ("3", "6"),
            ("3", "7"),
            ("3", "8"),
            ("3", "9"),
            ("3", "<s>"),
            ("4", "0"),
            ("4", "1"),
            ("4", "10"),
            ("4", "11"),
            ("4", "12"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "5"),
            ("4", "6"),
            ("4", "7"),
            ("4", "8"),
            ("4", "9"),
            ("4", "<s>"),
            ("9", "0"),
            ("9", "1"),
            ("9", "10"),
            ("9", "11"),
            ("9", "12"),
            ("9", "2"),
            ("9", "3"),
            ("9", "4"),
            ("9", "5"),
            ("9", "6"),
            ("9", "7"),
            ("9", "8"),
            ("9", "9"),
            ("9", "<s>"),
        }:
            return 9
        elif key in {("1", "</s>"), ("11", "</s>"), ("12", "</s>"), ("3", "</s>")}:
            return 5
        elif key in {("0", "</s>"), ("4", "</s>"), ("5", "</s>"), ("9", "</s>")}:
            return 14
        elif key in {("10", "</s>"), ("8", "</s>")}:
            return 6
        elif key in {("<s>", "</s>")}:
            return 3
        elif key in {("2", "</s>")}:
            return 8
        elif key in {("6", "</s>")}:
            return 10
        return 11

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_3_output):
        key = (num_attn_0_2_output, num_attn_0_3_output)
        if key in {(0, 0)}:
            return 2
        return 15

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 15}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2, 3, 13}:
            return k_position == 2
        elif q_position in {4, 8, 9, 10, 11, 12, 14}:
            return k_position == 1
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 3

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 2, 3}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 8
        elif q_position in {11, 4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 7
        elif q_position in {8, 9}:
            return k_position == 2
        elif q_position in {10, 12, 13, 14, 15}:
            return k_position == 1

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 5, 15}:
            return k_position == 15
        elif q_position in {1, 6}:
            return k_position == 9
        elif q_position in {2}:
            return k_position == 13
        elif q_position in {3}:
            return k_position == 12
        elif q_position in {8, 9, 4}:
            return k_position == 6
        elif q_position in {10, 7}:
            return k_position == 5
        elif q_position in {11}:
            return k_position == 4
        elif q_position in {12}:
            return k_position == 3
        elif q_position in {13}:
            return k_position == 2
        elif q_position in {14}:
            return k_position == 1

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 12, 15}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 14
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {10, 3}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 10
        elif q_position in {6, 7}:
            return k_position == 6
        elif q_position in {8, 14}:
            return k_position == 1
        elif q_position in {9}:
            return k_position == 0
        elif q_position in {11}:
            return k_position == 4
        elif q_position in {13}:
            return k_position == 2

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_3_output, attn_0_2_output):
        if attn_0_3_output in {"0", "1", "10", "4", "2", "12", "7"}:
            return attn_0_2_output == "<s>"
        elif attn_0_3_output in {"11"}:
            return attn_0_2_output == "11"
        elif attn_0_3_output in {"3", "9"}:
            return attn_0_2_output == "<pad>"
        elif attn_0_3_output in {"5"}:
            return attn_0_2_output == "5"
        elif attn_0_3_output in {"6"}:
            return attn_0_2_output == "6"
        elif attn_0_3_output in {"8"}:
            return attn_0_2_output == "8"
        elif attn_0_3_output in {"</s>"}:
            return attn_0_2_output == "4"
        elif attn_0_3_output in {"<s>"}:
            return attn_0_2_output == "1"

    num_attn_1_0_pattern = select(attn_0_2_outputs, attn_0_3_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_3_output, attn_0_2_output):
        if attn_0_3_output in {"0"}:
            return attn_0_2_output == "0"
        elif attn_0_3_output in {"9", "1", "11", "8", "4", "2", "7"}:
            return attn_0_2_output == "<s>"
        elif attn_0_3_output in {"10"}:
            return attn_0_2_output == "</s>"
        elif attn_0_3_output in {"12"}:
            return attn_0_2_output == "12"
        elif attn_0_3_output in {"3"}:
            return attn_0_2_output == "3"
        elif attn_0_3_output in {"5"}:
            return attn_0_2_output == "5"
        elif attn_0_3_output in {"6"}:
            return attn_0_2_output == "6"
        elif attn_0_3_output in {"</s>"}:
            return attn_0_2_output == "4"
        elif attn_0_3_output in {"<s>"}:
            return attn_0_2_output == "1"

    num_attn_1_1_pattern = select(attn_0_2_outputs, attn_0_3_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(q_attn_0_2_output, k_attn_0_2_output):
        if q_attn_0_2_output in {"6", "0", "11", "4", "2", "3", "12"}:
            return k_attn_0_2_output == "<s>"
        elif q_attn_0_2_output in {"1"}:
            return k_attn_0_2_output == "6"
        elif q_attn_0_2_output in {"10"}:
            return k_attn_0_2_output == "10"
        elif q_attn_0_2_output in {"5"}:
            return k_attn_0_2_output == "5"
        elif q_attn_0_2_output in {"7"}:
            return k_attn_0_2_output == "7"
        elif q_attn_0_2_output in {"8", "9"}:
            return k_attn_0_2_output == "</s>"
        elif q_attn_0_2_output in {"</s>"}:
            return k_attn_0_2_output == "<pad>"
        elif q_attn_0_2_output in {"<s>"}:
            return k_attn_0_2_output == "11"

    num_attn_1_2_pattern = select(attn_0_2_outputs, attn_0_2_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_3_output, attn_0_2_output):
        if attn_0_3_output in {"0", "9", "11", "2", "3", "7"}:
            return attn_0_2_output == "<s>"
        elif attn_0_3_output in {"1"}:
            return attn_0_2_output == "1"
        elif attn_0_3_output in {"10", "8"}:
            return attn_0_2_output == "</s>"
        elif attn_0_3_output in {"12", "6"}:
            return attn_0_2_output == "6"
        elif attn_0_3_output in {"4"}:
            return attn_0_2_output == "4"
        elif attn_0_3_output in {"5"}:
            return attn_0_2_output == "5"
        elif attn_0_3_output in {"</s>"}:
            return attn_0_2_output == "7"
        elif attn_0_3_output in {"<s>"}:
            return attn_0_2_output == "2"

    num_attn_1_3_pattern = select(attn_0_2_outputs, attn_0_3_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_2_output, attn_1_3_output):
        key = (attn_0_2_output, attn_1_3_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "10"),
            ("0", "2"),
            ("0", "4"),
            ("0", "5"),
            ("0", "6"),
            ("0", "9"),
            ("1", "2"),
            ("1", "5"),
            ("1", "6"),
            ("10", "2"),
            ("10", "5"),
            ("10", "6"),
            ("11", "2"),
            ("11", "6"),
            ("12", "0"),
            ("12", "1"),
            ("12", "10"),
            ("12", "11"),
            ("12", "12"),
            ("12", "2"),
            ("12", "3"),
            ("12", "4"),
            ("12", "5"),
            ("12", "6"),
            ("12", "7"),
            ("12", "8"),
            ("12", "9"),
            ("12", "</s>"),
            ("12", "<s>"),
            ("2", "0"),
            ("2", "1"),
            ("2", "10"),
            ("2", "11"),
            ("2", "12"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "5"),
            ("2", "6"),
            ("2", "7"),
            ("2", "8"),
            ("2", "9"),
            ("2", "</s>"),
            ("2", "<s>"),
            ("3", "2"),
            ("3", "5"),
            ("3", "6"),
            ("4", "0"),
            ("4", "10"),
            ("4", "12"),
            ("4", "2"),
            ("4", "4"),
            ("4", "5"),
            ("4", "6"),
            ("4", "9"),
            ("5", "2"),
            ("5", "5"),
            ("5", "6"),
            ("6", "0"),
            ("6", "1"),
            ("6", "10"),
            ("6", "11"),
            ("6", "12"),
            ("6", "2"),
            ("6", "3"),
            ("6", "4"),
            ("6", "5"),
            ("6", "6"),
            ("6", "7"),
            ("6", "8"),
            ("6", "9"),
            ("6", "</s>"),
            ("6", "<s>"),
            ("7", "2"),
            ("7", "5"),
            ("7", "6"),
            ("8", "2"),
            ("8", "5"),
            ("8", "6"),
            ("9", "2"),
            ("9", "6"),
            ("</s>", "2"),
            ("</s>", "6"),
            ("<s>", "2"),
            ("<s>", "6"),
        }:
            return 4
        return 13

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_1_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output):
        key = num_attn_1_3_output
        if key in {0}:
            return 1
        return 7

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_1_3_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0, 10, 15, 7}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 11
        elif q_position in {2}:
            return k_position == 12
        elif q_position in {3, 6}:
            return k_position == 0
        elif q_position in {4, 5}:
            return k_position == 8
        elif q_position in {8, 12}:
            return k_position == 2
        elif q_position in {9}:
            return k_position == 5
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {13, 14}:
            return k_position == 1

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, tokens)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0, 9, 6, 15}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 14
        elif q_position in {2}:
            return k_position == 8
        elif q_position in {3}:
            return k_position == 11
        elif q_position in {4}:
            return k_position == 0
        elif q_position in {5}:
            return k_position == 9
        elif q_position in {10, 7}:
            return k_position == 4
        elif q_position in {8, 12}:
            return k_position == 3
        elif q_position in {11}:
            return k_position == 2
        elif q_position in {13, 14}:
            return k_position == 1

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, tokens)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {0, 15}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 13
        elif q_position in {9, 2, 4}:
            return k_position == 4
        elif q_position in {3, 5}:
            return k_position == 0
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {8, 7}:
            return k_position == 15
        elif q_position in {10, 11, 12}:
            return k_position == 2
        elif q_position in {13, 14}:
            return k_position == 1

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, tokens)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0, 10, 11, 5}:
            return k_position == 3
        elif q_position in {1, 2}:
            return k_position == 13
        elif q_position in {3, 4}:
            return k_position == 9
        elif q_position in {12, 6}:
            return k_position == 2
        elif q_position in {9, 15, 7}:
            return k_position == 15
        elif q_position in {8}:
            return k_position == 4
        elif q_position in {13, 14}:
            return k_position == 1

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, tokens)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_0_output, attn_1_1_output):
        if attn_1_0_output in {"0"}:
            return attn_1_1_output == "0"
        elif attn_1_0_output in {"1"}:
            return attn_1_1_output == "1"
        elif attn_1_0_output in {"6", "9", "10", "8", "4", "3", "12", "7"}:
            return attn_1_1_output == "<s>"
        elif attn_1_0_output in {"11"}:
            return attn_1_1_output == "11"
        elif attn_1_0_output in {"2"}:
            return attn_1_1_output == "2"
        elif attn_1_0_output in {"5"}:
            return attn_1_1_output == "5"
        elif attn_1_0_output in {"<s>", "</s>"}:
            return attn_1_1_output == "<pad>"

    num_attn_2_0_pattern = select(attn_1_1_outputs, attn_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, ones)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_0_output, attn_1_1_output):
        if attn_1_0_output in {"</s>", "<s>", "0"}:
            return attn_1_1_output == "<pad>"
        elif attn_1_0_output in {"5", "1", "8", "4", "2", "3", "12", "7"}:
            return attn_1_1_output == "<s>"
        elif attn_1_0_output in {"10"}:
            return attn_1_1_output == "10"
        elif attn_1_0_output in {"11"}:
            return attn_1_1_output == "11"
        elif attn_1_0_output in {"6"}:
            return attn_1_1_output == "6"
        elif attn_1_0_output in {"9"}:
            return attn_1_1_output == "9"

    num_attn_2_1_pattern = select(attn_1_1_outputs, attn_1_0_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, ones)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_0_output, attn_1_1_output):
        if attn_1_0_output in {"0"}:
            return attn_1_1_output == "0"
        elif attn_1_0_output in {"1"}:
            return attn_1_1_output == "1"
        elif attn_1_0_output in {"5", "10", "11", "4", "2", "3", "7"}:
            return attn_1_1_output == "<s>"
        elif attn_1_0_output in {"<s>", "12"}:
            return attn_1_1_output == "12"
        elif attn_1_0_output in {"6", "</s>"}:
            return attn_1_1_output == "<pad>"
        elif attn_1_0_output in {"8"}:
            return attn_1_1_output == "8"
        elif attn_1_0_output in {"9"}:
            return attn_1_1_output == "9"

    num_attn_2_2_pattern = select(attn_1_1_outputs, attn_1_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, ones)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_0_output, attn_1_1_output):
        if attn_1_0_output in {"6", "0", "10", "11", "8", "4", "2", "12", "7"}:
            return attn_1_1_output == "<s>"
        elif attn_1_0_output in {"1"}:
            return attn_1_1_output == "1"
        elif attn_1_0_output in {"3"}:
            return attn_1_1_output == "3"
        elif attn_1_0_output in {"5", "<s>", "</s>"}:
            return attn_1_1_output == "<pad>"
        elif attn_1_0_output in {"9"}:
            return attn_1_1_output == "9"

    num_attn_2_3_pattern = select(attn_1_1_outputs, attn_1_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, ones)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_2_output, attn_1_1_output):
        key = (attn_0_2_output, attn_1_1_output)
        if key in {
            ("0", "1"),
            ("0", "11"),
            ("0", "2"),
            ("0", "8"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("1", "1"),
            ("1", "11"),
            ("1", "8"),
            ("1", "</s>"),
            ("1", "<s>"),
            ("11", "1"),
            ("11", "8"),
            ("11", "</s>"),
            ("11", "<s>"),
            ("12", "1"),
            ("12", "11"),
            ("12", "2"),
            ("12", "</s>"),
            ("12", "<s>"),
            ("2", "1"),
            ("2", "11"),
            ("2", "</s>"),
            ("2", "<s>"),
            ("6", "0"),
            ("6", "1"),
            ("6", "11"),
            ("6", "2"),
            ("6", "8"),
            ("6", "</s>"),
            ("6", "<s>"),
            ("7", "1"),
            ("7", "11"),
            ("7", "8"),
            ("7", "</s>"),
            ("7", "<s>"),
            ("8", "1"),
            ("8", "11"),
            ("8", "</s>"),
            ("8", "<s>"),
            ("</s>", "1"),
            ("</s>", "11"),
            ("</s>", "2"),
            ("</s>", "8"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "1"),
            ("<s>", "11"),
            ("<s>", "2"),
            ("<s>", "8"),
            ("<s>", "</s>"),
            ("<s>", "<s>"),
        }:
            return 0
        elif key in {
            ("0", "10"),
            ("0", "6"),
            ("10", "1"),
            ("10", "6"),
            ("10", "8"),
            ("10", "</s>"),
            ("10", "<s>"),
            ("12", "10"),
            ("12", "5"),
            ("12", "6"),
            ("5", "6"),
            ("5", "</s>"),
            ("5", "<s>"),
            ("6", "10"),
            ("6", "5"),
            ("6", "6"),
            ("</s>", "6"),
            ("<s>", "10"),
            ("<s>", "5"),
            ("<s>", "6"),
        }:
            return 8
        elif key in {("12", "8"), ("8", "8")}:
            return 13
        return 3

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_1_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_0_output):
        key = num_attn_1_0_output
        if key in {0}:
            return 0
        return 4

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_1_0_outputs]
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


print(run(["<s>", "3", "3", "2", "4", "</s>"]))
