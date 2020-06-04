"""
Huy Phi
CSE 163

This file implements test functions for my CSE 163 project
Tests the p-value calculations I made.
"""

from cse163_utils import assert_equals
from main import gen_p_values, find_sig_pvalues
import pandas as pd


def test_gen_p_values(data1, data2):
    """
    Tests the function gen_p_values by evaluating examples
    with known pvalues
    """
    assert_equals({"feature": 0.518}, gen_p_values(data1))
    assert_equals({"feature1": 0.518, "feature2": 0.000049},
                  gen_p_values(data2))


def test_find_sig_pvalues(data1, data2):
    """
    Tests the function find_sig_pvalues using examples with
    known levels of significance
    """
    assert_equals({"class": 0, "gender": 0}, find_sig_pvalues(data1, 0.05))
    assert_equals({"class": 0, "gender": 0, "feature2": 0.000049},
                  find_sig_pvalues(data2, 0.05))


def main():
    data1 = pd.read_csv("stat_test.csv")
    data2 = pd.read_csv("stat_test2.csv")
    test_gen_p_values(data1, data2)
    data1 = gen_p_values(data1)
    data2 = gen_p_values(data2)
    test_find_sig_pvalues(data1, data2)
    print("Tests passed! :^)")


if __name__ == "__main__":
    main()
