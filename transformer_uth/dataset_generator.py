"""
Script to generate different task datasets:

Some dataset inspired on https://github.com/greentfrapp/attention-primer

task1:
Counting letters
"ABCDABC" -> "2221"

task2:
Counting letters + Difference between A & B
"AABA"    -> "31002"
"""
import random
from collections import Counter

_LETTERS = "ABCD"
_MAX_LENGTH = 9


def _input_text() -> str:
    return "".join(random.choices(_LETTERS, k=random.randint(4, _MAX_LENGTH)))


def _output_task1(input_text: str) -> str:
    """Task 1 is count letters.

    * (numbers for each letter are always in the same position)

    example:
    input  -> "BBCCCCCCD"
    output -> "0261"
    """
    counter_letters = Counter(input_text)
    return "".join(str(counter_letters.get(letter, 0)) for letter in _LETTERS)


def _output_task2(input_text: str) -> str:
    """Task 2 is count letters adding absolute number difference between A & B.

    * last digit is absolute difference between numbers of A & numbers of B

    example:
    input  -> "ABA"
    output -> "21001"
    """
    count_letters = _output_task1(input_text)
    a, b, *_ = [int(i) for i in count_letters]
    difference = str(abs(a - b))
    return count_letters + difference
