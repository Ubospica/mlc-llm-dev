# pylint: disable=missing-module-docstring,missing-function-docstring
import os

import pytest
import tvm.testing
from tvm._ffi.base import TVMError

from mlc_chat.serve import BNFGrammar


def test_flatten():
    before = """main ::= or_test sequence_test nested_test empty_test
or_test ::= ([a] | "b") | "de" | "" | or_test | [^a-z]
sequence_test ::= [a] "a" ("b" ("c" | "d")) ("d" "e") sequence_test ""
nested_test ::= ("a" ("b" ("c" "d"))) | ("a" | ("b" | "c")) | nested_rest
nested_rest ::= ("a" | ("b" "c" | ("d" | "e" "f"))) | ((("g")))
empty_test ::= "d" | (("" | "" "") "" | "a" "") | ("" ("" | "")) "" ""
"""
    expected = """main ::= ((or_test sequence_test nested_test empty_test))
or_test ::= ("" | ([a]) | ([b]) | ([d] [e]) | (or_test) | ([a-z]))
sequence_test ::= (([a] [a] [b] sequence_test_choice [d] [e] sequence_test))
nested_test ::= (([a] [b] [c] [d]) | ([a]) | ([b]) | ([c]) | (nested_rest))
nested_rest ::= (([a]) | ([b] [c]) | ([d]) | ([e] [f]) | ([g]))
empty_test ::= ("" | ([d]) | ([a]))
sequence_test_choice ::= (([c]) | ([d]))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    normalized = bnf_grammar.to_normalized()
    after = normalized.to_string()
    assert after == expected


def test_unreachable_eliminator():
    before = """main ::= a
b ::= "a"
a ::= b
c ::= e b
d ::= c
e ::= d
"""
    expected = """main ::= ((a))
b ::= (([a]))
a ::= ((b))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    normalized = bnf_grammar.to_normalized()
    after = normalized.to_string()
    assert after == expected


def test_main_normalizer():
    before = """b ::= "b"
a ::= "a" b
main ::= "d" a c
c ::= "c" main | ""
"""
    expected = """main ::= (([d] a) | ([d] a c))
main_1 ::= (([d] a) | ([d] a c))
b ::= (([b]))
a ::= (([a] b))
c ::= (([c] main_1))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    normalized = bnf_grammar.to_normalized()
    after = normalized.to_string()
    assert after == expected


def test_epsilon_eliminator():
    before = """main ::= ep1 | ep2 | ep4 | ep5 | ep6
ep1 ::= "a" | ""
ep2 ::= ""
ep3 ::= "b" | ""
ep4 ::= ep1 ep2 ep3 "a" | ep3 ep1
ep5 ::= ep2 ep2 | "" | ep2
ep6 ::= ep7 | ""
ep7 ::= ep6 | ""
"""
    expected = """main ::= ("" | (ep1) | (ep4) | (ep6))
ep1 ::= (([a]))
ep3 ::= (([b]))
ep4 ::= (([a]) | (ep3 [a]) | (ep1 [a]) | (ep1 ep3 [a]) | (ep1) | (ep3) | (ep3 ep1))
ep6 ::= ((ep7))
ep7 ::= ((ep6))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    normalized = bnf_grammar.to_normalized()
    after = normalized.to_string()
    print(after)
    assert after == expected


def test_unit_production_eliminator():
    before = """main ::= u1 | u2 | u5 | u6 | u7
u1 ::= u3 | u2 "a"
u2 ::= u3 | "b"
u3 ::= "c"
u4 ::= u5
u5 ::= u4
u6 ::= u5 "a"
u7 ::= u4 u1
"""
    expected = """main ::= (([c]) | ([a]) | ([b]) | (u2 [a]) | (u2 [a]) | ([c]))
u2 ::= (([b]) | ([c]))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    normalized = bnf_grammar.to_normalized()
    after = normalized.to_string()
    assert after == expected


def test_left_recursion():
    before = """main ::= "a" l1 | "b" l2_0 | "c" l3_0 | "d" l4_3
l1 ::= l1 "a" | "b"
l2_0 ::= l2_1 "a" | "b"
l2_1 ::= l2_1 "c" | l2_0 "d" | "e"
l3_0 ::= l3_1 l3_2 "e"
l3_1 ::= ""
l3_2 ::= l3_1 l3_2
l4_0 ::= l4_1 "a"
l4_1 ::= l4_2 "b"
l4_2 ::= l4_3 "c"
l4_3 ::= l4_0 "d" | "e"
"""
    expected = """main ::= (([a] l1) | ([b] l2_0) | ([c] l3_0) | ([d] l4_3))
l1 ::= (([b]) | ([b] l1_recursion))
l2_0 ::= ((l2_1 [a]) | ([b]))
l2_1 ::= (([b] [d]) | ([b] [d] l2_1_recursion) | ([e]) | ([e] l2_1_recursion))
l3_0 ::= (([e]))
l4_3 ::= (([e]) | ([e] l4_3_recursion))
l1_recursion ::= (([a]) | ([a] l1_recursion))
l2_1_recursion ::= (([c]) | ([c] l2_1_recursion) | ([a] [d]) | ([a] [d] l2_1_recursion))
l4_3_recursion ::= (([c] [b] [a] [d]) | ([c] [b] [a] [d] l4_3_recursion))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    normalized = bnf_grammar.to_normalized()
    after = normalized.to_string()
    print(after)
    assert after == expected


def test_empty():
    before = """main ::= ""
"""
    expected = """main ::= ("")
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    normalized = bnf_grammar.to_normalized()
    after = normalized.to_string()
    print(after)
    assert after == expected


def test_error():
    with pytest.raises(TVMError, match="Rule a is not defined at line 1, column 11"):
        BNFGrammar.from_ebnf_string("main ::= a b")

    with pytest.raises(TVMError, match="Expect element at line 1, column 15"):
        BNFGrammar.from_ebnf_string('main ::= "a" |')

    with pytest.raises(TVMError, match='Expect " at line 1, column 15'):
        BNFGrammar.from_ebnf_string('main ::= "a" "')

    with pytest.raises(TVMError, match="Expect rule name at line 1, column 1"):
        BNFGrammar.from_ebnf_string('::= "a"')

    with pytest.raises(
        TVMError, match="Character range should not contain newline at line 1, column 12"
    ):
        BNFGrammar.from_ebnf_string("main ::= [a\n]")

    with pytest.raises(TVMError, match="Invalid escape sequence at line 1, column 11"):
        BNFGrammar.from_ebnf_string(r'main ::= "\@"')

    with pytest.raises(TVMError, match="Invalid escape sequence at line 1, column 11"):
        BNFGrammar.from_ebnf_string(r'main ::= "\uFF"')

    with pytest.raises(
        TVMError,
        match="Invalid character range: lower bound is larger than upper bound at "
        "line 1, column 14",
    ):
        BNFGrammar.from_ebnf_string(r"main ::= [Z-A]")

    with pytest.raises(TVMError, match="Expect ::= at line 1, column 6"):
        BNFGrammar.from_ebnf_string(r'main := "a"')

    with pytest.raises(TVMError, match="Rule main is defined multiple times at line 2, column 9"):
        BNFGrammar.from_ebnf_string('main ::= "a"\nmain ::= "b"')

    with pytest.raises(TVMError, match="There must be a rule named main at line 1, column 10"):
        BNFGrammar.from_ebnf_string('a ::= "a"')


if __name__ == "__main__":
    tvm.testing.main()
