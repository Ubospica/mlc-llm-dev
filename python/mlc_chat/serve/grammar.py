"""Classes handling the grammar guided generation of MLC LLM serving"""
from typing import List
import tvm._ffi
from tvm.runtime import Object

from ..tokenizer import Tokenizer
from . import _ffi_api


@tvm._ffi.register_object("mlc.serve.BNFGrammar")  # pylint: disable=protected-access
class BNFGrammar(Object):
    """This class stores the abstract syntax tree (AST) of the Backus-Naur Form (BNF) grammar and
    provides utilities to parse and print the AST. User should provide a BNF/EBNF (Extended
    Backus-Naur Form) grammar, and use from_ebnf_string to parse and simplify the grammar into an
    AST of BNF grammar.
    """

    @staticmethod
    def from_ebnf_string(
        ebnf_string: str, normalize: bool = True, simplify: bool = True
    ) -> "BNFGrammar":
        r"""Parse a BNF grammar from a string in BNF/EBNF format.

        This method accepts the EBNF notation from the W3C XML Specification
        (https://www.w3.org/TR/xml/#sec-notatPion), which is a popular standard, with the following
        changes:
        - Using # as comment mark instead of /**/
        - Using C-style unicode escape sequence \u01AB, \U000001AB, \xAB instead of #x0123
        - Do not support A-B (match A and not match B) yet

        See tests/python/serve/json.ebnf for an example.

        Parameters
        ----------
        ebnf_string : str
            The grammar string.

        normalize : bool
            Whether to normalize the grammar. Default: true. Only set to false for the purpose of
            testing.

            In The normalized form of a BNF grammar, every rule is in the form:
            `rule_name ::= ("" | (element1_1 element1_2 ...) | (element2_1 element2_2 ...) | ...)`.

            I.e. a list of choices, each choice is a sequence of elements. Elements can be a
            character class or a rule reference. And if the rule can be empty, the first choice
            will be an empty string.

        simplify : bool
            Whether to simplify the grammar to make matching more efficient. Default: true. Not
            implemented yet.

        Returns
        -------
        grammar : BNFGrammar
            The parsed BNF grammar.
        """
        return _ffi_api.BNFGrammarFromEBNFString(  # type: ignore  # pylint: disable=no-member
            ebnf_string, normalize, simplify
        )

    def to_string(self) -> str:
        """Print the BNF grammar to a string, in standard BNF format.

        Returns
        -------
        grammar_string : str
            The BNF grammar string.
        """
        return str(_ffi_api.BNFGrammarToString(self))  # type: ignore  # pylint: disable=no-member

    def __str__(self) -> str:
        return self.to_string()

    @staticmethod
    def from_json(json_string: str) -> "BNFGrammar":
        """Load a BNF grammar from the raw representation of the AST in JSON format.

        Parameters
        ----------
        json_string : str
            The JSON string.

        Returns
        -------
        grammar : BNFGrammar
            The loaded BNF grammar.
        """
        return _ffi_api.BNFGrammarFromJSON(json_string)  # type: ignore  # pylint: disable=no-member

    def to_json(self, prettify: bool = True) -> str:
        """Serialize the AST. Dump the raw representation of the AST to a JSON file.

        Parameters
        ----------
        prettify : bool
            Whether to format the JSON string. If False, all whitespaces will be removed.

        Returns
        -------
        json_string : str
            The JSON string.
        """
        return str(
            _ffi_api.BNFGrammarToJSON(self, prettify)  # type: ignore  # pylint: disable=no-member
        )

    @staticmethod
    def get_json_grammar() -> "BNFGrammar":
        """Get the grammar of standard JSON.

        Returns
        -------
        grammar : BNFGrammar
            The JSON grammar.
        """
        return _ffi_api.BNFGrammarGetJSONGrammar()


@tvm._ffi.register_object("mlc.serve.GrammarMatcher")  # pylint: disable=protected-access
class GrammarMatcher(Object):
    """Match character or string or tokens to the given BNF grammar. This class is the core logic
    of the grammar-guided generation.

    This class implements the non-deterministic pushdown automaton (NPDA) matching algorithm to
    match a string to a BNF grammar. It keep track of the current state of the matching process by
    maintaining several stacks internally as possible paths in the NPDA. Therefore, it supports
    continuous matching of characters and backtracking.

    It also supports detecting the rejected tokens at the current position, which helps the
    grammar-guided generation.

    Parameters
    ----------
    grammar : BNFGrammar
        The BNF grammar to match.

    max_rollback_steps : int
        The maximum number of steps to rollback when backtracking. Default: 0.
    """

    def __init__(self, grammar: BNFGrammar, max_rollback_steps: int = 0):
        self.__init_handle_by_constructor__(_ffi_api.GrammarMatcher, grammar, max_rollback_steps)  # type: ignore  # pylint: disable=no-member

    def accept_char(self, codepoint: int, drop_old: bool = True) -> bool:
        """Accept one unicode character to the current state.

        Parameters
        ----------
        codepoint : int
            The unicode codepoint of the character to be accepted.

        drop_old : bool
            If true, the old state will be dropped after accepting the new character when the number
            of states exceeds the limit of saved history.
        """
        return _ffi_api.GrammarMatcherAcceptChar(  # type: ignore  # pylint: disable=no-member
            self, codepoint, drop_old
        )

    def can_reach_end(self) -> bool:
        """Returns true if the matcher already reached the end of the grammar.

        Note
        ----
        Since the matcher maintains a non-deterministic state internally, even though the matcher
        reaches the end, it may still have other paths that can continue to accept new characters.
        """
        return _ffi_api.GrammarMatcherCanReachEnd(self)

    def match_complete_string(self, string: str) -> bool:
        """Check if a matcher can accept the complete string, and then reach the end of the
        grammar."""
        return _ffi_api.GrammarMatcherMatchCompleteString(self, string)

    def get_rejected_token_ids_for_tokenizer(
        self,
        grammar: BNFGrammar,
        tokenizer: "Tokenizer",
    ) -> List[int]:
        """Find the rejected tokens among all tokens in the tokenizer for the specified
        GrammarMatcher.

        Parameters
        ----------
        grammar : BNFGrammar
            The grammar associated to the matcher.
        tokenizer : Tokenizer
            The specified tokenizer.

        Returns
        -------
        rejected_token_ids : List[int]
            A list of rejected token ids.
        """

        return _ffi_api.GrammarMatcherGetRejectedTokenIdsForTokenizer(  # type: ignore  # pylint: disable=no-member
            self, grammar, tokenizer
        )
