/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.cc
 */

#include "grammar.h"

#include "grammar_parser.h"
#include "grammar_serializer.h"
#include "grammar_simplifier.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_REGISTER_OBJECT_TYPE(BNFGrammarNode);

std::ostream& operator<<(std::ostream& os, const BNFGrammar& grammar) {
  os << BNFGrammarPrinter(grammar).ToString();
  return os;
}

BNFGrammar BNFGrammar::FromEBNFString(const String& ebnf_string, bool unwrap_nesting_rules,
                                      bool simplify) {
  auto grammar = EBNFParser::Parse(ebnf_string);
  if (unwrap_nesting_rules) {
    grammar = NestedRuleUnwrapper(grammar).Apply();
  }
  if (simplify) {
    grammar = BNFGrammarSimplifier(grammar).Apply();
  }
  return grammar;
}

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarFromEBNFString")
    .set_body_typed([](String ebnf_string, bool unwrap_nesting_rules, bool simplify) {
      return BNFGrammar::FromEBNFString(ebnf_string, unwrap_nesting_rules, simplify);
    });

BNFGrammar BNFGrammar::FromJSON(const String& json_string) {
  return BNFJSONParser::Parse(json_string);
}

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarFromJSON").set_body_typed([](String json_string) {
  return BNFGrammar::FromJSON(json_string);
});

const std::string kJSONGrammarString = R"(
main ::= (
    "{" ws "}" ws |
    "{" members "}" ws |
    "[" ws "]" ws |
    "[" elements "]" ws |
    "\"" characters "\"" ws |
    [0-9] fraction exponent ws |
    [1-9] digits fraction exponent ws |
    "-" [0-9] fraction exponent ws |
    "-" [1-9] digits fraction exponent ws |
    "true" ws | "false" ws | "null" ws
    "\u0020" ws value ws |
    "\u000A" ws value ws |
    "\u000D" ws value ws |
    "\u0009" ws value ws
)
value ::= (
    "{" ws "}" |
    "{" members "}" |
    "[" ws "]" |
    "[" elements "]" |
    "\"" characters "\"" |
    [0-9] fraction exponent |
    [1-9] digits fraction exponent |
    "-" [0-9] fraction exponent |
    "-" [1-9] digits fraction exponent |
    "true" | "false" | "null"
)
object ::= "{" ws "}" | "{" members "}"
members ::= (
    "\"" characters "\"" ws ":" ws value ws |
    "\u0020" ws "\"" characters "\"" ws ":" ws value ws |
    "\u000A" ws "\"" characters "\"" ws ":" ws value ws |
    "\u000D" ws "\"" characters "\"" ws ":" ws value ws |
    "\u0009" ws "\"" characters "\"" ws ":" ws value ws |
    "\"" characters "\"" ws ":" ws value ws "," members |
    "\u0020" ws "\"" characters "\"" ws ":" ws value ws "," members |
    "\u000A" ws "\"" characters "\"" ws ":" ws value ws "," members |
    "\u000D" ws "\"" characters "\"" ws ":" ws value ws "," members |
    "\u0009" ws "\"" characters "\"" ws ":" ws value ws "," members |
)
array ::= "[" ws "]" | "[" elements "]"
elements ::= (
    "{" ws "}" ws elements_rest |
    "{" members "}" ws elements_rest |
    "[" ws "]" ws elements_rest |
    "[" elements "]" ws elements_rest |
    "\"" characters "\"" ws elements_rest |
    [0-9] fraction exponent ws elements_rest |
    [1-9] digits fraction exponent ws elements_rest |
    "-" [0-9] fraction exponent ws elements_rest |
    "-" [1-9] digits fraction exponent ws elements_rest |
    "true" ws elements_rest |
    "false" ws elements_rest |
    "null" ws elements_rest |
    "\u0020" ws value ws elements_rest |
    "\u000A" ws value ws elements_rest |
    "\u000D" ws value ws elements_rest |
    "\u0009" ws value ws elements_rest
)
elements_rest ::= "" | "," elements
characters ::= "" | [^"\\] characters | "\\" escape characters
escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
integer ::= [0-9] | [1-9] digits | "-" [0-9] | "-" [1-9] digits
digits ::= [0-9] | [0-9] digits
fraction ::= "" | "." digits
exponent ::= "" |  "e" sign digits | "E" sign digits
sign ::= "" | "+" | "-"
ws ::= "" | "\u0020" ws | "\u000A" ws | "\u000D" ws | "\u0009" ws
)";

BNFGrammar BNFGrammar::GetJSONGrammar() {
  static BNFGrammar grammar = BNFGrammar::FromEBNFString(kJSONGrammarString);
  return grammar;
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
