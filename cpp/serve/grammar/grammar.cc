/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.cc
 */

#include "grammar.h"

#include "grammar_parser.h"
#include "grammar_serializer.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_REGISTER_OBJECT_TYPE(BNFGrammarNode);

std::ostream& operator<<(std::ostream& os, const BNFGrammar& grammar) {
  os << BNFGrammarPrinter(grammar).ToString();
  return os;
}

BNFGrammar BNFGrammar::FromEBNFString(const String& ebnf_string) {
  return EBNFParser::Parse(ebnf_string);
}

BNFGrammar BNFGrammar::FromJSON(const String& json_string) {
  return BNFJSONParser::Parse(json_string);
}



BNFGrammar BNFGrammar::GetJSONGrammar() {
  return BNFJSONParser::GetJSONGrammar();
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

}  // namespace serve
}  // namespace llm
}  // namespace mlc
