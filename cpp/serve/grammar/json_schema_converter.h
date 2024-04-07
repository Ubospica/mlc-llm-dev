/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_JSON_SCHEMA_CONVERTER_H_
#define MLC_LLM_SERVE_GRAMMAR_JSON_SCHEMA_CONVERTER_H_

#include <optional>
#include <utility>

namespace mlc {
namespace llm {
namespace serve {

std::string JSONSchemaToEBNF(
    std::string schema, std::optional<int> indent = std::nullopt,
    std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
    bool strict_mode = true);

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_JSON_SCHEMA_CONVERTER_H_
