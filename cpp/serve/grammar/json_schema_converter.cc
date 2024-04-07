/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_state_matcher.cc
 */
#include <picojson.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

class IndentManager {
 public:
  IndentManager(std::optional<int> indent, const std::string& separator)
      : enable_newline_(indent.has_value()),
        indent_(indent.value_or(0)),
        separator_(separator),
        total_indent_(0),
        is_first_({true}) {}

  void StartIndent() {
    total_indent_ += indent_;
    is_first_.push_back(true);
  }

  void EndIndent() {
    total_indent_ -= indent_;
    is_first_.pop_back();
  }

  std::string NextSeparator(bool is_end = false) {
    std::string res = "";
    std::cout << "require sep: info " << is_first_.back() << " " << is_end
              << " res: " << (!is_first_.back() && !is_end) << std::endl;
    if (!is_first_.back() && !is_end) {
      res += separator_;
    }
    is_first_.back() = false;

    if (enable_newline_) {
      res += "\\n";
    }

    if (!is_end) {
      res += std::string(total_indent_, ' ');
    } else {
      res += std::string(total_indent_ - indent_, ' ');
    }

    return "\"" + res + "\"";
  }

  std::string GetBareSeparator() { return separator_; }

 private:
  bool enable_newline_;
  int indent_;
  std::string separator_;
  int total_indent_;
  std::vector<bool> is_first_;
  friend class JSONSchemaToEBNFConverter;
};

class JSONSchemaToEBNFConverter {
 public:
  JSONSchemaToEBNFConverter(
      const picojson::value& json_schema, std::optional<int> indent = std::nullopt,
      std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
      bool strict_mode = false)
      : json_schema_(json_schema), strict_mode_(strict_mode) {
    if (!separators.has_value()) {
      separators =
          (indent == std::nullopt) ? std::make_pair(", ", ": ") : std::make_pair(",", ": ");
    }
    indentManager_ = std::make_unique<IndentManager>(indent, separators->first);
    colon_ = separators->second;

    AddBasicRules();
  }

  std::string Convert() {
    CreateRuleFromSchema(json_schema_, "main");
    std::string res;
    for (auto& rule : rules_) {
      res += rule.first + " ::= " + rule.second + "\n";
    }
    return res;
  }

 private:
  std::unique_ptr<IndentManager> indentManager_;
  picojson::value json_schema_;
  bool strict_mode_;
  std::string colon_;
  std::vector<std::pair<std::string, std::string>> rules_;
  std::map<std::string, std::string> basic_rules_cache_;

  // The name of the basic rules
  inline static const std::string kBasicAny = "basic_any";
  inline static const std::string kBasicInteger = "basic_integer";
  inline static const std::string kBasicNumber = "basic_number";
  inline static const std::string kBasicString = "basic_string";
  inline static const std::string kBasicBoolean = "basic_boolean";
  inline static const std::string kBasicNull = "basic_null";
  inline static const std::string kBasicArray = "basic_array";
  inline static const std::string kBasicObject = "basic_object";

  // The name of the helper rules to construct basic rules
  inline static const std::string kBasicEscape = "basic_escape";
  inline static const std::string kBasicStringSub = "basic_string_sub";

  void AddBasicRules() {
    bool past_strict_mode = strict_mode_;
    strict_mode_ = false;

    auto past_indent_manager = std::move(indentManager_);
    indentManager_ =
        std::make_unique<IndentManager>(std::nullopt, past_indent_manager->GetBareSeparator());

    AddHelperRules();
    CreateBasicRule(picojson::value(true), kBasicAny);
    basic_rules_cache_[GetSchemaCacheIndex(picojson::value(picojson::object()))] = kBasicAny;
    CreateBasicRule(picojson::value(picojson::object{{"type", picojson::value("integer")}}),
                    kBasicInteger);
    CreateBasicRule(picojson::value(picojson::object{{"type", picojson::value("number")}}),
                    kBasicNumber);
    CreateBasicRule(picojson::value(picojson::object{{"type", picojson::value("string")}}),
                    kBasicString);
    CreateBasicRule(picojson::value(picojson::object{{"type", picojson::value("boolean")}}),
                    kBasicBoolean);
    CreateBasicRule(picojson::value(picojson::object{{"type", picojson::value("null")}}),
                    kBasicNull);
    CreateBasicRule(picojson::value(picojson::object{{"type", picojson::value("array")}}),
                    kBasicArray);
    CreateBasicRule(picojson::value(picojson::object{{"type", picojson::value("object")}}),
                    kBasicObject);

    strict_mode_ = past_strict_mode;
    indentManager_ = std::move(past_indent_manager);
  }

  void AddHelperRules() {
    rules_.push_back(std::make_pair(
        kBasicEscape, "[\"\\\\/bfnrt] | \"u\" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]"));
    rules_.push_back(std::make_pair(kBasicStringSub, "\"\" | [^\"\\\\\\r\\n] " + kBasicStringSub +
                                                         " | \"\\\\\" " + kBasicEscape + " " +
                                                         kBasicStringSub));
  }

  void CreateBasicRule(const picojson::value& schema, const std::string& name) {
    std::string rule_name = CreateRuleFromSchema(schema, name);
    basic_rules_cache_[GetSchemaCacheIndex(schema)] = rule_name;
  }

  std::string NextSeparator(bool is_end = false) { return indentManager_->NextSeparator(is_end); }

  static void WarnUnsupportedKeywords(const picojson::value& schema,
                                      const std::vector<std::string>& keywords) {
    if (schema.is<bool>()) {
      return;
    }

    ICHECK(schema.is<picojson::object>());
    WarnUnsupportedKeywords(schema.get<picojson::object>(), keywords);
  }

  static void WarnUnsupportedKeywords(const picojson::object& schema,
                                      const std::vector<std::string>& keywords) {
    for (const auto& keyword : keywords) {
      if (schema.find(keyword) != schema.end()) {
        LOG(WARNING) << "Keyword " << keyword << " is not supported in schema "
                     << picojson::value(schema);
      }
    }
  }

  std::string CreateRuleFromSchema(const picojson::value& schema,
                                   const std::string& rule_name_hint) {
    std::string idx = GetSchemaCacheIndex(schema);
    if (basic_rules_cache_.count(idx)) {
      return basic_rules_cache_[idx];
    }

    rules_.push_back(std::make_pair(rule_name_hint, VisitSchema(schema, rule_name_hint)));
    return rule_name_hint;
  }

  static picojson::value RemoveSkippedKeysRecursive(const picojson::value& obj) {
    static const std::unordered_set<std::string> kSkippedKeys = {
        "title",    "default",   "description", "examples", "deprecated",
        "readOnly", "writeOnly", "$comment",    "$schema",
    };
    if (obj.is<picojson::object>()) {
      picojson::object result;
      for (const auto& kv : obj.get<picojson::object>()) {
        if (kSkippedKeys.count(kv.first) == 0) {
          result[kv.first] = RemoveSkippedKeysRecursive(kv.second);
        }
      }
      return picojson::value(result);
    } else if (obj.is<picojson::array>()) {
      picojson::array result;
      for (const auto& item : obj.get<picojson::array>()) {
        result.push_back(RemoveSkippedKeysRecursive(item));
      }
      return picojson::value(result);
    }
    // If the object is neither an array nor an object, return it directly
    return obj;
  }

  std::string GetSchemaCacheIndex(const picojson::value& schema) {
    static const std::unordered_set<std::string> kSkippedKeys = {
        "title",    "default",   "description", "examples", "deprecated",
        "readOnly", "writeOnly", "$comment",    "$schema",
    };
    if (schema.is<picojson::object>()) {
      std::string result = "{";
      std::vector<std::pair<std::string, picojson::value>> sorted_kv;
      for (const auto& kv : schema.get<picojson::object>()) {
        if (kSkippedKeys.count(kv.first) == 0) {
          sorted_kv.push_back(kv);
        }
      }
      std::sort(sorted_kv.begin(), sorted_kv.end(),
                [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
      int idx = 0;
      for (const auto& [key, value] : sorted_kv) {
        if (idx != 0) {
          result += ",";
        }
        ++idx;
        result += "\"" + key + "\":" + GetSchemaCacheIndex(value);
      }
      return result + "}";
    } else if (schema.is<picojson::array>()) {
      std::string result = "[";
      int idx = 0;
      for (const auto& item : schema.get<picojson::array>()) {
        if (idx != 0) {
          result += ",";
        }
        ++idx;
        result += GetSchemaCacheIndex(item);
      }
      return result + "]";
    }
    // If the object is neither an array nor an object, return it directly
    return schema.serialize(false);
  }

  std::string VisitSchema(const picojson::value& schema, const std::string& rule_name) {
    if (schema.is<bool>()) {
      ICHECK(schema.get<bool>());
      return VisitAny(schema, rule_name);
    }

    WarnUnsupportedKeywords(schema, {
                                        "allof",
                                        "oneof",
                                        "not",
                                        "if",
                                        "then",
                                        "else",
                                        "dependentRequired",
                                        "dependentSchemas",
                                    });

    ICHECK(schema.is<picojson::object>());

    const auto& schema_obj = schema.get<picojson::object>();

    if (schema_obj.count("$ref")) {
      return VisitRef(schema_obj, rule_name);
    } else if (schema_obj.count("const")) {
      return VisitConst(schema_obj, rule_name);
    } else if (schema_obj.count("enum")) {
      return VisitEnum(schema_obj, rule_name);
    } else if (schema_obj.count("anyOf")) {
      return VisitAnyOf(schema_obj, rule_name);
    } else if (schema_obj.count("type")) {
      const std::string& type = schema_obj.at("type").get<std::string>();
      if (type == "integer") {
        return VisitInteger(schema_obj, rule_name);
      } else if (type == "number") {
        return VisitNumber(schema_obj, rule_name);
      } else if (type == "string") {
        return VisitString(schema_obj, rule_name);
      } else if (type == "boolean") {
        return VisitBoolean(schema_obj, rule_name);
      } else if (type == "null") {
        return VisitNull(schema_obj, rule_name);
      } else if (type == "array") {
        return VisitArray(schema_obj, rule_name);
      } else if (type == "object") {
        return VisitObject(schema_obj, rule_name);
      } else {
        LOG(FATAL) << "Unsupported type " << type << " in schema " << schema;
      }
    }

    return VisitAny(schema, rule_name);
  }

  std::string VisitRef(const picojson::object& schema, const std::string& rule_name) {
    ICHECK(schema.count("$ref"));
    picojson::value new_schema = URIToSchema(schema.at("$ref"));
    if (!new_schema.is<bool>()) {
      picojson::object new_schema_obj = new_schema.get<picojson::object>();
      for (const auto& [k, v] : schema) {
        if (k != "$ref") {
          new_schema_obj[k] = v;
        }
      }
      new_schema = picojson::value(new_schema_obj);
    }
    return VisitSchema(new_schema, rule_name);
  }

  picojson::value URIToSchema(const picojson::value& uri) {
    if (uri.get<std::string>().substr(0, 8) == "#/$defs/") {
      return json_schema_.get("$defs").get(uri.get<std::string>().substr(8));
    }
    LOG(WARNING) << "Now only support URI starting with '#/$defs/' but got " << uri;
    return picojson::value(true);
  }

  std::string VisitConst(const picojson::object& schema, const std::string& rule_name) {
    ICHECK(schema.count("const"));
    // TODO(yixin): Customize serialize to support indent logics
    return "\"" + JSONStrToPrintableStr(schema.at("const").serialize()) + "\"";
  }

  std::string VisitEnum(const picojson::object& schema, const std::string& rule_name) {
    ICHECK(schema.count("enum"));
    std::string result = "";
    int idx = 0;
    for (auto value : schema.at("enum").get<picojson::array>()) {
      if (idx != 0) {
        result += " | ";
      }
      ++idx;
      result += "(\"" + JSONStrToPrintableStr(value.serialize()) + "\")";
    }
    return result;
  }

  std::string JSONStrToPrintableStr(const std::string& json_str) {
    static const std::unordered_map<std::string, std::string> kReplaceMapping = {{"\\", "\\\\"},
                                                                                 {"\"", "\\\""}};
    std::string result = json_str;
    for (const auto& [k, v] : kReplaceMapping) {
      size_t pos = 0;
      while ((pos = result.find(k, pos)) != std::string::npos) {
        result.replace(pos, k.length(), v);
        pos += v.length();
      }
    }
    return result;
  }

  std::string VisitAnyOf(const picojson::object& schema, const std::string& rule_name) {
    ICHECK(schema.count("anyOf"));
    std::string result = "";
    int idx = 0;
    for (auto anyof_schema : schema.at("anyOf").get<picojson::array>()) {
      if (idx != 0) {
        result += " | ";
      }
      ++idx;
      result += CreateRuleFromSchema(anyof_schema, rule_name + "_" + std::to_string(idx));
    }
    return result;
  }

  std::string VisitAny(const picojson::value& schema, const std::string& rule_name) {
    return kBasicNumber + " | " + kBasicString + " | " + kBasicBoolean + " | " + kBasicNull +
           " | " + kBasicArray + " | " + kBasicObject;
  }

  std::string VisitInteger(const picojson::object& schema, const std::string& rule_name) {
    ICHECK(schema.count("type"));
    ICHECK(schema.at("type").get<std::string>() == "integer");
    WarnUnsupportedKeywords(schema, {
                                        "multipleOf",
                                        "minimum",
                                        "maximum",
                                        "exclusiveMinimum",
                                        "exclusiveMaximum",
                                    });
    return "(\"0\" | \"-\"? [1-9] [0-9]*) \".0\"?";
  }

  std::string VisitNumber(const picojson::object& schema, const std::string& rule_name) {
    ICHECK(schema.count("type"));
    ICHECK(schema.at("type").get<std::string>() == "number");
    WarnUnsupportedKeywords(schema, {
                                        "multipleOf",
                                        "minimum",
                                        "maximum",
                                        "exclusiveMinimum",
                                        "exclusiveMaximum",
                                    });
    return "(\"0\" | \"-\"? [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?";
  }
  std::string VisitString(const picojson::object& schema, const std::string& rule_name) {
    ICHECK(schema.count("type"));
    ICHECK(schema.at("type").get<std::string>() == "string");
    WarnUnsupportedKeywords(schema, {
                                        "minLength",
                                        "maxLength",
                                        "pattern",
                                        "format",
                                    });
    return "[\"] " + kBasicStringSub + " [\"]";
  }

  std::string VisitBoolean(const picojson::object& schema, const std::string& rule_name) {
    ICHECK(schema.count("type"));
    ICHECK(schema.at("type").get<std::string>() == "boolean");
    return "\"true\" | \"false\"";
  }

  std::string VisitNull(const picojson::object& schema, const std::string& rule_name) {
    ICHECK(schema.count("type"));
    ICHECK(schema.at("type").get<std::string>() == "null");
    return "\"null\"";
  }

  std::string VisitArray(const picojson::object& schema, const std::string& rule_name) {
    ICHECK(schema.count("type"));
    ICHECK(schema.at("type").get<std::string>() == "array");
    WarnUnsupportedKeywords(schema, {
                                        "uniqueItems",
                                        "contains",
                                        "minContains",
                                        "maxContains",
                                        "minItems",
                                        "maxItems",
                                    });

    std::string result = "\"[\"";

    indentManager_->StartIndent();

    // 1. Handle prefix items
    if (schema.count("prefixItems")) {
      const auto& prefix_items = schema.at("prefixItems").get<picojson::array>();
      for (int i = 0; i < prefix_items.size(); ++i) {
        ICHECK(prefix_items[i].is<picojson::object>());
        result += " " + NextSeparator() + " " +
                  CreateRuleFromSchema(prefix_items[i], rule_name + "_" + std::to_string(i));
      }
    }

    // 2. Find additional items
    picojson::value additional_item = picojson::value(false);
    std::string additional_suffix = "";

    if (schema.count("items") &&
        (!schema.at("items").is<bool>() || schema.at("items").get<bool>())) {
      additional_item = schema.at("items");
      additional_suffix = "item";
    }

    if (schema.count("items") == 0) {
      picojson::value unevaluated = schema.count("unevaluatedItems")
                                        ? schema.at("unevaluatedItems")
                                        : picojson::value(!strict_mode_);
      if (!unevaluated.is<bool>() || unevaluated.get<bool>()) {
        additional_item = unevaluated;
        additional_suffix = "uneval";
      }
    }

    // 3. Handle additional items and the end separator
    bool could_be_empty = false;
    if (additional_item.is<bool>() && !additional_item.get<bool>()) {
      result += " " + NextSeparator(true);
    } else {
      std::string additional_pattern =
          CreateRuleFromSchema(additional_item, rule_name + "_" + additional_suffix);
      std::cout << "additional_pattern: " << additional_pattern << std::endl;
      if (schema.count("prefixItems")) {
        result += " (" + NextSeparator() + " " + additional_pattern + ")* " + NextSeparator(true);
      } else {
        std::cout << "prev result: " << result << std::endl;
        result += " " + NextSeparator() + " " + additional_pattern + " (" + NextSeparator() + " " +
                  additional_pattern + ")* " + NextSeparator(true);
        std::cout << "updated result: " << result << std::endl;
        could_be_empty = true;
      }
    }

    indentManager_->EndIndent();

    result += " \"]\"";

    if (could_be_empty) {
      result = "(" + result + ") | \"[]\"";
    }

    return result;
  }
  std::string GetPropertyPattern(const std::string& prop_name, const picojson::value& prop_schema,
                                 const std::string& rule_name) {
    std::string key = "\"\\\"" + prop_name + "\\\"\"";
    std::string colon = "\"" + colon_ + "\"";
    std::string value = CreateRuleFromSchema(prop_schema, rule_name + "_" + prop_name);
    return key + " " + colon + " " + value;
  }

  std::string GetOtherPropertyPattern(const std::string& key_pattern,
                                      const picojson::value& prop_schema,
                                      const std::string& rule_name,
                                      const std::string& rule_name_suffix) {
    std::string colon = "\"" + colon_ + "\"";
    std::string value = CreateRuleFromSchema(prop_schema, rule_name + "_" + rule_name_suffix);
    return key_pattern + " " + colon + " " + value;
  }

  std::string GetPartialRuleForPropertiesAllOptional(
      const std::vector<std::pair<std::string, picojson::value>>& properties,
      const picojson::value& additional, const std::string& rule_name,
      const std::string& additional_suffix = "") {
    ICHECK(properties.size() >= 1);

    std::string first_sep = NextSeparator();
    std::string mid_sep = NextSeparator();
    std::string last_sep = NextSeparator(true);

    std::string res = "";

    std::vector<std::string> prop_patterns;
    for (const auto& [prop_name, prop_schema] : properties) {
      prop_patterns.push_back(GetPropertyPattern(prop_name, prop_schema, rule_name));
    }

    std::vector<std::string> rule_names(properties.size(), "");

    // construct the last rule
    std::string additional_prop_pattern;
    if (!additional.is<bool>() || additional.get<bool>()) {
      additional_prop_pattern =
          GetOtherPropertyPattern(kBasicString, additional, rule_name, additional_suffix);
      std::string last_rule_body = "(" + mid_sep + " " + additional_prop_pattern + ")*";
      std::string last_rule_name = rule_name + "_sub_" + std::to_string(properties.size() - 1);
      rules_.push_back(std::make_pair(last_rule_name, last_rule_body));
      rule_names.back() = last_rule_name;
    } else {
      rule_names.back() = "\"\"";
    }

    // construct 0~(len(properties) - 2) rules
    for (int i = properties.size() - 2; i >= 0; --i) {
      const std::string& prop_pattern = prop_patterns[i + 1];
      const std::string& last_rule_name = rule_names[i + 1];
      std::string cur_rule_body =
          last_rule_name + " | " + mid_sep + " " + prop_pattern + " " + last_rule_name;
      std::string cur_rule_name = rule_name + "_sub_" + std::to_string(i);
      rules_.push_back(std::make_pair(cur_rule_name, cur_rule_body));
      rule_names[i] = cur_rule_name;
    }

    // construct the main rule
    for (int i = 0; i < properties.size(); ++i) {
      if (i != 0) {
        res += " | ";
      }
      res += "(" + prop_patterns[i] + " " + rule_names[i] + ")";
    }

    if (!additional.is<bool>() || additional.get<bool>()) {
      res += " | " + additional_prop_pattern + " " + rule_names.back();
    }

    // add separators and the empty string option
    res = first_sep + " (" + res + ") " + last_sep;
    return res;
  }

  std::string GetPartialRuleForPropertiesContainRequired(
      const std::vector<std::pair<std::string, picojson::value>>& properties,
      const std::unordered_set<std::string>& required, const std::string& rule_name) {
    // Find the index of the first required property
    int first_required_idx = properties.size();
    for (int i = 0; i < properties.size(); ++i) {
      if (required.count(properties[i].first)) {
        first_required_idx = i;
        break;
      }
    }
    ICHECK(first_required_idx < properties.size());

    std::string res = NextSeparator();

    // Handle the properties before the first required property
    for (int i = 0; i < first_required_idx; ++i) {
      const auto& [prop_name, prop_schema] = properties[i];
      ICHECK(!prop_schema.is<bool>() || prop_schema.get<bool>());
      std::string property_pattern = GetPropertyPattern(prop_name, prop_schema, rule_name);
      res += " (" + property_pattern + " " + NextSeparator() + ")?";
    }

    // Handle the first required property
    const auto& [prop_name, prop_schema] = properties[first_required_idx];
    std::string property_pattern = GetPropertyPattern(prop_name, prop_schema, rule_name);
    res += " " + property_pattern;

    // Handle the properties after the first required property
    for (int i = first_required_idx + 1; i < properties.size(); ++i) {
      const auto& [prop_name, prop_schema] = properties[i];
      ICHECK(!prop_schema.is<bool>() || prop_schema.get<bool>());
      std::string property_pattern = GetPropertyPattern(prop_name, prop_schema, rule_name);
      if (required.count(prop_name)) {
        res += " " + NextSeparator() + " " + property_pattern;
      } else {
        res += " (" + NextSeparator() + " " + property_pattern + ")?";
      }
    }

    return res;
  }

  std::string VisitObject(const picojson::object& schema, const std::string& rule_name) {
    ICHECK(schema.count("type"));
    ICHECK(schema.at("type").get<std::string>() == "object");
    WarnUnsupportedKeywords(schema, {
                                        "patternProperties",
                                        "minProperties",
                                        "maxProperties",
                                        "propertyNames",
                                    });

    std::string result = "\"{\"";
    bool could_be_empty = false;

    indentManager_->StartIndent();

    // 1. Find additional properties
    picojson::value additional_property = picojson::value(false);
    std::string additional_suffix = "";

    if (schema.count("additionalProperties") && (!schema.at("additionalProperties").is<bool>() ||
                                                 schema.at("additionalProperties").get<bool>())) {
      additional_property = schema.at("additionalProperties");
      additional_suffix = "add";
    }

    if (schema.count("additionalProperties") == 0) {
      picojson::value unevaluated = schema.count("unevaluatedProperties")
                                        ? schema.at("unevaluatedProperties")
                                        : picojson::value(!strict_mode_);
      if (!unevaluated.is<bool>() || unevaluated.get<bool>()) {
        additional_property = unevaluated;
        additional_suffix = "uneval";
      }
    }

    // 2. Handle properties
    std::vector<std::pair<std::string, picojson::value>> properties;
    if (schema.count("properties")) {
      for (const auto& [prop_name, prop_schema] : schema.at("properties").get<picojson::object>()) {
        properties.push_back({prop_name, prop_schema});
      }
    }

    std::unordered_set<std::string> required;
    if (schema.count("required")) {
      for (const auto& required_prop : schema.at("required").get<picojson::array>()) {
        required.insert(required_prop.get<std::string>());
      }
    }

    bool is_all_properties_optional =
        std::all_of(properties.begin(), properties.end(),
                    [&](const auto& prop) { return required.count(prop.first) == 0; });

    if (is_all_properties_optional && properties.size() > 0) {
      // 3.1 Case 1: properties are defined and all properties are optional
      result += " " + GetPartialRuleForPropertiesAllOptional(properties, additional_property,
                                                             rule_name, additional_suffix);
      could_be_empty = true;
    } else if (properties.size() > 0) {
      // 3.2 Case 2: properties are defined and some properties are required
      result += " " + GetPartialRuleForPropertiesContainRequired(properties, required, rule_name);
      if (!additional_property.is<bool>() || additional_property.get<bool>()) {
        std::string other_property_pattern = GetOtherPropertyPattern(
            kBasicString, additional_property, rule_name, additional_suffix);
        result += " (" + NextSeparator() + " " + other_property_pattern + ")*";
      }
      result += " " + NextSeparator(true);
    } else if (!additional_property.is<bool>() || additional_property.get<bool>()) {
      // 3.3 Case 3: no properties are defined and additional properties are allowed
      std::string other_property_pattern =
          GetOtherPropertyPattern(kBasicString, additional_property, rule_name, additional_suffix);
      result += " " + NextSeparator() + " " + other_property_pattern + " (" + NextSeparator() +
                " " + other_property_pattern + ")* " + NextSeparator(true);
      could_be_empty = true;
    }

    result += " \"}\"";
    if (could_be_empty) {
      result = "(" + result + ") | \"{}\"";
    }
    return result;
  };
};

std::string JSONSchemaToEBNF(std::string schema, std::optional<int> indent,
                             std::optional<std::pair<std::string, std::string>> separators,
                             bool strict_mode) {
  picojson::value schema_value;
  std::string err = picojson::parse(schema_value, schema);
  if (!err.empty()) {
    LOG(FATAL) << "Failed to parse JSON: err. The JSON string is:" << schema;
  }
  JSONSchemaToEBNFConverter converter(schema_value, indent, separators, strict_mode);
  return converter.Convert();
}

TVM_REGISTER_GLOBAL("mlc.serve.DebugJSONSchemaToEBNF").set_body([](TVMArgs args, TVMRetValue* rv) {
  std::optional<int> indent;
  if (args[1].type_code() != kTVMNullptr) {
    indent = args[1];
  } else {
    indent = std::nullopt;
  }

  std::optional<std::pair<std::string, std::string>> separators;
  if (args[2].type_code() != kTVMNullptr) {
    Array<String> separators_arr = args[2];
    CHECK(separators_arr.size() == 2);
    separators = std::make_pair(separators_arr[0], separators_arr[1]);
  } else {
    separators = std::nullopt;
  }

  *rv = JSONSchemaToEBNF(args[0], indent, separators, args[3]);
});

}  // namespace serve
}  // namespace llm
}  // namespace mlc
