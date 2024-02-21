import transformers
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaTokenizer

from transformers import GPTNeoXTokenizerFast

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", trust_remote_code=True)
print(tokenizer.backend_tokenizer)
# tokenizer = AutoTokenizer.from_pretrained("BlinkDL/rwkv-4-world", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(
#     "togethercomputer/RedPajama-INCITE-Chat-3B-v1", trust_remote_code=True
# )

# str_used = "粪 水 123"
str_used = "abc	abc\n"
utf8_bytes = str_used.encode("utf-8")
# utf8_bytes = [229, 155, 191]
print(type(utf8_bytes))
utf8_bytes_values = list(utf8_bytes)
utf8_bytes_strs = [chr(i) for i in utf8_bytes_values]
print(utf8_bytes_strs)
print(utf8_bytes_values)
utf8_original = bytes(utf8_bytes).decode("utf-8")
print(utf8_original)

res = tokenizer.encode(str_used)
# res = [1, 29871, 231, 187, 176]
print(res)
tokenized = tokenizer.tokenize(str_used)
print(tokenized)
tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized)
print(tokenized_ids)
tmp = [tokenizer.decode([i]) for i in res]
print(tmp)
orig = tokenizer.decode(res)
print(orig)

print(type(tokenizer))
# print(tokenizer.get_vocab())
# print the mapping of id to token
print("Added tokens")
# print(tokenizer.added_tokens_encoder)
print(tokenizer.vocab_size)
# for i in range(tokenizer.vocab_size):
#     print((i, tokenizer.convert_ids_to_tokens(i)))
