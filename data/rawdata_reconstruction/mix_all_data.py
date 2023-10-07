import json
from random import sample, shuffle

# generate train data
data1 = json.load(open("dialogsum/dialogsum_train_question.json", "r"))
data2 = json.load(open("topiocqa/topiocqa_train_question.json", "r"))
data3 = json.load(open("alpacagpt4/alpaca_train_question.json", "r"))

data = data1 + data2 + data3
data = [{"id": str(d_i), "conversations": d["conversations"], "type": d["type"]} for d_i, d in enumerate(data)]
shuffle(data)

check_dict = {}
for d in data:
    check_dict[d["type"]] = check_dict.get(d["type"], 0) + 1
json.dump(sample(data, int(len(data) * 0.1)), open('../memochat_instructions/train_1k.json', 'w'), indent=2)
json.dump(data, open('../memochat_instructions/train_10k.json', 'w'), indent=2)
print(check_dict)


# generate test data
data1 = open("dialogsum/dialogsum_test_question.jsonl", "r").readlines()
data2 = open("topiocqa/topiocqa_test_question.jsonl", "r").readlines()
data = data1 + data2

check_dict = {}
w = open("../memochat_instructions/test.jsonl", "w")
for d_i, d in enumerate(data):
    d = json.loads(d.strip())
    d["question_id"] = str(d_i)
    check_dict[d["type"]] = check_dict.get(d["type"], 0) + 1
    w.write(json.dumps(d) + "\n")
w.close()
print(check_dict)
