import json
from random import random, choice

prompts = json.load(open("../../prompts.json", "r"))
system_insturction = prompts["chatting"]["system"]

data = json.load(open("alpaca_gpt4.json", "r"))
prob1 = 2000 / len(data)
new_data = []
num_count_1, num_count_2 = 0, 0
for d in data:
    if random() >= prob1:
        continue
    new_data_r = {"id": d["id"], "conversations": d["conversations"], "type": "chatting_alpacagpt4"}
    if (random() > 0.35) or (len(d["conversations"][1]["value"].split(" ")) < 32) or ("\n" not in d["conversations"][1]["value"]):
        new_data_r["conversations"][0]["value"] = system_insturction + "```\nRelated Evidences:\n" + "\nRecent Dialogs:\n" + "\n```\n\nUser Input:\n" + \
                                                  "user: " + new_data_r["conversations"][0]["value"] + " ### bot: "
        num_count_1 += 1
    elif len([ob for ob in new_data_r["conversations"][1]["value"].split("\n") if ob.strip() != ""]) >= 2:
        org_bots = [ob for ob in new_data_r["conversations"][1]["value"].split("\n") if ob.strip() != ""]
        select_i = choice(list(range(1, len(org_bots))))
        task_dialog = "user: " + new_data_r["conversations"][0]["value"] + " ### bot: " + "\n".join(org_bots[:select_i])
        new_user = choice(["continue", "keep going", "please continue", "tell me more"])
        new_data_r["conversations"][0]["value"] = system_insturction + "```\nRelated Evidences:\n" + "\nRecent Dialogs:\n" + task_dialog + "\n```\n\nUser Input:\n" + \
                                                  "user: " + new_user + " ### bot: "
        new_data_r["conversations"][1]["value"] = "\n".join(org_bots[select_i:])
        num_count_2 += 1
    else:
        continue
    new_data.append(new_data_r)
    print(new_data_r["conversations"][0]["value"])
    print(new_data_r["conversations"][1]["value"])
    print("=" * 20)
    if len(new_data) == 1698:
        break
json.dump(new_data, open("alpaca_train_question.json", "w"), indent=2)
print("{} continue chatting samples, {} normal chatting samples, {} all chatting samples".format(num_count_2, num_count_1, len(new_data)))
