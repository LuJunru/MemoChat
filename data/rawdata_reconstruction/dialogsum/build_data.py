import json
from random import choice, sample, shuffle, random

prompts = json.load(open("../../prompts.json", "r"))
data = open('raw_data/dialogsum.train.jsonl', 'r').readlines()
shuffle(data)

new_data = []
last_count = 0
count = 0
num_count = 0
num_count_1 = 0
num_count_2 = 0
num_count_3 = 0
while count < len(data):
    num_shift = choice([1] * 1 + [2] * 5 + [3] * 2 + [4] * 1 + [5] * 1)
    count += num_shift
    lines = data[last_count:count]
    
    new_row = {"id": "", "conversations": [{"from": "human", "value": ""}, {"from": "bot", "value": ""}], "type": ""}
    for line in lines:
        record = json.loads(line.strip())
        fname = record["fname"]
        if new_row["id"] == "":
            new_row["id"] = fname
        else:
            new_row["id"] += "_" + fname.split("_")[1]
    
    case_num = choice([1, 2, 3])

    # construct memo writing sample
    if case_num == 1 and num_shift <= 3:
        system_insturction = prompts["writing_dialogsum"]["system"]
        task_instruction = prompts["writing_dialogsum"]["instruction"]
    
        conversation = []
        answers = []
        for line in lines:
            record = json.loads(line.strip())
            t_lines = record["dialogue"].split("\n")
            for t_line in t_lines:
                if ": " not in t_line:
                    t_line = t_line.replace(":", ": ")
                t_line = t_line.replace("#Person1#", "user").replace("#Person2#", "bot")
                conversation.append('(line {}) {}'.format(len(conversation) + 1, t_line))
            answers.append({"topic": record["topic"], "summary": record["summary"].replace("#Person1#", "user").replace("#Person2#", "bot"), "start": len(conversation) - len(t_lines) + 1, "end": len(conversation)})
        new_row["conversations"][0]["value"] = system_insturction.replace("LINE", str(len(conversation))) + "\n\n```\nTask Conversation:\n" + "\n".join(conversation) + "\n```" + task_instruction.replace("LINE", str(len(conversation)))
        new_row["conversations"][1]["value"] = json.dumps(answers)
        new_row["type"] = "writing_dialogsum"

    # construct memo retrival sample
    if case_num == 2 or (case_num == 1 and num_shift >= 4):
        system_insturction = prompts["retrieval"]["system"]
        task_instruction = prompts["retrieval"]["instruction"]
        new_conversation = []
        turn_topics = []
        for line in lines:
            record = json.loads(line.strip())
            t_lines = record["dialogue"].split("\n")
            for t_i, t_line in enumerate(t_lines):
                if t_i % 2 == 1:
                    t_line_1 = t_lines[t_i - 1].split(":")[1].strip()
                    t_line = t_line.split(":")[1].strip()
                    new_conversation.append((t_line_1 + " " + t_line, record["topic"]))
                    turn_topics.append((record["topic"], record["summary"].replace("#Person1#", "user").replace("#Person2#", "bot")))
        select_query_sents = sample(new_conversation, choice(list(range(1, min(4, len(new_conversation) + 1)))))

        # create NOTO special cases with prob = 0.1
        if random() <= 0.1:
            seen_topics = set()
            for s_i, select_query_sent in enumerate(select_query_sents):
                seen_topics.add(select_query_sent[1])
                select_query_sents[s_i] = (select_query_sent[0], "NOTO")
            for t_i, t_ in enumerate(turn_topics):
                if t_ in seen_topics:
                    turn_topics[t_i] = ("NOTO", "None of the others.")

        query_sents = []
        answers = set()
        options = {t_[0]: ("({}) {}".format(str(t_i + 1), t_[0] + ". " + t_[1]), str(t_i + 1)) for t_i, t_ in enumerate(set(turn_topics + [("NOTO", "None of the others.")]))}
        for select_query_sent in select_query_sents:
            query_sents.append(select_query_sent[0])
            answers.add(options[select_query_sent[1]][1])
        task_case = "```\nQuery Sentence:\n" + " ".join(query_sents) + "\nTopic Options:\n" + "\n".join([v[0] for v in list(options.values())]) + "\n```"
        new_row["conversations"][0]["value"] = system_insturction.replace("OPTION", str(len(options.keys()))) + task_case + task_instruction.replace("OPTION", str(len(options.keys())))
        new_row["conversations"][1]["value"] = "#".join(list(sorted(answers)))
        new_row["type"] = "retrieval_dialogsum"

    # construct chatting with memo sample
    if case_num == 3:
        system_insturction = prompts["chatting"]["system"]
        task_instruction = prompts["chatting"]["instruction"]
        data_set = {
            "Recent Dialogs": "", 
            "Related Topics": "", 
            "Related Summaries": "", 
            "Related Dialogs": "", 
            "User Input": "",
            "answer": ""
        }

        """ 
        1 - pick the last dialog as current dialog
        2 - random pick 1 turn in current dialog, then select 5 previous turns as recent dialogs (may across to last dialog)
        3 - random pick 1 ~ 3 turns after the picked turn in (2) as related dialogs
        4 - Also, use a lower prob to random pick 1 turn from other dialog to use as related dialogs. This turn is not related, but just to support cross-topic scenarios.
        """
        cur_dialog = json.loads(lines[-1].strip())
        cur_dialog_lines = cur_dialog["dialogue"].split("\n")
        select_cur_dialog_index = choice(list(range(len(cur_dialog_lines) // 2)))
        select_cur_dialog_sent = cur_dialog_lines[2 * select_cur_dialog_index]
        if "#Person1#" in select_cur_dialog_sent:
            user_p = "#Person1#"
            bot_p = "#Person2#"
        else:
            user_p = "#Person2#"
            bot_p = "#Person1#"
        data_set["User Input"] = "user: " + select_cur_dialog_sent.split(":")[1].strip() + " ### bot: "
        data_set["answer"] = cur_dialog_lines[2 * select_cur_dialog_index + 1].split(":")[1].strip()
        data_set["Recent Dialogs"] = cur_dialog_lines[max(0, (2 * select_cur_dialog_index - 10)):(2 * select_cur_dialog_index)]

        pr_dialogs = lines[:-1]
        if len(pr_dialogs) >= 1 and len(data_set["Recent Dialogs"]) < 10:
            data_set["Recent Dialogs"] = json.loads(pr_dialogs[-1].strip())["dialogue"].split("\n")[-(10 - len(data_set["Recent Dialogs"])):] + data_set["Recent Dialogs"]
            turn_last_dialog = json.loads(pr_dialogs[-1].strip())
            turn_last_dialog["dialogue"] = "\n".join(turn_last_dialog["dialogue"].split("\n")[:-(10 - len(data_set["Recent Dialogs"]))])
            pr_dialogs[-1] = json.dumps(turn_last_dialog)
        data_set["Recent Dialogs"] = " ### ".join(["user: " + dn.split(":")[1].strip() if user_p in dn else "bot: " + dn.split(":")[1].strip() for dn in data_set["Recent Dialogs"]])

        if len(cur_dialog_lines[2 * select_cur_dialog_index + 2:]) > 0:
            related_materials = [(cur_dialog_lines[2 * select_cur_dialog_index + 2:], cur_dialog["summary"], cur_dialog["topic"])]
        else:
            related_materials = []
        if random() < 0.1 and len(pr_dialogs) >= 1:
            chosen_dialog_c = choice(list(range(min(len(pr_dialogs), 3))))
            chosen_dialogs = sample(pr_dialogs, chosen_dialog_c)
            for chosen_dialog in chosen_dialogs:
                chosen_dialog_r = json.loads(chosen_dialog.strip())
                if len(chosen_dialog_r["dialogue"].split("\n")[0]) > 0:
                    related_materials.append((chosen_dialog_r["dialogue"].split("\n"), chosen_dialog_r["summary"], chosen_dialog_r["topic"]))
        data_set["Related Topics"] = [rs[2] for rs in related_materials]
        data_set["Related Summaries"] = [rs[1].replace(user_p, "user").replace(bot_p, "bot") for rs in related_materials]
        data_set["Related Dialogs"] = [" ### ".join(["user: " + dn.split(":")[1].strip() if user_p in dn else "bot: " + dn.split(":")[1].strip() for dn in rs[0]]) for rs in related_materials]

        task_case = "```\nRelated Evidences:\n" + "\n".join(["({}) {}".format(r_tsd_i + 1, {
                        "Related Topics": data_set["Related Topics"][r_tsd_i], 
                        "Related Summaries": data_set["Related Summaries"][r_tsd_i], 
                        "Related Dialogs": data_set["Related Dialogs"][r_tsd_i]
                    }) for r_tsd_i in range(len(data_set["Related Topics"]))]) + "\n\nRecent Dialogs:\n" + data_set["Recent Dialogs"] + \
                    "\n```\n\nUser Input:\n" + data_set["User Input"]
        new_row["conversations"][0]["value"] = system_insturction + task_case + task_instruction
        new_row["conversations"][1]["value"] = data_set["answer"]
        new_row["type"] = "chatting_dialogsum"

        if len(data_set["Related Topics"]) > 1:
            print(new_row["conversations"][0]["value"])
            print(new_row["conversations"][1]["value"])
            print("=" * 20)

    if len(new_row["conversations"][0]["value"].split(" ") + new_row["conversations"][1]["value"].split(" ")) > 1400:
        count -= num_shift
        continue
    else:
        if case_num == 1:
            num_count_1 += 1
        elif case_num == 2:
            num_count_2 +=1
        else:
            num_count_3 += 1

    last_count = count
    new_data.append(new_row)
    num_count += 1

json.dump(new_data, open('dialogsum_train_question.json', 'w'), indent=2)
print("{} memo writing samples, {} memo retrival samples, {} chat with memo samples, {} all samples".format(num_count_1, num_count_2, num_count_3, num_count))
