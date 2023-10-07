import json
from random import sample, choice, random

prompts = json.load(open("../../prompts.json", "r"))
data = json.load(open('raw_data/topiocqa_dev.json', 'r'))

last_c_no = 0
count = 0
count_1 = 0
count_2 = 0
w = open('topiocqa_test_question.jsonl', 'w')
for ri, row in enumerate(data):
    if (ri + 1 == len(data)) or (row["Conversation_no"] != data[ri + 1]["Conversation_no"]):
        new_row = {"question_id": row["Conversation_no"], "text": "", "answer": "", "type": ""}
        case_num = choice([1, 2])

        # construct memo writing sample
        if case_num == 1:
            system_insturction = prompts["writing_topiocqa"]["system"]
            task_instruction = prompts["writing_topiocqa"]["instruction"]
            new_conversation = []
            for li, line in enumerate(row["Context"] + [row["Question"], row["Answer"]]):
                if li % 2 == 0:
                    new_conversation.append("(line {}) user: ".format(li + 1) + line)
                else:
                    new_conversation.append("(line {}) bot: ".format(li + 1) + line)
            new_row["text"] = system_insturction.replace("LINE", str(li + 1)) + "\n\n```\nTask Conversation:\n" + "\n".join(new_conversation) + "\n```" + task_instruction.replace("LINE", str(li + 1))
            turn_topics = [rl["Topic"] for rl in data[last_c_no:(ri + 1)]]
            assert len(turn_topics) == len(new_conversation) // 2
            row_answer = []
            last_topic = ""
            for ti, turn_topic in enumerate(turn_topics):
                if turn_topic != last_topic:
                    row_answer.append({"topic": turn_topic, "start": 2 * ti + 1, "end": 2 * ti + 2})
                    last_topic = turn_topic
                else:
                    row_answer[-1]["end"] = 2 * ti + 2
            new_row["answer"] = json.dumps(row_answer)
            new_row["type"] = "writing_topiocqa"
            count_1 += 1

        # construct memo retrival sample
        if case_num == 2:
            system_insturction = prompts["retrieval"]["system"]
            task_instruction = prompts["retrieval"]["instruction"]
            turn_topics = [rl["Topic"] for rl in data[last_c_no:(ri + 1)]]
            new_conversation = []
            raw_conversation = row["Context"] + [row["Question"], row["Answer"]]
            for li, line in enumerate(raw_conversation):
                if li % 2 == 1:
                    new_conversation.append((raw_conversation[li - 1] + " " + line, turn_topics[li // 2]))
            select_query_sents = sample(new_conversation, choice(list(range(1, min(4, len(new_conversation) + 1)))))

            # create NOTO special cases with prob = 0.1
            if random() <= 0.1:
                seen_topics = set()
                for s_i, select_query_sent in enumerate(select_query_sents):
                    seen_topics.add(select_query_sent[1])
                    select_query_sents[s_i] = (select_query_sent[0], "NOTO. None of the others.")
                for t_i, t_ in enumerate(turn_topics):
                    if t_ in seen_topics:
                        turn_topics[t_i] = "NOTO. None of the others."

            query_sents = []
            answers = set()
            options = {t_: ("({}) {}".format(str(t_i + 1), t_), str(t_i + 1)) for t_i, t_ in enumerate(set(turn_topics + ["NOTO. None of the others."]))}
            for select_query_sent in select_query_sents:
                query_sents.append(select_query_sent[0])
                answers.add(options[select_query_sent[1]][1])
            task_case = "```\nQuery Sentence:\n" + " ".join(query_sents) + "\nTopic Options:\n" + "\n".join([v[0] for v in list(options.values())]) + "\n```"
            new_row["text"] = system_insturction.replace("OPTION", str(len(options.keys()))) + task_case + task_instruction.replace("OPTION", str(len(options.keys())))
            new_row["answer"] = "#".join(list(sorted(answers)))
            new_row["type"] = "retrieval_topiocqa"
            count_2 += 1

        print(new_row["text"])
        print(new_row["answer"])
        print("=" * 20)
        w.write(json.dumps(new_row) + "\n")
        count += 1
        last_c_no = ri + 1

w.close()
print("{} memo writing samples, {} memo retrival samples, {} all samples".format(count_1, count_2, count))
