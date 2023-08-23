import json
import re
import string
import sys
import random
from argparse import ArgumentParser
from collections import Counter

from evaluate import load
bertscore = load("bertscore")

refer_file_path = sys.argv[1]
input_file_path = sys.argv[2]

conversations = open(refer_file_path, "r").readlines()
conversations_dict = {}
for conversation in conversations:
    conv_l = json.loads(conversation.strip())
    conversations_dict[conv_l["question_id"]] = (conv_l["text"], conv_l["answer"], conv_l["type"])

class Metrics():
    def __init__(self):
        pass

    def __normalize_text(self, s_text):
        """Lower text and remove punctuation, storys and extra whitespace."""
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s_text))))

    def __normalize_model_outputs(self, model_text, type_category):
        """post process of memo writing outputs"""
        extracted_elements = [re.sub(r'\s+', ' ', mt.replace('"', '').replace("'", "")) for mt in re.findall(r"'[^']*'|\"[^\"]*\"|\d+", model_text)]
        model_outputs = []
        ti = 0

        if "dialogsum" in type_category:
            while ti + 7 < len(extracted_elements):
                if extracted_elements[ti] == "topic" and extracted_elements[ti + 2] == "summary" and extracted_elements[ti + 4] == "start" and extracted_elements[ti + 6] == "end":
                    try:
                        model_outputs.append({"topic": extracted_elements[ti + 1], "summary": extracted_elements[ti + 3], "start": int(extracted_elements[ti + 5]), "end": int(extracted_elements[ti + 7])})
                    except:
                        pass
                ti += 1
        else:
            while ti + 5 < len(extracted_elements):
                if extracted_elements[ti] == "topic" and extracted_elements[ti + 2] == "start" and extracted_elements[ti + 4] == "end":
                    try:
                        model_outputs.append({"topic": extracted_elements[ti + 1], "start": int(extracted_elements[ti + 3]), "end": int(extracted_elements[ti + 5])})
                    except:
                        pass
                ti += 1
        
        return model_outputs

    def __get_class_span_dict__(self, label, checkitem_k):
        class_span = {}
        for i in range(len(label)):
            checkitem_i = self.__normalize_text(label[i][checkitem_k])
            class_span[(label[i]['start'], label[i]['end'])] = class_span.get((label[i]['start'], label[i]['end']), []) + [checkitem_i]
        return class_span

    def __get_intersect_by_entity__(self, pred_class_span, label_class_span):
        '''
        return the count of correct entity
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label,[])))))
        return cnt

    def __get_bertscore_by_entity__(self, pred_class_span, label_class_span):
        '''
        return the count of correct entity
        '''
        cnt = 0
        for label in label_class_span:
            if label in pred_class_span:
                references = [label_class_span[label]]
                prediction = [pred_class_span[label][0]]
                result = bertscore.compute(predictions=prediction, references=references, model_type="microsoft/deberta-xlarge-mnli")["precision"][0]
                cnt += result
        return cnt

    def __get_cnt__(self, label_class_span):
        '''
        return the count of entities
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(label_class_span[label])
            # cnt += 1  # set as 1 if we have multiple references
        return cnt
                
    def metrics_by_entity_(self, pred, label, checkitem_k):
        '''
        return entity level count of total prediction, true labels, and correct prediction
        '''
        pred_class_span = self.__get_class_span_dict__(pred, checkitem_k)
        label_class_span = self.__get_class_span_dict__(label, checkitem_k)
        pred_cnt = self.__get_cnt__(pred_class_span)
        label_cnt = self.__get_cnt__(label_class_span)
        if checkitem_k == "topic":
            correct_cnt = self.__get_intersect_by_entity__(pred_class_span, label_class_span)
        elif checkitem_k == "summary":
            correct_cnt = self.__get_bertscore_by_entity__(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt

    def p_r_f1_by_entity(self, pc, lc, cc):
        precision = cc / (pc + 1e-8)
        recall = cc / (lc + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return round(precision * 100, 2), round(recall * 100, 2), round(f1 * 100, 2)

    def metrics_by_entity_files(self, pred_file, checkitem_k, type_key):
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        for l_i, line in enumerate(open(pred_file, "r").readlines()):
            eles = json.loads(line.strip())

            if (type_key not in conversations_dict[eles["question_id"]][2]) or (conversations_dict[eles["question_id"]][2] == "writing_topiocqa" and checkitem_k == "summary"):
                continue
            if type_key == "writing":
                model_text = self.__normalize_model_outputs(eles["text"], conversations_dict[eles["question_id"]][2])
                label_i = json.loads(conversations_dict[eles["question_id"]][1])
            elif type_key == "retrieval":
                model_text = [{"topic": v, "start": 0, "end": 0} for v in set(eles["text"].split("#"))]
                label_i = [{"topic": v, "start": 0, "end": 0} for v in set(conversations_dict[eles["question_id"]][1].split("#"))]
            else:
                model_text = [{"summary": eles["text"], "start": 0, "end": 0}]
                label_i = [{"summary": conversations_dict[eles["question_id"]][1], "start": 0, "end": 0}]

            p_cnt, l_cnt, c_cnt = self.metrics_by_entity_(model_text, label_i, checkitem_k)
            p_i, r_i, f_i = self.p_r_f1_by_entity(p_cnt, l_cnt, c_cnt)
            # if p_i + r_i + f_i != 0:
            #     print("Q ID: " + str(eles["question_id"]) + "\n")
            #     print(conversations_dict[eles["question_id"]][0] + "\n")
            #     # print("Raw Ouput: " + eles["text"] + "\n")
            #     print("Model: {}".format(model_text) + "\n")
            #     print("Refer: {}".format(label_i) + "\n")
            #     print("Case P/R/F1: {}%, {}%, {}%".format(p_i, r_i, f_i))
            #     print("=" * 20)
            pred_cnt += p_cnt
            label_cnt += l_cnt
            correct_cnt += c_cnt
        return self.p_r_f1_by_entity(pred_cnt, label_cnt, correct_cnt)

calculate_metrics = Metrics()
p_a, r_a, f1_a = calculate_metrics.metrics_by_entity_files(input_file_path, 'topic', 'writing')  # both
print("Overall P/R/F1 of topic: {}%, {}%, {}%".format(p_a, r_a, f1_a))
p_b, r_b, f1_b = calculate_metrics.metrics_by_entity_files(input_file_path, 'summary', 'writing')  # dialogsum
print("Overall P/R/F1 of summary: {}%, {}%, {}%".format(p_b, r_b, f1_b))
_, _, f1 = calculate_metrics.metrics_by_entity_files(input_file_path, "topic", "retrieval")  # both
print("Retrival F1: {}%".format(f1))
p, _, _ = calculate_metrics.metrics_by_entity_files(input_file_path, "summary", "chatting")  # dialogsum
print("Chatting similarity: {}%".format(p))
