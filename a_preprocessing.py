#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/5/28 16:49
# @Author : WK
import numpy as np
import json
from a_buildModel import BuildModel

class DataPreprocessing():
    def __init__(self, **kwargs):
        self.labels_len = kwargs.get('labels_len', 1700)
        self.windows = kwargs.get('windows', 50)
        self.ranges = int(self.labels_len / self.windows)
        self.diagnosis_file = kwargs.get('diagnosis_file', './data/ICD_code_cancer.txt')

    def positional_embedding(self):
        model_size = 768
        PE = np.zeros((self.labels_len, model_size))
        for i in range(self.labels_len):
            for j in range(model_size):
                if j % 2 == 0:
                    PE[i, j] = np.sin(i / 10000 ** (j / model_size))
                else:
                    PE[i, j] = np.cos(i / 10000 ** ((j - 1) / model_size))
        return PE

    def labeEmbding(self, tokenizer, bert_model):

        label_file = open(self.diagnosis_file, 'r', encoding='utf8')
        input_l, diagnosis_list = [], []
        for line in label_file.readlines():
            label = line.strip().split()[0]
            diagnosis_list.append(line.split('\t')[0])
            label_1, label_2 = tokenizer.encode(first=label, max_len=16)
            label_embding = bert_model.predict([np.array([label_1]), np.array([label_2])])[0, 0, :]
            label_embding = np.around(label_embding, decimals=4)
            input_l.append(label_embding)
        input_l = np.array(input_l)
        PE = self.positional_embedding()
        input_l = np.vstack((input_l, np.array([[0] * 768] * (self.labels_len-len(diagnosis_list)))))
        input_l += PE
        diagnosis_list += [''] * (self.labels_len-len(diagnosis_list))
        return input_l, diagnosis_list


    def read_data(self, data_filename, task_name, input_l, diagnosis_list, tokenizer):

        data_file = open(data_filename, 'r', encoding='utf8')
        y_sim, x_index, y_range, y_all = [], [], [], []
        inhospital_x1, inhospital_x2, indiagnosis_x1, indiagnosis_x2 = [], [], [], []
        outdiagnosis_x1, outdiagnosis_x2, intreat_x1, intreat_x2 = [], [], [], []

        # print(self.windows)
        for line in data_file.readlines():
            line = json.loads(line)
            if line != '':
                inp1_x1, inp1_x2 = tokenizer.encode(first=line['data_1'], max_len=128)
                inp2_x1, inp2_x2 = tokenizer.encode(first=line['data_2'], max_len=128)
                inp3_x1, inp3_x2 = tokenizer.encode(first=line['data_4'], max_len=128)
                inp4_x1, inp4_x2 = tokenizer.encode(first=line['data_6'], max_len=128)

                diagnosis_index = diagnosis_list.index(line['diagnosis_nation'])
                sim_index = diagnosis_index % self.windows
                y_sim.append(sim_index)
                range_index = diagnosis_index // self.windows
                x_index.append(input_l[range_index * self.windows: (range_index + 1) * self.windows])
                y_range.append(range_index)
                y_all.append(diagnosis_index)

                inhospital_x1.append(inp1_x1)
                inhospital_x2.append(inp1_x2)
                indiagnosis_x1.append(inp2_x1)
                indiagnosis_x2.append(inp2_x2)
                outdiagnosis_x1.append(inp3_x1)
                outdiagnosis_x2.append(inp3_x2)
                intreat_x1.append(inp4_x1)
                intreat_x2.append(inp4_x2)

        inhospital_x1 = np.array(inhospital_x1)
        inhospital_x2 = np.array(inhospital_x2)
        indiagnosis_x1 = np.array(indiagnosis_x1)
        indiagnosis_x2 = np.array(indiagnosis_x2)

        outdiagnosis_x1 = np.array(outdiagnosis_x1)
        outdiagnosis_x2 = np.array(outdiagnosis_x2)
        intreat_x1 = np.array(intreat_x1)
        intreat_x2 = np.array(intreat_x2)
        x_index = np.array(x_index)
        y_sim = np.array(y_sim)
        y_range = np.array(y_range)
        y_all = np.array(y_all)

        print(inhospital_x1.shape, indiagnosis_x1.shape, outdiagnosis_x1.shape, intreat_x1.shape,
              x_index.shape, y_sim.shape, y_range.shape, y_all.shape)
        indices = np.arange(inhospital_x1.shape[0])
        np.random.shuffle(indices)
        inhospital_x1 = inhospital_x1[indices]
        inhospital_x2 = inhospital_x2[indices]
        indiagnosis_x1 = indiagnosis_x1[indices]
        indiagnosis_x2 = indiagnosis_x2[indices]

        outdiagnosis_x1 = outdiagnosis_x1[indices]
        outdiagnosis_x2 = outdiagnosis_x2[indices]
        intreat_x1 = intreat_x1[indices]
        intreat_x2 = intreat_x2[indices]
        x_index = x_index[indices]
        y_sim = y_sim[indices]
        y_range = y_range[indices]
        y_all = y_all[indices]

        if task_name == 'retrieval':
            x_data = [inhospital_x1, inhospital_x2, indiagnosis_x1, indiagnosis_x2,
                            outdiagnosis_x1, outdiagnosis_x2, intreat_x1, intreat_x2, x_index]
            return x_data, y_sim
        elif task_name == 'localization':
            x_data = [inhospital_x1, inhospital_x2, indiagnosis_x1, indiagnosis_x2,
                      outdiagnosis_x1, outdiagnosis_x2, intreat_x1, intreat_x2]
            return x_data, y_range
        else:
            print("""erro task name! please choose task_name == 'retrieval' or task_name == 'localization' """)





        # x_data_train = [inhospital_x1, inhospital_x2, indiagnosis_x1, indiagnosis_x2,
        #                 outdiagnosis_x1, outdiagnosis_x2, intreat_x1, intreat_x2, x_index]
        # return x_data_train, y_sim, y_range


    def positional_embedding(self):
        model_size = 768
        PE = np.zeros((self.labels_len, model_size))
        for i in range(self.labels_len):
            for j in range(model_size):
                if j % 2 == 0:
                    PE[i, j] = np.sin(i / 10000 ** (j / model_size))
                else:
                    PE[i, j] = np.cos(i / 10000 ** ((j - 1) / model_size))
        return PE

    def sim_evaluation(self, y_sim_pre, y_data):
        a, b, c = 0, 0, 0
        print('pre shape', y_sim_pre.shape)
        for one2, one4 in zip(y_sim_pre, y_data):
            max5_index = one2.argsort()[-5:][::-1]
            y_sim_index = one4
            b += 1
            # print(y_sim_index, max5_index)
            # print(one2, np.sum(one2))
            if y_sim_index in max5_index:
                a += 1
            if y_sim_index == one2.argmax():
                c += 1
        print('acc', a / b, c / b, a, b, c)

    def range_evaluation(self, y_range_pre, y_data):
        a, b, c = 0, 0, 0
        for one1, one3 in zip(y_range_pre, y_data):
            b += 1
            if one1.argmax() == one3:
                a += 1
        print('acc', a / b, c / b, a, b)


if __name__ == '__main__':
    train_file = '../301_all_model_0424/data/train_data.jsonl'
    valid_file = '../301_all_model_0424/data/valid_data.jsonl'
    test_file = '../301_all_model_0424/data/test_data.jsonl'
    x_data_train, y_train_sim, y_train_range = DataPreprocessing().read_data(train_file, 'retrieval')



