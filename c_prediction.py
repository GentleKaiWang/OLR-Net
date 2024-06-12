#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/6/12 11:52
# @Author : WK

from a_buildModel import BuildModel
from a_preprocessing import DataPreprocessing
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow as tf
import numpy as np
import json


test_file = './data/test_data.jsonl'
i = 50

processing = DataPreprocessing(windows=i)
buildmodel = BuildModel(windows=i)

tokenizer, bert_model = buildmodel.load_bert()
input_l, diagnosis_list = processing.labeEmbding(tokenizer, bert_model)
# x_range_train, x_sim_train, y_train = processing.read_data(train_file, 'prediction', input_l, diagnosis_list, tokenizer)
# x_range_valid, x_sim_valid, y_valid = processing.read_data(valid_file, 'prediction', input_l, diagnosis_list, tokenizer)

rangeModel, range_embdmodel = buildmodel.RangeModel(input_l, is_tuning=False)
simModel, sim_embdmodel = buildmodel.SimModel()

simModel.load_weights('./model/a1_simModel.h5')
rangeModel.load_weights('./model/b1_rangeModel.h5')
print('model load weights over')

a, b, c, d = 0, 0, 0, 0
result_pre_sim, result_pre_range = [], []
for line in open(test_file, 'r', encoding='utf-8').readlines():
    line = json.loads(line)
    if line != '':
        b += 1
        inhospital_x1, inhospital_x2 = tokenizer.encode(first=line['data_1'], max_len=128)
        indiagnosis_x1, indiagnosis_x2 = tokenizer.encode(first=line['data_2'], max_len=128)
        outdiagnosis_x1, outdiagnosis_x2 = tokenizer.encode(first=line['data_4'], max_len=128)
        intreat_x1, intreat_x2 = tokenizer.encode(first=line['data_6'], max_len=128)
        diagnosis_index = diagnosis_list.index(line['diagnosis_nation'])

        inhospital_x1 = np.array([inhospital_x1])
        inhospital_x2 = np.array([inhospital_x2])
        indiagnosis_x1 = np.array([indiagnosis_x1])
        indiagnosis_x2 = np.array([indiagnosis_x2])
        outdiagnosis_x1 = np.array([outdiagnosis_x1])
        outdiagnosis_x2 = np.array([outdiagnosis_x2])
        intreat_x1 = np.array([intreat_x1])
        intreat_x2 = np.array([intreat_x2])

        x_range_test = [inhospital_x1, inhospital_x2, indiagnosis_x1, indiagnosis_x2,
                        outdiagnosis_x1, outdiagnosis_x2, intreat_x1, intreat_x2]
        x_sim_test = [np.repeat(inhospital_x1, processing.ranges, axis=0), np.repeat(inhospital_x2, processing.ranges, axis=0),
                      np.repeat(indiagnosis_x1, processing.ranges, axis=0), np.repeat(indiagnosis_x2, processing.ranges, axis=0),
                      np.repeat(outdiagnosis_x1, processing.ranges, axis=0), np.repeat(outdiagnosis_x2, processing.ranges, axis=0),
                      np.repeat(intreat_x1, processing.ranges, axis=0), np.repeat(intreat_x2, processing.ranges, axis=0),
                      input_l.reshape((processing.ranges, i, 768))]

        y_range_pre = rangeModel.predict(x_range_test)
        y_sim_pre = simModel.predict(x_sim_test)

        y_range_pre_T = np.repeat(y_range_pre.T, i, axis=1)
        y_sim_all = np.multiply(y_range_pre_T, y_sim_pre)
        y_sim_all = y_sim_all.reshape((1700,))
        sim_max5 = y_sim_all.argsort()[-5:][::-1]
        print([diagnosis_list[i] for i in sim_max5])
        # print(diagnosis_list[sim_max5])



