#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/5/28 16:42
# @Author : WK

from a_buildModel import BuildModel
from a_preprocessing import DataPreprocessing
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow as tf
train_file = './data/train_data.jsonl'
valid_file = './data/valid_data.jsonl'
test_file = './data/test_data.jsonl'

i = 50
processing = DataPreprocessing(windows=i)
buildmodel = BuildModel(windows=i)
print(processing.windows)

tokenizer, bert_model = buildmodel.load_bert()
input_l, diagnosis_list = processing.labeEmbding(tokenizer, bert_model)
x_data_train, y_sim_train = processing.read_data(train_file, 'retrieval', input_l, diagnosis_list,
                                                          tokenizer)
x_data_valid, y_sim_valid = processing.read_data(valid_file, 'retrieval', input_l, diagnosis_list,
                                                          tokenizer)
x_data_test, y_sim_test = processing.read_data(test_file, 'retrieval', input_l, diagnosis_list, tokenizer)


simModel, embd_model = buildmodel.SimModel()
simModel.compile(loss='sparse_categorical_crossentropy',
                 optimizer=Adam(1e-3),
                 metrics='accuracy')

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, min_delta=0.0001,
                                            restore_best_weights=True)
callback_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=5,
                                                   min_delta=0.0001,
                                                   factor=0.2, min_lr=0.00001)
print('training start')
simModel.fit(x_data_train, y_sim_train,
          validation_data=(x_data_valid, y_sim_valid),
          epochs=200,
          batch_size=32,
          # verbose=2,
          callbacks=[callback, callback_LR])

simModel.save_weights('./model/a1_simModel.h5')
embd_model.save_weights('./model/a1_simModel_embd.h5')
print('model save over', str(i))
y_sim_pre = simModel.predict(x_data_test)
processing.sim_evaluation(y_sim_pre, y_sim_test)


