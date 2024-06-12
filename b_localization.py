#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/5/28 21:19
# @Author : WK
from a_buildModel import BuildModel
from a_preprocessing import DataPreprocessing
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow as tf

i = 50
processing = DataPreprocessing(windows=i)
buildmodel = BuildModel(windows=i)

train_file = './data/train_data.jsonl'
valid_file = './data/valid_data.jsonl'
test_file = './data/test_data.jsonl'
tokenizer, bert_model = buildmodel.load_bert()
input_l, diagnosis_list = processing.labeEmbding(tokenizer, bert_model)
x_data_train, y_range_train = processing.read_data(train_file, 'localization', input_l, diagnosis_list, tokenizer)
x_data_valid, y_range_valid = processing.read_data(valid_file, 'localization', input_l, diagnosis_list, tokenizer)
x_data_test, y_range_test = processing.read_data(test_file, 'localization', input_l, diagnosis_list, tokenizer)

rangeModel, embd_model = buildmodel.RangeModel(input_l, is_tuning=True, embd_h5='./model/a1_simModel_embd{0}.h5'.format(str(i)))
rangeModel.compile(loss='sparse_categorical_crossentropy',
                 optimizer=Adam(1e-3),
                 metrics='accuracy')

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, min_delta=0.0001,
                                            restore_best_weights=True)
callback_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=5,
                                                   min_delta=0.0001,
                                                   factor=0.2, min_lr=0.00001)
print('training start')
rangeModel.fit(x_data_train, y_range_train,
          validation_data=(x_data_valid, y_range_valid),
          epochs=200,
          batch_size=32,
          # verbose=2,
          callbacks=[callback, callback_LR])

rangeModel.save_weights('./model/b1_rangeModel.h5')
embd_model.save_weights('./model/b1_rangeModel_embd.h5')
print('model save over')
y_range_pre = rangeModel.predict(x_data_test)
processing.range_evaluation(y_range_pre, y_range_test)