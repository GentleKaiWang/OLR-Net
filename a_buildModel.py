#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/5/28 14:54
# @Author : WK
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer
import codecs
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, regularizers
import tensorflow as tf


np.random.seed(123)


class BuildModel():
    def __init__(self, **kwargs):
        self.BN_train = kwargs.get('BN_train', True)
        self.outRatio = kwargs.get('outRatio', 0.4)
        self.labels_len = kwargs.get('labels_len', 1700)
        self.windows = kwargs.get('windows', 50)
        self.is_GPU = kwargs.get('is_GPU', 1)
        self.bert_path = kwargs.get('bert_path', '/home/vico/wk/project/bert-master/weight/chinese_roberta_wwm_ext_L-12_H-768_A-12')
        self.ranges = self.labels_len / self.windows
        # self.bert_path = 'D:/pythonProgram/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12'

    def load_bert(self):
        """
        load bert model
        :param is_GPU: Whether to use GPU，1：Yes，0：No
        :return: bert tokenizer、model
        """
        if self.is_GPU == 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            physical_device = tf.config.experimental.list_physical_devices("GPU")
            tf.config.experimental.set_memory_growth(physical_device[0], True)
            config_path = self.bert_path + '/bert_config.json'
            checkpoint_path = self.bert_path + '/bert_model.ckpt'
            dict_path = self.bert_path + '/vocab.txt'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            config_path = self.bert_path + '/bert_config.json'
            checkpoint_path = self.bert_path + '/bert_model.ckpt'
            dict_path = self.bert_path + '/vocab.txt'

        token_dict = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        tokenizer = Tokenizer(token_dict)

        bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
        for l in bert_model.layers:
            l.trainable = False
        return tokenizer, bert_model

    def gelu(self, input_tensor):
        cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.sqrt(2.0)))
        return input_tensor * cdf

    def self_attention(self, x, dim, times):
        x1 = Dense(dim,
                  kernel_initializer='glorot_uniform',
                  bias_initializer='zeros',
                  kernel_regularizer=regularizers.l2(0.01))(x)
        x1 = Lambda(self.gelu)(x1)
        x1 = LayerNormalization()(x1)

        x2 = Dense(dim,
                  kernel_initializer='glorot_uniform',
                  bias_initializer='zeros',
                  kernel_regularizer=regularizers.l2(0.01))(x)
        x2 = Lambda(self.gelu)(x2)
        x2 = LayerNormalization()(x2)

        x3 = Dense(dim,
                  kernel_initializer='glorot_uniform',
                  bias_initializer='zeros',
                  kernel_regularizer=regularizers.l2(0.01))(x)
        x3 = Lambda(self.gelu)(x3)
        x3 = LayerNormalization()(x3)

        x2_T = Permute((2, 1))(x2)
        attention = Lambda(lambda x: tf.matmul(x[0], x[1]) / dim ** 0.5)([x1, x2_T])
        attention = Softmax(name=times)(attention)
        x3 = Lambda(lambda x: tf.matmul(x[0], x[1]))([attention, x3])
        x3 = Add()([x, x3])
        x3 = LayerNormalization()(x3)
        x3 = Dropout(self.outRatio)(x3)
        return x3

    def do_senModel(self):
        input_m = Input((128, 768))
        m1 = self.self_attention(input_m, 768, 'sen1')
        m1 = Lambda(lambda x: tf.reduce_mean(x, axis=1))(m1)
        sen_model = Model(input_m, m1, name='sen_model')
        # sen_model.summary()
        return sen_model


    def do_txtembdModel(self):
        inhospital_inp = Input((768))
        indiagnosis_inp = Input((768))
        outdiagnosis_inp = Input((768))
        intreat_inp = Input((768))

        inhospital = Lambda(lambda x: tf.expand_dims(x, axis=1))(inhospital_inp)
        indiagnosis = Lambda(lambda x: tf.expand_dims(x, axis=1))(indiagnosis_inp)
        outdiagnosis = Lambda(lambda x: tf.expand_dims(x, axis=1))(outdiagnosis_inp)
        intreat = Lambda(lambda x: tf.expand_dims(x, axis=1))(intreat_inp)

        txt_embd = Concatenate(axis=1)([inhospital, indiagnosis, outdiagnosis, intreat])
        txt_embd = self.self_attention(txt_embd, 768, '1')
        txt_embd = Lambda(lambda x: tf.reduce_mean(x, axis=1))(txt_embd)
        txt_embd_model = Model([inhospital_inp, indiagnosis_inp, outdiagnosis_inp, intreat_inp],
                               txt_embd, name='txt_embd_model')
        # txt_embd_model.summary()
        return txt_embd_model


    def labeEmbding(self, diagnosis_file, tokenizer, bert_model):
        # print(diagnosis_file)
        label_file = open(diagnosis_file, 'r', encoding='utf8')
        input_l = []
        for line in label_file.readlines():
            label = line.split(' ')[0]
            label_1, label_2 = tokenizer.encode(first=label, max_len=16)
            label_embding = bert_model.predict([np.array([label_1]), np.array([label_2])])[0, 0, :]
            label_embding = np.around(label_embding, decimals=4)
            input_l.append(label_embding)
        input_l = np.array(input_l)
        return input_l

    def do_rangeModel(self, input_l):
        txt_inp = Input((768,))
        input_l = Lambda(lambda x: tf.constant(x, dtype=tf.float32))(input_l)
        txt_embdings = Lambda(lambda x: tf.expand_dims(x, axis=-1))(txt_inp)
        txt_mapping = Lambda(lambda x: tf.matmul(x[0], x[1]) / 768 ** 0.5)([input_l, txt_embdings])
        if self.windows == 17:
            txt_mapping2 = Conv1D(filters=16, kernel_size=20, strides=17,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))(txt_mapping)
            txt_mapping2 = Lambda(self.gelu)(txt_mapping2)
            txt_mapping2 = LayerNormalization()(txt_mapping2)

        elif self.windows == 20:
            txt_mapping1 = Conv1D(filters=16, kernel_size=10, strides=5,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))(txt_mapping)
            txt_mapping1 = Lambda(self.gelu)(txt_mapping1)
            txt_mapping1 = LayerNormalization()(txt_mapping1)

            txt_mapping2 = Conv1D(filters=32, kernel_size=10, strides=4,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))(txt_mapping1)
            txt_mapping2 = Lambda(self.gelu)(txt_mapping2)
            txt_mapping2 = LayerNormalization()(txt_mapping2)


        elif self.windows == 25:
            txt_mapping1 = Conv1D(filters=16, kernel_size=10, strides=5,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))(txt_mapping)
            txt_mapping1 = Lambda(self.gelu)(txt_mapping1)
            txt_mapping1 = LayerNormalization()(txt_mapping1)

            txt_mapping2 = Conv1D(filters=32, kernel_size=10, strides=5,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))(txt_mapping1)
            txt_mapping2 = Lambda(self.gelu)(txt_mapping2)
            txt_mapping2 = LayerNormalization()(txt_mapping2)
        elif self.windows == 34:
            txt_mapping1 = Conv1D(filters=16, kernel_size=20, strides=17,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))(txt_mapping)
            txt_mapping1 = Lambda(self.gelu)(txt_mapping1)
            txt_mapping1 = LayerNormalization()(txt_mapping1)

            txt_mapping2 = Conv1D(filters=32, kernel_size=5, strides=2,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))(txt_mapping1)
            txt_mapping2 = Lambda(self.gelu)(txt_mapping2)
            txt_mapping2 = LayerNormalization()(txt_mapping2)
        elif self.windows == 50:
            txt_mapping1 = Conv1D(filters=16, kernel_size=20, strides=10,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))(txt_mapping)
            txt_mapping1 = Lambda(self.gelu)(txt_mapping1)
            txt_mapping1 = LayerNormalization()(txt_mapping1)

            txt_mapping2 = Conv1D(filters=32, kernel_size=10, strides=5,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))(txt_mapping1)
            txt_mapping2 = Lambda(self.gelu)(txt_mapping2)
            txt_mapping2 = LayerNormalization()(txt_mapping2)
        elif self.windows == 68:
            txt_mapping1 = Conv1D(filters=16, kernel_size=20, strides=17,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))(txt_mapping)
            txt_mapping1 = Lambda(self.gelu)(txt_mapping1)
            txt_mapping1 = LayerNormalization()(txt_mapping1)

            txt_mapping2 = Conv1D(filters=32, kernel_size=5, strides=4,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))(txt_mapping1)
            txt_mapping2 = Lambda(self.gelu)(txt_mapping2)
            txt_mapping2 = LayerNormalization()(txt_mapping2)
        elif self.windows == 85:
            txt_mapping1 = Conv1D(filters=16, kernel_size=20, strides=17,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))(txt_mapping)
            txt_mapping1 = Lambda(self.gelu)(txt_mapping1)
            txt_mapping1 = LayerNormalization()(txt_mapping1)

            txt_mapping2 = Conv1D(filters=32, kernel_size=10, strides=5,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))(txt_mapping1)
            txt_mapping2 = Lambda(self.gelu)(txt_mapping2)
            txt_mapping2 = LayerNormalization()(txt_mapping2)
        elif self.windows == 100:
            txt_mapping1 = Conv1D(filters=16, kernel_size=20, strides=10,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))(txt_mapping)
            txt_mapping1 = Lambda(self.gelu)(txt_mapping1)
            txt_mapping1 = LayerNormalization()(txt_mapping1)

            txt_mapping2 = Conv1D(filters=32, kernel_size=20, strides=10,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))(txt_mapping1)
            txt_mapping2 = Lambda(self.gelu)(txt_mapping2)
            txt_mapping2 = LayerNormalization()(txt_mapping2)


        txt_mapping3 = Flatten()(txt_mapping2)
        txt_mapping3 = Dropout(self.outRatio)(txt_mapping3)
        txt_mapping3 = Dense(self.ranges,
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             kernel_regularizer=regularizers.l2(0.01))(txt_mapping3)
        y = Softmax()(txt_mapping3)

        # y = Lambda(lambda x: x / tf.repeat(tf.expand_dims(tf.reduce_sum(x, axis=1), axis=-1), tf.shape(x)[1], axis=-1))(y)
        topk_model = Model(txt_inp, y, name='range_model')
        # topk_model.summary()
        return topk_model

    def do_simModel(self):
        txt_inp = Input((768,))
        x_index_inp = Input((self.windows, 768))

        txt = Lambda(lambda x: tf.expand_dims(x, axis=-1))(txt_inp)

        txt_mapping = Lambda(lambda x: tf.matmul(x[0], x[1]) / 768 ** 0.5)([x_index_inp, txt])
        txt_mapping = Flatten()(txt_mapping)
        y = Softmax()(txt_mapping)
        model = Model([txt_inp, x_index_inp], y, name='sim_model')
        # model.summary()
        return model


    def SimModel(self):
        tokenizer, bert_model = self.load_bert()
        sen_model = self.do_senModel()
        txt_embd_model = self.do_txtembdModel()
        sim_model = self.do_simModel()

        inhospital_x1 = Input((128,))
        inhospital_x2 = Input((128,))
        indiagnosis_x1 = Input((128, ))
        indiagnosis_x2 = Input((128, ))
        outdiagnosis_x1 = Input((128, ))
        outdiagnosis_x2 = Input((128, ))
        intreat_x1 = Input((128,))
        intreat_x2 = Input((128,))
        range_inp = Input((self.windows, 768))

        inhospital_x = bert_model([inhospital_x1, inhospital_x2])
        indiagnosis_x = bert_model([indiagnosis_x1, indiagnosis_x2])
        outdiagnosis_x = bert_model([outdiagnosis_x1, outdiagnosis_x2])
        intreat_x = bert_model([intreat_x1, intreat_x2])

        inhospital_x = sen_model(inhospital_x)
        indiagnosis_x = sen_model(indiagnosis_x)
        outdiagnosis_x = sen_model(outdiagnosis_x)
        intreat_x = sen_model(intreat_x)

        txt_embding = txt_embd_model([inhospital_x, indiagnosis_x, outdiagnosis_x, intreat_x])
        y_sim = sim_model([txt_embding, range_inp])

        simModel = Model([inhospital_x1, inhospital_x2, indiagnosis_x1, indiagnosis_x2,
                       outdiagnosis_x1, outdiagnosis_x2, intreat_x1, intreat_x2, range_inp], y_sim, name='simmodel')
        embd_model = Model([inhospital_x1, inhospital_x2, indiagnosis_x1, indiagnosis_x2,
                       outdiagnosis_x1, outdiagnosis_x2, intreat_x1, intreat_x2], txt_embding, name='Embdmodel')
        # simModel.compile(loss='sparse_categorical_crossentropy',
        #                     optimizer=Adam(1e-3),
        #                     metrics='accuracy')
        # simModel.summary()
        return simModel, embd_model


    def RangeModel(self, input_l, is_tuning=False, embd_h5=''):
        tokenizer, bert_model = self.load_bert()
        sen_model = self.do_senModel()
        txt_embd_model = self.do_txtembdModel()
        range_model = self.do_rangeModel(input_l)

        inhospital_x1 = Input((128,))
        inhospital_x2 = Input((128,))
        indiagnosis_x1 = Input((128, ))
        indiagnosis_x2 = Input((128, ))
        outdiagnosis_x1 = Input((128, ))
        outdiagnosis_x2 = Input((128, ))
        intreat_x1 = Input((128,))
        intreat_x2 = Input((128,))

        inhospital_x = bert_model([inhospital_x1, inhospital_x2])
        indiagnosis_x = bert_model([indiagnosis_x1, indiagnosis_x2])
        outdiagnosis_x = bert_model([outdiagnosis_x1, outdiagnosis_x2])
        intreat_x = bert_model([intreat_x1, intreat_x2])

        inhospital_x = sen_model(inhospital_x)
        indiagnosis_x = sen_model(indiagnosis_x)
        outdiagnosis_x = sen_model(outdiagnosis_x)
        intreat_x = sen_model(intreat_x)

        txt_embding = txt_embd_model([inhospital_x, indiagnosis_x, outdiagnosis_x, intreat_x])
        embd_model = Model([inhospital_x1, inhospital_x2, indiagnosis_x1, indiagnosis_x2,
                            outdiagnosis_x1, outdiagnosis_x2, intreat_x1, intreat_x2], txt_embding, name='Embdmodel')
        if is_tuning == True:
            embd_model.load_weights(embd_h5)
        # for line in embd_model.layers:
        #     line.trainable = True
        txt_embd = embd_model([inhospital_x1, inhospital_x2, indiagnosis_x1, indiagnosis_x2,
                            outdiagnosis_x1, outdiagnosis_x2, intreat_x1, intreat_x2])

        y_range = range_model(txt_embd)

        rangemodel = Model([inhospital_x1, inhospital_x2, indiagnosis_x1, indiagnosis_x2,
                       outdiagnosis_x1, outdiagnosis_x2, intreat_x1, intreat_x2], y_range, name='rangemodel')
        # rangemodel.summary()
        return rangemodel, embd_model


    def read_lineBERT(self, line, x1, x2):
        if line != '':
            inp1_x1, inp1_x2 = tokenizer.encode(first=line, max_len=128)
        else:
            inp1_x1 = [0] * 128
            inp1_x2 = [0] * 128
        x1.append(inp1_x1)
        x2.append(inp1_x2)
        return x1, x2


    def positional_embedding(self):
        maxlen = 1700
        model_size = 768
        PE = np.zeros((maxlen, model_size))
        for i in range(maxlen):
            for j in range(model_size):
                if j % 2 == 0:
                    PE[i, j] = np.sin(i / 10000 ** (j / model_size))
                else:PE[i, j] = np.cos(i / 10000 ** ((j-1) / model_size))
        return PE


    def evaluation(self, test_file, sim_model, range_model, diagnosis_list, input_l, name):
        # data_file = './data/test_data.jsonl'
        data_file = open(test_file, 'r', encoding='utf8')


        a, b, c, d = 0, 0, 0, 0
        result_pre_sim, result_pre_range = [], []
        x_range = input_l.reshape((34, 50, 768))
        for line in data_file.readlines():
            y_index = []
            inhospital_x1, inhospital_x2, indiagnosis_x1, indiagnosis_x2 = [], [], [], []
            outdiagnosis_x1, outdiagnosis_x2, intreat_x1, intreat_x2 = [], [], [], []
            b += 1
            line = json.loads(line)
            inhospital_x1, inhospital_x2 = self.read_lineBERT(line['data_1'], inhospital_x1, inhospital_x2)
            indiagnosis_x1, indiagnosis_x2 = self.read_lineBERT(line['data_2'], indiagnosis_x1, indiagnosis_x2)
            outdiagnosis_x1, outdiagnosis_x2 = self.read_lineBERT(line['data_4'], outdiagnosis_x1, outdiagnosis_x2)
            intreat_x1, intreat_x2 = self.read_lineBERT(line['data_6'], intreat_x1, intreat_x2)

            line_label = line['diagnosis_nation']
            pos_index = diagnosis_list.index(line_label)
            y_index.append(pos_index)

            inhospital_x1 = np.array(inhospital_x1)
            inhospital_x2 = np.array(inhospital_x2)
            indiagnosis_x1 = np.array(indiagnosis_x1)
            indiagnosis_x2 = np.array(indiagnosis_x2)

            outdiagnosis_x1 = np.array(outdiagnosis_x1)
            outdiagnosis_x2 = np.array(outdiagnosis_x2)
            intreat_x1 = np.array(intreat_x1)
            intreat_x2 = np.array(intreat_x2)

            x_data_test_embd = [inhospital_x1, inhospital_x2, indiagnosis_x1, indiagnosis_x2,
                                outdiagnosis_x1, outdiagnosis_x2, intreat_x1, intreat_x2]
            x_data_test_sim = [np.repeat(inhospital_x1, 34, axis=0), np.repeat(inhospital_x2, 34, axis=0),
                               np.repeat(indiagnosis_x1, 34, axis=0),np.repeat(indiagnosis_x2, 34, axis=0),
                               np.repeat(outdiagnosis_x1, 34, axis=0),np.repeat(outdiagnosis_x2, 34, axis=0),
                               np.repeat(intreat_x1, 34, axis=0),np.repeat(intreat_x2, 34, axis=0)]

            y_range_pre = range_model.predict(x_data_test_embd)
            y_sim_pre = sim_model.predict(x_data_test_sim + [x_range])

            y_range_pre_T = np.repeat(y_range_pre.T, 50, axis=1)
            y_sim_all = np.multiply(y_range_pre_T, y_sim_pre)
            y_sim_all = y_sim_all.reshape((1700,))
            result_pre_sim.append(y_sim_all)
            result_pre_range.append(y_range_pre[0])
            sim_max5 = y_sim_all.argsort()[-5:][::-1]
            if pos_index in sim_max5:
                a += 1
            if pos_index == sim_max5[0]:
                c += 1

            if b % 200 == 0 or b == 2627:
                print('acc', a / b, c / b,  d /b, a, b, c)
            # if b == 1000:
            #     break

        result_pre_sim = np.array(result_pre_sim)
        result_pre_range = np.array(result_pre_range)
        print(result_pre_sim.shape)
        # np.save('./data/{0}/0_result_pre_1700sim7-3.npy'.format(name), result_pre_sim)
        # np.save('./data/{0}/0_result_pre_1700range7-3.npy'.format(name), result_pre_range)

        print('acc', a / b, c / b, d /b, a, b)


    def predict(self, test_file, diagnosis_list, input_l, windows, name):
        sim_model = self.SimModel(windows)
        range_model = self.RangeModel(input_l)
        # embd_model = EmbdModel()
        sim_model.load_weights('../301_all_model_0707/Allmodel/a1_simModel7.h5')
        # embd_model.load_weights('./Allmodel/a1_simModel_embd2.h5')
        range_model.load_weights('../301_all_model_0707/Allmodel/b1_rangeModel3.h5')
        self.evaluation(test_file, sim_model, range_model, diagnosis_list, input_l, name)


if __name__ == '__main__':
    build = BuildModel()
    diagnosis_file = '../301_all_model_0707/data/ICD_code_cancer.txt'
    tokenizer, bert_model = build.load_bert(1)
    # windows = 50
    train_file = '../301_all_model_0424/data/train_data.jsonl'
    test_file = '../301_all_model_0424/data/test_data.jsonl'
    valid_file = '../301_all_model_0424/data/valid_data.jsonl'

    PE = build.positional_embedding()
    input_l = build.labeEmbding(diagnosis_file, tokenizer, bert_model)
    input_l = np.vstack((input_l, np.array([[0] * 768] * 13)))
    input_l += PE

    diagnosis_file = open(diagnosis_file, encoding='utf8')
    diagnosis_list = []
    for line in diagnosis_file.readlines():
        diagnosis_list.append(line.split('\t')[0])
    diagnosis_list += [''] * 13
    # train(train_file, test_file, diagnosis_list, input_l, windows)
    # test(test_file, diagnosis_list, input_l, windows)
    build.predict(test_file, diagnosis_list, input_l, windows, 'test')
    build.predict(train_file, diagnosis_list, input_l, windows, 'train')
    build.predict(valid_file, diagnosis_list, input_l, windows, 'valid')