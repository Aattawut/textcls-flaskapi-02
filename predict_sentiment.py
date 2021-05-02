#!/usr/bin/python
# -*-coding: utf-8 -*-
# from __future__ import absolute_import
######
import botnoi as bn
import pickle
import pandas as pd
from sklearn.svm import LinearSVC
import numpy as np
import joblib


def trainmodel(modelFileName='sentiment.mod'):
    # get data

    templist1 = ['อุณหภูมิสามสิบเจ็ดจุดสององศาครับ', 'อุณหภูมิสามสิบเจ็ดจุดห้าองศาค่ะ',
                 'อุณหภูมิสามสิบเจ็ดจุดแปดองศาค่ะ', 'อุณหภูมิสามสิบเเปดจุดศูนย์องศาครับ', 'อุณหภูมิสามสิบแปดจุดห้าองศา']
    templist2 = ['อุณหภูมิสามสิบแปดจุดเก้าองศาครับ', 'อุณหภูมิสามสิบเก้าจุดศูนย์องศาค่ะ',
                 'อุณหภูมิสามสิบเก้าจุดสององศา', 'อุณหภูมิสามสิบเก้าจุดห้าองศาครับ', 'อุณหภูมิสามสิบเก้าจุดห้าองศาค่ะ']
    templist3 = ['อุณหภูมิสามสิบเก้าจุดหกองศาค่ะ', 'อุณหภูมิสามสิบเก้าจุดเจ็ดองศาครับ',
                 'อุณหภูมิสามสิบเก้าจุดแปดองศาครับ', 'อุณหภูมิสี่สิบจุดศูนย์องศาค่ะ', 'อุณหภูมิสี่สิบจุดสามองศา']
    # 0 ไม่เป็น 1 เป็น
    headachelist0 = ['ไม่มีอาการปวดหัว', 'ไม่มีอาการปวดหัวครับ',
                     'ไม่ปวดหัวนะครับ', 'ไม่ปวดหัว', 'ไม่ปวดหัวเลย']
    headachelist1 = ['มีอาการปวดหัว', 'มีอาการปวดหัวเล็กน้อยครับ',
                     'ปวดหัวครับ', 'ปวดเเต่ไม่มาก', 'ปวดหัวนิดหน่อย']
    # extract feature
    temp1 = [bn.nlp.text(sen).getw2v_light() for sen in templist1]
    temp2 = [bn.nlp.text(sen).getw2v_light() for sen in templist2]
    temp3 = [bn.nlp.text(sen).getw2v_light() for sen in templist3]

    headache0 = [bn.nlp.text(sen).getw2v_light() for sen in headachelist0]
    headache1 = [bn.nlp.text(sen).getw2v_light() for sen in headachelist1]
    # create training set
    nlpdataset = pd.DataFrame()
    nlpdataset['feature'] = temp1 + temp2 + temp3 + headache0 + headache1
    nlpdataset['label'] = ['0']*5 + ['1']*5 + ['2']*5 + ['0']*5 + ['1']*5
    # train model
    clf = LinearSVC()
    mod = clf.fit(
        np.vstack(nlpdataset['feature'].values), nlpdataset['label'].values)
    # save model
    #pickle.dump(mod, open(modelFileName, 'wb'))
    joblib.dump(mod, modelFileName)
    return 'model created'


# load model
#mod = pickle.load(open('sentiment.mod', 'rb'))
mod = joblib.load('sentiment.mod')


def get_sentiment(sen):
    feat = bn.nlp.text(sen).getw2v_light()
    res = mod.predict([feat])[0]
    return {'result': res}
    # return jsonify(res)
