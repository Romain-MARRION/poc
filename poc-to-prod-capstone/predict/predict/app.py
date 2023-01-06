#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:54:51 2023

@author: rmarrion
"""

from flask import Flask, render_template, request
from predict.predict.run import TextPredictionModel

app = Flask(__name__)


@app.route('/', methods=['GET'])
def form():
    return render_template('form.html')


@app.route('/hello', methods=['GET', 'POST'])
def hello():
    artefacts_path = "/home/rmarrion/Downloads/Capstone-20221116/poc-to-prod-capstone/poc-to-prod-capstone/train/data/artefacts/2023-01-03-12-06-56"
    model = TextPredictionModel.from_artefacts(artefacts_path)
    prediction = model.predict([request.form['text']], top_k=int(request.form['top_k']))
    return render_template('prediction.html', pred=prediction)


if __name__ == "__main__":
    app.run()
