import os
from flask import Flask, Markup, render_template, request
from flask_cors import CORS
from Model import json_func
# from google.cloud import bigquery

# Initialize flask application
app = Flask(__name__)
CORS(app)

# Define API route
@app.route("/")
def home():
    return render_template("index.html")
    # return True

@app.route("/cin")
def cin_func():
    cin_num=request.args.get("cin")
    op = json_func(cin_num)
    print(op)
    return op

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True, port = 8431)
