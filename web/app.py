from flask import Flask, render_template, json, request
app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/run', methods=["POST"])
def run():
    file = request.files['sourceFile']
    learning_rate = request.form['learningRate']
    max_inter = request.form['maxInter']
    algorithm = request.form['algorithm']
    percent = request.form['percent']

    #TODO: add validation(optional), call machine learning module, return path to the picture (and data?)

    response = app.response_class(
        response=json.dumps("data"),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == '__main__':
   app.run(debug = True)
