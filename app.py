from flask import Flask, render_template, request
from search import cosine_similarity_T
from flask import jsonify
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def index_post():
    text = request.form['text']
    df = cosine_similarity_T(10, text)
    return render_template('result.html', tables=[df.to_html(classes='data')], titles=df.columns.values)


if __name__ == "__main__":
    app.run()
