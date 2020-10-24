from flask import Flask, render_template, request
from search import cosine_similarity_T
from flask import jsonify
from ocr import ocr
import re
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def index_post():
    text = request.form['text']
    df = cosine_similarity_T(10, text).head(6)
    return render_template('index.html', frame=df)


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        file_name = f.filename
        f.save(file_name)
        text = str(ocr(str(file_name)))
        text = re.findall(r'\w+', text)
        text = ' '.join(text)
        df = cosine_similarity_T(10, text).head(6)
        return render_template('index.html', frame=df)


if __name__ == "__main__":
    app.run()
