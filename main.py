from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
import os
import shutil
import tagger
import enhancement


if not os.path.exists('/tmp/instance'):
    os.mkdir('/tmp/instance')
if not os.path.exists('/tmp/instance/uploads'):
    os.mkdir('/tmp/instance/uploads')

app = Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.JPG', '.png', '.PNG']
app.config['UPLOAD_PATH'] = '/tmp/instance/uploads'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # this prevents browsers from caching the return files


@app.errorhandler(400)
def invalid_type(e):
    return "File type not a .png or .jpg", 400


@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('main.html', files=files)


@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    extensions = tuple(app.config['UPLOAD_EXTENSIONS'])
    if uploaded_file.filename != '' and uploaded_file.filename.endswith(extensions):
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], uploaded_file.filename))
    elif not uploaded_file.filename.endswith(extensions):
        print(uploaded_file.filename + " was not image file!")
        abort(400)
    return redirect((url_for('index')))


@app.route('/Photos/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


@app.route('/Photos', methods=['GET'])
def zip_and_download():
    shutil.make_archive('/tmp/instance/uploads', 'zip', '/tmp/instance/uploads')
    return send_from_directory('/tmp/instance', 'uploads.zip')


@app.route('/gallery')
def gallery():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('gallery.html', files=files)


@app.route('/clear')
def clear_gallery():
    to_remove = [f for f in os.listdir(app.config['UPLOAD_PATH'])]
    for f in to_remove:
        os.remove(os.path.join(app.config['UPLOAD_PATH'], f))
    return redirect(url_for('gallery'))


if __name__ == "__main__":
    # Running locally
    app.config['UPLOAD_PATH'] = 'uploads'
    shutil.rmtree('uploads', True)
    os.mkdir('uploads')

    tagger.initialize(app.config['UPLOAD_PATH'], 'tagger/face_sample')
    enhancement.init()

    app.run(host="127.0.0.1", port=8080, debug=True)
