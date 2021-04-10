from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
import os
import zipfile

app = Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png']
app.config['UPLOAD_PATH'] = '/tmp'


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


@app.route('/download', methods=['GET'])
def zip_and_download():
    path = app.config['UPLOAD_PATH']
    ziph = zipfile.ZipFile('Photos.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))
    ziph.close()
    return send_from_directory(directory=os.path.dirname(app.instance_path), filename="Photos.zip")


if __name__ == "__main__":
    app.config['UPLOAD_PATH'] = 'uploads'

    old_files = [f for f in os.listdir('uploads')]
    for f in old_files:
        os.remove(os.path.join('uploads', f))

    app.run(host="127.0.0.1", port=8080, debug=True)
