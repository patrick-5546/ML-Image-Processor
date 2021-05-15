from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
import os
import shutil
from tagger import Tagger
# import enhancement

if not os.path.exists('./tmp'):
    os.mkdir('./tmp')
if not os.path.exists('./tmp/instance'):
    os.mkdir('./tmp/instance')
if not os.path.exists('./tmp/instance/uploads'):
    os.mkdir('./tmp/instance/uploads')

app = Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.JPG']
app.config['UPLOAD_PATH'] = './tmp/instance/uploads'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # this prevents browsers from caching the return files

local = False
tagger = Tagger(app.config['UPLOAD_PATH'], 'Test_Photos/face_sample')


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
    tagger.save_tags()
    shutil.make_archive(app.config['UPLOAD_PATH'], 'zip', app.config['UPLOAD_PATH'])
    if not local:
        return send_from_directory('/tmp/instance', 'uploads.zip')
    else:
        return send_from_directory('tmp/instance', 'uploads.zip')


@app.route('/gallery')
def gallery():
    files = os.listdir(app.config['UPLOAD_PATH'])
    if len(files) == 0:
        return render_template('empty_gallery.html')
    elif len(files) > 20:
        return render_template('large_gallery.html', files=files, tags=tagger.get_all_tags())
    return render_template('gallery.html', files=files, tags=tagger.get_all_tags())


@app.route('/clear')
def clear_gallery():
    to_remove = [f for f in os.listdir(app.config['UPLOAD_PATH'])]
    for f in to_remove:
        os.remove(os.path.join(app.config['UPLOAD_PATH'], f))
    return redirect(url_for('gallery'))


@app.route('/tags/<filename>', methods=['GET'])
def edit_tags(filename):
    all_tags = tagger.get_all_tags()
    if filename in all_tags:
        tags = all_tags[filename]
    else:
        tags = ""
    return render_template('tag.html', file=filename, tags=tags)


@app.route('/tagging/<filename>', methods=['POST'])
def update_tags(filename):
    all_tags = tagger.get_all_tags()
    if filename in all_tags:
        old_tags_str = all_tags[filename]
    else:
        old_tags_str = ""
    old_tags = old_tags_str.split("; ")

    form_data = request.form
    new_tags_str = ""
    for _, value in form_data.items():
        new_tags_str = value
    new_tags = new_tags_str.split("; ")

    for old_tag in old_tags:
        if old_tag not in new_tags:
            tagger.exclude_tag(filename, old_tag)
    for new_tag in new_tags:
        if new_tag not in old_tags:
            tagger.add_tag(filename, new_tag)
            print("Adding " + new_tag)

    return redirect(url_for('gallery'))


@app.route('/tagger')
def tag_photos():
    # Tag photos
    tagger.tag_all_images()
    return redirect(url_for('index'))


if __name__ == "__main__":
    # Running locally
    local = True
    app.config['UPLOAD_PATH'] = 'tmp/instance/uploads'
    shutil.rmtree('uploads', True)
    os.mkdir('uploads')

    old_files = [f for f in os.listdir(app.config['UPLOAD_PATH'])]
    for f in old_files:
        os.remove(os.path.join(app.config['UPLOAD_PATH'], f))

    # enhancement.init()

    app.run(host="127.0.0.1", port=8080, debug=True)
