from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from convert_mp3_to_wav import convert_mp3_to_wav
from user_input_prediction import predict_genre

UPLOAD_FOLDER = os.path.join(os.path.expanduser("~"), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    main_title = "Ocimum Basilicum"
    return render_template('index.html', main_title=main_title)

@app.route('/upload-page')
def upload_page():
    main_title = "Genre Classification"
    return render_template('upload.html', main_title=main_title)

@app.route('/prepare')
def prepare():
    main_title = "Prepare Data Set"
    return render_template('prepare.html', main_title=main_title)

@app.route('/upload', methods=['POST'])
def song_upload():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})

    filename = secure_filename(file.filename)
    new_filename = os.path.join(app.config['UPLOAD_FOLDER'], "user_input" + os.path.splitext(filename)[1])

    file.save(new_filename)

    dst_wav = os.path.join(app.config['UPLOAD_FOLDER'], "converted_user_input.wav")
    convert_mp3_to_wav(dst_wav)
    modelpath = "model_checkpoint.ckpt"
    prediction = predict_genre(modelpath)

    return jsonify({'status': 'success', 'message': f'The genre is {prediction}', 'filename': filename})

# @app.route('/upload-folder', methods=['POST'])
# def folder_upload():
#     if 'folder' not in request.files:
#         return jsonify({'status': 'error', 'message': 'No folder part'})

#     folder = request.files['folder']

#     if folder.filename == '':
#         return jsonify({'status': 'error', 'message': 'No selected folder'})

#     folder_path = os.path.join(app.config['UPLOAD_FOLDER'], 'user_input_folder')
#     os.makedirs(folder_path, exist_ok=True)

#     folder.save(folder_path)

#     return jsonify({'status': 'success', 'message': 'Folder uploaded successfully', 'folder_path': folder_path})


if __name__ == '__main__':
    app.run(debug=True)
