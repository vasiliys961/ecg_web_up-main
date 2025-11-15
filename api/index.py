import os
import sys
from flask import Flask, render_template, request, redirect, url_for, flash

# Важно: для Vercel template_folder должен ссылаться на "../templates"
app = Flask(__name__, template_folder="../templates")
app.secret_key = 'supersecretkey'  # Для flash-сообщений

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'ecg_file' not in request.files or request.files['ecg_file'].filename == '':
        flash('Файл не выбран')
        return redirect(url_for('index'))

    file = request.files['ecg_file']

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Пути для анализа
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, '..', 'models')
            analysis_scripts_path = os.path.join(base_dir, '..', 'analysis_scripts')
            if analysis_scripts_path not in sys.path:
                sys.path.insert(0, analysis_scripts_path)

            try:
                from predict_csv import analyze   # для упрощения структуры
            except ImportError:
                # Fallback – для старой структуры через пакет
                from analysis_scripts.predict_csv import analyze

            # Анализ
            results = analyze(filepath, model_dir=os.path.abspath(models_dir))

            return render_template('results.html', results=results, filename=filename)

        except Exception as e:
            flash(f'Ошибка анализа: {e}')
            return redirect(url_for('index'))
    else:
        flash('Недопустимый формат файла. Разрешен только CSV.')
        return redirect(url_for('index'))

# В Vercel app.run() НЕ нужен!

