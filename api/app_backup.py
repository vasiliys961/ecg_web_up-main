# # ecg_web_app/app.py
# app.py
"""
Основной файл веб-приложения Flask для анализа ЭКГ.
"""

import os
import sys

# --- Добавляем текущую директорию в путь поиска модулей ---
# Это решает проблему "ModuleNotFoundError: No module named 'analysis_scripts'"
# Получаем абсолютный путь к директории, где лежит этот файл (app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Добавляем её в sys.path, если её там ещё нет
if current_dir not in sys.path:
    sys.path.insert(0, current_dir) # insert(0, ...) ставит в начало, приоритет выше
# -----------------------------------------------------------

print(f"Корневая директория проекта: {os.getcwd()}")
print(f"Директория app.py: {current_dir}")
# print(f"sys.path (первые 3 элемента): {sys.path[:3]}") # Для отладки

from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__, template_folder='templates')
app.secret_key = 'supersecretkey' # Для flash-сообщений

# --- Настройки ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'} # Пока только CSV
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- УВЕЛИЧЕНИЕ ЛИМИТА РАЗМЕРА ФАЙЛА ---
# Установите максимальный размер файла, например, 100MB (100 * 1024 * 1024 байт)
# Это должно быть достаточно для файла ~60MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 100 Megabytes
# ---------------------------------------

# Убедимся, что папка для загрузок существует
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Проверяет, разрешено ли расширение файла."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Маршрут для главной страницы (GET /)"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Маршрут для обработки загрузки (POST /upload)"""
    # 1. Проверка, есть ли файл в запросе
    if 'ecg_file' not in request.files:
        flash('Файл не выбран')
        return redirect(request.url)
    
    file = request.files['ecg_file']
    
    # 2. Проверка, не пустой ли файл
    if file.filename == '':
        flash('Файл не выбран')
        return redirect(request.url)

    # 3. Проверка разрешенного формата
    if file and allowed_file(file.filename):
        # 4. Сохранение файла на сервере
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"✅ Файл сохранен: {filepath}")

        # --- ИНТЕГРАЦИЯ АНАЛИЗА ---
        try:
            # --- ВАЖНО: Импорт внутри функции и путь к моделям ---
            # Формируем абсолютный путь к родительской директории папки models
            models_parent_dir = current_dir # Это '/Users/maxmobiles.ru/Documents/ecg_web_app'
            print(f"Родительская директория для папки models: {models_parent_dir}")

            # --- ЯВНО ДОБАВЛЯЕМ analysis_scripts в sys.path ---
            # Это дополнительная мера предосторожности
            analysis_scripts_path = os.path.join(current_dir, 'analysis_scripts')
            if analysis_scripts_path not in sys.path:
                sys.path.insert(0, analysis_scripts_path)
                print(f"Добавлен путь к analysis_scripts в sys.path: {analysis_scripts_path}")
            # ----------------------------------------------------

            # --- Попытка импорта с обработкой ошибок ---
            try:
                # Пробуем прямой импорт из пакета
                from analysis_scripts.predict_csv import analyze
                print("✅ Импорт через 'from analysis_scripts.predict_csv import analyze' успешен.")
                analyze_function = analyze # Для совместимости с кодом ниже
            except ImportError as ie1:
                print(f"⚠️ Прямой импорт не удался: {ie1}")
                try:
                    # Альтернатива 1: Импорт модуля, затем функции
                    import analysis_scripts.predict_csv as predict_csv_module
                    analyze_function = predict_csv_module.analyze
                    print("✅ Импорт через 'import analysis_scripts.predict_csv' успешен.")
                except ImportError as ie2:
                    print(f"⚠️ Импорт модуля не удался: {ie2}")
                    try:
                        # Альтернатива 2: Добавляем analysis_scripts в sys.path по-другому и пробуем снова
                        # Убедимся, что основная директория проекта в sys.path
                        if current_dir not in sys.path:
                            sys.path.insert(0, current_dir)
                        # Пробуем импорт снова
                        from analysis_scripts.predict_csv import analyze
                        print("✅ Повторный импорт через 'from analysis_scripts.predict_csv import analyze' успешен.")
                        analyze_function = analyze # Для совместимости с кодом ниже
                    except ImportError as ie3:
                        print(f"❌ Все попытки импорта неудачны: {ie3}")
                        raise ImportError(f"Не удалось импортировать 'analyze' из 'analysis_scripts.predict_csv'. "
                                          f"Проверьте структуру папок, наличие __init__.py и содержимое файлов. "
                                          f"Ошибки: 1) {ie1}, 2) {ie2}, 3) {ie3}")
            # --- КОНЕЦ ИМПОРТА ---

            # --- Вызов функции анализа ---
            print("🔮 Начинаем анализ файла...")
            # Формируем путь к папке, СОДЕРЖАЩЕЙ файлы моделей и нормализации
            # Это папка models, а не её родительская директория
            models_dir_path = os.path.join(models_parent_dir, 'models') # '/Users/maxmobiles.ru/Documents/ecg_web_app/models'
            print(f"Путь к папке с моделями и нормализацией: {models_dir_path}")
            
            # Проверка существования папки models и ключевых файлов (отладка)
            # Эта проверка не обязательна, если вы уверены в наличии файлов,
            # но помогает при отладке.
            # if not os.path.exists(models_dir_path):
            #     raise FileNotFoundError(f"Папка моделей не найдена: {models_dir_path}")
            # required_files = ["ecg_model.pth", "ecg_1dcnn_best.pth", "ecg_resnet1d_features_best.pth", "ecg_train_mean.npy", "ecg_train_std.npy"]
            # missing_files = [f for f in required_files if not os.path.exists(os.path.join(models_dir_path, f))]
            # if missing_files:
            #      raise FileNotFoundError(f"В папке моделей отсутствуют файлы: {missing_files}")

            # Выполняем анализ, передавая путь к папке models
            results = analyze_function(filepath, model_dir=models_dir_path) 
            
            print(f"✅ Анализ завершен. Найдено {len(results)} вероятных диагнозов.")
            
            # Отображаем страницу с результатами
            return render_template('results.html', results=results, filename=filename)
            # --- КОНЕЦ ВЫЗОВА АНАЛИЗА ---
            
        except FileNotFoundError as e:
            print(f"❌ Ошибка файловой системы при анализе: {e}")
            flash(f'Ошибка: Не найдены необходимые файлы модели или нормализации. {e}')
            return redirect(url_for('index'))
        except ValueError as e:
            print(f"❌ Ошибка данных при анализе: {e}")
            flash(f'Ошибка: Неверный формат CSV-файла. {e}')
            return redirect(url_for('index'))
        except ImportError as e:
            print(f"❌ Ошибка импорта: {e}")
            import traceback
            traceback.print_exc()
            flash(f'Ошибка импорта модуля анализа: {e}')
            return redirect(url_for('index'))
        except Exception as e:
            print(f"❌ Неожиданная ошибка анализа: {e}")
            import traceback
            traceback.print_exc()
            flash(f'Ошибка анализа файла: {e}')
            return redirect(url_for('index'))
        # --- КОНЕЦ ИНТЕГРАЦИИ АНАЛИЗА ---
        
    else:
        flash('Недопустимый формат файла. Разрешен: CSV.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5002) # Изменили порт
