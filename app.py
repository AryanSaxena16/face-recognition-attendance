import os
import io
import base64
from datetime import date, datetime

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

import cv2
import numpy as np
from PIL import Image
import json

app = Flask(__name__)


# Config
app.config.from_object('config.Config')

# Init DB, LoginManager
mysql = MySQL(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# Load Haar cascade for face detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id_, username, role):
        self.id = id_
        self.username = username
        self.role = role

    @staticmethod
    def get(user_id):
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT id, username, role FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            return User(user[0], user[1], user[2])
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# Role check decorator
def roles_required(*roles):
    def decorator(func):
        @login_required
        def wrapper(*args, **kwargs):
            if current_user.role not in roles:
                flash('Unauthorized access.', 'danger')
                return redirect(url_for('dashboard'))
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

# ============================
# Helper functions for LBPH model and data
# ============================

def save_face_samples(user_id, samples=20):
    # This function is ideal for offline/special client-side capture.
    # For fully web-based, capture multiple frames from webcam client side.
    face_cascade = haar_cascade
    user_folder = os.path.join(app.config['DATASET_PATH'], str(user_id))
    os.makedirs(user_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    while count < samples:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            file_path = os.path.join(user_folder, f'{count + 1}.jpg')
            cv2.imwrite(file_path, face_roi)
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'{count}/{samples}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Capturing faces - Press ESC to cancel', frame)
            if count >= samples:
                break
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def train_lbph_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_samples = []
    ids = []
    face_cascade = haar_cascade
    dataset_path = app.config['DATASET_PATH']

    for user_id in os.listdir(dataset_path):
        user_folder = os.path.join(dataset_path, user_id)
        if not os.path.isdir(user_folder):
            continue
        label = int(user_id)
        for file in os.listdir(user_folder):
            img_path = os.path.join(user_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces = face_cascade.detectMultiScale(img)
            if len(faces) == 0:
                continue
            (x, y, w, h) = faces[0]
            face_roi = img[y:y+h, x:x+w]
            face_samples.append(face_roi)
            ids.append(label)
    if len(face_samples) == 0:
        raise RuntimeError("No training faces found")

    recognizer.train(face_samples, np.array(ids))
    model_path = os.path.join(app.root_path, 'trainer.yml')
    recognizer.save(model_path)
    print(f"Trained LBPH model saved to {model_path}")


def train_lbph_face_recognizer(dataset_path=None, model_save_path=None):
    """
    Trains the LBPH face recognizer on the dataset stored in dataset_path.
    Saves the trained model to model_save_path.
    """
    if dataset_path is None:
        dataset_path = app.config['DATASET_PATH']
    if model_save_path is None:
        model_save_path = os.path.join(app.root_path, 'trainer.yml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_samples = []
    labels = []

    # Iterate through each user folder in dataset
    for user_id in os.listdir(dataset_path):
        user_folder = os.path.join(dataset_path, user_id)
        if not os.path.isdir(user_folder):
            continue
        try:
            label = int(user_id)
        except ValueError:
            continue  # ignore folders not named by integer user IDs

        for image_name in os.listdir(user_folder):
            image_path = os.path.join(user_folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces = face_cascade.detectMultiScale(img)
            if len(faces) == 0:
                continue

            # Take first detected face in image
            (x, y, w, h) = faces[0]
            face_roi = img[y:y+h, x:x+w]

            face_samples.append(face_roi)
            labels.append(label)

    if len(face_samples) == 0:
        raise ValueError("No training data found in dataset.")

    recognizer.train(face_samples, np.array(labels))
    recognizer.save(model_save_path)
    print(f'[INFO] Trained LBPH model saved to {model_save_path}')
# ================
# Routes
# ================

@app.context_processor
def inject_user():
    return dict(user=current_user)

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT id, password_hash, role FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()
        cursor.close()

        if user and check_password_hash(user[1], password):
            user_obj = User(user[0], username, user[2])
            login_user(user_obj)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

@app.route('/employees/add', methods=['GET', 'POST'])
@login_required
@roles_required('Admin')
def add_employee():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        fullname = request.form.get('fullname', '').strip()
        password = request.form.get('password', '')
        photo_data_array_json = request.form.get('photo_data_array', '')

        print("Raw photo_data_array_json received:", photo_data_array_json[:100])  # print first 100 chars

        if not username or not fullname or not password or not photo_data_array_json:
            flash('All fields and at least one photo required.', 'danger')
            return redirect(url_for('add_employee'))

        try:
            photo_data_array = json.loads(photo_data_array_json)
            print(f"Decoded photo_data_array size: {len(photo_data_array)}")
            for i, p in enumerate(photo_data_array):
                print(f"Photo {i} prefix check: {p[:30]}")  # print first 30 chars
            if not isinstance(photo_data_array, list) or len(photo_data_array) == 0:
                flash('No photos captured.', 'danger')
                return redirect(url_for('add_employee'))
        except Exception as e:
            print("Error decoding photo JSON:", e)
            flash('Invalid photo data format.', 'danger')
            return redirect(url_for('add_employee'))

        first_photo = photo_data_array[0]
        if not first_photo.startswith('data:image/jpeg;base64,'):
            flash('Invalid photo data format.', 'danger')
            return redirect(url_for('add_employee'))

        try:
            _, encoded_img = first_photo.split(',', 1)
            img_bytes = base64.b64decode(encoded_img)
        except Exception:
            flash('Error decoding photo data.', 'danger')
            return redirect(url_for('add_employee'))

        filename_safe = secure_filename(f"{username}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        upload_folder = app.config['UPLOAD_PATH']
        os.makedirs(upload_folder, exist_ok=True)
        profile_photo_path = os.path.join(upload_folder, filename_safe)

        try:
            with open(profile_photo_path, 'wb') as f:
                f.write(img_bytes)
        except Exception:
            flash('Failed to save profile photo.', 'danger')
            return redirect(url_for('add_employee'))

        hashed_pw = generate_password_hash(password)
        cursor = mysql.connection.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, password_hash, role, fullname, photo) VALUES (%s, %s, %s, %s, %s)",
                (username, hashed_pw, 'Employee', fullname, filename_safe)
            )
            mysql.connection.commit()
            user_id = cursor.lastrowid
        except Exception as e:
            print(f"DB error adding user: {e}")
            mysql.connection.rollback()
            flash('Error adding employee to database. Possibly duplicate username.', 'danger')
            cursor.close()
            return redirect(url_for('add_employee'))
        cursor.close()

        dataset_user_folder = os.path.join(app.config['DATASET_PATH'], str(user_id))
        os.makedirs(dataset_user_folder, exist_ok=True)

        for idx, photo_b64 in enumerate(photo_data_array):
            if not photo_b64.startswith('data:image/jpeg;base64,'):
                print(f"Skipping invalid photo at index {idx}")
                continue
            try:
                _, encoded = photo_b64.split(',', 1)
                img_bytes = base64.b64decode(encoded)
            except Exception as e:
                print(f"Error decoding photo at index {idx}: {e}")
                continue
            photo_path = os.path.join(dataset_user_folder, f"{idx + 1}.jpg")
            try:
                with open(photo_path, 'wb') as f:
                    f.write(img_bytes)
            except Exception as e:
                print(f"Failed saving photo {idx} : {e}")

        try:
            train_lbph_face_recognizer()
        except Exception as e:
            flash(f'Face recognition training failed: {str(e)}', 'warning')

        flash('Employee added and photos saved successfully.', 'success')
        return redirect(url_for('employees'))

    return render_template('add_employee.html')

@app.route('/employees')
@roles_required('Admin')
def employees():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT id, username, fullname, role FROM users WHERE role='Employee'")
    employees = cursor.fetchall()
    cursor.close()
    return render_template('employees.html', employees=employees)




@app.route('/mark_attendance', methods=['GET', 'POST'])
@roles_required('Employee', 'Admin')
def mark_attendance():
    if request.method == 'GET':
        return render_template('mark_attendance.html')

    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({'message': f'JSON error: {str(e)}'}), 400

    if not data or 'image' not in data:
        return jsonify({'message': 'No image provided'}), 400

    try:
        _, base64_img = data['image'].split(',',1)
        img_bytes = base64.b64decode(base64_img)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'message': f'Image decode error: {str(e)}'}), 400

    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    faces = haar_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return jsonify({'message': 'No face detected'}), 400
    if len(faces) > 1:
        return jsonify({'message': 'Multiple faces detected. Show only your face.'}), 400

    (x,y,w,h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]

    model_path = os.path.join(app.root_path, 'trainer.yml')
    if not os.path.exists(model_path):
        return jsonify({'message': 'Recognition model not found. Please ask admin to add employees.'}), 500

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    label, confidence = recognizer.predict(face_roi)

    CONFIDENCE_THRESHOLD = 60
    if label != current_user.id or confidence > CONFIDENCE_THRESHOLD:
        return jsonify({'message': 'Face does not match stored photo. Attendance denied.'}), 403

    today = date.today()
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM attendance WHERE user_id=%s AND attendance_date=%s", (current_user.id, today))
    attendance_exists = cursor.fetchone()
    cursor.close()

    if attendance_exists:
        return jsonify({'message': 'Attendance already marked today.'}), 200

    now = datetime.now().time()
    cursor = mysql.connection.cursor()
    cursor.execute("INSERT INTO attendance (user_id, attendance_date, attendance_time) VALUES (%s, %s, %s)",
                   (current_user.id, today, now))
    mysql.connection.commit()
    cursor.close()

    return jsonify({'message': 'Attendance marked successfully.'}), 200

@app.route('/attendance_report')
@login_required
@roles_required('Admin')  # Or roles you want to allow
def attendance_report():
    cursor = mysql.connection.cursor()
    query = """
        SELECT a.id, u.id AS employee_id, u.fullname, a.attendance_date, a.attendance_time
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        ORDER BY a.attendance_date, a.attendance_time
    """
    cursor.execute(query)
    attendance_records = cursor.fetchall()
    cursor.close()
    return render_template('attendance_report.html', records=attendance_records)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_PATH'], exist_ok=True)
    os.makedirs(app.config['DATASET_PATH'], exist_ok=True)
    app.run(debug=True)
