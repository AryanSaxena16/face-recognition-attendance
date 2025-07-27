# Face Recognition Based Attendance System

This project is a web-based employee attendance system that uses face recognition technology to automate attendance marking. It leverages Flask for the backend, OpenCV for face detection and recognition, and MySQL for data storage.

---

## Features

- Secure user authentication with Flask-Login
- Role-based access: Admin and Employee
- Employee registration with multiple webcam face photo captures
- LBPH face recognition for attendance marking
- Attendance report with employee names and timestamps
- Webcam integration with JavaScript and HTML5
- Automated training of face recognition model after adding employees

---

## Technologies Used

- Python 3.x
- Flask
- Flask-Login
- Flask-MySQLdb
- OpenCV (opencv-contrib-python)
- Werkzeug Security
- MySQL database
- JavaScript (for webcam capture)
- HTML/CSS/Jinja2 templates

---

## Installation and Setup

1. **Clone the repository**:

git clone https://github.com/AryanSaxena16/face-recognition-attendance.git
cd face-recognition-attendance


2. **Create and activate a virtual environment (optional but recommended):**

python -m venv venv

On Windows
venv\Scripts\activate

On macOS/Linux
source venv/bin/activate


3. **Install dependencies:**

pip install -r requirements.txt


4. **Configure MySQL database:**

- Create a database and required tables (`users`, `attendance`, etc.).
- Update your Flask app configuration `app.config` with your MySQL connection details.

5. **Set configuration variables:**

- `UPLOAD_PATH`: Directory to store profile photos.
- `DATASET_PATH`: Directory to store face samples dataset.
- `SECRET_KEY` for Flask session management.(optional)

6. **Run the Flask app:**

python app.py


7. **Access the app:**

Open your browser and navigate to `http://localhost:5000`

---

## Usage

- Log in as an Admin to add new employees using webcam photo captures.
- Employees log in and mark attendance by capturing their face.
- Admins can view attendance reports with employee details.

---

## Folder Structure

/project_root
│
├── app.py
├── uploads/ # Stores profile photos
├── dataset/ # Stores face samples per user for training
├── templates/ # HTML templates
├── static/ # Static assets like CSS, JS
├── requirements.txt
└── README.md


---

## Important Notes

- Webcam requires user permission to access the camera.
- For production, secure database credentials and use environment variables for configuration.
- Large folders like `uploads/` and `dataset/` can be excluded from Git with `.gitignore`.
- Retraining the model happens automatically after adding employees.
- Ensure good lighting and multiple face samples for better accuracy.

---

## Contact

Your Name - aryansaxena2001@gmail.com
Project Link: https://github.com/AryanSaxena16/face-recognition-attendance


