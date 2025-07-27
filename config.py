import os

class Config:
    SECRET_KEY = os.urandom(32)  # Secure session
    MYSQL_HOST = 'localhost'
    MYSQL_USER = 'mave'
    MYSQL_PASSWORD = 'qwerty12345'
    MYSQL_DB = 'attendance_db'
    
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # Max upload 2MB
    
    UPLOAD_EXTENSIONS = ['.jpg', '.png']
    UPLOAD_PATH = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
    DATASET_PATH= os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
