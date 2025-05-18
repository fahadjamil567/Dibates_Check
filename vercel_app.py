from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return {"message": "Disease Detection API"}

@app.route('/health')
def health():
    return {"status": "healthy"} 