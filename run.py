from vercel_app import app

if __name__ == '__main__':
    print("Starting Flask server...")
    print("You can access the API at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True) 