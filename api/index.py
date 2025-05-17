from http.server import BaseHTTPRequestHandler
import streamlit.web.bootstrap as bootstrap
import os

def run_streamlit():
    bootstrap.run(
        "app.py",
        "streamlit run",
        [],
        {
            "server.address": "0.0.0.0",
            "server.port": int(os.environ.get("PORT", 8080)),
            "server.headless": True,
            "browser.serverAddress": "0.0.0.0",
        },
    )

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            run_streamlit()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"App is running")
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(str(e).encode()) 