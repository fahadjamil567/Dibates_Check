from http.server import BaseHTTPRequestHandler
import streamlit as st
from app import app

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # Run the Streamlit app
        try:
            st.script_runner.get_script_run_ctx()
            app()
            self.wfile.write(str(st._get_widget_states()).encode())
        except Exception as e:
            self.wfile.write(str(e).encode()) 