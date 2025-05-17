import os
import sys
import streamlit as st
from streamlit.web.bootstrap import run

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def handler(event, context):
    try:
        # Set Streamlit configuration
        os.environ['STREAMLIT_SERVER_PORT'] = '8080'
        os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
        
        # Import app after setting environment variables
        from app import app
        
        # Run the application
        return run(
            app,
            command_line="",
            args=[],
            flag_options={
                'server.address': '0.0.0.0',
                'server.port': 8080,
                'server.headless': True,
                'browser.serverAddress': '0.0.0.0',
            }
        )
    except Exception as e:
        print(f"Error running Streamlit app: {str(e)}")
        raise e 