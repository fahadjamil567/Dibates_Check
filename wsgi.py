import streamlit as st
from app import app

def handler(event, context):
    try:
        st.script_runner.get_script_run_ctx()
        return app()
    except Exception as e:
        return {
            'statusCode': 500,
            'body': str(e)
        } 