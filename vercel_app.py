from app import app
import streamlit as st
from streamlit.web.bootstrap import run

def handler(event, context):
    return run(app, command_line="", args=[]) 