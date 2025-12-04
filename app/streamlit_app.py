"""
Streamlit application for RAG data-readiness tool.
"""

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    st.title("RAG Data-Readiness Tool")
    st.write("Powered by Cohere")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Main interface
    query = st.text_input("Enter your question:")

    if st.button("Submit"):
        if query:
            st.write("Processing query...")
            # Add query logic here
        else:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
