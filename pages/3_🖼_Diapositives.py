import streamlit as st
import base64

st.set_page_config(
    page_title="Diapositives",
    page_icon="🖼",
)

st.title("🖼 Diapositives")

def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML using embed tag
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# Example usage
file_path = "./text/slides.pdf"  # Replace with your PDF file path
displayPDF(file_path)

