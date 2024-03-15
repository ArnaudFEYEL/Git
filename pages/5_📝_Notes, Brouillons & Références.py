import streamlit as st

st.set_page_config(
    page_title="Notes, Brouillons & Références",
    page_icon="📝",
)

st.title("📝 Notes, Brouillons & Références")

def main():
    # Display bibliography section
    st.markdown("## Bibliographie")

    # Citations
    st.markdown("1. <u>Neural Ordinary Differential Equations</u> Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. University of Toronto, Vector Institute. {rtqichen, rubanova, jessebett, duvenaud}@cs.toronto.edu",
                 unsafe_allow_html=True)

if __name__ == "__main__":
    main()
