import streamlit as st
import os
import time
import re


st.set_page_config(
    page_title="Piste de réflexions",
    page_icon="🔬",
)

st.title("🔬 Piste de réflexions")

st.write(r"""L'une des questions naturelles dans la méthode d'optimisation de $\nabla L_{\theta}$ est 
         de se demander ce qu'il se passe si $L$ n'est pas différentiable en tout points.
         Par exemple, la fonction de """)

st.write(r"""As a proof-of-concept we will now test if Neural ODE can indeed restore true dynamics function using sampled data.
To test this we will specify an ODE, evolve it and sample points on its trajectory, and then restore it.
First, we'll test a simple linear ODE. Dynamics is given with a matrix.
$$
\frac{dz}{dt} = \begin{bmatrix}-0.1 & -1.0\\1.0 & -0.1\end{bmatrix} z
$$
         """)

# Function to read PNG files corresponding to each iteration
def read_png_files(folder_path):
    png_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')], key=sort_key)
    return png_files

# Custom sorting function to sort filenames based on their numeric part
def sort_key(filename):
    return int(re.search(r'\d+', filename).group())

# Main function
def main_2_1():
    st.title('Evolution du résultat')

    # Select folder containing PNG files
    folder_path = "data/dim2.1"
    st.empty()  # Placeholder for the slider

    # Check if folder path is provided
    if folder_path:
        # Read PNG files from the folder
        png_files = read_png_files(folder_path)

        # Placeholder for the selected image
        selected_image_placeholder = st.empty()

        # Progress through the iterations automatically
        for i in range(len(png_files) * 10):  # Progress through each step of 10
            time.sleep(0.01)  # Adjust the speed of progression
            iteration = i // 10 * 10  # Get the current iteration

            # Display the selected PNG file
            selected_png = os.path.join(folder_path, png_files[iteration // 10])
            selected_image_placeholder.image(selected_png, use_column_width=True)

        # Button to restart the progress
        restart_button = st.button("Recommencer l'animation")
        if restart_button:
            start_progress = True  # Set the flag to start progress again
            
def main_2_2():
    
    # Select folder containing PNG files
    folder_path = "data/dim2.2"
    st.empty()  # Placeholder for the slider

    # Check if folder path is provided
    if folder_path:
        # Read PNG files from the folder
        png_files = read_png_files(folder_path)

        # Placeholder for the selected image
        selected_image_placeholder = st.empty()

        # Progress through the iterations automatically
        for i in range(len(png_files) * 10):  # Progress through each step of 10
            time.sleep(0.01)  # Adjust the speed of progression
            iteration = i // 10 * 10  # Get the current iteration

            # Display the selected PNG file
            selected_png = os.path.join(folder_path, png_files[iteration // 10])
            selected_image_placeholder.image(selected_png, use_column_width=True)

        # Button to restart the progress
        restart_button = st.button("Recommencer l'animation")
        if restart_button:
            start_progress = True  # Set the flag to start progress again
            
if __name__ == '__main__':
    main_2_1()
    main_2_2()

