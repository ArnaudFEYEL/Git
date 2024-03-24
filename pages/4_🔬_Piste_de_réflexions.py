import streamlit as st
import os
import time
import re
import numpy as np
import pandas as pd 
import subprocess
import matplotlib.pyplot as plt
import sys
import torch
sys.path.append('sub_code/')  
import NEDO_original
import NEDO_GeLU
import NEDO_Leaky_ReLU

import base64

import LEDO_original
import LEDO_1
import LEDO_2
import LEDO_3
import LEDO_4


#tester larchitecture de JAX et sinon dire d'utliser autre chiose 


# Set page title and icon
st.set_page_config(
    page_title="Piste de réflexions",
    page_icon="🔬",
)

# Main title
st.title("🔬 Piste de réflexions")

# Create a sidebar for navigation
page = st.sidebar.radio("Aller à", ["NEDOS", "Latent ODE", "Normalizing Flow"])

# Define the content for each page
if page == "NEDOS":
    st.markdown("<h1 style='text-align: center; color: white;'>Neural ODEs", unsafe_allow_html=True)

    st.write(r"""En guise de preuve pratique, nous allons maintenant tester si un NODE peut effectivement restaurer la vraie fonction de dynamique à l'aide de données échantillonnées. Pour ce faire, nous spécifierons un ODE, le ferons évoluer et échantillonnerons des points sur sa trajectoire, puis le restaurerons. 
         Tout d'abord, nous testerons un ODE linéaire simple. 
         La dynamique est donnée la matrice ci dessous matrice.
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

    # Graph plot function
    def plot_graph_example():
        st.title("Evolution du résultat de l'exemple")

        # Start button to initiate the animation
        if st.button("Commencer l'animation", key="start_animation"):
            # Select folder containing PNG files
            folder_path = "data/plot_example"

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
                restart_button = st.button("Recommencer l'animation", key="restart_button")
                if restart_button:
                    start_progress = True  # Set the flag to start progress again
                
    def plot_graph_user():
        st.title('Evolution du résultat')

        # Start button to initiate the animation
        if st.button("Commencer l'animation du code original", key="start_user_animation_orginal"):
            # Select folder containing PNG files
            folder_path = "data/user_try/NEDO_original"

            # Check if folder path is provided
            if folder_path:
                # Read PNG files from the folder
                png_files = read_png_files(folder_path)

                # Placeholder for the selected image
                selected_image_placeholder = st.empty()

                # Progress through the iterations automatically
                for i in range(len(png_files) * 10):  # Progress through each step of 50 to be faster
                    time.sleep(0.0000001)  # Adjust the speed of progression
                    iteration = i // 10 * 10  # Get the current iteration

                    # Display the selected PNG file
                    selected_png = os.path.join(folder_path, png_files[iteration // 10])
                    selected_image_placeholder.image(selected_png, use_column_width=True)

                # Button to restart the progress
                restart_button = st.button("Recommencer l'animation", key="restart_button")
                if restart_button:
                    start_progress = True  # Set the flag to start progress again
        
        if st.button("Commencer l'animation du code avec GeLU", key="start_user_animation_GeLU"):
            # Select folder containing PNG files
            folder_path = "data/user_try/NEDO_GeLU"

            # Check if folder path is provided
            if folder_path:
                # Read PNG files from the folder
                png_files = read_png_files(folder_path)

                # Placeholder for the selected image
                selected_image_placeholder = st.empty()

                # Progress through the iterations automatically
                for i in range(len(png_files) * 10):  # Progress through each step of 50 to be faster
                    time.sleep(0.0000001)  # Adjust the speed of progression
                    iteration = i // 10 * 10  # Get the current iteration

                    # Display the selected PNG file
                    selected_png = os.path.join(folder_path, png_files[iteration // 10])
                    selected_image_placeholder.image(selected_png, use_column_width=True)

                # Button to restart the progress
                restart_button = st.button("Recommencer l'animation", key="restart_button")
                if restart_button:
                    start_progress = True  # Set the flag to start progress again
        
        if st.button("Commencer l'animation du code avec Leaky ReLU", key="start_user_animation_Leaky_ReLU"):
            # Select folder containing PNG files
            folder_path = "data/user_try/NEDO_Leaky_ReLU"

            # Check if folder path is provided
            if folder_path:
                # Read PNG files from the folder
                png_files = read_png_files(folder_path)

                # Placeholder for the selected image
                selected_image_placeholder = st.empty()

                # Progress through the iterations automatically
                for i in range(len(png_files) * 10):  # Progress through each step of 50 to be faster
                    time.sleep(0.0000001)  # Adjust the speed of progression
                    iteration = i // 10 * 10  # Get the current iteration

                    # Display the selected PNG file
                    selected_png = os.path.join(folder_path, png_files[iteration // 10])
                    selected_image_placeholder.image(selected_png, use_column_width=True)

                # Button to restart the progress
                restart_button = st.button("Recommencer l'animation", key="restart_button")
                if restart_button:
                    start_progress = True  # Set the flag to start progress again                
    
    def create_matrix():
        st.title("À votre tour !")
        st.write(r"""Essayez de résoudre votre équation différentielle dans $\mathbb{R}^{2}$ avec un NODE. 
             Nous restons dans le cadre d'une équation différencielle *linéaire* pour simplifier le modèle.""")
        st.write(r"""C'est à dire un problème de la forme 
             $$\frac{dz}{dt} = Az$$ avec $A \in \mathcal{M}_{2}(\mathbb{R})$.""")


        st.write(r"""Commencez par entrer votre matrice $A$.""")

        matrix = [[0,0], [0,0]]
        value_00 = float(st.number_input(f"Ligne 1, Colonne 1", key = "00"))
        matrix[0][0] = value_00
        value_01 = float(st.number_input(f"Ligne 1, Colonne 2", key = "01"))
        matrix[0][1] = value_01
        value_10 = float(st.number_input(f"Ligne 2, Colonne 1", key = "10"))
        matrix[1][0] = value_10
        value_11 = float(st.number_input(f"Ligne 2, Colonne 2", key = "11"))
        matrix[1][1] = value_11

        print(type(matrix))
        
        return matrix
    
    def matrix_to_torch(matrix):
        
        if st.button("Enregistrer la matrice", key = "save_torch_matrix"):
            
            #Converting Matrix to torch
            matrix = torch.tensor(matrix)
            #matrix = matrix.clone().detach()
            #matrix.requires_grad_(True)
            
            #Saving for call later
            torch.save(matrix, 'data/matrix.pth')
        
    def show_matrix(matrix):
        
        st.write("Votre matrice A en input vérifie l'EDO suivante :")
        nice_matrix = np.zeros((2,2))
        
        nice_matrix[0,0] = matrix[0][0]
        nice_matrix[0,1] = matrix[0][1]
        nice_matrix[1,0] = matrix[1][0]
        nice_matrix[1,1] = matrix[1][1]

        # Convert the matrix to a LaTeX string
        
        latex_matrix = r"\begin{bmatrix}"
        for row in nice_matrix:
            for value in row:
                latex_matrix += f"{value} & "
            latex_matrix = latex_matrix[:-2]  # Remove the last '& ' from each row
            latex_matrix += r"\\"
        latex_matrix += r"\end{bmatrix}"
        
        # Display the LaTeX equation
        st.latex(fr"\frac{{dz}}{{dt}} = {latex_matrix} z")
            
    def check_conditions(A):
        st.write("Il est commode de vérifier les conditions de stabilité de la matrice en input pour un meilleur résultat numérique.")
        # Stability: Check if real parts of eigenvalues are negative
        eigenvalues, _ = np.linalg.eig(A)
        is_stable = np.all(np.real(eigenvalues) < 0)
        
        if is_stable:
            st.write("Les parties réelles des valeurs propres sont négatives et non nulles. ✅")
        else:
            st.write(f"Les parties réelles des valeurs propres ({eigenvalues}) sont positives ou nulles. ❌")

        # Non-degeneracy: Check if matrix is non-singular
        is_nonsingular = np.linalg.det(A) != 0
        if is_nonsingular:
            st.write("La matrice est non dégénérée. ✅")
        else:
            st.write("La matrice est dégénérée. ❌")

        # Real Eigenvalues: Check if eigenvalues are real
        is_real_eigenvalues = np.all(np.imag(eigenvalues) == 0)
        if is_real_eigenvalues:
            st.write("Les valeurs propres de la matrice sont réelles. ✅")
        else:
            st.write("Les valeurs propres de la matrice ne sont pas réelles. ✅")

    def choose_parameters():
        st.write(r""" Le point clé de la méthode NODE est qu'elle repose sur la méthode de l'adjoint pour calculer $\nabla_{\theta}L$. 
                 Cela implique donc directement l'utilisation d'une fonction de perte $L$ différentiable, ou au moins dérivable. 
                 Pour appliquer la méthode NODE, vous avez le choix du nombre d'itérations. 
                 Nous recommandons au moins 1000 itérations : l'algorithme peut prendre du temps à converger.""")
            
        # Prompt the user to enter the number of iterations
        iterations = st.number_input("Enter the number of iterations:", min_value=1, step=1)
        
        # Check if the input is not a multiple of 10
        if iterations % 10 != 0:
            # Round up to the nearest number divisible by 10
            iterations = np.ceil(iterations / 10) * 10
            st.write(f"Nombre d'itérations ajusté: {int(iterations)}")
        
        # Open a file in write mode
        with open("data/iteration.py", "w") as file:
        # Write the integer value as a Python variable assignment
            file.write(f"user_it = {int(iterations)}")
        
    def Picard_PB():
        st.write(r"""Le théorème d'existence de Picard stipule que la solution à un problème de valeur initiale existe 
                et est unique si l'équation différentielle est uniformément continue de Lipschitz en $z$ et continue en $t$. 
                Ce théorème s'applique à notre modèle si le réseau neuronal a des poids finis et utilise des non-linéarités de Lipschitz, 
                telles que tanh ou relu.
                En pratique, le nombre de poids est toujours finis et la plupart des fonctions d'activations sont Lipschitz et non linéaires.  
                Alors, on peut se demander si le choix de telles fonctions impacte la convergence du réseaux vers la solution..
                """)
    
    def try_code_original():
        # Button to run the code
        if st.button("Essayer le code soit même ", key = "original"):
            
            # Execute the code
             NEDO_original.main()

    def try_code_GeLU():
        # Button to run the code
        if st.button("Essayer le code soit même avec la fonction d'activation GeLU (Lipshitz)", key = "GeLU"):
            
            # Execute the code
             NEDO_GeLU.main()
             
    def try_code_Leaky_ReLU():
        # Button to run the code
        if st.button("Essayer le code soit même avec la fonction d'activation Leaky ReLU (Lipshitz)", key = "Leaky_ReLU"):
            
            # Execute the code
             NEDO_Leaky_ReLU.main()
             
    def main():
        start_progress = False 
        plot_graph_example()
        matrix = create_matrix()
        show_matrix(matrix)
        check_conditions(matrix)
        matrix_to_torch(matrix)
        choose_parameters()
        Picard_PB()
        try_code_original()
        try_code_GeLU()
        try_code_Leaky_ReLU()
        plot_graph_user()
        
    if __name__ == '__main__':
        main()

elif page == "Latent ODE":
    # Noor
    st.markdown("<h1 style='text-align: center; color: white;'>Neural ODEs", unsafe_allow_html=True)

    st.write("""Nous allons maintenant entraîner une EDO latente afin de répliquer la distribution d'un ensemble de données de certains oscillateurs à amplitudes décroissantes.""") 
    
    st.write("""Les observations sont issues d'une équation de la forme:""")  
    st.latex(r"""y(t) = e^{At} y_0""")

    st.write(r"""avec $y_0, y(t) \in \mathbb{R}^2$, où $y_0 \sim \mathcal{N}(0,I_{2*2})$""")
    st.write(r"""$A \in \mathbb{R}^{2 * 2}$ telle que les valeurs propres de $A$ soient complexes avec des composantes réelles négatives.""")


    """🔎 Ce qui est vraiment intéressant dans cet exemple est l'échantillonnage de manière irrégulière des données soujacentes.
        \n ➡️ Nous notons donc des temps d'observation différents pour différents éléments du lot.
    """
    
    # Function to read PNG files corresponding to each iteration
    def read_png_files(folder_path):
        png_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')], key=sort_key)
        return png_files

    # Custom sorting function to sort filenames based on their numeric part
    def sort_key(filename):
        return int(re.search(r'\d+', filename).group())

    # Graph plot function
    def plot_graph_example():
        st.title("Evolution du résultat de l'exemple")
        st.write(r""" Regardons les résultats dans le cadre d'un exemple avec une matrice
            $$ 
            A =
            \begin{bmatrix}
            -0.1 & 1.3 \\
            -1 & -0.1
            \end{bmatrix}
            $$
            """)
        # Start button to initiate the animation
        if st.button("Commencer l'animation", key="start_animation"):
            # Select folder containing PNG files
            folder_path = "data/user_try/LEDO_original"

            # Check if folder path is provided
            if folder_path:
                # Read PNG files from the folder
                png_files = read_png_files(folder_path)

                # Placeholder for the selected image
                selected_image_placeholder = st.empty()

                # Progress through the iterations automatically
                for i in range(len(png_files)):  # Progress through each step of 10
                    time.sleep(0.07)  # Adjust the speed of progression

                    # Display the selected PNG file
                    if i < len(png_files):
                        selected_png = os.path.join(folder_path, png_files[i])
                        selected_image_placeholder.image(selected_png, use_column_width=True)
            
            # Create another st.empty() element for the restart button
            restart_button_placeholder = st.empty()
    
            # Button to restart the progress
            restart_button = st.button("Recommencer l'animation", key="restart_button")
            if restart_button:
                start_progress = True  # Set the flag to start progress again
        """On obtient bien la forme voulue!"""

    def plot_solution(A):
        # Generate random initial condition y_0 ~ N(0, I)
        y_0 = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2))
    
        # Define time points for plotting
        t = np.sort(np.random.uniform(0, 3, 250))  # Select 250 random points in [0, 3]
    
        # Compute y(t) = exp(At) * y_0
        y_t = np.dot(np.exp(A * t[:, None, None]), y_0)
    
        # Plot the solution
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(t, y_t[:, 0], label='y_1(t)')
        plt.plot(t, y_t[:, 1], label='y_2(t)')
        plt.title('Distribution de y(t)')
        plt.xlabel('Temps (t)')
        plt.ylabel('y(t)')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)


    def Matrix_choices():
        st.write(r"Choisir la matrice $A$:")

        option = st.selectbox(
            "Select an option:",
            ("Matrice 1", "Matrice 2", "Matrice 3", "Matrice 4")
        )

        if option == "Matrice 1":
            A = np.array([[-1, 0], [0, -1]])
            st.write(r"""
            $$ 
            A =
            \begin{bmatrix}
            -1 & 0 \\
            0 & -1
            \end{bmatrix}
            $$
            """)
            plot_solution(A)
        elif option == "Matrice 2":
            A = np.array([[-2, 0], [0, -3]])
            st.write(r"""
            $$ 
            A =
            \begin{bmatrix}
            -2 & 0 \\
            0 & -3
            \end{bmatrix}
            $$
            """)
            plot_solution(A)
        elif option == "Matrice 3":
            A = np.array([[-2, 1], [-1, -2]])            
            st.write(r"""
            $$ 
            A =
            \begin{bmatrix}
            -2 & 1 \\
            -1 & -2
            \end{bmatrix}
            $$
            """)
            plot_solution(A)
        elif option == "Matrice 4":
            A = np.array([[-1, 2], [-1, -1]])            
            st.write(r"""
            $$ 
            A =
            \begin{bmatrix}
            -1 & 2 \\
            -1 & -1
            \end{bmatrix}
            $$
            """)
            plot_solution(A)
        
        return option
    

    def plot_graph_user(option):
        st.title('Evolution du résultat')

        # Start button to initiate the animation
        if st.button("Commencer l'animation du code avec la matrice choisie", key="start_user_animation_orginal"):

            # Select folder containing PNG files
            if option == "Matrice 1":
                folder_path = "data/user_try/latent_plots_1"
            elif option == "Matrice 2":
                folder_path = "data/user_try/latent_plots_2"
            elif option == "Matrice 3":
                folder_path = "data/user_try/latent_plots_3"
            elif option == "Matrice 4":
                folder_path = "data/user_try/latent_plots_4"

            # Check if folder path is provided
            if folder_path:
                # Read PNG files from the folder
                png_files = read_png_files(folder_path)

                # Placeholder for the selected image
                selected_image_placeholder = st.empty()

                # Progress through the iterations automatically
                for i in range(len(png_files)):  # Progress through each step of 10
                    time.sleep(0.07)  # Adjust the speed of progression

                    # Display the selected PNG file
                    if i < len(png_files):
                        selected_png = os.path.join(folder_path, png_files[i])
                        selected_image_placeholder.image(selected_png, use_column_width=True)
            
            # Create another st.empty() element for the restart button
            restart_button_placeholder = st.empty()
    
            # Button to restart the progress
            restart_button = st.button("Recommencer l'animation", key="restart_button")
            if restart_button:
                start_progress = True  # Set the flag to start progress again
        """On obtient bien la forme voulue!"""       
    
    def main():
        plot_graph_example()
        st.title("À votre tour !")
        option = Matrix_choices()
        plot_graph_user(option)
    

    if __name__ == "__main__":
        main()
    
elif page == "Normalizing Flow":
    #Malek

    st.write("This is the content of Normalizing Flow")

    st.write("Nous allons étudier de code")
    st.write(r"""Ce code permet de visualiser l'évolution de la distribution du Continuous Normalizing Flows de t = 0 à t = T,
             en prenant en entrée une image (dans notre exemple c'est le logo de GitHub). L'image est transformée en une représentation en 2D,
                puis cette représentation est utilisée pour entraîner le modèle CNF à apprendre la distribution de probabilité des données sous-jacentes.
                """)
        
    # Pré-requis
    st.write("Installer `torchdiffeq` de https://github.com/rtqichen/torchdiffeq.")
    # Visualisation des résultats
    st.write("Afin de visualiser les résulats, nous exécutons le code ci-dessous.")
    
    st.write("python train_img2d.py --img imgs/github.png --save github_flow")

    st.write(r"""L'exécution de ce code dure environ 1h30. Il retourne les informations de suivi de l'entraînement du modèle CNF à chaque itération.
    Le nombre d'itérations de l'entraînement est 10000. La moyenne cumulative de toutes les pertes est environ 3.47.
    Le temps moyen pris par le modèle pour évaluer une seule étape dans le CNF est 0.5000""")

    file_ = open("./sub_code/cnf/assets/github_flow.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.image(data_url, use_column_width=True)

    ## Explication du processus 
    ### Prétraitement de l'image
    st.write(r"""L'image est chargée et prétraitée pour être utilisée comme données d'entraînement pour le modèle CNF. 
    Dans ce script, l'image est transformée en une représentation en 2D qui peut être interprétée comme des coordonnées dans un espace bidimensionnel. """)
    ### Construction du Modèle CNF
    st.write(r"""Le script construit le modèle CNF en fonction des arguments fournis, tels que la configuration des couches, le nombre de blocs CNF, etc. Le modèle CNF apprendra à modéliser la distribution de probabilité sous-jacente des données d'entraînement.""")
    ### Entraînement du Modèle
    st.write(r"""Le modèle CNF est entraîné sur les données d'entraînement. Pendant l'entraînement, le modèle ajuste ses paramètres pour minimiser la différence entre la distribution de probabilité qu'il apprend et la distribution des données réelles.""")
    ### Évaluation et Visualisation
    st.write(r"""À intervalles réguliers, le modèle est évalué sur un ensemble de validation pour surveiller ses performances. De plus, des visualisations sont effectuées pour observer comment le modèle transforme les données au fil du temps.""")
    ### Sauvegarde du Modèle
    st.write(r"""Une fois l'entraînement terminé, le meilleur modèle est sauvegardé pour une utilisation ultérieure.""")