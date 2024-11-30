import tkinter as tk
from tkinter import messagebox, ttk
import joblib
import pandas as pd
from modelo import train_light  # Assuming the train_light function is in a separate module named modelo

class DietaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recomendador de Dietas")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        # Model and related variables
        self.model = None
        self.label_encoders = None
        self.features = None
        
        # Attempt to load pre-trained model
        self.load_model()

        # Initialize variables
        self.nombre = tk.StringVar()
        self.edad = tk.StringVar()
        self.estatura = tk.StringVar()
        self.peso = tk.StringVar()
        self.sexo = tk.StringVar()
        self.tension = tk.StringVar()
        self.diabetes = tk.StringVar()
        self.fisico = tk.StringVar()
        self.actividad = tk.StringVar()

        # Clear user information file
        open("informacion_usuario.txt", "w").close()

        # Center window on screen
        self.center_window_on_screen()

        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create widgets
        self.create_widgets()

    def load_model(self):
        """Attempt to load pre-trained model"""
        try:
            self.model = joblib.load('diet_recommendation_model.pkl')
            self.label_encoders = joblib.load('label_encoders.pkl')
            self.features = ['Sex_Male', 'Age', 'Height', 'Weight', 'Hypertension', 
                             'Diabetes_Yes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type']
            print("Modelo cargado exitosamente")
        except FileNotFoundError:
            messagebox.showinfo("Modelo", "No se encontró un modelo entrenado. Por favor, entrene un modelo primero.")

    def center_window_on_screen(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 800) // 2
        y = (screen_height - 600) // 2
        self.root.geometry(f"800x600+{x}+{y}")

    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.main_frame)
        title_frame.pack(fill="x", pady=(0, 20))
        tk.Label(title_frame, text="Recomendador de Dietas", font=("Arial", 16, "bold")).pack()

        # Form frame
        form_frame = tk.Frame(self.main_frame, width=600)
        form_frame.pack(pady=(0, 20))
        form_frame.pack_propagate(False)

        # Configure grid columns
        form_frame.grid_columnconfigure(0, weight=1)
        form_frame.grid_columnconfigure(1, weight=1)
        form_frame.grid_columnconfigure(2, weight=1)

        # First row
        self.create_entry_grid(form_frame, "Nombre:", self.nombre, 0, 0)
        self.create_entry_grid(form_frame, "Edad (años):", self.edad, 0, 1)
        self.create_entry_grid(form_frame, "Estatura (cm):", self.estatura, 0, 2)

        # Second row
        self.create_entry_grid(form_frame, "Peso (kg):", self.peso, 1, 0)
        self.create_sex_grid(form_frame, 1, 1)
        self.create_option_grid(form_frame, "Tienes hipertensión?:", self.tension, ["Yes", "No"], 1, 2)

        # Third row
        self.create_option_grid(form_frame, "Tienes diabetes?:", self.diabetes, ["Yes", "No"], 2, 0)
        self.create_option_grid(form_frame, "Nivel de actividad:", self.actividad, ["Bajo", "Moderado", "Alto"], 2, 1)
        self.create_option_grid(form_frame, "Objetivo físico:", self.fisico, ["Pérdida de Peso", "Ganancia Muscular"], 2, 2)

        # Training button
        training_frame = tk.Frame(self.main_frame)
        training_frame.pack(fill="x", pady=(0, 20))
    

        # Save button
        button_frame = tk.Frame(self.main_frame)
        button_frame.pack(fill="x", pady=(0, 20))
        self.save_button = tk.Button(button_frame, text="Agregar", command=self.guardar_informacion, bd=5)
        self.save_button.pack()

        # Treeview
        self.create_treeview()

    def create_entry_grid(self, parent, label_text, variable, row, col):
        """Create an entry grid with label and entry"""
        frame = tk.Frame(parent)
        frame.grid(row=row, column=col, padx=10, pady=5, sticky="nsew")
        
        # Label
        label = tk.Label(frame, text=label_text, font=("Arial", 10))
        label.pack()
        
        # Entry
        entry = tk.Entry(frame, textvariable=variable, font=("Arial", 10), justify="center", width=20)
        entry.pack()

    def create_sex_grid(self, parent, row, col):
        """Create sex selection grid"""
        frame = tk.Frame(parent)
        frame.grid(row=row, column=col, padx=10, pady=5, sticky="nsew")
        
        # Label
        label = tk.Label(frame, text="Sexo:", font=("Arial", 10))
        label.pack()
        
        # Radio buttons
        tk.Radiobutton(frame, text="Masculino", variable=self.sexo, value="M").pack()
        tk.Radiobutton(frame, text="Femenino", variable=self.sexo, value="F").pack()

    def create_option_grid(self, parent, label_text, variable, options, row, col):
        """Create option menu grid"""
        frame = tk.Frame(parent)
        frame.grid(row=row, column=col, padx=10, pady=5, sticky="nsew")
        
        # Label
        label = tk.Label(frame, text=label_text, font=("Arial", 10))
        label.pack()
        
        # Option menu
        menu = tk.OptionMenu(frame, variable, *options)
        menu.config(cursor="hand2")
        menu.pack()

    def create_treeview(self):
        """Create treeview for displaying users and diets"""
        tree_frame = tk.Frame(self.main_frame)
        tree_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(tree_frame, columns=("ID", "Info", "Dieta"), show="headings", height=10)
        self.tree.heading("ID", text="ID", anchor="center")
        self.tree.heading("Info", text="Información", anchor="center")
        self.tree.heading("Dieta", text="Dieta", anchor="center")

        self.tree.column("ID", anchor="center", width=200)
        self.tree.column("Info", anchor="center", width=200)
        self.tree.column("Dieta", anchor="center", width=200)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.tree.bind('<ButtonRelease-1>', self.handle_click)

    def handle_click(self, event):
        """Handle treeview click events"""
        region = self.tree.identify_region(event.x, event.y)
        if region == "cell":
            row_id = self.tree.identify_row(event.y)
            col = self.tree.identify_column(event.x)
            item = self.tree.item(row_id)
            nombre = item['values'][0]

            if col == "#2":
                self.ver_info(nombre)
            elif col == "#3":
                self.ver_dieta(nombre)

    def train_lightgbm_model(self):
        """Train LightGBM model"""
        try:
            self.model, self.label_encoders, self.features = train_light()
            messagebox.showinfo("Éxito", "Modelo LightGBM entrenado correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar el modelo: {str(e)}")

    def guardar_informacion(self):
        """Save user information and recommend diet"""
        # Validate inputs
        if self.nombre.get().isdigit():
            messagebox.showwarning("Datos inválidos", "Por favor, ingresa un nombre válido.")
            return

        try:
            edad = int(self.edad.get())
            estatura = float(self.estatura.get())
            peso = float(self.peso.get())
        except ValueError:
            messagebox.showwarning("Datos inválidos", "Por favor, ingresa valores numéricos válidos para edad, estatura y peso.")
            return

        if not all([self.nombre.get(), self.edad.get(), self.estatura.get(), self.peso.get(),
                    self.sexo.get(), self.tension.get(), self.diabetes.get(),
                    self.fisico.get(), self.actividad.get()]):
            messagebox.showwarning("Campos incompletos", "Por favor, llena toda la información.")
            return

        # Prepare user data for diet recommendation
        nombre = self.nombre.get()
        sexo = 1 if self.sexo.get() == "M" else 0
        tension = 1 if self.tension.get() == "Yes" else 0
        diabetes = 1 if self.diabetes.get() == "Yes" else 0

        bmi = peso / ((estatura / 100) ** 2)

        actividad = {"Bajo": 0, "Moderado": 1, "Alto": 2}.get(self.actividad.get(), -1)
        fisico = 0 if self.fisico.get() == "Pérdida de Peso" else 1

        # Recommend diet if model is loaded
        dieta_recomendada = "No recomendada"
        if self.model and self.label_encoders:
            user_data = pd.DataFrame({
                'Sex_Male': [sexo],
                'Age': [edad],
                'Height': [estatura],
                'Weight': [peso],
                'Hypertension': [tension],
                'Diabetes_Yes': [diabetes],
                'BMI': [bmi],
                'Level': [actividad],
                'Fitness Goal': [fisico],
                'Fitness Type': [0]  # Default to 0 (Cardio)
            })

            diet_pred = self.model.predict(user_data)
            diet_pred_classes = [i.argmax() for i in diet_pred]
            dieta_recomendada = self.label_encoders['Diet'].inverse_transform(diet_pred_classes)[0]

        # Save user information
        with open("informacion_usuario.txt", "a") as file:
            file.write(f"Nombre: {nombre}\n")
            file.write(f"Edad: {edad}\n")
            file.write(f"Estatura: {estatura}\n")
            file.write(f"Peso: {peso}\n")
            file.write(f"Sexo: {sexo}\n")
            file.write(f"Hipertensión: {tension}\n")
            file.write(f"Diabetes: {diabetes}\n")
            file.write(f"Nivel de Actividad: {actividad}\n")
            file.write(f"Objetivo de acondicionamiento físico: {fisico}\n")
            file.write(f"Dieta Recomendada: {dieta_recomendada}\n")
            file.write(f"{'-' * 30}\n")

        # Insert into treeview with formatted diet
        # Insert into treeview with formatted diet
        self.tree.insert("", "end", values=(nombre, "👁️ Ver Info", "👁️ Ver Dieta"))


        self.limpiar_campos()

    def limpiar_campos(self):
        """Clear input fields"""
        for var in [self.nombre, self.edad, self.estatura, self.peso, self.sexo,
                    self.tension, self.diabetes, self.fisico, self.actividad]:
            var.set("")

    def ver_info(self, nombre):
        """View user information without showing diet"""
        try:
            with open("informacion_usuario.txt", "r") as file:
                content = file.read()
                usuarios = content.split("-" * 30)  # Separar cada usuario por los delimitadores
                for usuario in usuarios:
                    if f"Nombre: {nombre}" in usuario:
                        # Filtrar las líneas que no contienen información de la dieta
                        info_lines = []
                        in_dieta_section = False  # Flag to detect diet section
                        for line in usuario.split("\n"):
                            if "Dieta Recomendada:" in line:
                                in_dieta_section = True  # Start of diet section
                            elif in_dieta_section and not line.strip(): 
                                # End of diet section (empty line after diet)
                                in_dieta_section = False
                            elif not in_dieta_section:  # Add only non-diet lines
                                info_lines.append(line.strip())

                        # Ahora convertimos los valores numéricos en texto antes de mostrar la información
                        info = "\n".join(info_lines)
                        
                        # Convertir valores numéricos a texto
                        info = info.replace("Sexo: 1", "Sexo: Masculino").replace("Sexo: 0", "Sexo: Femenino")
                        info = info.replace("Hipertensión: 1", "Hipertensión: Sí").replace("Hipertensión: 0", "Hipertensión: No")
                        info = info.replace("Diabetes: 1", "Diabetes: Sí").replace("Diabetes: 0", "Diabetes: No")
                        info = info.replace("Actividad: 0", "Actividad: Bajo").replace("Actividad: 1", "Actividad: Moderado").replace("Actividad: 2", "Actividad: Alto")
                        info = info.replace("Objetivo de acondicionamiento físico: 0", "Objetivo de acondicionamiento físico: Pérdida de Peso").replace("Objetivo de acondicionamiento físico: 1", "Objetivo de acondicionamiento físico: Mantenimiento o Aumento de Peso")

                        # Mostrar la información
                        messagebox.showinfo(f"Información de {nombre}", info)
                        return
                messagebox.showwarning("No encontrado", f"No se encontró información de {nombre}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo recuperar la información: {str(e)}")

    def ver_dieta(self, nombre):
        """Ver recomendación de dieta"""
        try:
            with open("informacion_usuario.txt", "r") as file:
                content = file.read()
                usuarios = content.split("-" * 30)  # Separar cada usuario por los delimitadores
                for usuario in usuarios:
                    if f"Nombre: {nombre}" in usuario:
                        # Buscar la sección de dieta
                        dieta_start = "Dieta Recomendada:"
                        dieta_lines = []
                        dieta_found = False
                        
                        for line in usuario.split("\n"):
                            if dieta_start in line:
                                dieta_found = True  # Iniciar la sección de dieta
                            if dieta_found:
                                # Añadir cada línea de la dieta
                                dieta_lines.append(line.strip())
                            if dieta_found and line.strip() == "":
                                break  # Finalizar la búsqueda de la dieta cuando encontramos una línea vacía
                                
                        # Ahora formatear correctamente la dieta
                        dieta_formateada = "\n".join(dieta_lines).strip()

                        # Reemplazar categorías y ingredientes
                        dieta_formateada = dieta_formateada.replace("Vegetables:", "\n\nVegetales:")
                        dieta_formateada = dieta_formateada.replace("Protein Intake:", "\n\nProteínas:")
                        dieta_formateada = dieta_formateada.replace("Juice :", "\n\nJugos:")

                        # Asegurar que los ingredientes estén en líneas separadas
                        dieta_formateada = dieta_formateada.replace(';', ';\n')
                        dieta_formateada = dieta_formateada.replace(':', ':\n')

                        # Mostrar la dieta organizada
                        messagebox.showinfo(f"Dieta Recomendada para {nombre}", f"{dieta_formateada}")
                        return

                messagebox.showwarning("No encontrado", f"No se encontró dieta para {nombre}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo recuperar la dieta: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DietaApp(root)
    root.mainloop()
