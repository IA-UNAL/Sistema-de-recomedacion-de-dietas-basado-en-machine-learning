import tkinter as tk
from tkinter import messagebox, ttk
import os
from cabe  import train_random
from cabe1  import train_xgb
from cabe2  import train_light
from cabe3  import train_svc
from cabe4  import train_mlp

class DietaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recomendador de Dietas")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        self.model = None
        self.label_encoders = None
        self.features = None
        self.scaler = None

        # Inicializar las variables
        self.nombre = tk.StringVar()
        self.edad = tk.StringVar()
        self.estatura = tk.StringVar()
        self.peso = tk.StringVar()
        self.sexo = tk.StringVar()
        self.tension = tk.StringVar()
        self.diabetes = tk.StringVar()
        self.fisico = tk.StringVar()
        self.actividad = tk.StringVar()

        # Limpiar el archivo de información de usuarios
        open("informacion_usuario.txt", "w").close()

        # Centrar la ventana en la pantalla
        self.center_window_on_screen()

        # Crear el frame principal
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Crear los widgets
        self.create_widgets()

    def center_window_on_screen(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 800) // 2
        y = (screen_height - 600) // 2
        self.root.geometry(f"800x600+{x}+{y}")

    def create_widgets(self):
        # Título
        title_frame = tk.Frame(self.main_frame)
        title_frame.pack(fill="x", pady=(0, 20))
        tk.Label(title_frame, text="Recomendador de Dietas", font=("Arial", 16, "bold")).pack()

        # Frame para el formulario (3x3 grid)
        form_frame = tk.Frame(self.main_frame, width=600)
        form_frame.pack(pady=(0, 20))
        form_frame.pack_propagate(False)

        # Configurar el grid para que las columnas tengan el mismo ancho
        form_frame.grid_columnconfigure(0, weight=1)
        form_frame.grid_columnconfigure(1, weight=1)
        form_frame.grid_columnconfigure(2, weight=1)

        # Primera fila
        self.create_entry_grid(form_frame, "Nombre:", self.nombre, 0, 0)
        self.create_entry_grid(form_frame, "Edad (años):", self.edad, 0, 1)
        self.create_entry_grid(form_frame, "Estatura (cm):", self.estatura, 0, 2)

        # Segunda fila
        self.create_entry_grid(form_frame, "Peso (kg):", self.peso, 1, 0)
        self.create_sex_grid(form_frame, 1, 1)
        self.create_option_grid(form_frame, "Tienes hipertensión?:", self.tension, ["Yes", "No"], 1, 2)

        # Tercera fila
        self.create_option_grid(form_frame, "Tienes diabetes?:", self.diabetes, ["Yes", "No"], 2, 0)
        self.create_option_grid(form_frame, "Nivel de actividad:", self.actividad, ["Bajo", "Moderado", "Alto"], 2, 1)
        self.create_option_grid(form_frame, "Objetivo físico:", self.fisico, ["Pérdida de Peso", "Ganancia Muscular"], 2, 2)

        # Botones de entrenamiento
        training_frame = tk.Frame(self.main_frame)
        training_frame.pack(fill="x", pady=(0, 20))
        button_container = tk.Frame(training_frame)
        button_container.pack(expand=True)


        # Botón guardar
        button_frame = tk.Frame(self.main_frame)
        button_frame.pack(fill="x", pady=(0, 20))
        self.save_button = tk.Button(button_frame, text="Agregar", command=self.guardar_informacion, bd=5)
        self.save_button.pack()

        # Treeview
        self.create_treeview()

    def create_entry_grid(self, parent, label_text, variable, row, col):
        frame = tk.Frame(parent)
        frame.grid(row=row, column=col, padx=10, pady=5, sticky="nsew")
        tk.Label(frame, text=label_text, font=("Arial", 10)).pack()
        tk.Entry(frame, textvariable=variable, font=("Arial", 10), justify="center", width=20).pack()

    def create_sex_grid(self, parent, row, col):
        frame = tk.Frame(parent)
        frame.grid(row=row, column=col, padx=10, pady=5, sticky="nsew")
        tk.Label(frame, text="Sexo:", font=("Arial", 10)).pack()
        tk.Radiobutton(frame, text="Masculino", variable=self.sexo, value="M").pack()
        tk.Radiobutton(frame, text="Femenino", variable=self.sexo, value="F").pack()

    def create_option_grid(self, parent, label_text, variable, options, row, col):
        frame = tk.Frame(parent)
        frame.grid(row=row, column=col, padx=10, pady=5, sticky="nsew")
        tk.Label(frame, text=label_text, font=("Arial", 10)).pack()
        menu = tk.OptionMenu(frame, variable, *options)
        menu.config(cursor="hand2")
        menu.pack()

    def create_treeview(self):
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

    def train_model_1(self):
        self.entrenar_modelo("RandomForest")

    def train_model_2(self):
        self.entrenar_modelo("XGB")

    def train_model_3(self):
        self.entrenar_modelo("LightGBM")
        
    def train_model_4(self):
        self.entrenar_modelo("SVC")

    def train_model_5(self):
        self.entrenar_modelo("MLP")

    def entrenar_modelo(self, modelo_nombre):
        try:
            if os.path.exists('diet_recommendation_model.pkl') and os.path.exists('label_encoders.pkl'):
                messagebox.showinfo("Info", "Ya hay un modelo entrenado")
                return

            # Verifica si el archivo existe y tiene contenido
            if os.path.exists('modelo_actual.txt') and os.path.getsize('modelo_actual.txt') > 0:
                with open('modelo_actual.txt', 'r') as f:
                    modelo_actual = f.read().strip()
            else:
                # Si el archivo está vacío, se guarda el nombre del modelo que se pasó
                modelo_actual = modelo_nombre
                with open('modelo_actual.txt', 'w') as f:
                    f.write(modelo_actual)

            # Dependiendo del nombre del modelo, se entrena el modelo correspondiente
            if modelo_actual == 'RandomForest':
                self.model, self.label_encoders, self.features = train_random()
            elif modelo_actual == 'XGB':
                self.model, self.label_encoders, self.features = train_xgb()
            elif modelo_actual == 'LightGBM':
                self.model, self.label_encoders, self.features = train_light()
            elif modelo_actual == 'SVC':
                self.model, self.scaler, self.label_encoders, self.features = train_svc()
            elif modelo_actual == 'MLP':
                self.model, self.label_encoders, self.features = train_mlp()
            else:
                messagebox.showerror("Error", f"Modelo no reconocido: {modelo_actual}")
                return

            # Guarda el nombre del modelo entrenado en el archivo
            with open('modelo_actual.txt', 'w') as f:
                f.write(modelo_actual)

            messagebox.showinfo("Éxito", f"Modelo {modelo_actual} entrenado correctamente")

        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar el modelo: {str(e)}")

    def ver_info(self, nombre):
        try:
            with open("informacion_usuario.txt", "r") as file:
                usuarios = file.read().split("-----")
                for usuario in usuarios:
                    if f"Nombre: {nombre}" in usuario:
                        messagebox.showinfo(f"Información de {nombre}", usuario)
                        return
            messagebox.showwarning("No encontrado", f"No se encontró información de {nombre}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo recuperar la información: {str(e)}")

    def ver_dieta(self, nombre):
        import os
        if os.path.exists('diet_recommendation_model.pkl') and os.path.exists('label_encoders.pkl'):
            try:
                # Leer qué modelo está entrenado
                with open('modelo_actual.txt', 'r') as f:
                    modelo_actual = f.read()
                messagebox.showinfo("Dieta", f"Dieta1 para {nombre} usando modelo {modelo_actual}")
            except:
                messagebox.showinfo("Dieta", f"Dieta1 para {nombre}")
        else:
            messagebox.showerror("Error", "Entrena un modelo primero")

    def guardar_informacion(self):
    # Validar que el nombre no sea un número
        if self.nombre.get().isdigit():
            messagebox.showwarning("Datos inválidos", "Por favor, ingresa un nombre válido.")
            return

        # Validar que edad, estatura y peso sean números
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

        nombre = self.nombre.get()
        sexo = 1 if self.sexo.get() == "M" else 0
        tension = 1 if self.tension.get() == "Yes" else 0
        diabetes = 1 if self.diabetes.get() == "Yes" else 0
        actividad = {"Bajo": 0, "Moderado": 1, "Alto": 2}.get(self.actividad.get(), -1)
        fisico = 0 if self.fisico.get() == "Pérdida de Peso" else 1

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
            file.write(f"{'-' * 30}\n")

        self.tree.insert("", "end", values=(nombre, "👁️ Ver Info", "🔍 Dieta"))
        self.limpiar_campos()

    def limpiar_campos(self):
        for var in [self.nombre, self.edad, self.estatura, self.peso, self.sexo,
                    self.tension, self.diabetes, self.fisico, self.actividad]:
            var.set("")

    def clear_model(self):
        if os.path.exists('diet_recommendation_model.pkl') and os.path.exists('label_encoders.pkl'):
            os.remove('diet_recommendation_model.pkl')
            os.remove('label_encoders.pkl')
            if os.path.exists('modelo_actual.txt'):
                os.remove('modelo_actual.txt')
            messagebox.showinfo("Éxito", "Modelo eliminado correctamente")
        else:
            messagebox.showinfo("Info", "No hay modelo para eliminar.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DietaApp(root)
    root.mainloop()
