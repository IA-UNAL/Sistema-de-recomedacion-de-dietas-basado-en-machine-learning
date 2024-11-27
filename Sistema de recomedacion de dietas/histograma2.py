import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo Excel proporcionado
file_path = 'diet_data.xlsx'  # Ruta del archivo
data = pd.read_excel(file_path)

# Visualizar las primeras filas del archivo para validar los datos
print(data.head())

# Visualizar la distribución de las clases en la columna 'Diet'
plt.figure(figsize=(10, 6))
sns.countplot(
    y=data['Diet'],  # Cambia a eje 'y' para clases largas
    order=data['Diet'].value_counts().index,  # Ordenar por frecuencia descendente
    palette='viridis'  # Paleta de colores
)
plt.title('Distribución de las Clases en la columna Diet', fontsize=16)
plt.xlabel('Frecuencia', fontsize=12)
plt.ylabel('Diet', fontsize=12)
plt.show()