import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo Excel proporcionado
file_path = 'diet_data.xlsx'  # Asegúrate de que el archivo esté en el mismo directorio
data = pd.read_excel(file_path)

# Renombrar las dietas como 'Dieta 1', 'Dieta 2', ...
diet_mapping = {original: f"Dieta {i+1}" for i, original in enumerate(data['Diet'].unique())}

print("DIETAS: ", diet_mapping)
#diet_mapping ={1,2,3,4,5,6,7,8,9,10}
data['Diet'] = data['Diet'].map(diet_mapping)

# Calcular el porcentaje que representa cada clase en la columna 'Diet'
percentages = data['Diet'].value_counts(normalize=True) * 100

# Mostrar los porcentajes en consola
print("Porcentajes de las clases (Diet):")
print(percentages)

# Crear un orden específico para las etiquetas de las barras
custom_labels = [
    'Dieta 4', 'Dieta 1', 'Dieta 6', 'Dieta 9', 'Dieta 8', 
    'Dieta 10', 'Dieta 7', 'Dieta 5', 'Dieta 3', 'Dieta 2'
]

# Visualizar la distribución de las clases en la columna 'Diet' con porcentajes
plt.figure(figsize=(10, 6))
ax = sns.countplot(
    y=data['Diet'],  # Cambia a eje 'y' para clases largas
    order=data['Diet'].value_counts().index,  # Orden descendente basado en frecuencia
    palette='viridis'  # Paleta de colores
)

# Cambiar las etiquetas de las barras en el orden deseado
for i, container in enumerate(ax.containers):
    percentage = percentages[i]
    ax.bar_label(
        container, 
        labels=[f'{custom_labels[i]} ({percentage:.1f}%)' for _ in container], 
        label_type='edge'
    )

plt.title('Distribución de las Clases en la columna Diet (en %)', fontsize=16)
plt.xlabel('Frecuencia', fontsize=12)
plt.ylabel('Diet', fontsize=12)
plt.show()