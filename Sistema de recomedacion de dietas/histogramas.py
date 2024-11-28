import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo Excel proporcionado
file_path = 'diet_data.xlsx'  # Asegúrate de que el archivo esté en el mismo directorio
data = pd.read_excel(file_path)

# Renombrar las dietas como 'Dieta 1', 'Dieta 2', ...
diet_mapping = {original: f"Dieta {i+1}" for i, original in enumerate(data['Diet'].unique())}
data['Diet'] = data['Diet'].map(diet_mapping)

# Calcular el porcentaje que representa cada clase en la columna 'Diet'
percentages = data['Diet'].value_counts(normalize=True) * 100

# Mostrar los porcentajes en consola
print("Porcentajes de las clases (Diet):")
print(percentages)

# Visualizar la distribución de las clases en la columna 'Diet' con porcentajes
plt.figure(figsize=(10, 6))
ax = sns.countplot(
    y=data['Diet'],  # Cambia a eje 'y' para clases largas
    order=data['Diet'].value_counts().index,  # Ordenar por frecuencia descendente
    palette='viridis'  # Paleta de colores
)

# Añadir los porcentajes como etiquetas en las barras
for container, percentage in zip(ax.containers, percentages):
    ax.bar_label(container, labels=[f'{percentage:.1f}%' for _ in container], label_type='edge')

plt.title('Distribución de las Clases en la columna Diet (en %)', fontsize=16)
plt.xlabel('Porcentaje', fontsize=12)
plt.ylabel('Diet', fontsize=12)
plt.show()

# Determinar si el dataset está desbalanceado
# Obtener la clase mayoritaria y minoritaria
max_percentage = percentages.max()
min_percentage = percentages.min()
imbalance_ratio = max_percentage / min_percentage

mayoritaria = percentages.idxmax()
minoritaria = percentages.idxmin()

# Mostrar resultados
print(f"Clase mayoritaria: {mayoritaria} - {max_percentage:.2f}%")
print(f"Clase minoritaria: {minoritaria} - {min_percentage:.2f}%")
print(f"Relación entre la clase mayoritaria y la minoritaria: {imbalance_ratio:.2f}")

if imbalance_ratio > 4:
    print("El dataset está desbalanceado.")
else:
    print("El dataset no está significativamente desbalanceado.")