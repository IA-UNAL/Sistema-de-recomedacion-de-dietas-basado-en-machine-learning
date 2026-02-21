# Sistema de Recomendación para Soluciones Nutricionales Personalizadas utilizando Técnicas de Aprendizaje Automático

Este proyecto implementa un sistema de recomendación basado en inteligencia artificial para generar soluciones nutricionales personalizadas, promoviendo estilos de vida saludables mediante dietas adaptadas a las características individuales de los usuarios.

## Características

- **Personalización**: Recomendaciones ajustadas según edad, peso, altura, estado de salud y metas nutricionales.
- **Modelos de aprendizaje automático**:
  - LightGBM (mejor desempeño)
  - Random Forest
  - XGBoost
  - Support Vector Classifier (SVC)
  - Multi-Layer Perceptron (MLP)
- **Interfaz interactiva**: Diseñada con Tkinter para generar recomendaciones en tiempo real.

## Metodología

- **Datos**: Conjunto de datos de 14,589 registros con variables como edad, peso, altura, estado de salud, etc.
- **Procesamiento**:
  - Codificación `one-hot` y `LabelEncoder` para variables categóricas.
  - Manejo de desbalance de clases mediante técnicas avanzadas.
- **Optimización**: Búsqueda de hiperparámetros con `GridSearch` y validación cruzada.

## Resultados

El modelo **LightGBM** obtuvo el mejor rendimiento en todas las métricas:

| Modelo            | Precisión | Recall   | F1-Score | AUC       | Media Geométrica |
|-------------------|-----------|----------|----------|-----------|------------------|
| LightGBM          | 0.972764  | 0.973269 | 0.972937 | 0.997934  | 0.984269         |
| XGBoost           | 0.971914  | 0.972241 | 0.972010 | 0.997808  | 0.983530         |
| Random Forest     | 0.950179  | 0.949966 | 0.949951 | 0.994462  | 0.969969         |
| SVC               | 0.919142  | 0.918095 | 0.917951 | 0.993561  | 0.949211         |
| MLP               | 0.912529  | 0.911583 | 0.911549 | 0.979207  | 0.946237         |

## Estructura del proyecto

### [EntrenarModelos](./EntrenarModelos)
Esta carpeta contiene todo el análisis necesario para evaluar y seleccionar el modelo con mejor rendimiento. Incluye:
- Preprocesamiento de datos.
- Pruebas de distintos algoritmos de aprendizaje automático.
- Código para la optimización de hiperparámetros.
- Generación de los archivos `.pkl` del modelo final.

### [Interfaz](./Interfaz)
En esta carpeta se encuentra la versión final de la interfaz desarrollada con Tkinter. Utiliza el modelo entrenado con LightGBM para realizar recomendaciones de dietas personalizadas a partir de los archivos `.pkl`. La interfaz está lista para ser utilizada en tiempo real por los usuarios.

##  Publicación asociada

Este proyecto está respaldado por la siguiente publicación científica:

Pérez-Baquero, R. S., Pérez-Baquero, E. B., Palomino-Ramírez, I., & Hoyos-Sánchez, J. P. (2024). *Recommendation System for Personalized Nutritional Solutions Using Machine Learning Techniques*. Revista Facultad de Ingeniería, 33(70), e18720. https://doi.org/10.19053/uptc.01211129.v33.n70.2024.18720

[![Publication](https://img.shields.io/badge/Publication-Revista%20Facultad%20de%20Ingeniería-green)](https://doi.org/10.19053/uptc.01211129.v33.n70.2024.18720)
