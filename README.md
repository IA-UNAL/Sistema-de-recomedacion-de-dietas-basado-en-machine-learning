# Sistema de Recomendación de Dietas Personalizadas

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
