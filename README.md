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

- ## Resultados

El modelo **LightGBM** obtuvo el mejor rendimiento en todas las métricas:

| Modelo            | Precisión | Recall   | F1-Score | AUC       |
|-------------------|-----------|----------|----------|-----------|
| LightGBM         | 0.972764     | 0.973269    | 0.972937   | 0.997934     |
| XGBoost          | 0.971914    | 0.972    | 0.972241  | 0.972010    |
| Random Forest    | 0.950179     | 0.949966  | 0.949951   | 0.994462     |
| SVC              | 0.919142    | 0.918095   | 0.917951   | 0.993561    |
| MLP              | 0.912529    | 0.911583   | 0.911549   | 0.979207    |
