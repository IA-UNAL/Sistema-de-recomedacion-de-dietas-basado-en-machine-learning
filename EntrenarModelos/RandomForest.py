import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier #MODELO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (
    accuracy_score,  #Exactitud
    roc_auc_score,
    precision_score,  #Presiscion
    recall_score,  #Sensibilidad
    f1_score,  
    confusion_matrix,
    roc_curve, 
    auc
)

def find_best_params(X_train, y_train):
    # Definir el modelo base
    rf_model = RandomForestClassifier(bootstrap=True, random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200, 500, 1000],
    }
    # Configurar el GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring='precision_weighted',  
        cv=3,  # Cross-validation con 3 particiones
        verbose=2,
        n_jobs=-1  # 
    )

    grid_search.fit(X_train, y_train)
    print("\n📊 Mejores parámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"  - {param}: {value}")

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Grafica la matriz de confusión
    
    Parámetros:
    - y_true: Etiquetas verdaderas
    - y_pred: Etiquetas predichas
    - classes: Nombres de las clases
    """
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    
    plt.title('Matriz de Confusión - RandomForest')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.savefig('confusion_matrix_rf.png')


def plot_roc_curve(y_test, y_pred_proba, classes,model_name):
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    # Calcular ROC curve y ROC area para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr["weighted"], tpr["weighted"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
    roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    mean_tpr /= len(classes)
    fpr["weighted"] = all_fpr
    tpr["weighted"] = mean_tpr
    roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])
    
    # Devolver los valores de fpr, tpr y auc
    return fpr, tpr, roc_auc, model_name

def train_diet_recommendation_model(data_path='diet_data.xlsx',model_name='RandomForest'):
    """
    Entrena el modelo de recomendación de dietas utilizando Random Forest
    """
    data = pd.read_excel(data_path)

    data = pd.get_dummies(data, columns=['Diabetes', 'Sex'], prefix=['Diabetes', 'Sex'], drop_first=True)
    data['Hypertension'] = data['Hypertension'].map({'Yes': 1, 'No': 0})

    features = ['Sex_Male', 'Age', 'Height', 'Weight', 'Hypertension', 
                'Diabetes_Yes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type']
    target = 'Diet'

    # Codificación de etiquetas
    label_encoders = {}
    for col in ['Level', 'Fitness Goal', 'Fitness Type', target]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Dividir datos
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #find_best_params(X_train, y_train)  #ACTIVAR PARA GRIDSEARCH

    # Configuración del modelo Random Forest
    rf_model = RandomForestClassifier(n_estimators=1000,bootstrap=True,  random_state=42) #PARAMETRO

    # Entrenamiento del modelo
    rf_model.fit(X_train, y_train)

    # Predicciones
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)

    print("RandomForest")
    # Métricas de evaluación
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, rf_model.predict_proba(X_test), multi_class='ovr')
    g_mean = geometric_mean_score(y_test, y_pred, average='weighted')

    print(f'Accuracy: {accuracy:.6f}')
    print(f'AUC: {auc:.6f}')
    print(f'Precision: {precision:.6f}')
    print(f'Recall (Sensibilidad): {recall:.6f}')
    print(f'F1 Score: {f1:.6f}')
    print(f'G_mean: {g_mean:.6f}')

    diet_classes = label_encoders['Diet'].classes_
    
    # Renombrar las clases para el gráfico
    diet_classes_renamed = [f'Dieta{i+1}' for i in range(len(diet_classes))]
    print("\nContenido de las Dietas (Renombradas):")
    for i, diet in enumerate(diet_classes_renamed):
        print(f'{diet}: {diet_classes[i]}')

    # Guardar modelo y codificadores
    #joblib.dump(rf_model, 'diet_recommendation_model_random.pkl')
    #joblib.dump(label_encoders, 'label_encoders_random.pkl')

    plot_confusion_matrix(y_test, y_pred, diet_classes_renamed)

    print("\n📊 Matriz de confusión guardada como 'confusion_matrix_rf.png'")

    fpr, tpr, roc_auc, model_name = plot_roc_curve(y_test, y_pred_proba, diet_classes_renamed, model_name)
    
    return fpr, tpr, roc_auc, model_name , rf_model, label_encoders, features

def validate_input(prompt, valid_options=None, input_type=int):
    """
    Valida la entrada con restricciones
    """
    while True:
        try:
            user_input = input(prompt)
            if valid_options and user_input not in valid_options:
                raise ValueError("Opción inválida")
            return input_type(user_input)
        except ValueError as e:
            print(f"Error: {e}. Por favor, intente de nuevo.")

def validate_numeric_input(prompt, min_value, max_value):
    """
    Valida entradas numéricas con límites establecidos
    """
    while True:
        try:
            value = float(input(prompt))
            if min_value <= value <= max_value:
                return value
            else:
                print(f"Por favor, ingrese un valor entre {min_value} y {max_value}")
        except ValueError:
            print("Entrada inválida. Debe ser un número.")

def recomendar_dieta(model, label_encoders,features):
    """
    Función para recomendar dieta basada en input del usuario
    """
    try:
        # Validación de entradas
        sex = validate_input("Sexo (1 para Male, 0 para Female): ", ['0', '1'])
        age = validate_numeric_input("Edad: ", 0, 120)
        height = validate_numeric_input("Altura (cm): ", 50, 250)
        weight = validate_numeric_input("Peso (kg): ", 20, 300)
        
        hypertension = validate_input("¿Tienes hipertensión? (1 para Yes, 0 para No): ", ['0', '1'])
        diabetes = validate_input("¿Tienes diabetes? (1 para Yes, 0 para No): ", ['0', '1'])
        
        bmi = weight / ((height / 100) ** 2)
        
        level = validate_input("Nivel de actividad (0 para Bajo, 1 para Moderado, 2 para Alto): ", ['0', '1', '2'])
        fitness_goal = validate_input("Objetivo de acondicionamiento físico (0 para Perdida de Peso, 1 para Ganancia Muscular): ", ['0', '1'])
        fitness_type = validate_input("Tipo de acondicionamiento físico (0 para Cardio, 1 para Fuerza): ", ['0', '1'])

        user_data = {
            'Sex_Male': int(sex),
            'Age': age,
            'Height': height,
            'Weight': weight,
            'Hypertension': int(hypertension),
            'Diabetes_Yes': int(diabetes),
            'BMI': bmi,
            'Level': int(level),
            'Fitness Goal': int(fitness_goal),
            'Fitness Type': int(fitness_type)
        }

        user_df = pd.DataFrame([user_data])

        diet_pred = model.predict(user_df)
        predicted_diet_name = label_encoders['Diet'].inverse_transform(diet_pred)[0]

        print(f"\n🍽️ La dieta recomendada para usted es: {predicted_diet_name}")
        print("\nRecomendaciones adicionales:")
        print("1. Consulte siempre con un nutricionista antes de iniciar cualquier dieta.")
        print("2. Adapte la dieta a sus necesidades y condiciones personales.")
        print("3. Mantenga un equilibrio nutricional y realice actividad física regular.")

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

def main():
    try:
        # Intentar cargar modelo previamente entrenado
        model = joblib.load('diet_recommendation_model_random.pkl')
        label_encoders = joblib.load('label_encoders_random.pkl')
        features = ['Sex_Male', 'Age', 'Height', 'Weight', 'Hypertension', 
                    'Diabetes_Yes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type']
    except FileNotFoundError:
        print("No se encontró un modelo previamente entrenado. Entrenando nuevo modelo...")
        _, _, _, _ ,model, label_encoders, features = train_diet_recommendation_model()

    # Opción para el usuario
    print("\n--- Sistema de Recomendación de Dietas ---")
    print("1. Entrenar nuevo modelo")
    print("2. Recomendar dieta")
    
    opcion = validate_input("Seleccione una opción (1/2): ", ['1', '2'])
    
    if opcion == 1:
        _, _, _, _ ,model, label_encoders, features = train_diet_recommendation_model()
    
    # Ejecutar recomendación de dieta
    recomendar_dieta(model, label_encoders, features)

if __name__ == "__main__":
    main()