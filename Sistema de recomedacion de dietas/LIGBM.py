import pandas as pd
import joblib
import lightgbm as lgb
from  lightgbm  import LGBMClassifier #MODELO
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import label_binarize
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

def find_best_params(X_train, y_train,y):
    lgb_model = LGBMClassifier(    objective='multiclass',
    num_class=len(y.unique()), 
    boosting_type='gbdt',
    metric='multi_logloss',
    verbose=-1,
    )

    # Definir el rango de parámetros a explorar
    param_grid = {
        'n_estimators': [100, 200, 300, 400,500, 1000],
        "learning_rate":[0.01, 0.05, 0.1],
        "num_leaves":[32, 64, 128, 256],
    }

    # Configurar el GridSearchCV
    grid_search = GridSearchCV(
        estimator=lgb_model,
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
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    
    plt.title('Matriz de Confusión - lightgbm')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


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
    
    return fpr, tpr, roc_auc, model_name


def train_diet_recommendation_model(data_path='diet_data.xlsx',model_name='LightGBM'):
    """
    Entrena el modelo de recomendación de dietas utilizando LightGBM
    """
    data = pd.read_excel(data_path)

    data = pd.get_dummies(data, columns=['Diabetes', 'Sex'], prefix=['Diabetes', 'Sex'], drop_first=True)
    data['Hypertension'] = data['Hypertension'].map({'Yes': 1, 'No': 0})

    features = ['Sex_Male', 'Age', 'Height', 'Weight', 'Hypertension', 
                'Diabetes_Yes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type']
    target = 'Diet'

    label_encoders = {}
    for col in ['Level', 'Fitness Goal', 'Fitness Type', target]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #find_best_params(X_train,y_train,y) #ACTIVAR PARA GRIDSEARCH

    # Configuración de LightGBM
    params = {
        'objective': 'multiclass',
        'num_classes': len(y.unique()),
        'boosting_type': 'gbdt',
        'metric': 'multi_logloss',
        'learning_rate': 0.05 ,
        'num_leaves': 128,
        'verbose': -1
    }

    # Crear datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Entrenamiento del modelo
    lgb_model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )

    # Predicciones
    y_pred = lgb_model.predict(X_test)
    y_pred_classes = [i.argmax() for i in y_pred]

    print("LighGBM")
    # Métricas de evaluación
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    g_mean = geometric_mean_score(y_test, y_pred_classes, average='weighted') 


    print(f'Accuracy: {accuracy:.6f}')
    print(f'AUC: {auc:.6f}')
    print(f'Precision: {precision:.6f}')
    print(f'Recall (Sensibilidad): {recall:.6f}')
    print(f'F1 Score: {f1:.6f}')
    print(f'G_mean: {g_mean:.6f}')

    diet_classes = label_encoders['Diet'].classes_

    diet_classes_renamed = [f'Dieta{i+1}' for i in range(len(diet_classes))]

    print("\nContenido de las Dietas (Renombradas):")
    for i, diet in enumerate(diet_classes_renamed):
        print(f'{diet}: {diet_classes[i]}')

    fpr, tpr, roc_auc, model_name = plot_roc_curve(y_test, y_pred, diet_classes_renamed, model_name)
    
    return fpr, tpr, roc_auc, model_name , lgb_model, label_encoders, features

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
        diet_pred_classes = [i.argmax() for i in diet_pred]
        predicted_diet_name = label_encoders['Diet'].inverse_transform(diet_pred_classes )[0]
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
        model = joblib.load('diet_recommendation_model.pkl')
        label_encoders = joblib.load('label_encoders_light.pkl')
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
