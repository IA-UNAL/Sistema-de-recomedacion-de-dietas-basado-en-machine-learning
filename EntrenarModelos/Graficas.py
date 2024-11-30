import matplotlib.pyplot as plt
import math
from RandomForest import train_diet_recommendation_model as model1  
from XGB import train_diet_recommendation_model as model2  
from LIGBM import train_diet_recommendation_model as model3
from SVC import train_diet_recommendation_model as model4 
from MLP import train_diet_recommendation_model as model5  

def truncar_a_3_decimales(valor):
    return math.floor(valor * 1000) / 1000

def plot_all_roc_curves():
    # Entrenar cada modelo y obtener sus curvas ROC
    fpr_rf, tpr_rf, roc_auc_rf, model_name_rf,_, _, _ = model1()
    fpr_xgb, tpr_xgb, roc_auc_xgb, model_name_xgb,_, _, _ = model2()
    fpr_lgb, tpr_lgb, roc_auc_lgb, model_name_lgb,_, _, _ = model3()
    fpr_svc, tpr_svc, roc_auc_svc, model_name_svc,_, _, _ = model4()
    fpr_mlp, tpr_mlp, roc_auc_mlp, model_name_mlp,_, _, _ = model5()

    # Graficar todas las curvas ROC
    plt.figure(figsize=(10, 8))
    # Aplicación del truncamiento y formateo en el plot
    plt.plot(fpr_rf["weighted"], tpr_rf["weighted"], 
            label=f'{model_name_rf} (AUC = {truncar_a_3_decimales(roc_auc_rf["weighted"]):.3f})', 
            linestyle='-', linewidth=3.5, color='blue')

    plt.plot(fpr_xgb["weighted"], tpr_xgb["weighted"], 
            label=f'{model_name_xgb} (AUC = {truncar_a_3_decimales(roc_auc_xgb["weighted"]):.3f})', 
            linestyle='--', linewidth=3.5, color='orange')

    plt.plot(fpr_svc["weighted"], tpr_svc["weighted"], 
            label=f'{model_name_svc} (AUC = {truncar_a_3_decimales(roc_auc_svc["weighted"]):.3f})', 
            linestyle=':', linewidth=3.5, color='green')

    plt.plot(fpr_lgb["weighted"], tpr_lgb["weighted"], 
            label=f'{model_name_lgb} (AUC = {truncar_a_3_decimales(roc_auc_lgb["weighted"]):.3f})', 
            linestyle=':', linewidth=3.5, color='red')

    plt.plot(fpr_mlp["weighted"], tpr_mlp["weighted"], 
            label=f'{model_name_mlp} (AUC = {truncar_a_3_decimales(roc_auc_mlp["weighted"]):.3f})', 
            linestyle='-.', linewidth=3.5, color='purple')

        # Agregar la línea diagonal de referencia
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Guardar la imagen
    plt.savefig('roc_curves_all_models.png')
    plt.close()
    
    print("\n📊 Todas las curvas ROC han sido guardadas como 'roc_curves_all_models.png'")

# Llamar a la función para graficar las curvas
plot_all_roc_curves()
