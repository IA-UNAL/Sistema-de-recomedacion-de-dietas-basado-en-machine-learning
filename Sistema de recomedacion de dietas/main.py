import matplotlib.pyplot as plt
from RandomForest import train_diet_recommendation_model as model1  
from xgboost import train_diet_recommendation_model as model2  
from lightgbm import train_diet_recommendation_model as model3
from SVC import train_diet_recommendation_model as model4 
from MLP import train_diet_recommendation_model as model5  


def plot_all_roc_curves():
    # Entrenar cada modelo y obtener sus curvas ROC
    fpr_rf, tpr_rf, roc_auc_rf, model_name_rf = model1()
    fpr_xgb, tpr_xgb, roc_auc_xgb, model_name_xgb = model2()
    fpr_lgb, tpr_lgb, roc_auc_lgb, model_name_lgb = model3()
    fpr_svc, tpr_svc, roc_auc_svc, model_name_svc = model4()
    fpr_mlp, tpr_mlp, roc_auc_mlp, model_name_mlp = model5()

    # Graficar todas las curvas ROC
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_rf["weighted"], tpr_rf["weighted"], label=f'{model_name_rf} (AUC = {roc_auc_rf["weighted"]:.6f})', linestyle=':',linewidth=3)
    plt.plot(fpr_xgb["weighted"], tpr_xgb["weighted"], label=f'{model_name_xgb} (AUC = {roc_auc_xgb["weighted"]:.6f})', linestyle=':',linewidth=3)
    plt.plot(fpr_svc["weighted"], tpr_svc["weighted"], label=f'{model_name_svc} (AUC = {roc_auc_svc["weighted"]:.6f})', linestyle=':',linewidth=3)
    plt.plot(fpr_lgb["weighted"], tpr_lgb["weighted"], label=f'{model_name_lgb} (AUC = {roc_auc_lgb["weighted"]:.6f})', linestyle=':',linewidth=3)
    plt.plot(fpr_mlp["weighted"], tpr_mlp["weighted"], label=f'{model_name_mlp} (AUC = {roc_auc_mlp["weighted"]:.6f})', linestyle=':',linewidth=3)
    # Agregar la línea diagonal de referencia
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC de Modelos')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Guardar la imagen
    plt.savefig('roc_curves_all_models.png')
    plt.close()
    
    print("\n📊 Todas las curvas ROC han sido guardadas como 'roc_curves_all_models.png'")

# Llamar a la función para graficar las curvas
plot_all_roc_curves()
