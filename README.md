# 📝README: Proyecto de Machine Learning - Herramienta de Predicción Salarial para el Sector IT
👩‍💻 Autor: Maria Emilia Tartaglia 

📅Fecha: Enero 2025

### 📑 **Tabla de Contenidos**
- [📌 Presentación del Proyecto](#-presentación-del-proyecto)
  - [Introducción: Problema y Solución](#introducción)
  - [Dataset y Análisis Inicial](#dataset-y-análisis-inicial)
- [🛠️ Estructura del Proyecto](#️-estructura-del-proyecto)
  - [Análisis Exploratorio de Datos (EDA)](#análisis-exploratorio-de-datos-eda)
  - [Feature Engineering](#feature-engineering)
  - [Entrenamiento del Modelo](#entrenamiento-del-modelo)
  - [Pipeline Automático](#pipeline-automático)
  - [Despliegue en Streamlit](#despliegue-en-streamlit)
- [🔍 Conclusión y Propuesta de Valor](#-conclusión-y-propuesta-de-valor)
  - [Conclusión](#conclusión)
  - [Propuesta de Valor](#propuesta-de-valor)
  - [Siguientes Pasos](#siguientes-pasos)

 _____________________________________________________________________________________________
### 📌 **Presentación del Proyecto**
##### 🎯 *Introducción*

Este proyecto se desarrolló un modelo de predicción de salarios en el sector tecnológico en España. La herramienta se dirige a dos públicos principales:
- 👤 Particulares: Para estimar su salario potencial según su perfil y recibir orientación sobre qué habilidades, tecnologías o industrias podrían incrementar su compensación.
- 💼 Reclutadores: Para obtener estimaciones salariales personalizadas que ayuden a construir ofertas más competitivas y alineadas con el mercado.

*Objetivo del Proyecto*: El objetivo es crear una aplicación accesible y práctica para particulares (orientar sobre habilidades y experiencias que incrementen su valor en el mercado laboral) y reclutadores (proveer herramientas para hacer propuestas salariales alineadas con las tendencias del mercado, basadas en datos reales y actualizados). 

##### 📁 *Dataset y Análisis Inicial*

- **Fuente de Datos**: Encuestas anuales de Stack Overflow 2023 y 2024, de las muestras que tenían datos sobre la Compensación Total Anual, y que residen en Espaá (total de 1,934 registros y 397 variables). 

🔗 Se pueden conseguir las mismas en este [link de desacarga](https://survey.stackoverflow.co/) 

🛠️ *Estructura del Proyecto*
```python
├── data/                     # .txt con link de descarga
├── notebooks/                # Notebooks con análisis y modelado
    ├── PrimerNotebook.ipynb
    ├── 2. Sof_23.ipynb
    ├── 3. Sof_24.ipynb
    ├── 4. OutliersTarget.ipynb
    ├── 5. Feature_selection.ipynb
    ├── 6. VotingModelo.ipynb
    ├── 7. PipelineModelo.ipynb 
    ├──custom_preprocessor.py
├── Pickles/ 
    ├── data_2023.pickle # Exportado luego de la primera limpieza (PrimerNotebook)
    ├── data_2024.pickle # Exportado luego de la primera limpieza (PrimerNotebook)
    ├── data_23.pickle   # Exportado para hacer merge con el dataset final 
    ├── df_final.pikle
    ├── Scaler.pkl
    ├── MiModelo.pkl
    ├── Pipeline.pkl
├── app.py                    # Aplicación Streamlit
├── requirements.txt          # Dependencias del proyecto
```

- **Preparación del Dataset:**
  - *Limpieza de datos*: Eliminación de valores atípicos, nulos y ajustes en escalas (mensual a anual).
  - *Combinación de datasets 2023 y 2024* tras confirmar similitudes en la distribución de la variable objetivo (CompTotal).
  - *Codificación* de variables categóricas con métodos como OneHotEncoder y Target Encoding.
 
- **Feature Engineering**
*Selección de Variables*:
- Proceso iterativo basado en el rendimiento del modelo con Random Forest.
- Reducción de 400 a 200 variables relevantes, seleccionando finalmente un top 35 características para el modelo final.
*Variables Destacadas*:
- YearsCodePro (años de experiencia profesional).
- DevType (tipo de rol).
- - LanguageWantToWorkWith (lenguajes de programación deseados para trabajar en un futuro ).
ToolsTechHaveWorkedWith (herramientas utilizadas).

______________________________________________________________________________________________________________________________________________________________________
### 🔍 **Modelos Probados**: Se probaron varios algoritmos de regresión:
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- ⭐ Voting Regressor (modelo final)

🏆 Mejor Modelo: Voting Regressor

Combinación de `Random Forest`, `Gradient Boosting`, y `XGBoost`

##### Métricas de Evaluación
*Cross-Validation (promedio):*
- R²: 0.48
- MSE: 53530181.38
- MAE: 5,572 €
- RMSE: 7316.4323 €
- MAPE: 16.37 %
*Test Set:*
- R²: 0.54
- MSE: 46037445.11
- MAE: 5,199 €
- MAPE: 15.55 %

##### Pipeline Automatizado
- *Integración de preprocesamiento y modelo predictivo*: Pipeline completo desarrollado con joblib.
- *Preprocesamiento*: Incluye normalización, codificación y eliminación de valores faltantes.
- *Modelo Predictivo*: Voting Regressor entrenado con las 35 variables clave.

##### 🚀 Despliegue en Streamlit
Se desarrolló una aplicación web que permite a los usuarios:
- Ingresar información sobre su perfil profesional (experiencia, tecnologías, educación, etc.).
- Obtener una predicción salarial en tiempo real.
- Visualizar cómo sus características impactan en el rango salarial.
- La aplicación está diseñada para ser accesible y fácil de usar, facilitando la toma de decisiones informadas.

🔗 Para correr la app: ```python streamlit run app.py```
____________________________________________________________________________________________________________________________________________________________________
### ✅ **Conclusiones**
- Valor para Particulares: Orientación personalizada para optimizar su perfil profesional y maximizar sus ingresos.
- Valor para Reclutadores: Estimaciones salariales competitivas basadas en datos reales, libres de sesgos.

*Propuesta de Valor*: Combina datos sin sesgos, algoritmos avanzados y una interfaz amigable para resolver un problema real del sector IT.

### Siguientes Pasos
*Personalización*: Agregar filtros por región, industria, o características específicas.
*Ampliar el Dataset*: Incorporar datos de nuevas encuestas y fuentes externas.
*Integración*: Desarrollar una API para permitir que plataformas externas utilicen el modelo.
*Expansión*: Extender el modelo a sectores fuera del IT.
____________________________________________________________________________________________________________________________________________________________________
### 🛠️ **Tecnologías Utilizadas**

El desarrollo de este proyecto incluyó una variedad de herramientas y tecnologías modernas para garantizar un flujo de trabajo eficiente y un modelo predictivo robusto. Las principales tecnologías utilizadas fueron:

**🖥️ Lenguajes de Programación**
- Python 3.10.15: Utilizado para todo el desarrollo del proyecto, desde el análisis exploratorio de datos hasta el despliegue del modelo en Streamlit.

**📚 Bibliotecas de Machine Learning y Preprocesamiento**
- Pandas, NumPy
- Scikit-learn
- XGBoost (para utilizarlo se debe hacer ```python pip install xgboost```)
- Joblib (para utilizarlo se debe hacer ```python pip install jolib```)

3. **Visualización**
- Matplotlib y Seaborn: Creación de gráficos para análisis exploratorio y visualización de distribuciones.

4. **Despliegue Web**
- Streamlit: Desarrollo de una interfaz web interactiva para predicción de salarios en tiempo real. (Para utilizarlo se debe hacer ```python pip install streamlit```)

5. **Gestión de Datos*
- Pickle: Almacenamiento y carga de datos procesados, pipelines y modelos entrenados.
