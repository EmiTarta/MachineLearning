# Definición de la clase CustomPreprocessor
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
import pickle
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesador personalizado para transformar el conjunto de datos. 
    Realiza mapeos, codificación de múltiples etiquetas y otras transformaciones necesarias.
    """
    def __init__(self, target_columns, multi_label_columns, edlevel_mapping, industry_mapping, frequency_mapping):
        # Constructor de la clase
        # Inicializa los atributos de la clase con los valores proporcionados
        self.target_columns = target_columns  # Lista de columnas a codificar usando target encoding
        self.multi_label_columns = multi_label_columns  # Lista de columnas a codificar usando MultiLabelBinarizer
        self.edlevel_mapping = edlevel_mapping  # Diccionario para mapear los niveles educativos a valores numéricos
        self.industry_mapping = industry_mapping  # Diccionario para mapear las industrias a valores numéricos
        self.frequency_mapping = frequency_mapping  # Diccionario para mapear las frecuencias a valores numéricos
        self.target_map = {}  # Diccionario para almacenar los mapeos de target encoding
    
    def fit(self, X, y=None):
        """
        Ajusta el preprocesador a los datos.
        
        Args:
            X (pd.DataFrame): DataFrame de entrada.
            y (pd.Series, optional): Serie con la variable objetivo. Por defecto es None.
        
        Returns:
            self: Instancia ajustada del preprocesador.
        """
        if y is not None:  # Solo calcula el target encoding si se proporciona la variable objetivo 'y'
            for col in self.target_columns:
                X_copy = X.copy() # Copia para no modificar el original
                X_copy[col] = X_copy[col].fillna('Unknown')  # Rellenar valores faltantes con 'Unknown'
                X_copy[col] = X_copy[col].apply(lambda x: x.split(',') if isinstance(x, str) else [x])  # Dividir strings por comas
                exploded_df = X_copy.copy()
                exploded_df['target'] = y  # Añadir la variable objetivo al DataFrame temporal
                exploded_df = exploded_df.explode(col)  # Aplanar listas a valores individuales
                exploded_df[col] = exploded_df[col].astype(str)  # Asegurar que no haya listas ni valores no hashables
                self.target_map[col] = exploded_df.groupby(col)['target'].mean().to_dict()  # Calcular la media de la variable objetivo para cada categoría y almacenarla en un diccionario
        return self


    def transform(self, X, y=None):
        """
        Transforma los datos de acuerdo a las reglas definidas.
        
        Args:
            X (pd.DataFrame): DataFrame de entrada.
            y (pd.Series, optional): No se usa en este caso. Por defecto es None.
        
        Returns:
            pd.DataFrame: DataFrame transformado.
        """
        # Procesar YearsCodePro y YearsCode
        for col in ['YearsCodePro', 'YearsCode']:
            moda = X[col].mode()[0] # Calcular la moda de la columna
            X[col] = X[col].fillna(moda).replace({'Less than 1 year': 0, 'More than 50 years': 50}).astype(int) # Imputar valores faltantes con la moda, reemplazar valores de texto y convertir a entero

        # Procesar columnas ordinales
        X['EdLevel'] = X['EdLevel'].map(self.edlevel_mapping).fillna(-1) # Mapear los niveles educativos a valores numéricos y rellenar faltantes con -1

        # Procesar Industry
        X['Industry_Category'] = X['Industry'].map(self.industry_mapping).fillna('Otros Servicios') # Mapear las industrias a categorías y rellenar faltantes con 'Otros Servicios'
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Crear un codificador OneHotEncoder
        encoded = encoder.fit_transform(X[['Industry_Category']])  # Ajustar y transformar la columna 'Industry_Category'
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Industry_Category']), index=X.index) # Crear un DataFrame con las columnas codificadas
        X = pd.concat([X, encoded_df], axis=1).drop(['Industry_Category', 'Industry'], axis=1) # Concatenar las columnas codificadas al DataFrame original y eliminar las columnas originales

        # Procesar columnas de frecuencia
        for col in ['Frequency_1', 'Frequency_2']:
            median_value = X[col].map(self.frequency_mapping).median() # Calcular la mediana de la columna mapeada
            X[col] = X[col].map(self.frequency_mapping).fillna(median_value) # Mapear las frecuencias a valores numéricos y rellenar faltantes con la mediana

        # Target Encoding para las columnas seleccionadas
        for col in self.target_columns:
            X[col] = X[col].fillna('Unknown')
            X[col] = X[col].apply(lambda x: x.split(',') if isinstance(x, str) else [x])
            X[col] = X[col].apply(lambda cats: [str(cat) for cat in cats])  # Convertir todos los valores a strings
            X[f'{col}_encoded'] = X[col].apply(
                lambda cats: sum(self.target_map[col].get(cat, 0) for cat in cats) / len(cats) if cats else 0
            )
            X.drop(col, axis=1, inplace=True)


        # Procesar columnas con múltiples valores (MultiLabelBinarizer)
        for col in self.multi_label_columns:
            X[col] = X[col].fillna('').str.split(';') # Rellenar valores faltantes con una cadena vacía y dividir strings por punto y coma
            mlb = MultiLabelBinarizer() # Crear un objeto MultiLabelBinarizer
            encoded_values = mlb.fit_transform(X[col]) # Ajustar y transformar la columna
            encoded_df = pd.DataFrame(encoded_values, columns=[f"{col}_{c}" for c in mlb.classes_], index=X.index)  # Crear un DataFrame con las columnas codificadas
            X = pd.concat([X, encoded_df], axis=1).drop(col, axis=1)

        # Procesar Employment
        if 'Employment' in X.columns:
            X['Employment'] = X['Employment'].replace('Retired', 'I prefer not to say')  # Reemplazar 'Retired' por 'I prefer not to say'
            X['is_full_time'] = X['Employment'].str.contains('Employed, full-time', na=False).astype(int)  # Crear columna indicando si es empleo a tiempo completo
            X['is_part_time'] = X['Employment'].str.contains('Employed, part-time', na=False).astype(int)  # Crear columna indicando si es empleo a tiempo parcial
            X['is_independent'] = X['Employment'].str.contains('Independent contractor, freelancer, or self-employed', na=False).astype(int)  # Crear columna indicando si es trabajador independiente
            X['num_jobs'] = X['Employment'].str.split(';').str.len().fillna(0).astype(int)  # Crear columna con el número de empleos
            X['is_other_employment'] = ((X['is_full_time'] == 0) & (X['is_part_time'] == 0) &
                                        (X['is_independent'] == 0)).astype(int)  # Crear columna indicando si no tiene ninguno de los empleos anteriores
            X.drop(columns=['Employment'], inplace=True)  # Eliminar la columna original

        # Procesar AISent
        if 'AISent' in X.columns:
            labels61 = {   # Definir el mapeo para AISent
                'Very favorable': 5, 
                'Favorable': 4, 
                'Indifferent': 3, 
                'Unfavorable': 2,
                'Very unfavorable': 1,
                'Unsure': 0
            }
            X['AISent'] = X['AISent'].fillna('Unsure').map(labels61).fillna(-1)
            
        return X
    def get_feature_names_out(self, input_features=None):
        """
        Simula las transformaciones y devuelve las columnas resultantes tras el preprocesamiento.
        """
        columnas_resultantes = []
        if 'YearsCodePro' in input_features:
            columnas_resultantes.append('YearsCodePro')
        if 'YearsCode' in input_features:
            columnas_resultantes.append('YearsCode')
        if 'EdLevel' in input_features:
            columnas_resultantes.append('EdLevel')
        if 'AISent' in input_features:
            columnas_resultantes.append('AISent')
        if 'Frequency_2' in input_features:
            columnas_resultantes.append('Frequency_2')
        if 'Frequency_1' in input_features:
            columnas_resultantes.append('Frequency_1')
        if 'Industry' in input_features:
            columnas_resultantes.extend([
                'Industry_Category_Industria y Energía',
                'Industry_Category_Otros Servicios',
                'Industry_Category_Salud y Educación',
                'Industry_Category_Servicios Financieros',
                'Industry_Category_Tecnología y Servicios Digitales'
            ])
        if 'LearnCodeOnline' in input_features:
            columnas_resultantes.append('LearnCodeOnline_encoded')
        if 'DevType' in input_features:
            columnas_resultantes.append('DevType_encoded')
        if 'LearnCode' in input_features:
            columnas_resultantes.append('LearnCode_encoded')
        if 'CodingActivities' in input_features:
            columnas_resultantes.append('CodingActivities_encoded')
        if 'DatabaseHaveWorkedWith' in input_features:
            columnas_resultantes.extend([
                'DatabaseHaveWorkedWith_', 'DatabaseHaveWorkedWith_Cassandra', 'DatabaseHaveWorkedWith_Clickhouse',
                'DatabaseHaveWorkedWith_Cloud Firestore', 'DatabaseHaveWorkedWith_Cockroachdb', 'DatabaseHaveWorkedWith_Cosmos DB', 
                'DatabaseHaveWorkedWith_Couch DB', 'DatabaseHaveWorkedWith_Couchbase', 
                'DatabaseHaveWorkedWith_Databricks SQL','DatabaseHaveWorkedWith_Datomic', 'DatabaseHaveWorkedWith_DuckDB', 'DatabaseHaveWorkedWith_Dynamodb', 
                'DatabaseHaveWorkedWith_Firebase Realtime Database', 'DatabaseHaveWorkedWith_Firebird', 'DatabaseHaveWorkedWith_H2',
                'DatabaseHaveWorkedWith_IBM DB2', 'DatabaseHaveWorkedWith_InfluxDB', 'DatabaseHaveWorkedWith_Microsoft Access', 
                'DatabaseHaveWorkedWith_Microsoft SQL Server', 'DatabaseHaveWorkedWith_MongoDB', 'DatabaseHaveWorkedWith_Neo4J', 
                'DatabaseHaveWorkedWith_Oracle', 'DatabaseHaveWorkedWith_Presto', 'DatabaseHaveWorkedWith_RavenDB', 
                'DatabaseHaveWorkedWith_SQLite', 'DatabaseHaveWorkedWith_Snowflake', 'DatabaseHaveWorkedWith_Solr','DatabaseHaveWorkedWith_Supabase',  
                'DatabaseHaveWorkedWith_BigQuery', 'DatabaseHaveWorkedWith_Elasticsearch',
                'DatabaseHaveWorkedWith_MariaDB', 'DatabaseHaveWorkedWith_MySQL',
                'DatabaseHaveWorkedWith_PostgreSQL', 'DatabaseHaveWorkedWith_Redis'
        ])
        if 'LanguageWantToWorkWith' in input_features:
            columnas_resultantes.extend([
                'LanguageWantToWorkWith_', 'LanguageWantToWorkWith_Ada', 'LanguageWantToWorkWith_Apex',  
                'LanguageWantToWorkWith_Assembly','LanguageWantToWorkWith_Bash/Shell (all shells)','LanguageWantToWorkWith_C',
                'LanguageWantToWorkWith_C#','LanguageWantToWorkWith_C++','LanguageWantToWorkWith_Clojure','LanguageWantToWorkWith_Crystal',
                'LanguageWantToWorkWith_Dart','LanguageWantToWorkWith_Delphi','LanguageWantToWorkWith_Elixir','LanguageWantToWorkWith_Erlang',
                'LanguageWantToWorkWith_F#','LanguageWantToWorkWith_Fortran','LanguageWantToWorkWith_GDScript','LanguageWantToWorkWith_Go',
                'LanguageWantToWorkWith_Groovy','LanguageWantToWorkWith_HTML/CSS','LanguageWantToWorkWith_Haskell','LanguageWantToWorkWith_Java',
                'LanguageWantToWorkWith_JavaScript','LanguageWantToWorkWith_Julia',
                'LanguageWantToWorkWith_Kotlin','LanguageWantToWorkWith_Lisp','LanguageWantToWorkWith_Lua','LanguageWantToWorkWith_MATLAB',
                'LanguageWantToWorkWith_MicroPython','LanguageWantToWorkWith_Nim','LanguageWantToWorkWith_OCaml','LanguageWantToWorkWith_Objective-C',
                'LanguageWantToWorkWith_PHP','LanguageWantToWorkWith_Perl','LanguageWantToWorkWith_PowerShell','LanguageWantToWorkWith_Prolog',
                'LanguageWantToWorkWith_Python',  'LanguageWantToWorkWith_SQL', 'LanguageWantToWorkWith_R','LanguageWantToWorkWith_Ruby','LanguageWantToWorkWith_Rust',
                'LanguageWantToWorkWith_SAS','LanguageWantToWorkWith_Scala','LanguageWantToWorkWith_Solidity','LanguageWantToWorkWith_Swift',
                'LanguageWantToWorkWith_TypeScript','LanguageWantToWorkWith_VBA','LanguageWantToWorkWith_Visual Basic (.Net)','LanguageWantToWorkWith_Zephyr',
                'LanguageWantToWorkWith_Zig'
            ])
        if 'LanguageHaveWorkedWith' in input_features:
            columnas_resultantes.extend([
                'LanguageHaveWorkedWith_Python', 'LanguageHaveWorkedWith_JavaScript',
                'LanguageHaveWorkedWith_SQL', 'LanguageHaveWorkedWith_TypeScript','LanguageHaveWorkedWith_','LanguageHaveWorkedWith_Ada','LanguageHaveWorkedWith_Apex','LanguageHaveWorkedWith_Assembly',
                'LanguageHaveWorkedWith_Bash/Shell (all shells)','LanguageHaveWorkedWith_C','LanguageHaveWorkedWith_C#','LanguageHaveWorkedWith_C++',
                'LanguageHaveWorkedWith_Clojure','LanguageHaveWorkedWith_Cobol','LanguageHaveWorkedWith_Crystal','LanguageHaveWorkedWith_Dart',
                'LanguageHaveWorkedWith_Delphi','LanguageHaveWorkedWith_Elixir','LanguageHaveWorkedWith_Erlang','LanguageHaveWorkedWith_F#',
                'LanguageHaveWorkedWith_Fortran','LanguageHaveWorkedWith_GDScript','LanguageHaveWorkedWith_Go','LanguageHaveWorkedWith_Groovy',
                'LanguageHaveWorkedWith_HTML/CSS','LanguageHaveWorkedWith_Haskell','LanguageHaveWorkedWith_Java','LanguageHaveWorkedWith_Julia',
                'LanguageHaveWorkedWith_Kotlin','LanguageHaveWorkedWith_Lisp','LanguageHaveWorkedWith_Lua','LanguageHaveWorkedWith_MATLAB',
                'LanguageHaveWorkedWith_MicroPython','LanguageHaveWorkedWith_Nim','LanguageHaveWorkedWith_OCaml','LanguageHaveWorkedWith_Objective-C',
                'LanguageHaveWorkedWith_PHP','LanguageHaveWorkedWith_Perl','LanguageHaveWorkedWith_PowerShell','LanguageHaveWorkedWith_Prolog',
                'LanguageHaveWorkedWith_R','LanguageHaveWorkedWith_Raku','LanguageHaveWorkedWith_Ruby','LanguageHaveWorkedWith_Rust',
                'LanguageHaveWorkedWith_SAS','LanguageHaveWorkedWith_Scala','LanguageHaveWorkedWith_Solidity','LanguageHaveWorkedWith_Swift',
                'LanguageHaveWorkedWith_VBA','LanguageHaveWorkedWith_Visual Basic (.Net)','LanguageHaveWorkedWith_Zig'])
        if 'ToolsTechHaveWorkedWith' in input_features:
            columnas_resultantes.extend([
                'ToolsTechHaveWorkedWith_','ToolsTechHaveWorkedWith_APT','ToolsTechHaveWorkedWith_Ansible','ToolsTechHaveWorkedWith_Ant',
                'ToolsTechHaveWorkedWith_Bun','ToolsTechHaveWorkedWith_CMake','ToolsTechHaveWorkedWith_Cargo','ToolsTechHaveWorkedWith_Catch2',
                'ToolsTechHaveWorkedWith_Chef','ToolsTechHaveWorkedWith_Chocolatey','ToolsTechHaveWorkedWith_Composer','ToolsTechHaveWorkedWith_Dagger',
                'ToolsTechHaveWorkedWith_Docker','ToolsTechHaveWorkedWith_GNU GCC','ToolsTechHaveWorkedWith_Godot','ToolsTechHaveWorkedWith_Google Test',
                'ToolsTechHaveWorkedWith_Gradle','ToolsTechHaveWorkedWith_Homebrew','ToolsTechHaveWorkedWith_Kubernetes','ToolsTechHaveWorkedWith_LLVM’s Clang',
                'ToolsTechHaveWorkedWith_MSBuild','ToolsTechHaveWorkedWith_MSVC','ToolsTechHaveWorkedWith_Make','ToolsTechHaveWorkedWith_Maven (build tool)',
                'ToolsTechHaveWorkedWith_Meson','ToolsTechHaveWorkedWith_Ninja','ToolsTechHaveWorkedWith_Nix','ToolsTechHaveWorkedWith_NuGet',
                'ToolsTechHaveWorkedWith_Pacman','ToolsTechHaveWorkedWith_Pip','ToolsTechHaveWorkedWith_Podman','ToolsTechHaveWorkedWith_Pulumi',
                'ToolsTechHaveWorkedWith_Puppet','ToolsTechHaveWorkedWith_QMake','ToolsTechHaveWorkedWith_SCons','ToolsTechHaveWorkedWith_Terraform',
                'ToolsTechHaveWorkedWith_Unity 3D','ToolsTechHaveWorkedWith_Unreal Engine','ToolsTechHaveWorkedWith_Visual Studio Solution','ToolsTechHaveWorkedWith_Vite',
                'ToolsTechHaveWorkedWith_Wasmer','ToolsTechHaveWorkedWith_Webpack','ToolsTechHaveWorkedWith_Yarn','ToolsTechHaveWorkedWith_bandit',
                'ToolsTechHaveWorkedWith_doctest','ToolsTechHaveWorkedWith_lest','ToolsTechHaveWorkedWith_npm','ToolsTechHaveWorkedWith_pnpm'])
        if 'Employment' in input_features:
            columnas_resultantes.extend([
                'is_full_time', 'is_part_time', 'is_independent', 'num_jobs', 'is_other_employment'])
        return columnas_resultantes