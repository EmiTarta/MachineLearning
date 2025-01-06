import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os

# Configuración de la página
st.set_page_config(
    page_title="Predice tu salario!",
    page_icon=":rocket:",
    layout="wide",
)
# Estilo CSS para personalizar los colores
st.markdown("""
    <style>
    body {
        background-color: #e6f7ff; /* Fondo azul claro */
        color: #004080; /* Texto en azul oscuro */
    }
    h1, h2, h3, h4 {
        color: #003366; /* Títulos en azul oscuro */
    }
    .stButton>button {
        background-color: #004080; /* Botón azul oscuro */
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #0066cc; /* Botón azul claro al pasar el mouse */
    }
    </style>
""", unsafe_allow_html=True)


# Obtener el directorio base del archivo actual (app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Ruta absoluta para la imagen
image_path = os.path.join(current_dir, "image.jpeg")
# Usar la ruta absoluta en st.image
st.image(image_path, use_container_width=True, caption="Explorando el mundo IT")

# Obtener el directorio base del archivo actual
current_dir = os.path.dirname(os.path.abspath(__file__))
# Crear una ruta absoluta al archivo Pipeline.pkl
pipeline_path = os.path.join(current_dir, "Pipeline.pkl")
# Cargar el pipeline
with open(pipeline_path, 'rb') as f:
    pipeline = joblib.load(f)

# Título de la aplicación
st.title("Predicción de Salarios en el Sector IT")

# Crear el formulario dinámico
st.header("Por favor, responde las siguientes preguntas")
# (Tu formulario continúa aquí...)

# Preguntas del formulario
dev_type = st.selectbox(
    "¿Cuál es tu perfil?",
    ["Academic researcher", "Blockchain", "Cloud infrastructure engineer", "Data or business analyst", "Data engineer", "Data scientist or machine learning specialist", 
    "Database administrator", "Designer", "Developer Advocate", "Developer, AI", "Developer, back-end", "Developer, desktop or enterprise applications", 
    "Developer, embedded applications or devices", "Developer Experience", "Developer, front-end", "Developer, full-stack", "Developer, game or graphics", 
    "Developer, mobile", "Developer, QA or test", "DevOps specialist", "Educator", "Engineer, site reliability", "Engineering manager", "Hardware Engineer", 
    "Marketing or sales professional", "Product manager", "Project manager", "Research & Development role", "Scientist", "Senior Executive (C-Suite, VP, etc.)", 
    "Student", "System administrator", "Security professional", "Other"]
)
ed_level = st.selectbox(
    "¿Cuál es tu nivel educativo?",
    ['Master’s degree (M.A., M.S., M.Eng., MBA, etc.)',
       'Bachelor’s degree (B.A., B.S., B.Eng., etc.)',
       'Professional degree (JD, MD, Ph.D, Ed.D, etc.)',
       'Some college/university study without earning a degree',
       'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)',
       'Associate degree (A.A., A.S., etc.)', 'Something else',
       'Primary/elementary school']
)
years_code = st.slider("¿Cuántos años totales de experiencia en programación tienes?", min_value=0, max_value=50, value=5)
years_code_pro = st.slider("¿Hace cuántos años programas profesionalmente?", min_value=0, max_value=50, value=5)

learn_code = st.multiselect(
    "¿Cómo aprendiste a programar?",
    ["Books / Physical media", "Coding Bootcamp", "Colleague", "Friend or family member", "Online Courses or Certification", "On the job training", 
     "Other online resources (e.g., videos, blogs, forum, online community)", "School (i.e., University, College, etc)", "Other (please specify):"]
)
learn_code_online = st.multiselect(
    "¿Dónde aprendiste a programar en línea?",
    ["Stack Overflow", "Blogs", "Online community (including social media)", "Video-based online courses", "Online challenges (e.g., daily or weekly coding challenges)",
    "Coding sessions (live or recorded)", "Written-based online courses", "How-to videos", "Auditory material (e.g., podcasts)", "Technical documentation", "Certification videos", 
    "Books", "Games that teach programming", "AI chatbot or other tool", "Written tutorials", "Interactive tutorial", "Other"]
)
language_have_worked_with = st.multiselect(
    "¿Con qué lenguajes de programación has trabajado?",
    ['Python', 'JavaScript', 'SQL', 'TypeScript', 'Ada', 'Apex', 'Assembly', 'Bash/Shell (all shells)', 'C', 'C#', 'C++', 'Clojure', 'Cobol', 
     'Crystal', 'Dart', 'Delphi', 'Elixir', 'Erlang', 'F#', 'Fortran', 'GDScript', 'Go', 'Groovy', 'HTML/CSS',  'Haskell', 'Java', 'Julia', 
     'Kotlin', 'Lisp', 'Lua', 'MATLAB',  'MicroPython', 'Nim', 'OCaml', 'Objective-C', 'PHP', 'Perl', 'PowerShell', 'Prolog', 'R', 'Raku', 
     'Ruby', 'Rust', 'SAS', 'Scala', 'Solidity', 'Swift', 'VBA', 'Visual Basic (.Net)', 'Zig']
)
language_want_to_work_with = st.multiselect(
    "¿Con qué lenguajes te gustaría trabajar?",
    ['Python', 'JavaScript', 'SQL', 'TypeScript', 'Ada', 'Apex', 'Assembly', 'Bash/Shell (all shells)', 'C', 'C#', 'C++', 'Clojure', 'Cobol', 
     'Crystal', 'Dart', 'Delphi', 'Elixir', 'Erlang', 'F#', 'Fortran', 'GDScript', 'Go', 'Groovy', 'HTML/CSS',  'Haskell', 'Java', 'Julia', 
     'Kotlin', 'Lisp', 'Lua', 'MATLAB',  'MicroPython', 'Nim', 'OCaml', 'Objective-C', 'PHP', 'Perl', 'PowerShell', 'Prolog', 'R', 'Raku', 
     'Ruby', 'Rust', 'SAS', 'Scala', 'Solidity', 'Swift', 'VBA', 'Visual Basic (.Net)', 'Zig']
)
database_have_worked_with = st.multiselect(
    "¿Con qué bases de datos has trabajado?",
    ['Cassandra', 'Clickhouse','Cloud Firestore', 'Cockroachdb', 'Cosmos DB', 'Couch DB', 'Couchbase', 'Databricks SQL','Datomic', 
     'DuckDB', 'Dynamodb', 'Firebase Realtime Database', 'Firebird', 'H2','IBM DB2', 'InfluxDB', 'Microsoft Access', 
     'HMicrosoft SQL Server', 'MongoDB', 'Neo4J', 'Oracle', 'Presto', 'RavenDB', 'SQLite', 'Snowflake', 'Solr','Supabase', 
     'BigQuery', 'Elasticsearch', 'MariaDB', 'MySQL', 'PostgreSQL', 'Redis']
)
tools_tech_have_worked_with = st.multiselect(
    "¿Con qué herramientas/tecnologías has trabajado?",
    ['APT', 'Ansible', 'Ant', 'Bun', 'CMake', 'Cargo', 'Catch2', 'Chef', 'Chocolatey', 'Composer', 'Dagger', 'Docker', 'GNU GCC', 'Godot', 
    'Google Test', 'Gradle', 'Homebrew', 'Kubernetes', 'LLVM’s Clang', 'MSBuild', 'MSVC','Make','Maven (build tool)', 'Meson', 'Ninja', 'Nix',
    'NuGet', 'Pacman', 'Pip', 'Podman', 'Pulumi', 'Puppet', 'QMake', 'SCons', 'Terraform','Unity 3D', 'Unreal Engine','Visual Studio Solution',
    'Vite','Wasmer','Webpack','Yarn','bandit','doctest','lest','npm','pnpm']
)
coding_activities = st.multiselect(
    "¿En qué actividades de programación participas?",
    ["Hobby", "Freelance/contract work", "Contribute to open-source projects", "Bootstrapping a business", "School or academic work", 
     "Professional development or self-paced learning from online courses", "I don’t code outside of work", "Other"]
)
employment = st.selectbox(
    "¿Cuál es tu situación laboral?",
    ["Employed, full-time", "Employed, part-time", "Independent contractor, freelancer, or self-employed", "Not employed, but looking for work", 
     "Not employed, and not looking for work", "Student, full-time ", "Student, part-time", "Retired", "I prefer not to say"]
)
industry = st.selectbox(
    "¿En qué industria trabajas?",
    ['Insurance', 'Healthcare', 'Computer Systems Design and Services', 'Media & Advertising Services', 'Software Development', 'Fintech', 
     'Retail and Consumer Services', 'Internet, Telecomm or Information Services', 'Other:', 'Energy', 'Transportation, or Supply Chain', 
     'Banking/Financial Services', 'Manufacturing', 'Higher Education', 'Government']
)
frequency_2 = st.selectbox(
    "En tu trabajo como programador, ¿qué tan frecuente interactúas con personas fuera de tu equipo inmediato?",
    ['1-2 times a week', '3-5 times a week', 'Never',
       '6-10 times a week', '10+ times a week']
)
frequency_1 = st.selectbox(
    "En tu trabajo como programador, ¿qué tan frecuente necesitas ayuda de personas fuera de tu equipo inmediato?",
    ['1-2 times a week', '3-5 times a week', 'Never',
       '6-10 times a week', '10+ times a week']
)
ai_sent = st.selectbox(
    "¿Qué opinas sobre la IA?",
    ['Favorable', 'Unsure', 'Indifferent', 'Very favorable', 'Unfavorable', 'Very unfavorable']
)
# Crear el DataFrame estructurado
new_data = pd.DataFrame({
    'YearsCodePro': [years_code_pro],
    'LearnCodeOnline': [';'.join(learn_code_online)],
    'DevType': [dev_type],
    'LearnCode': [';'.join(learn_code)],
    'CodingActivities': [';'.join(coding_activities)],
    'DatabaseHaveWorkedWith': [';'.join(database_have_worked_with)],
    'YearsCode': [years_code],
    'LanguageWantToWorkWith': [';'.join(language_want_to_work_with)],
    'LanguageHaveWorkedWith': [';'.join(language_have_worked_with)],
    'EdLevel': [ed_level],
    'Employment': [employment],
    'ToolsTechHaveWorkedWith': [';'.join(tools_tech_have_worked_with)],
    'AISent': [ai_sent],
    'Industry': [industry],
    'Frequency_2': [frequency_2],
    'Frequency_1': [frequency_1]
})


# Al final, donde realizas la predicción:
if st.button("Predecir Salario"):
    # Preprocesar los datos
    preprocessed_data = pipeline.named_steps['preprocessor'].transform(new_data)
    column_names = pipeline.named_steps['preprocessor'].get_feature_names_out(input_features=new_data.columns)
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=column_names).fillna(0)

    # Realizar la predicción
    predicted_salary_log = pipeline.named_steps['model'].predict(preprocessed_df)
    predicted_salary = np.expm1(predicted_salary_log)

    # Mostrar el resultado
    st.subheader("Resultado de la Predicción")
    st.write(f"El salario anual bruto predicho es: **${predicted_salary[0]:,.2f}**")