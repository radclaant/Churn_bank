import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

st.set_page_config(page_title="Predicción de Churn Bancario", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# --- FUNCIONES DE CARGA Y PROCESAMIENTO ---
@st.cache_data
def cargar_datos():
    df = pd.read_csv('dataset_churn_banco_2.csv', encoding='utf-8')
    df.columns = df.columns.str.replace('antiguedad_años', 'antiguedad_anos').str.replace('quejas_ultimo_año', 'quejas_ultimo_ano')
    df['canal_preferido'] = df['canal_preferido'].str.replace('Telefnico', 'Telefónico').str.replace('App Mvil', 'App Móvil')
    df['region'] = df['region'].str.replace('Amazona', 'Amazonía').str.replace('Pacfica', 'Pacífica').str.replace('Orinoqua', 'Orinoquía')
    
    # Feature Engineering para el dashboard
    df['ratio_saldo_salario'] = df['saldo_promedio_cop'] / (df['salario_mensual_cop'] + 1)
    df['score_actividad'] = df['num_transacciones_mes'] / (df['meses_sin_uso_app'] + 1)
    df['score_insatisfaccion'] = (df['meses_sin_uso_app'] * df['quejas_ultimo_ano']) / (df['nps_score'] + 1)
    df['contacto_abandonado'] = df['dias_desde_ultimo_contacto'] * df['meses_sin_uso_app']
    
    # Asignación manual rápida de clusters para el dashboard basado en los resultados del EDA
    # Para visualización si el modelo no estuviera disponible, pero usaremos K-means rápido
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    vars_cluster = ['edad', 'antiguedad_anos', 'salario_mensual_cop', 'saldo_promedio_cop', 
                    'num_transacciones_mes', 'nps_score', 'meses_sin_uso_app', 'quejas_ultimo_ano',
                    'score_insatisfaccion']
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[vars_cluster])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster_perfil'] = kmeans.fit_predict(X_cluster)
    cluster_nombres = {0: 'Riesgo Moderado', 1: 'Alto Riesgo (Insatisfechos)', 2: 'Bajo Riesgo (Fieles)'}
    df['Nombre_Perfil'] = df['cluster_perfil'].map(cluster_nombres)
    
    return df

@st.cache_resource
def cargar_modelo():
    if os.path.exists('modelo_churn_pipeline.pkl'):
        with open('modelo_churn_pipeline.pkl', 'rb') as f:
            return pickle.load(f)
    return None

df = cargar_datos()
modelo = cargar_modelo()

# --- CÁLCULO DE PREDICCIONES ---
# Si el modelo está disponible, predecimos la probabilidad de churn
if modelo is not None:
    X_pred = df.drop(columns=['cliente_id', 'churn', 'cluster_perfil', 'Nombre_Perfil'], errors='ignore')
    df['Probabilidad_Churn'] = modelo.predict_proba(X_pred)[:, 1]
    # Usamos el umbral óptimo encontrado en el EDA (0.21)
    df['Prediccion_Churn'] = (df['Probabilidad_Churn'] >= 0.21).astype(int)
else:
    # Si no, usamos el valor real para demostración
    df['Probabilidad_Churn'] = df['churn'].astype(float)
    df['Prediccion_Churn'] = df['churn']

df['Riesgo_Nivel'] = pd.cut(df['Probabilidad_Churn'], bins=[0, 0.21, 0.5, 1.0], labels=['Bajo', 'Medio', 'Alto'])

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
st.sidebar.title("Banco UniColombia")
st.sidebar.markdown("---")
vista_seleccionada = st.sidebar.radio("Seleccione una vista:", 
                                      ["📊 KPIs Generales", "🗺️ Mapa de Riesgo", "👥 Segmentación", "⚡ Lista de Acción", "💰 Simulador Financiero"])
st.sidebar.markdown("---")
st.sidebar.info("Panel de Control - Prevención de Abandono (Churn) a 90 días.")

# --- VISTA 1: KPIs ---
if vista_seleccionada == "📊 KPIs Generales":
    st.title("📊 Panel de Control: KPIs de Retención")
    st.markdown("Visión general del estado actual de la cartera y proyecciones financieras.")
    
    total_clientes = len(df)
    clientes_riesgo = df['Prediccion_Churn'].sum()
    tasa_riesgo = (clientes_riesgo / total_clientes) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Clientes Activos", f"{total_clientes:,}")
    col2.metric("Clientes en Riesgo", f"{clientes_riesgo:,}", f"{tasa_riesgo:.1f}% del portafolio", delta_color="inverse")
    
    margen_perdido = 2400000
    costo_campana = 150000
    efectividad = 0.30
    
    riesgo_financiero = clientes_riesgo * margen_perdido
    inversion_campana = clientes_riesgo * costo_campana
    retencion_esperada = int(clientes_riesgo * efectividad)
    ahorro_proyectado = retencion_esperada * margen_perdido - inversion_campana
    
    col3.metric("Riesgo Financiero Proyectado", f"${riesgo_financiero/1000000:,.1f} M")
    col4.metric("Ahorro Neto Estimado (ROI)", f"${ahorro_proyectado/1000000:,.1f} M", "Asumiendo 30% efectividad")
    
    st.markdown("---")
    st.subheader("Tendencia de Probabilidad de Churn")
    fig_hist = px.histogram(df, x="Probabilidad_Churn", nbins=20, color="Riesgo_Nivel", 
                            color_discrete_map={"Bajo": "#2ecc71", "Medio": "#f1c40f", "Alto": "#e74c3c"},
                            title="Distribución de la Probabilidad de Abandono")
    fig_hist.add_vline(x=0.21, line_dash="dash", line_color="white", annotation_text="Umbral Óptimo de Acción (21%)")
    st.plotly_chart(fig_hist, use_container_width=True)

# --- VISTA 2: MAPA DE RIESGO ---
elif vista_seleccionada == "🗺️ Mapa de Riesgo":
    st.title("🗺️ Mapa de Riesgo y Canales")
    st.markdown("Análisis del riesgo por segmento, canal y región.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        riesgo_segmento = df.groupby('segmento_cliente')['Probabilidad_Churn'].mean().reset_index()
        fig_seg = px.bar(riesgo_segmento, x='segmento_cliente', y='Probabilidad_Churn', color='Probabilidad_Churn',
                         color_continuous_scale='Reds', title="Riesgo Promedio por Segmento")
        st.plotly_chart(fig_seg, use_container_width=True)
        
        riesgo_region = df.groupby('region')['Probabilidad_Churn'].mean().reset_index()
        fig_reg = px.pie(riesgo_region, names='region', values='Probabilidad_Churn', hole=0.4,
                         color_discrete_sequence=px.colors.sequential.RdBu, title="Distribución de Riesgo por Región")
        st.plotly_chart(fig_reg, use_container_width=True)
        
    with col2:
        riesgo_canal = df.groupby('canal_preferido')['Probabilidad_Churn'].mean().reset_index()
        fig_canal = px.bar(riesgo_canal, x='canal_preferido', y='Probabilidad_Churn', color='Probabilidad_Churn',
                           color_continuous_scale='Oranges', title="Riesgo Promedio por Canal Preferido")
        st.plotly_chart(fig_canal, use_container_width=True)
        
        fig_box = px.box(df, x="canal_preferido", y="Probabilidad_Churn", color="canal_preferido",
                         title="Dispersión de Probabilidad por Canal")
        st.plotly_chart(fig_box, use_container_width=True)

# --- VISTA 3: SEGMENTACIÓN ---
elif vista_seleccionada == "👥 Segmentación":
    st.title("👥 Segmentación de Clientes (Perfiles de Riesgo)")
    st.markdown("Características promedio de los clusters identificados.")
    
    perfiles = df.groupby('Nombre_Perfil')[['nps_score', 'quejas_ultimo_ano', 'meses_sin_uso_app', 'score_insatisfaccion', 'Probabilidad_Churn']].mean().reset_index()
    
    fig_radar = go.Figure()
    for i, row in perfiles.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['nps_score']/10, row['quejas_ultimo_ano']/4, row['meses_sin_uso_app']/12, row['Probabilidad_Churn']],
            theta=['NPS (Normalizado)', 'Quejas (Norm)', 'Inactividad App (Norm)', 'Prob. Churn'],
            fill='toself',
            name=row['Nombre_Perfil']
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Comparativa Radial de Perfiles")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig_radar, use_container_width=True)
    with col2:
        st.markdown("### Detalles del Perfil")
        for i, row in perfiles.iterrows():
            st.info(f"**{row['Nombre_Perfil']}**\n- NPS Medio: {row['nps_score']:.1f}\n- Quejas Medias: {row['quejas_ultimo_ano']:.1f}\n- Riesgo Churn: {row['Probabilidad_Churn']*100:.1f}%")
            
    fig_scatter = px.scatter(df, x="score_insatisfaccion", y="Probabilidad_Churn", color="Nombre_Perfil", 
                             size="saldo_promedio_cop", hover_data=['cliente_id', 'canal_preferido'],
                             title="Relación entre Score de Insatisfacción y Probabilidad de Churn")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- VISTA 4: LISTA DE ACCIÓN ---
elif vista_seleccionada == "⚡ Lista de Acción":
    st.title("⚡ Lista de Acción Recomendada")
    st.markdown("Clientes priorizados para campaña de retención. Busque un cliente específico o filtre por segmentos de riesgo.")
    
    # Buscador por ID de Cliente
    search_id = st.text_input("🔍 Buscar por ID de Cliente (ej. C-101 o similar):").strip()
    
    # Filtros
    col1, col2 = st.columns(2)
    riesgo_filtro = col1.multiselect("Filtrar por Nivel de Riesgo:", ['Alto', 'Medio', 'Bajo'], default=['Alto', 'Medio'])
    canal_filtro = col2.multiselect("Canal Preferido:", df['canal_preferido'].unique(), default=df['canal_preferido'].unique())
    
    df_filtrado = df[(df['Riesgo_Nivel'].isin(riesgo_filtro)) & (df['canal_preferido'].isin(canal_filtro))].copy()
    
    # Asignación de acción recomendada
    def recomendar_accion(row):
        if row['quejas_ultimo_ano'] > 2:
            return "Llamada prioritaria: Resolver queja/PQR."
        elif row['meses_sin_uso_app'] > 6 and row['canal_preferido'] in ['Digital', 'App Móvil']:
            return "Enviar SMS/Email: Beneficios app y descuento."
        elif row['nps_score'] < 4:
            return "Contacto telefónico: Entrevista de satisfacción y oferta."
        else:
            return "Campaña genérica de retención."
            
    df_filtrado['Accion_Recomendada'] = df_filtrado.apply(recomendar_accion, axis=1)
    
    if search_id:
        # Filtrar por coincidencia en el ID
        df_filtrado = df_filtrado[df_filtrado['cliente_id'].astype(str).str.contains(search_id, case=False)]
    
    # Ordenar por riesgo
    df_filtrado = df_filtrado.sort_values(by='Probabilidad_Churn', ascending=False)
    
    # Si se selecciona o encuentra un único cliente, mostrar su ficha detallada
    if len(df_filtrado) == 1:
        cliente = df_filtrado.iloc[0]
        st.markdown("---")
        st.subheader(f"👤 Ficha de Detalle de Cliente: {cliente['cliente_id']}")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Probabilidad de Churn", f"{cliente['Probabilidad_Churn']*100:.1f}%")
            st.write(f"**Nivel de Riesgo:** {cliente['Riesgo_Nivel']}")
            st.write(f"**Perfil (Cluster):** {cliente['Nombre_Perfil']}")
            st.write(f"**Segmento Comercial:** {cliente['segmento_cliente']}")
            
        with c2:
            st.write(f"**Edad:** {cliente['edad']} años")
            st.write(f"**Antigüedad:** {cliente['antiguedad_anos']} años")
            st.write(f"**NPS Score (Último):** {cliente['nps_score']} / 10")
            st.write(f"**Quejas Último Año:** {cliente['quejas_ultimo_ano']}")
            
        with c3:
            st.write(f"**Meses sin uso de App:** {cliente['meses_sin_uso_app']}")
            st.write(f"**Días desde último contacto:** {cliente['dias_desde_ultimo_contacto']}")
            st.write(f"**Saldo Promedio:** ${cliente['saldo_promedio_cop']:,.0f} COP")
            st.write(f"**Salario Mensual:** ${cliente['salario_mensual_cop']:,.0f} COP")
            
        st.success(f"🎯 **Acción Comercial Recomendada:** {cliente['Accion_Recomendada']}")
        st.markdown("---")
        
    elif len(df_filtrado) == 0:
        st.warning("⚠️ No se encontraron clientes que coincidan con la búsqueda o filtros aplicados.")
        
    # Mostrar tabla
    st.dataframe(df_filtrado[['cliente_id', 'Probabilidad_Churn', 'Riesgo_Nivel', 'Nombre_Perfil', 'canal_preferido', 'nps_score', 'Accion_Recomendada']].style.background_gradient(subset=['Probabilidad_Churn'], cmap='Reds'), use_container_width=True)
    
    # Exportar a CSV
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Descargar Lista de Acción Filtrada (CSV)", data=csv, file_name='lista_accion_churn_filtrada.csv', mime='text/csv')

# --- VISTA 5: SIMULADOR FINANCIERO ---
elif vista_seleccionada == "💰 Simulador Financiero":
    st.title("💰 Simulador de Impacto Económico")
    st.markdown("Estime y simule el impacto financiero de las campañas de retención variando los costos del negocio y la efectividad.")
    
    # Parámetros del simulador en columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        costo_c = st.number_input("Costo de Campaña por Cliente (COP):", min_value=10000, max_value=5000000, value=150000, step=10000)
        margen_c = st.number_input("Margen Anual por Cliente Perdido (COP):", min_value=100000, max_value=20000000, value=2400000, step=100000)
    
    with col2:
        efectividad_c = st.slider("Efectividad de la Campaña (Tasa de Retención %):", min_value=5, max_value=100, value=30, step=5) / 100.0
        umbral_c = st.slider("Umbral de Probabilidad para Accionar (%):", min_value=1, max_value=99, value=21, step=1) / 100.0

    with col3:
        st.markdown("### ¿Cómo se calcula?")
        st.caption("**Falso Positivo (FP):** Se gasta campaña en cliente que no iba a hacer churn.")
        st.caption("**Falso Negativo (FN):** Se pierde el margen del cliente por no contactarlo.")
        st.caption("**Verdadero Positivo (TP):** Se contacta al cliente. Se logra retener el % de efectividad y se pierde el resto.")

    # Cálculos dinámicos
    if 'Probabilidad_Churn' in df:
        probs = df['Probabilidad_Churn'].values
    else:
        probs = df['churn'].values
        
    y_real = df['churn'].values
    y_pred = (probs >= umbral_c).astype(int)
    
    # Matriz
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_real, y_pred)
    if len(cm.ravel()) == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = sum((y_real == 0) & (y_pred == 0))
        fp = sum((y_real == 0) & (y_pred == 1))
        fn = sum((y_real == 1) & (y_pred == 0))
        tp = sum((y_real == 1) & (y_pred == 1))

    total_churners = sum(y_real)
    
    # Costo "No hacer nada"
    costo_nada = total_churners * margen_c
    
    # Costo "Contactar a todos"
    costo_todos = (len(df) * costo_c) + (total_churners * (1 - efectividad_c) * margen_c)
    
    # Costo Estrategia del Modelo
    costo_modelo = (fp * costo_c) + (fn * margen_c) + (tp * (costo_c + (1 - efectividad_c) * margen_c))
    
    ahorro_modelo = costo_nada - costo_modelo
    inversion_c = (tp + fp) * costo_c
    roi_modelo = (ahorro_modelo / inversion_c * 100) if inversion_c > 0 else 0
    
    # Mostrar Métricas de Impacto
    st.markdown("---")
    st.subheader("Resultados de la Simulación")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clientes a Contactar", f"{tp + fp} ({((tp + fp)/len(df)*100):.1f}% de la base)")
    m2.metric("Inversión en Campañas", f"${inversion_c:,.0f} COP")
    m3.metric("Ahorro Neto Proyectado", f"${ahorro_modelo:,.0f} COP", f"{((ahorro_modelo/costo_nada)*100):.1f}% vs No Hacer Nada")
    m4.metric("Retorno de la Inversión (ROI)", f"{roi_modelo:.1f}%")
    
    # Gráfico Comparativo
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=['No Hacer Nada', 'Contactar Todos', 'Estrategia del Modelo'],
        y=[costo_nada, costo_todos, costo_modelo],
        marker_color=['#e74c3c', '#f39c12', '#2ecc71'],
        text=[f"${costo_nada:,.0f}", f"${costo_todos:,.0f}", f"${costo_modelo:,.0f}"],
        textposition='auto'
    ))
    fig_comp.update_layout(title="Comparativa de Costo Total por Estrategia (COP)", yaxis_title="Costo Total (COP)", showlegend=False)
    
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Resumen de Casos en tabla
    st.subheader("Desglose de Decisiones de Campaña")
    d1, d2 = st.columns(2)
    with d1:
        st.write(f"**Verdaderos Positivos (TP) - Churners capturados:** {tp} (Se evita la pérdida de {int(tp * efectividad_c)} clientes)")
        st.write(f"**Falsos Positivos (FP) - Campañas enviadas en vano:** {fp} (Costo innecesario: ${fp * costo_c:,.0f} COP)")
    with d2:
        st.write(f"**Falsos Negativos (FN) - Churners no detectados:** {fn} (Costo de oportunidad: ${fn * margen_c:,.0f} COP)")
        st.write(f"**Verdaderos Negativos (TN) - Clientes sanos no contactados:** {tn} (Ahorro de campaña de ${tn * costo_c:,.0f} COP)")
