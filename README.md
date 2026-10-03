# Analisis-Sentadilla
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import plotly.graph_objects as go

def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radianes = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angulo = np.abs(radianes*180.0/np.pi)
    return angulo if angulo <= 180.0 else 360 - angulo

st.title("Análisis Bioinstrumental: Sentadilla")
st.write("Sube un video de perfil para evaluar la técnica mediante el ángulo de flexión de rodilla y cadera.")

# Recibir datos desde un archivo (Requerimiento del instructivo)
archivo_video = st.file_uploader("Cargar video (.mp4, .mov)", type=["mp4", "mov"])

if archivo_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(archivo_video.read())
    cap = cv2.VideoCapture(tfile.name)
    
    mp_pose = mp.solutions.pose
    angulos_rodilla = []
    angulos_cadera = []
    
    st.info("Procesando los datos asociados al gesto seleccionado... por favor espera.")
    
    # Procesar los datos
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Extracción de coordenadas del lado derecho
                cadera = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                rodilla = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                tobillo = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                hombro = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                
                # Análisis de la variable cuantitativa
                ang_rodilla = calcular_angulo(cadera, rodilla, tobillo)
                ang_cadera = calcular_angulo(hombro, cadera, rodilla)
                
                angulos_rodilla.append(ang_rodilla)
                angulos_cadera.append(ang_cadera)

    cap.release()
    
    # Mostrar resultados mediante representación gráfica
    st.subheader("Resultados de la Evaluación")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=angulos_rodilla, mode='lines', name='Ángulo de Rodilla', line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=angulos_cadera, mode='lines', name='Ángulo de Cadera', line=dict(color='red')))
    fig.add_hline(y=90, line_dash="dot", annotation_text="Profundidad Óptima (90°)", line_color="green")
    
    fig.update_layout(title="Variación del Ángulo Articular durante el Gesto", xaxis_title="Fotogramas (Tiempo)", yaxis_title="Grados (°)")
    st.plotly_chart(fig)
    st.success("Análisis completado exitosamente.")
