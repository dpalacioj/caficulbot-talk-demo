#app/web.py
import streamlit as st
import tempfile
import time
from audio_recorder_streamlit import audio_recorder
import requests
import io
import os
from faster_whisper import WhisperModel
from PIL import Image

st.set_page_config(
    page_title="CaficulBot",
    page_icon="🌱"
)

API_URL = "http://localhost:8000"

@st.cache_resource
def load_whisper_model():
    """Cargar modelo Whisper una sola vez"""
    return WhisperModel("small", device="cpu", compute_type="int8")

whisper_model = load_whisper_model()

def transcribe_audio(audio_bytes):
    """Transcribe audio usando Whisper local"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        segments, info = whisper_model.transcribe(
            tmp_file_path, 
            beam_size=5,
            language="es"
        )
        
        transcript = " ".join([segment.text.strip() for segment in segments])
        
        print(f"[DEBUG] Audio transcrito: '{transcript}'")
        
        os.unlink(tmp_file_path)
        
        if not transcript or transcript.isspace():
            return "No se pudo transcribir el audio. Por favor, intenta de nuevo."
        
        return transcript
    except Exception as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        return f"Error al transcribir: {str(e)}"

def query_api(question, image=None, max_tokens=200):
    """Consultar la API de FastAPI"""
    try:
        print(f"[DEBUG] Enviando pregunta a la API: '{question}'")
        
        data = {
            "question": question,
            "max_tokens": max_tokens
        }
        
        files = None
        if image is not None:
            if hasattr(image, 'read'):
                image.seek(0)
                image_bytes = image.read()
            else:
                image_bytes = image
            
            print(f"[DEBUG] Tamaño de imagen a enviar: {len(image_bytes)} bytes")
            
            files = {
                "image": ("image.jpg", io.BytesIO(image_bytes), "image/jpeg")
            }
        
        response = requests.post(
            f"{API_URL}/ask",
            data=data,
            files=files
        )
        
        print(f"[DEBUG] Respuesta status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answer = result["answer"]
            print(f"[DEBUG] Respuesta del modelo: '{answer[:100]}...'")
            print(f"[DEBUG] Imagen procesada: {result.get('has_image', False)}")
            return answer
        else:
            return f"Error en la API: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error al conectar con la API: {str(e)}"

def get_image_bytes(image_object):
    """Convierte de manera segura un objeto de imagen a bytes"""
    if image_object is None:
        return None
    
    if hasattr(image_object, 'read'):
        image_object.seek(0)
        return image_object.read()
    elif isinstance(image_object, bytes):
        return image_object
    else:
        return bytes(image_object)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if "pending_image" not in st.session_state:
    st.session_state.pending_image = None

if "pending_camera_image" not in st.session_state:
    st.session_state.pending_camera_image = None

if "pending_image_bytes" not in st.session_state:
    st.session_state.pending_image_bytes = None

if "pending_camera_image_bytes" not in st.session_state:
    st.session_state.pending_camera_image_bytes = None

st.title("🌱 CaficulBot")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat")
    
    message_container = st.container(height=400)
    with message_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                if message.get("type") == "image":
                    with st.chat_message("user"):
                        st.write(f"📷 Foto enviada con pregunta: {message.get('question', 'Análisis de imagen')}")
                        if isinstance(message["content"], bytes):
                            st.image(message["content"], width=200)
                else:
                    with st.chat_message("user"):
                        st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])
    
    st.markdown("### 💬 Enviar mensaje")
    
    if st.session_state.pending_image_bytes is not None:
        st.info("📷 Imagen cargada. Escribe tu pregunta sobre la imagen.")
        st.image(st.session_state.pending_image_bytes, width=150)
    
    if st.session_state.pending_camera_image_bytes is not None:
        st.info("📸 Foto tomada. Escribe tu pregunta sobre la imagen.")
        st.image(st.session_state.pending_camera_image_bytes, width=150)
    
    tab1, tab2 = st.tabs(["⌨️ Texto", "🎤 Voz"])
    
    with tab1:
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Escribe tu pregunta aquí:", placeholder="Ej: ¿Qué enfermedad tiene esta planta?")
            submit_button = st.form_submit_button("Enviar", use_container_width=True, type="primary")
            
            if submit_button and user_input:
                if st.session_state.pending_image_bytes is not None or st.session_state.pending_camera_image_bytes is not None:
                    image_bytes = st.session_state.pending_image_bytes or st.session_state.pending_camera_image_bytes
                    
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": image_bytes,
                        "type": "image",
                        "question": user_input
                    })
                    
                    with st.spinner("Analizando imagen..."):
                        response = query_api(user_input, image_bytes)
                    
                    st.session_state.pending_image = None
                    st.session_state.pending_camera_image = None
                    st.session_state.pending_image_bytes = None
                    st.session_state.pending_camera_image_bytes = None
                else:
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    
                    with st.spinner("Consultando..."):
                        response = query_api(user_input)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                st.rerun()
    
    with tab2:
        st.info("🎤 Presiona el botón del micrófono para grabar tu pregunta")
        
        audio_bytes = audio_recorder(
            text="",
            icon_size="3x",
            recording_color="#e74c3c",
            neutral_color="#6c757d",
            sample_rate=16000
        )
        
        if audio_bytes:
            print(f"[DEBUG] Tamaño del audio grabado: {len(audio_bytes)} bytes")
            
            st.audio(audio_bytes, format="audio/wav")
            
            if len(audio_bytes) < 1000:
                st.warning("El audio grabado parece estar vacío. Por favor, intenta grabar de nuevo.")
            else:
                with st.spinner("Transcribiendo audio..."):
                    transcribed_text = transcribe_audio(audio_bytes)
            
            if transcribed_text and not transcribed_text.startswith("Error"):
                st.success(f"📝 **Transcripción:** {transcribed_text}")
                
                if st.button("📤 Enviar pregunta", use_container_width=True, type="primary"):
                    if transcribed_text and not transcribed_text.isspace():
                        if st.session_state.pending_image_bytes is not None or st.session_state.pending_camera_image_bytes is not None:
                            image_bytes = st.session_state.pending_image_bytes or st.session_state.pending_camera_image_bytes
                            
                            st.session_state.messages.append({
                                "role": "user", 
                                "content": image_bytes,
                                "type": "image",
                                "question": transcribed_text
                            })
                            
                            with st.spinner("Analizando imagen..."):
                                response = query_api(transcribed_text, image_bytes)
                            
                            st.session_state.pending_image = None
                            st.session_state.pending_camera_image = None
                            st.session_state.pending_image_bytes = None
                            st.session_state.pending_camera_image_bytes = None
                        else:
                            st.session_state.messages.append({"role": "user", "content": transcribed_text})
                            
                            with st.spinner(f"Consultando sobre: '{transcribed_text}'"):
                                response = query_api(transcribed_text)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        st.rerun()
                    else:
                        st.error("La transcripción está vacía. Por favor, graba tu pregunta de nuevo.")
            else:
                st.error(transcribed_text)

with col2:
    if st.button("📸 Tomar Foto", use_container_width=True, type="primary"):
        st.session_state.camera_active = not st.session_state.camera_active
    
    if st.session_state.camera_active:
        st.info("Cámara activa - Toma la foto")
        
        picture = st.camera_input("Capturar imagen")
        
        if picture is not None:
            if st.button("✅ Usar esta foto", use_container_width=True, type="primary"):
                picture.seek(0)
                st.session_state.pending_camera_image_bytes = picture.read()
                st.session_state.pending_camera_image = picture
                st.session_state.camera_active = False
                st.rerun()
        
        if st.button("❌ Cancelar", use_container_width=True):
            st.session_state.camera_active = False
            st.rerun()
    else:
        st.info("Presiona 'Tomar Foto' para activar la cámara")
    
    st.markdown("---")
    st.markdown("**Alternativa: Subir imagen**")
    uploaded_file = st.file_uploader("Selecciona una imagen", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.button("📤 Cargar imagen", use_container_width=True, type="primary"):
            uploaded_file.seek(0)
            st.session_state.pending_image_bytes = uploaded_file.read()
            st.session_state.pending_image = uploaded_file
            st.rerun()
    
    if st.session_state.pending_image_bytes is not None or st.session_state.pending_camera_image_bytes is not None:
        if st.button("🗑️ Quitar imagen", use_container_width=True):
            st.session_state.pending_image = None
            st.session_state.pending_camera_image = None
            st.session_state.pending_image_bytes = None
            st.session_state.pending_camera_image_bytes = None
            st.rerun()