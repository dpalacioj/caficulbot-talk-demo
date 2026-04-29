# 🔧 Troubleshooting - Problemas Comunes y Soluciones

## Problema: "Los datos no aparecen después de insertarlos"

### Síntoma:
- Insertaste datos con `curl POST` o DBeaver
- La base de datos muestra los datos en SQLite
- Pero el chatbot responde "0 unidades" o no encuentra los datos

### Causa Raíz:
**Uvicorn está en modo `--reload`**, lo que significa que:
1. El servicio se reinicia automáticamente cuando detecta cambios en archivos `.py`
2. Al reiniciarse, SQLite puede perder transacciones no confirmadas
3. Si insertaste datos justo antes de un reinicio, se pierden

### Solución Rápida:
**Volver a insertar los datos después de que los servicios estén estables**

```bash
# 1. Verificar que los servicios estén corriendo
curl http://localhost:8000/health
curl http://localhost:8001
curl http://localhost:8002

# 2. Insertar datos de nuevo
curl -X POST http://localhost:8001/inventarioingresar/ \
  -H "Content-Type: application/json" \
  -d '{"producto": "fertilizante", "cantidad": 75}'

curl -X POST http://localhost:8001/inventarioingresar/ \
  -H "Content-Type: application/json" \
  -d '{"producto": "fertilizante NPK", "cantidad": 50}'

# 3. Verificar inmediatamente
sqlite3 app/databases/inventario/inventario.db "SELECT * FROM inventario;"
curl "http://localhost:8001/inventarioconsultar/?producto=fertilizante"
```

### Solución Permanente:
**Desactivar auto-reload en producción**

Edita `app/run-local.sh` y cambia:
```bash
# Antes:
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Después:
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

**Nota:** Sin `--reload`, necesitarás reiniciar los servicios manualmente cuando cambies código.

---

## Problema: "DBeaver no muestra los nuevos datos"

### Síntoma:
- Los datos existen en SQLite (verificado con `sqlite3` en terminal)
- DBeaver no los muestra

### Causa:
**DBeaver cachea los resultados** y no se actualiza automáticamente cuando los datos cambian externamente.

### Solución:
1. **Refrescar la tabla**: Clic derecho en la tabla → **"Refresh"** o presiona **F5**
2. **Reconectar**: Clic derecho en la conexión → **"Disconnect"** → Doble clic para reconectar
3. **Re-ejecutar query**: Ejecuta `SELECT * FROM inventario;` nuevamente

### Verificar Ruta de Conexión:
Asegúrate de que DBeaver esté conectado a la ruta correcta:
```
/Users/david.palacio/Documents/caficulbot-talk-demo/app/databases/inventario/inventario.db
/Users/david.palacio/Documents/caficulbot-talk-demo/app/databases/gastos/gastos.db
```

---

## Problema: "El modelo genera tools que no existen (enfermedad_consulta)"

### Síntoma:
- Preguntas sobre enfermedades generan JSON como:
  ```json
  {"tool": "enfermedad_consulta", "argumentos": "enfermedad=roya"}
  ```
- La respuesta es el JSON crudo, no una respuesta útil

### Causa:
**El modelo fue fine-tuneado con más herramientas de las que están implementadas**

Durante el fine-tuning se incluyeron tools como:
- `enfermedad_consulta`
- `plaga_consulta`
- `cosecha_registro`
- Etc.

Pero en `app/api.py` solo están implementados:
- `inventario_consulta`
- `gastos_consulta`

### Solución Temporal:
El modelo eventualmente aprenderá que esas herramientas no funcionan y responderá directamente. Por ahora, reformula la pregunta:

**En vez de:**
```
❌ "¿Cómo puedo tratar esta enfermedad?"
```

**Usa:**
```
✅ "¿Cómo se trata la roya del café?"
✅ "Dame información sobre la broca"
✅ "Explícame cómo combatir plagas"
```

### Solución Permanente:
Implementar las herramientas faltantes o modificar `app/api.py` para ignorar tools desconocidos:

```python
# En línea 291, cambiar:
else:
    final_answer = answer

# Por:
else:
    if tool_name:
        # Tool desconocido, intentar extraer respuesta antes del JSON
        print(f"[WARNING] Tool '{tool_name}' no implementado")
        clean_answer = answer.split('{"tool"')[0].strip()
        final_answer = clean_answer if clean_answer else answer
    else:
        final_answer = answer
```

---

## Problema: "El servicio tarda mucho en responder (30-60 segundos)"

### Síntoma:
- Las consultas tardan más de 30 segundos
- El navegador muestra "Loading..." por mucho tiempo

### Causa:
**El modelo está corriendo en CPU** en vez de GPU.

### Verificación:
```bash
tail -100 app/logs/main_api.log | grep "Cargando modelo"
```

Deberías ver:
```
[INFO] Cargando modelo en dispositivo: mps    (macOS)
[INFO] Cargando modelo en dispositivo: cuda   (Linux/Windows con GPU)
```

Si ves `cpu`, el modelo está corriendo lento.

### Solución:
**macOS (Apple Silicon):**
```bash
# Verificar MPS
python3 -c "import torch; print('MPS disponible:', torch.backends.mps.is_available())"

# Si muestra False, reinstalar PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

**Linux con GPU NVIDIA:**
```bash
# Verificar CUDA
nvidia-smi

# Reinstalar PyTorch con CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

---

## Problema: "Connection refused" al hacer requests

### Síntoma:
```
curl: (7) Failed to connect to localhost port 8000: Connection refused
```

### Causa:
Los servicios no están corriendo.

### Solución:
```bash
# 1. Verificar procesos
ps aux | grep uvicorn

# 2. Si no hay procesos, iniciar servicios
cd /Users/david.palacio/Documents/caficulbot-talk-demo
./app/run-local.sh

# 3. Esperar 30-60 segundos para que el modelo cargue

# 4. Verificar health
curl http://localhost:8000/health
```

Deberías ver:
```json
{"status": "healthy", "model_loaded": true}
```

---

## Problema: "El modelo no se encuentra en ../models"

### Síntoma:
```
ValueError: El modelo no se encuentra en ../models
```

### Causa:
El modelo no está descargado.

### Solución:
```bash
# Desde el directorio raíz del proyecto
python download.py

# O desde app/
python app/download.py
```

El modelo se descargará a `./models/` (puede tardar varios minutos dependiendo de la conexión).

---

## Problema: "ModuleNotFoundError" al iniciar servicios

### Síntoma:
```
ModuleNotFoundError: No module named 'transformers'
```

### Causa:
Dependencias no instaladas o entorno virtual no activado.

### Solución:
```bash
# 1. Activar entorno virtual
cd /Users/david.palacio/Documents/caficulbot-talk-demo/app
source venv/bin/activate

# 2. Reinstalar dependencias
pip install -r requirements.txt

# 3. Reiniciar servicios
./run-local.sh
```

---

## Comandos Útiles para Debugging

### Ver logs en tiempo real:
```bash
# API principal (modelo)
tail -f app/logs/main_api.log

# Inventario
tail -f app/logs/inventario.log

# Gastos
tail -f app/logs/gastos.log

# Todos a la vez
tail -f app/logs/*.log
```

### Verificar estado de servicios:
```bash
# Health checks
curl http://localhost:8000/health  # Main API
curl http://localhost:8001         # Inventario
curl http://localhost:8002         # Gastos
curl http://localhost:8003         # Cosecha
curl http://localhost:8004         # Ingresos
curl http://localhost:8501         # Streamlit

# Procesos corriendo
ps aux | grep uvicorn
ps aux | grep streamlit
```

### Ver contenido de bases de datos:
```bash
# Con formato bonito
sqlite3 app/databases/inventario/inventario.db << 'EOF'
.headers on
.mode column
SELECT * FROM inventario;
EOF

# Query directa
sqlite3 app/databases/inventario/inventario.db "SELECT * FROM inventario;"
sqlite3 app/databases/gastos/gastos.db "SELECT * FROM gastos ORDER BY año DESC, mes DESC;"
```

### Reiniciar servicios limpios:
```bash
# Matar todos los procesos
pkill -f uvicorn
pkill -f streamlit

# Limpiar logs
rm -f app/logs/*.log

# Reiniciar
./app/run-local.sh
```

### Probar function calling rápidamente:
```bash
# Inventario
curl -X POST http://localhost:8000/ask \
  -F "question=¿Cuánto fertilizante hay?" \
  -F "max_tokens=200"

# Gastos
curl -X POST http://localhost:8000/ask \
  -F "question=¿Cuánto gastamos en agosto 2025?" \
  -F "max_tokens=200"

# Conocimiento (sin function calling)
curl -X POST http://localhost:8000/ask \
  -F "question=¿Qué es la roya del café?" \
  -F "max_tokens=200"
```

---

## Estado Actual de las Bases de Datos

### Inventario:
```
1. abono: 30 unidades
2. fertilizante: 75 unidades
3. fertilizante NPK: 50 unidades
```

### Gastos:
```
1. abono (6/2025): $300,000
2. herramientas (6/2025): $70,000
3. abono (7/2025): $12,000
4. fertilizante (8/2025): $450,000
5. fertilizante NPK (8/2025): $280,000
6. fertilizante (9/2025): $320,000
```

Para repoblar datos si se pierden, ejecuta:
```bash
./populate_demo_data.sh  # Si lo creaste
# O copia los comandos curl del script demo_fertilizante.sh
```
