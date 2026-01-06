#app/api.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from transformers import pipeline
import torch
import uvicorn
from typing import Optional, Dict, Any, Union, List
from PIL import Image
import io
import requests
import json
import os

app = FastAPI(title="Coffee Expert API", version="1.0.0")

pipe = None

INVENTORY_API_BASE_URL = "http://localhost:8001"
EXPENSES_API_BASE_URL = "http://localhost:8002"
PRODUCTION_API_BASE_URL = "http://localhost:8003"
INCOME_API_BASE_URL = "http://localhost:8004"


#For the MVP we will use only two tools: inventory and expenses
SYSTEM_PROMPT = """Eres un experto en café de Colombia con amplio conocimiento sobre cultivo, procesamiento, variedades, manejo de plagas y enfermedades.
Vas a responder preguntas sobre café de manera natural y directa, sin usar formato JSON, a menos que se indique lo contrario.

INSTRUCCIONES CRÍTICAS:
- Por defecto, SIEMPRE responde en lenguaje natural, NO en formato JSON
- SOLO usa herramientas en estos casos específicos:
  * Si preguntan "¿cuánto hay de X?" o "¿qué cantidad tenemos de X?" → usa inventario_consulta
  * Si preguntan "¿cuánto gastamos en mes/año?" → usa gastos_consulta
- Para TODAS las demás preguntas (saludos, enfermedades, plagas, consejos, análisis de imágenes, etc.) responde DIRECTAMENTE sin usar herramientas

Ejemplos de cuando NO usar herramientas:
- Cuando te saluden
- "¿Cómo tratar la roya?" → Responde directamente

Ejemplos de cuando SÍ usar herramientas:
- "¿Cuánto fertilizante tenemos?" → {"tool": "inventario_consulta", "argumentos": "producto=fertilizante"}
- "¿Cuánto gastamos en enero 2024?" → {"tool": "gastos_consulta", "argumentos": "mes=1,año=2024"}
"""

SYSTEM_PROMPT_IMAGE = """Eres un experto en café de Colombia con amplio conocimiento sobre cultivo, procesamiento, variedades, 
manejo de plagas y enfermedades, y todas las prácticas agrícolas relacionadas con el café colombiano.

Analiza la imagen proporcionada y responde de manera natural y directa en español. NO uses formato JSON ni herramientas.
Proporciona un análisis detallado de lo que observas en la imagen, identificando posibles problemas, enfermedades, 
estado de la planta o cualquier aspecto relevante relacionado con el café."""


@app.on_event("startup")
async def load_model():
    global pipe
    
    local_model_path = "./models"
    
    if not os.path.exists(local_model_path):
        raise ValueError(f"El modelo no se encuentra en {local_model_path}")
    
    pipe = pipeline(
        "image-text-to-text",
        model=local_model_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        truncation=True,
        local_files_only=True
    )

def consultar_inventario_api(producto: str) -> Dict[str, Any]:
    try:
        response = requests.get(f"{INVENTORY_API_BASE_URL}/inventarioconsultar/", params={"producto": producto})
        return response.json()
    except Exception as e:
        return {"error": f"Error al consultar inventario: {str(e)}"}


def consultar_gastos_api(mes: int, año: int) -> Dict[str, Any]:
    try:
        response = requests.get(f"{EXPENSES_API_BASE_URL}/gastosconsultar/", params={"mes": mes, "año": año})
        return response.json()
    except Exception as e:
        return {"error": f"Error al consultar gastos: {str(e)}"}


def parse_tool_call(response):
    try:
        response_json = json.loads(response.strip())
        tool_name = response_json.get("tool")
        
        argumentos_raw = response_json.get("argumentos")
        
        if argumentos_raw is None:
            return None, None
            
        if isinstance(argumentos_raw, str):
            argumentos = {}
            for arg in argumentos_raw.split(","):
                if "=" in arg:
                    key, value = arg.strip().split("=", 1)
                    argumentos[key.strip()] = value.strip()
                else:
                    argumentos["producto"] = arg.strip()
                    
        elif isinstance(argumentos_raw, dict):
            argumentos = argumentos_raw
        else:
            argumentos = {"producto": str(argumentos_raw)}
            
        return tool_name, argumentos
        
    except Exception as e:
        print(f"Error parsing tool call: {e}")
        print(f"Response was: {response}")
        return None, None


def extract_content_from_response(raw_response: Union[str, List, Dict]) -> str:
    """Extract text content from various response formats"""
    
    if isinstance(raw_response, str):
        return raw_response
    
    if isinstance(raw_response, list):
        for item in raw_response:
            if isinstance(item, str):
                return item
            elif isinstance(item, dict) and "text" in item:
                return item["text"]
        return str(raw_response)
    
    if isinstance(raw_response, dict):
        for key in ["text", "content", "response", "answer"]:
            if key in raw_response:
                return extract_content_from_response(raw_response[key])
        return str(raw_response)
    
    return str(raw_response)


def extract_response_content(answer: Union[str, List, Dict]) -> str:
    """Extrae el contenido de respuesta si viene en formato JSON o estructurado"""
    
    text_content = extract_content_from_response(answer)
    
    try:
        response_json = json.loads(text_content.strip())
        if isinstance(response_json, dict) and "respuesta" in response_json:
            return response_json["respuesta"]
        return text_content
    except (json.JSONDecodeError, ValueError, AttributeError):
        return text_content


@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    max_tokens: int = Form(200),
    image: Optional[UploadFile] = File(None)
):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Modelo aún cargando")

    try:
        is_image_request = image is not None
        
        if is_image_request:
            print(f"[DEBUG] Procesando imagen con pregunta: {question}")
            
            image_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            modified_question = f"Sobre esta imagen de café: {question}"
            
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_IMAGE}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": modified_question}]}
            ]
            
            output = pipe(
                text=messages, 
                images=pil_image, 
                max_new_tokens=max_tokens, 
                temperature=0.8,
                repetition_penalty=1.2,
                top_p=0.95, 
                use_cache=False,
                truncation=True
            )
            
            if output and len(output) > 0:
                generated_text = output[0].get("generated_text", [])
                if generated_text and len(generated_text) > 0:
                    last_message = generated_text[-1]
                    if isinstance(last_message, dict) and "content" in last_message:
                        raw_answer = last_message["content"]
                    else:
                        raw_answer = str(last_message)
                else:
                    raw_answer = "No pude generar una respuesta para la imagen."
            else:
                raw_answer = "No se generó respuesta del modelo."
            
            final_answer = extract_response_content(raw_answer)
            
            print(f"[DEBUG] Respuesta para imagen: {final_answer[:100]}...")
            
            return JSONResponse(content={
                "question": question,
                "answer": final_answer,
                "has_image": True,
                "tools_used": None
            })

        else:
            print(f"[DEBUG] Procesando texto: {question}")
            
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": question}]}
            ]
            
            output = pipe(
                text=messages, 
                max_new_tokens=max_tokens,
                temperature=0.7, 
                repetition_penalty=1.1, 
                top_p=0.95, 
                use_cache=False,
                truncation=True
            )
            
            if output and len(output) > 0:
                generated_text = output[0].get("generated_text", [])
                if generated_text and len(generated_text) > 0:
                    last_message = generated_text[-1]
                    if isinstance(last_message, dict) and "content" in last_message:
                        raw_answer = last_message["content"]
                    else:
                        raw_answer = str(last_message)
                else:
                    raw_answer = "No pude generar una respuesta."
            else:
                raw_answer = "No se generó respuesta del modelo."
            
            answer = extract_response_content(raw_answer)

            tool_name, argumentos = parse_tool_call(extract_content_from_response(raw_answer))
            tools_used = None
            final_answer = answer

            if tool_name == "inventario_consulta" and argumentos and "producto" in argumentos:
                resultado = consultar_inventario_api(argumentos["producto"])
                tools_used = {"inventario_consulta": resultado}
                if isinstance(resultado, (int, float)):
                    final_answer = f"Quedan disponibles: {resultado} unidades de {argumentos['producto']}."
                elif isinstance(resultado, dict):
                    if "error" in resultado:
                        final_answer = resultado["error"]
                    else:
                        cantidad = resultado.get("cantidad", resultado.get("unidades", resultado))
                        final_answer = f"Quedan disponibles: {cantidad} unidades de {argumentos['producto']}."
                else:
                    final_answer = f"Quedan disponibles: {resultado}"
                    

            elif tool_name == "gastos_consulta" and argumentos and "mes" in argumentos and "año" in argumentos:
                resultado = consultar_gastos_api(int(argumentos["mes"]), int(argumentos["año"]))
                tools_used = {"gastos_consulta": resultado}
                if isinstance(resultado, (int, float)):
                    final_answer = f"El gasto total en {argumentos['mes']}/{argumentos['año']} fue de: ${resultado}."
                elif isinstance(resultado, dict):
                    if "error" in resultado:
                        final_answer = resultado["error"]
                    else:
                        total_gasto = resultado.get("total", resultado.get("gasto", 0))
                        final_answer = f"El gasto total en {argumentos['mes']}/{argumentos['año']} fue de: ${total_gasto}."
                else:
                    final_answer = f"El gasto total en {argumentos['mes']}/{argumentos['año']} fue de: ${resultado}"
            else:
                final_answer = answer
            
            print(f"[DEBUG] Respuesta para texto: {final_answer[:100]}...")

            return JSONResponse(content={
                "question": question,
                "answer": final_answer,
                "has_image": False,
                "tools_used": tools_used
            })
            
    except Exception as e:
        print(f"[ERROR] Error procesando pregunta: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "question": question,
                "answer": f"Error al procesar la pregunta: {str(e)}",
                "has_image": is_image_request if 'is_image_request' in locals() else False,
                "tools_used": None,
                "error": True
            }
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy" if pipe else "loading", "model_loaded": pipe is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)