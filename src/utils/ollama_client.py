import base64
import json
from openai import OpenAI

# Conexión local a Ollama
client = OpenAI(
    base_url="http://192.168.0.21:11434/v1/",
    api_key="ollama"
)

def analyze_image_with_ollama(image_path: str, model: str, prompt: str, retry_non_json: bool = True):
    """
    Envía una imagen al modelo (LLaVA o BakLLaVA) para descripción y clasificación.
    Retorna un dict con las claves esperadas o fallback básico.
    """

    # Codificar la imagen en base64
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        return {"error": f"No se pudo leer la imagen: {e}"}

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
                    ],
                }
            ],
            max_tokens=500,
        )
        print(response)
        raw_output = response.choices[0].message.content.strip()


        try:
            return json.loads(raw_output)
        except Exception:
            if not retry_non_json:
                raise

            # Intentar rescatar el bloque JSON con regex
            import re
            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass

            # Fallback genérico
            return {
                "description": raw_output[:200],
                "category": "Sin clasificar",
                "subcategory": None,
                "tags": []
            }

    except Exception as e:
        return {"error": str(e)}
