from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from typing import List

# Datos y etiquetas de categorías
data = [
    # Saludos
    {"category": "Saludo", "phrase": "hola"},
    {"category": "Saludo", "phrase": "¿cómo estás?"},
    {"category": "Saludo", "phrase": "buenos días"},
    {"category": "Saludo", "phrase": "buenas tardes"},
    {"category": "Saludo", "phrase": "buenas noches"},
    {"category": "Saludo", "phrase": "¿qué tal?"},
    
    # Despedidas
    {"category": "Despedida", "phrase": "adiós"},
    {"category": "Despedida", "phrase": "hasta luego"},
    {"category": "Despedida", "phrase": "nos vemos"},
    {"category": "Despedida", "phrase": "me despido"},
    {"category": "Despedida", "phrase": "chau"},

    # Información general
    {"category": "Información general", "phrase": "¿Qué servicios ofrecen?"},
    {"category": "Información general", "phrase": "Quiero saber más sobre ustedes"},
    {"category": "Información general", "phrase": "¿Qué productos tienen?"},
    {"category": "Información general", "phrase": "Háblame sobre la empresa"},

    # Contacto
    {"category": "Información de contacto", "phrase": "¿Cómo puedo contactarlos?"},
    {"category": "Información de contacto", "phrase": "¿Cuál es el número de teléfono?"},
    {"category": "Información de contacto", "phrase": "¿Tienen algún correo electrónico?"},
    {"category": "Información de contacto", "phrase": "¿Dónde están ubicados?"},
    {"category": "Información de contacto", "phrase": "¿Cómo puedo llegar a sus oficinas?"},

    # Horarios
    {"category": "Horario de atención", "phrase": "¿Cuál es el horario de atención?"},
    {"category": "Horario de atención", "phrase": "¿En qué días están abiertos?"},
    {"category": "Horario de atención", "phrase": "¿Están abiertos los fines de semana?"},
    {"category": "Horario de atención", "phrase": "¿A qué hora abren?"},
    {"category": "Horario de atención", "phrase": "¿A qué hora cierran?"},
    
    # Precios
    {"category": "Precios", "phrase": "¿Cuánto cuesta el producto?"},
    {"category": "Precios", "phrase": "¿Cuál es el precio del servicio?"},
    {"category": "Precios", "phrase": "¿Tienen algún descuento?"},
    {"category": "Precios", "phrase": "¿Cuánto cuesta la suscripción mensual?"},
    {"category": "Precios", "phrase": "¿Cuál es el costo de envío?"},

    # Devoluciones
    {"category": "Política de devolución", "phrase": "¿Cuál es la política de devoluciones?"},
    {"category": "Política de devolución", "phrase": "¿Puedo devolver el producto?"},
    {"category": "Política de devolución", "phrase": "¿Cuánto tiempo tengo para hacer una devolución?"},
    {"category": "Política de devolución", "phrase": "¿Necesito el recibo para una devolución?"},
    {"category": "Política de devolución", "phrase": "¿Es posible cambiar un producto?"},

    # Soporte técnico
    {"category": "Soporte técnico", "phrase": "Tengo un problema técnico"},
    {"category": "Soporte técnico", "phrase": "¿Cómo puedo solucionar un error?"},
    {"category": "Soporte técnico", "phrase": "¿Tienen soporte técnico?"},
    {"category": "Soporte técnico", "phrase": "Mi aplicación no funciona"},
    {"category": "Soporte técnico", "phrase": "¿Pueden ayudarme con un problema en mi cuenta?"},
]

# Extraer frases y categorías
phrases = [item["phrase"] for item in data]
categories = [item["category"] for item in data]

# Crear y entrenar el clasificador
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(phrases, categories)

# Aplicación FastAPI
app = FastAPI()

# Modelo para solicitudes
class MessageRequest(BaseModel):
    message: str

# Diccionario de respuestas mejorado
responses = {
    "Saludo": "¡Hola! ¿Cómo puedo ayudarte hoy?",
    "Despedida": "Gracias por contactarnos. ¡Hasta pronto!",
    "Información general": "Ofrecemos una variedad de servicios y productos. Por favor, visita nuestro sitio web para más información.",
    "Información de contacto": "Puedes contactarnos al 123-456-7890 o enviar un correo a contacto@ejemplo.com.",
    "Horario de atención": "Atendemos de lunes a viernes de 9:00 a 18:00 y los sábados hasta las 14:00.",
    "Precios": "Consulta precios y descuentos en nuestro catálogo en línea.",
    "Política de devolución": "Aceptamos devoluciones dentro de los 30 días con recibo original.",
    "Soporte técnico": "Por favor, describe tu problema técnico para que podamos ayudarte rápidamente.",
}

# Ruta principal del chatbot
@app.post("/chat")
async def chat(request: MessageRequest):
    user_message = request.message
    predicted_category = model.predict([user_message])[0]
    response_text = responses.get(predicted_category, "Lo siento, no entiendo tu pregunta. ¿Puedes reformularla?")
    return {"category": predicted_category, "response": response_text}

# Ruta para verificar el estado
@app.get("/")
async def root():
    return {"message": "Chatbot mejorado y listo para ayudarte."}
