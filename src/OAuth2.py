from flask import Flask, request, jsonify
import jwt
from datetime import datetime, timedelta

app = Flask(__name__)

# Clave secreta para firmar el token
SECRET_KEY = "supersecreto"
ALGORITHM = "HS256"

# Base de datos falsa de usuarios
users_db = {"usuario1": "123456", "admin": "adminpass"}

# Función para generar un token JWT
def create_access_token(username: str):
    expire = datetime.utcnow() + timedelta(minutes=30)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# Endpoint para autenticación (obtención de token)
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if username in users_db and users_db[username] == password:
        token = create_access_token(username)
        return jsonify({"access_token": token.decode("utf-8"), "token_type": "bearer"})

    return jsonify({"error": "Credenciales incorrectas"}), 401

# Middleware para verificar el token
def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return {"error": "Token expirado"}
    except jwt.InvalidTokenError:
        return {"error": "Token inválido"}

# Ruta protegida que requiere autenticación
@app.route("/profile", methods=["GET"])
def profile():
    token = request.headers.get("Authorization")
    
    if not token:
        return jsonify({"error": "Token requerido"}), 401

    token = token.split(" ")[1]  # Eliminar el prefijo "Bearer"
    user_data = verify_token(token)

    if "error" in user_data:
        return jsonify(user_data), 401

    return jsonify({"message": f"Bienvenido, {user_data['sub']}!"})

if __name__ == "__main__":
    app.run(debug=True)
