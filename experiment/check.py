import jwt
from datetime import datetime, timedelta

print("JWT module loaded from:", jwt.__file__)

SECRET_KEY = "testsecret"
ALGORITHM = "HS256"

payload = {
    "email": "test@example.com",
    "exp": datetime.utcnow() + timedelta(minutes=5)
}

token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
print("JWT:", token)

decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
print("Decoded:", decoded)


