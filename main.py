import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import cv2
import numpy as np
import face_recognition
import uvicorn

app = FastAPI()

SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:sarth1234@localhost/face_recognition"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the User model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, unique=True, index=True)  # Ensure unique names
    imageEncoding = Column(String(length=6000))
    attendance_marked = Column(Boolean, default=False)  # New column added

# Create the tables in the database
Base.metadata.create_all(bind=engine)

@app.post("/upload_face/")
async def upload_face(name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        encode = face_recognition.face_encodings(img)[0]
        encoded_bytes = base64.b64encode(encode)
        encoded_str = encoded_bytes.decode('utf-8')  
    except IndexError as e:
        raise HTTPException(status_code=400, detail="No face found in the image.")
    
    # Save user info to the database
    db = SessionLocal()
    user = db.query(User).filter(User.name == name).first()
    if user:
        user.imageEncoding = encoded_str  # Update existing user
    else:
        user = User(name=name, imageEncoding=encoded_str)
        db.add(user)  # Add new user
    db.commit()
    db.refresh(user)
    return {"message": f"Face encoding for {name} added successfully."}

@app.post("/recognize/")
async def recognize_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Process the image and recognize
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    db = SessionLocal()
    users = db.query(User).all()
    encodeListKnown = [np.frombuffer(base64.b64decode(user.imageEncoding), dtype=np.float64) for user in users]
    classNames = [user.name for user in users]
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            user = db.query(User).filter(User.name == name).first()
            user.attendance_marked = True
            db.commit()
            return {"message": f"Attendance marked for {name}"}
    
    return {"message": "No face recognized"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
