import os
import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types
import firebase_admin
from firebase_admin import credentials, db

def main():
    print("Initializing IRIS")
    
    # 1. Load configuration safely from .env file
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("CRITICAL ERROR: GEMINI_API_KEY not found in .env file.")
        return

    # 2. Securely initialize Firebase Admin SDK using your credentials
    try:
        cred = credentials.Certificate("iris-firebase-credentials.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://iris-50bfb-default-rtdb.asia-southeast1.firebasedatabase.app/'
        })
        print("-> Secured Firebase connection established successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Firebase: {e}")
        return

    # 3. Initialize the modern Google GenAI client
    client = genai.Client(api_key=gemini_key)

    # 4. Interface with the webcam
    # We try port 0 first, which is standard for USB webcams on Raspberry Pi
    camera_port = 0 
    camera = cv2.VideoCapture(camera_port)
    
    if not camera.isOpened():
        print("ERROR: Webcam could not be accessed. Verify connection or check camera port.")
        return

    print("-> Capturing image environment...")
    retval, frame = camera.read()
    camera.release()  # Release the hardware camera immediately so other processes aren't blocked

    if not retval:
        print("ERROR: Failed to capture frames from webcam.")
        return

    # 5. Optimization: Compress image straight to memory bytes (Zero Disk I/O delay)
    retval, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()

    print("-> Image captured. Sending optimized payload to Gemini 2.5 Flash...")

    # 6. Assistive Prompt Engineering 
    # This guides the model to act specifically as a mobility and context helper
    system_instruction = (
        "You are IRIS, an assistive AI companion for a blind person. "
        "Describe what is directly in front of the user clearly and concisely. "
        "Prioritize immediate obstacles, items within arm's reach, or text if visible. "
        "Describe spatial orientation (e.g., 'on your left', 'right in front of you'). "
        "Keep your description brief (under 3 sentences) so it can be smoothly spoken aloud."
    )

    try:
        # Request content generation using highly efficient multimodal Gemini 2.5 Flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                system_instruction
            ]
        )
        
        description = response.text.strip()
        print(f"\n[IRIS Generated Scene Analysis]:\n\"{description}\"\n")

        # 7. Push the string safely to your Firebase Realtime database node
        # This replaces the old legacy 'RaspberryString' location with bulletproof writing rules
        ref = db.reference('/RaspberryString')
        ref.set(description)
        print("-> Scene analysis successfully synchronized to Firebase Database!")

    except Exception as e:
        print(f"ERROR during processing or transmission: {e}")

if __name__ == "__main__":
    main()
