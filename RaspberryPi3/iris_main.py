import os
import cv2
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
import firebase_admin
from firebase_admin import credentials, db

def process_and_analyze_image(client, system_instruction):
    """
    Handles hardware camera capture, optimizes image payloads in memory, 
    and requests scene analysis from Gemini 2.5 Flash.
    """
    # 1. Interface with the webcam
    camera_port = 0 
    camera = cv2.VideoCapture(camera_port)
    
    if not camera.isOpened():
        print("ERROR: Webcam could not be accessed. Verify connection.")
        return "Error: Could not access the Raspberry Pi camera module."

    print("📸 Trigger received! Capturing image environment...")
    retval, frame = camera.read()
    camera.release()  # Release hardware lock immediately

    if not retval:
        print("ERROR: Failed to capture frames from webcam.")
        return "Error: Camera failed to capture a clear frame."

    # 2. Optimization: Compress image straight to memory bytes (Zero Disk I/O delay)
    retval, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()

    print("🧠 Sending optimized payload to Gemini 2.5 Flash...")

    try:
        # 3. Request content generation
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                system_instruction
            ]
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Exception: {e}")
        return "Error: Gemini analysis engine timed out."

def main():
    print("Initializing IRIS Core Engine...")
    
    # Load configuration safely from .env file
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("CRITICAL ERROR: GEMINI_API_KEY not found in .env file.")
        return

    # Securely initialize Firebase Admin SDK
    try:
        cred = credentials.Certificate("iris-firebase-credentials.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://iris-50bfb-default-rtdb.asia-southeast1.firebasedatabase.app/'
        })
        print("-> Secured Firebase connection established successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Firebase: {e}")
        return

    # Initialize the Google GenAI client
    client = genai.Client(api_key=gemini_key)

    # Assistive Prompt Engineering 
    system_instruction = (
        "You are IRIS, an assistive AI companion for a blind person. "
        "Describe what is directly in front of the user clearly and concisely. "
        "Prioritize immediate obstacles, items within arm's reach, or text if visible. "
        "Describe spatial orientation (e.g., 'on your left', 'right in front of you'). "
        "Keep your description brief (under 3 sentences) so it can be smoothly spoken aloud."
    )

    # Get reference to the communications node
    ref = db.reference('/RaspberryString')
    print("\n🚀 IRIS 2.0 Pi Core Online. Awaiting voice triggers from app...")

    # Infinite processing loop
    while True:
        try:
            # Check the current status in the cloud node
            current_status = ref.get()

            if current_status == "TRIGGER":
                # 1. Run the camera capture and analysis pipeline
                description = process_and_analyze_image(client, system_instruction)
                print(f"[IRIS Scene Output]: \"{description}\"")

                # 2. Push the result right back into the exact same node!
                # This overwrites "TRIGGER" so your app intercepts the speech text instantly
                ref.set(description)
                print("-> Scene analysis pushed back to app via Firebase.")

        except Exception as e:
            print(f"Loop Alert: {e}")

        # Sleep for 1 second between checks to manage Pi CPU cycles cleanly
        time.sleep(1)

if __name__ == "__main__":
    main()