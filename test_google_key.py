import os
from dotenv import load_dotenv
import google.generativeai as genai

def test_google_api_key():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please ensure you have a .env file with GOOGLE_API_KEY set.")
        return

    print(f"Found API Key: {api_key[:4]}...{api_key[-4:]}")
    
    try:
        genai.configure(api_key=api_key)
        
        print("Attempting to list available models...")
        models = list(genai.list_models())
        
        print(f"✅ Success! Authentication worked. Found {len(models)} models.")
        print("First 5 models available:")
        for model in models[:5]:
            print(f" - {model.name}")
            
    except Exception as e:
        print(f"❌ API Key validation failed: {str(e)}")

if __name__ == "__main__":
    test_google_api_key()
