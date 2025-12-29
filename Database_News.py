import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# --- AI MODEL LOADING ---
MODEL_NAME = "facebook/bart-large-cnn"
print(f"IVORY Engine: Loading AI Model ({MODEL_NAME})...")
try:
    # Use CPU for maximum compatibility
    DEVICE = "cpu"
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    MODEL.eval()
    print(f"✅ AI Model Loaded on {DEVICE.upper()}.")
except Exception as e:
    print(f"❌ FATAL MODEL ERROR: {e}")
    TOKENIZER, MODEL, DEVICE = None, None, None

def summarize_text(text):
    """Summarizes text safely."""
    if not MODEL or not text or len(text) < 50:
        return text[:200] + "..." 
    
    clean_text = " ".join(text.split())[:3000] 
    
    try:
        inputs = TOKENIZER(clean_text, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
        with torch.no_grad():
            summary_ids = MODEL.generate(
                inputs["input_ids"], 
                max_length=130, 
                min_length=30, 
                length_penalty=2.0, 
                num_beams=4, 
                early_stopping=True
            )
        return TOKENIZER.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"   ⚠️ AI Summary Failed: {e}")
        return clean_text[:200] + "..."

# --- NEW LOC.GOV API FUNCTION ---
def search_chronicling_america(query, page=1):
    print(f"🔎 Searching LOC.GOV (New API) for: '{query}'")
    
    # 1. NEW ENDPOINT: The new home for the API
    base_url = "https://www.loc.gov/collections/chronicling-america/"
    
    params = {
        "q": query,
        "fo": "json",           # 'fo' = format (must be json)
        "fa": "original_format:newspaper", # Filter for newspapers
        "sp": page,             # Page number
        "c": 5                  # Count (limit results)
    }

    try:
        # 2. NO API KEY NEEDED - Just be polite with User-Agent
        headers = {'User-Agent': 'IvoryArchive/1.0 (Educational Use)'}
        
        response = requests.get(base_url, params=params, headers=headers, timeout=20)
        response.raise_for_status() 
        data = response.json()
    except Exception as e:
        print(f"❌ API Connection Error: {e}")
        return []

    results = data.get('results', [])
    print(f"   ✅ Found {len(results)} items.")

    processed_articles = []
    
    for item in results:
        try:
            # 3. PARSE DATA
            title = item.get('title', 'Untitled Record')
            date = item.get('date', 'Unknown Date')
            
            # 4. GET TEXT (from description field in new API)
            description_list = item.get('description', [])
            full_text = " ".join(description_list) if description_list else ""
            
            if len(full_text) < 50:
                full_text = f"{title}. {full_text}"

            # 5. SUMMARIZE
            summary = summarize_text(full_text)

            # 6. LINK
            link = item.get('id') or item.get('url')

            processed_articles.append({
                "title": title,
                "link": link,
                "source": "Library of Congress Archive",
                "date": date,
                "summary": summary
            })
            
        except Exception as e:
            print(f"   ⚠️ Error parsing item: {e}")
            continue

    return processed_articles