import torch
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURATION ---
MODEL_DIR = "./final_model"
TEST_FILE = "test_samples"
OUTPUT_FILE = "responses.txt"
MAX_LEN = 512

# --- 1. LOAD MODEL ---
print(f"Loading model from {MODEL_DIR}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    model.eval()
except Exception as e:
    print(f"Error: {e}")
    exit()

# --- 2. SCRAPER FUNCTION ---
def get_text_from_url(url):
    print(f"Scraping: {url}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url.strip(), headers=headers, timeout=10)
        
        if response.status_code != 200:
            return f"Error: Status {response.status_code}"
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        text = " ".join(text.split()) # Clean whitespace
        
        if len(text) < 50:
            return "Error: Webpage text too short."
            
        return text
    except Exception as e:
        return f"Error scraping: {e}"

# --- 3. INFERENCE LOOP ---
print(f"Reading {TEST_FILE}...")
try:
    with open(TEST_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print("Error: test_samples.txt not found!")
    exit()

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for url in urls:
        raw_text = get_text_from_url(url)
        
        if raw_text.startswith("Error"):
            summary = f"[FAILED] {raw_text}"
        else:
            # --- CRITICAL FIX: SMART TRUNCATION ---
            # 1. Define the wrapper text
            prefix = "Summarize this news article: "
            suffix = "\nResponse:"
            
            # 2. Calculate how many tokens the wrapper takes
            wrapper_len = len(tokenizer.encode(prefix + suffix))
            
            # 3. Calculate safe length for the article (512 - wrapper - buffer)
            safe_len = MAX_LEN - wrapper_len - 5
            
            # 4. Tokenize article and chop it
            article_tokens = tokenizer.encode(raw_text)
            truncated_article_tokens = article_tokens[:safe_len]
            truncated_text = tokenizer.decode(truncated_article_tokens)
            
            # 5. Build the final safe prompt
            prompt = f"{prefix}{truncated_text}{suffix}"
            
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=100,
                    do_sample=True,          
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Robust extraction
            if "Response:" in decoded:
                summary = decoded.split("Response:")[-1].strip()
            else:
                # If separator is missing, take the text after the prompt end
                summary = decoded[len(prompt):].strip()

        # Save result
        print(f" -> Summary: {summary[:100]}...") 
        out.write(f"URL: {url}\nSummary: {summary}\n{'-'*50}\n")

print(f"Done! Check {OUTPUT_FILE}")