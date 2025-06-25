# llm_handler.py
import requests
import re

class LLMHandler:
    def __init__(self, ollama_url="http://localhost:11434/api/generate", model="mistral"):
        self.ollama_url = ollama_url
        self.model = model

    def generate_sql(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }

        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()
            output = result.get("response", "").strip()
            print(f"[LLM Output]\n{output}\n")
            return self.extract_sql_from_response(output)
        except Exception as e:
            print(f"LLM Error: {e}")
            return "SELECT 1;"  # fallback

    def extract_sql_from_response(self, response: str) -> str:
        matches = re.findall(r"(SELECT\s+.*?;)", response, re.IGNORECASE | re.DOTALL)
        if matches:
            return matches[0].replace('\n', ' ').strip()
        for line in response.splitlines():
            if line.strip().upper().startswith("SELECT"):
                return line.strip()
        return response.strip()
