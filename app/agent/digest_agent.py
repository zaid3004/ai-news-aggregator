# app/agent/digest_agent.py

import os
import json
import time
import random
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class DigestOutput(BaseModel):
    title: str
    summary: str


PROMPT = """You are an expert AI news analyst specializing in summarizing technical articles, research papers, and video content about artificial intelligence.

Your role is to create concise, informative digests that help readers quickly understand the key points and significance of AI-related content.

Guidelines:
- Create a compelling title (5-10 words)
- Write a 2-3 sentence summary
- Focus on key insights and implications
- Be clear and concise
- No fluff
- Respond ONLY with valid JSON. No explanation. No markdown. No backticks.
"""


class DigestAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OLLAMA_API_KEY"),
            base_url="http://localhost:11434/v1"
        )

        # Free-tier fallback models (rotate if needed)
        self.models = [
            "mistral"
        ]

        self.system_prompt = PROMPT

        # HARD throttle (critical for free tier)
        self.request_delay = 6  # seconds


    def generate_digest(
        self,
        title: str,
        content: str,
        article_type: str
    ) -> Optional[DigestOutput]:

        user_prompt = (
            f"Create a digest for this {article_type}:\n"
            f"Title: {title}\n"
            f"Content: {content[:4000]}\n\n"
            f"Respond ONLY with valid JSON matching this schema:\n"
            f"{DigestOutput.model_json_schema()}"
        )

        for model in self.models:
            for attempt in range(3):
                try:
                    print(f"→ Trying model: {model} (attempt {attempt+1})")

                    response = self.client.chat.completions.create(
                        model=model,
                        temperature=0.5,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )

                    raw = response.choices[0].message.content.strip()

                    # Clean response
                    raw = raw.replace("```json", "").replace("```", "").strip()

                    parsed = json.loads(raw)

                    # throttle AFTER success
                    time.sleep(self.request_delay)

                    return DigestOutput(**parsed)

                except Exception as e:
                    err = str(e)

                    # Rate limit handling
                    if "429" in err:
                        wait = (2 ** attempt) * 5 + random.uniform(1, 3)
                        print(f"⚠ Rate limited. Waiting {wait:.2f}s...")
                        time.sleep(wait)
                        continue

                    # JSON issues → retry once
                    elif "Expecting value" in err or "JSON" in err:
                        print("⚠ Bad JSON, retrying...")
                        time.sleep(3)
                        continue

                    else:
                        print(f"❌ Error with model {model}: {e}")
                        break  # move to next model

        print("❌ All models failed")
        return None
    
    def safe_parse(raw):
        try:
            return json.loads(raw)
        except:
            # crude recovery
            start = raw.find("{")
            end = raw.rfind("}") + 1
            return json.loads(raw[start:end])