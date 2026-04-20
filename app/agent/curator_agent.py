import os
import json
from typing import List
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class RankedArticle(BaseModel):
    digest_id: str = Field(description="The ID of the digest (article_type:article_id)")
    relevance_score: float = Field(description="Relevance score from 0.0 to 10.0", ge=0.0, le=10.0)
    rank: int = Field(description="Rank position (1 = most relevant)", ge=1)
    reasoning: str = Field(description="Brief explanation of ranking")


class CuratorAgent:
    def __init__(self, user_profile: dict):
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = "llama-3.3-70b-versatile"
        self.user_profile = user_profile
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        interests = "\n".join(f"- {i}" for i in self.user_profile["interests"])
        preferences = "\n".join(f"- {k}: {v}" for k, v in self.user_profile["preferences"].items())

        return f"""
You are an expert AI news curator.

Rank articles based on:
1. Relevance
2. Technical depth
3. Novelty
4. Practical value

User:
Name: {self.user_profile["name"]}
Background: {self.user_profile["background"]}
Expertise: {self.user_profile["expertise_level"]}

Interests:
{interests}

Preferences:
{preferences}
"""

    def rank_digests(self, digests: List[dict]) -> List[RankedArticle]:
        if not digests:
            return []

        digest_list = "\n\n".join([
            f"ID: {d['id']}\nTitle: {d['title']}\nSummary: {d['summary']}\nType: {d['article_type']}"
            for d in digests
        ])

        user_prompt = f"""
Rank these {len(digests)} AI articles:

{digest_list}

Return ONLY valid JSON:

{{
  "articles": [
    {{
      "digest_id": "...",
      "relevance_score": 0.0,
      "rank": 1,
      "reasoning": "..."
    }}
  ]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
            )

            content = response.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()

            data = json.loads(content)

            return [RankedArticle(**item) for item in data["articles"]]

        except Exception as e:
            print("❌ Curator parsing failed:", e)
            print("RAW OUTPUT:\n", content if 'content' in locals() else "No content")

            # fallback: simple ranking
            sorted_digests = sorted(digests, key=lambda x: x.get("relevance_score", 0), reverse=True)

            fallback = []
            for i, d in enumerate(sorted_digests):
                fallback.append(RankedArticle(
                    digest_id=d["id"],
                    relevance_score=d.get("relevance_score", 5.0),
                    rank=i + 1,
                    reasoning="Fallback ranking"
                ))

            return fallback