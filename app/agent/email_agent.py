import os
from datetime import datetime
from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class EmailIntroduction(BaseModel):
    greeting: str
    introduction: str


class RankedArticleDetail(BaseModel):
    digest_id: str
    rank: int
    relevance_score: float
    title: str
    summary: str
    url: str
    article_type: str
    reasoning: Optional[str] = None


class EmailDigestResponse(BaseModel):
    introduction: EmailIntroduction
    articles: List[RankedArticleDetail]
    total_ranked: int
    top_n: int

    def to_markdown(self) -> str:
        md = f"{self.introduction.greeting}\n\n"
        md += f"{self.introduction.introduction}\n\n"
        md += "---\n\n"

        for a in self.articles:
            md += f"## {a.title}\n\n"
            md += f"{a.summary}\n\n"
            md += f"[Read more →]({a.url})\n\n"
            md += "---\n\n"

        return md


EMAIL_PROMPT = """Write a short, engaging AI news email introduction.

Requirements:
- Greet user by name
- Mention today's date
- 2-3 sentences
- Highlight key themes
- Professional but slightly conversational
"""


class EmailAgent:
    def __init__(self, user_profile: dict):
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = "llama-3.3-70b-versatile"
        self.user_profile = user_profile

    def generate_introduction(self, ranked_articles: List) -> EmailIntroduction:
        current_date = datetime.now().strftime('%B %d, %Y')

        if not ranked_articles:
            return EmailIntroduction(
                greeting=f"Hey {self.user_profile['name']}, here is your daily digest for {current_date}.",
                introduction="No major AI updates today."
            )

        summaries = "\n".join([
            f"{i+1}. {a.title if hasattr(a, 'title') else a.get('title')}"
            for i, a in enumerate(ranked_articles[:10])
        ])

        user_prompt = f"""
User: {self.user_profile['name']}
Date: {current_date}

Top articles:
{summaries}

Write the introduction.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EMAIL_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
            )

            content = response.choices[0].message.content.strip()

            return EmailIntroduction(
                greeting=f"Hey {self.user_profile['name']}, here is your daily digest of AI news for {current_date}.",
                introduction=content
            )

        except Exception as e:
            print("❌ Email generation failed:", e)

            return EmailIntroduction(
                greeting=f"Hey {self.user_profile['name']}, here is your daily digest of AI news for {current_date}.",
                introduction="Here are the top AI updates curated for you."
            )

    def create_email_digest_response(
        self,
        ranked_articles: List[RankedArticleDetail],
        total_ranked: int,
        limit: int = 10
    ) -> EmailDigestResponse:

        top_articles = ranked_articles[:limit]
        intro = self.generate_introduction(top_articles)

        return EmailDigestResponse(
            introduction=intro,
            articles=top_articles,
            total_ranked=total_ranked,
            top_n=limit
        )