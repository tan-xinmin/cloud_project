from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import urlparse, quote_plus
import httpx
import os
import asyncio
import xml.etree.ElementTree as ET
import json
 
app = FastAPI(title="News Search Backend", version="1.0.0")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "api_key_here")
NEWSAPI_URL = "https://newsapi.org/v2/everything"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
LMSTUDIO_MODEL = "gemma-3-4b"

CREDIBLE_SOURCES = {
    # Wire services
    "reuters.com", "apnews.com",
    # Broadcast & major outlets
    "bbc.co.uk", "bbc.com", "cnn.com", "nbcnews.com",
    "cbsnews.com", "npr.org", "pbs.org", "wired.com", "m.economictimes.com",
    # Newspapers
    "nytimes.com", "nypost.com", "washingtonpost.com", "theguardian.com", "wsj.com",
    "ft.com", "economist.com", "independent.co.uk", "telegraph.co.uk",
    "independent.ie", "theatlantic.com", "time.com", "newsweek.com",
    # Science
    "scientificamerican.com", "science.org", "nature.com", "newscientist.com",
    # Business / Finance
    "bloomberg.com", "forbes.com",
    # International
    "aljazeera.com", "aljazeera.net", "dw.com", "france24.com", "abc.net.au", "scmp.com",
    # Singapore — primary
    "straitstimes.com", "channelnewsasia.com", "todayonline.com",
    "mothership.sg", "asiaone.com",
    # Singapore — vernacular
    "beritaharian.sg", "zaobao.com.sg", "tamilmurasu.com.sg",
    # Singapore — govt / official
    "gov.sg", "mha.gov.sg", "spf.gov.sg", "scdf.gov.sg", "lta.gov.sg"
}

SG_DOMAINS = "straitstimes.com,channelnewsasia.com,todayonline.com,mothership.sg,asiaone.com,beritaharian.sg,lta.gov.sg"
 
SOURCE_NAME_TO_DOMAIN = {
    "the straits times": "straitstimes.com",
    "straits times": "straitstimes.com",
    "cna": "channelnewsasia.com",
    "channel newsasia": "channelnewsasia.com",
    "channelnewsasia": "channelnewsasia.com",
    "today": "todayonline.com",
    "todayonline": "todayonline.com",
    "mothership": "mothership.sg",
    "mothership.sg": "mothership.sg",
    "asiaone": "asiaone.com",
    "stomp": "straitstimes.com",
    "mustsharenews.com": "mustsharenews.com",
    "reuters": "reuters.com",
    "associated press": "apnews.com",
    "ap news": "apnews.com",
    "bbc": "bbc.com",
    "bbc news": "bbc.com",
    "cnn": "cnn.com",
    "the new york times": "nytimes.com",
    "new york times": "nytimes.com",
    "the washington post": "washingtonpost.com",
    "washington post": "washingtonpost.com",
    "the guardian": "theguardian.com",
    "bloomberg": "bloomberg.com",
    "al jazeera": "aljazeera.com",
    "al jazeera english": "aljazeera.com",
    "the economist": "economist.com",
    "financial times": "ft.com",
    "scientific american": "scientificamerican.com",
    "abc news (au)": "abc.net.au",
    "france 24": "france24.com",
    "dw": "dw.com",
    "npr": "npr.org",
    "time": "time.com",
    "time magazine": "time.com",
    "newsweek": "newsweek.com",
    "the atlantic": "theatlantic.com",
    "berita harian": "beritaharian.sg",
    "lianhe zaobao": "zaobao.com.sg",
}
 
class SearchInput(BaseModel):
    query: str
 
@app.get("/")
async def root():
    return {"message": "News Search Backend is running!"}
 
def make_params(query: str, page_size: int = 10, domains: str = None) -> dict:
    p = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY
    }
    if domains:
        p["domains"] = domains
    return p
 
def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except:
        return ""
 
def check_credible(domain: str) -> bool:
    return any(domain == src or domain.endswith("." + src) for src in CREDIBLE_SOURCES)
 
async def fetch_google_news_rss(client: httpx.AsyncClient, query: str, geo: str = "SG") -> list:
    url = f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=en-{geo}&gl={geo}&ceid={geo}:en"
    print(f"  Google News RSS URL: {url}")
    try:
        resp = await client.get(url, timeout=10, follow_redirects=True)
        print(f"  Google News RSS status: {resp.status_code}")
        if resp.status_code != 200:
            return []
 
        root = ET.fromstring(resp.text)
        articles = []
        for item in root.findall(".//item"):
            title       = item.findtext("title", "")
            link        = item.findtext("link", "")
            pub_date    = item.findtext("pubDate", "")
            description = item.findtext("description", "")
            source_el   = item.find("source")
            source_name = source_el.text.strip() if source_el is not None else ""
            real_domain = SOURCE_NAME_TO_DOMAIN.get(source_name.lower(), "")
            if not real_domain and source_el is not None:
                source_url = source_el.get("url", "")
                real_domain = get_domain(source_url) if source_url else ""
 
            # Prefer source homepage over the broken Google redirect link.
            # Google RSS <link> values are news.google.com/rss/articles/CBMi… redirects
            # that are frequently truncated or expire — never surface them to users.
            source_homepage = source_el.get("url", "") if source_el is not None else ""
            display_url = source_homepage if source_homepage else link

            articles.append({
                "title": title,
                "source": source_name,
                "url": display_url,          # stable URL shown to users
                "real_domain": real_domain,
                "published_at": pub_date,
                "description": description,
            })
 
        print(f"  Google News RSS returned {len(articles)} articles, keeping first 5")
        return articles[:5]
 
    except Exception as e:
        print(f"  Google News RSS error: {e}")
        return []
 
async def llm_verify_articles(client: httpx.AsyncClient, claim: str, articles: list) -> dict:
    """
    Ask LMStudio to judge whether the found articles actually confirm
    the claim as a fact — not speculative, not analysis, not hypothetical.
    Returns: { "confirmed": bool, "reasoning": str, "confirmed_titles": [str] }
    """
    if not articles:
        return {"confirmed": False, "reasoning": "No articles to evaluate.", "confirmed_titles": []}
 
    # Build a numbered list of titles + sources for the LLM
    article_lines = "\n".join([
        f"{i+1}. [{a['source']}] {a['title']}"
        for i, a in enumerate(articles)
    ])
 
    prompt = f"""You are a strict fact-checking assistant. Your job is to determine whether news article headlines CONFIRM a specific claim as an established fact.
 
CLAIM: "{claim}"
 
ARTICLE HEADLINES:
{article_lines}
 
RULES:
- Only mark an article as CONFIRMING if the headline reports the claim as something that HAS HAPPENED or IS HAPPENING right now as a fact.
- Do NOT count articles that are: speculative ("could", "might", "what if", "how to", "plans to"), analytical ("pros and cons", "debate"), hypothetical, opinion pieces, or about future goals.
- Do NOT count articles that merely mention the topic without confirming the claim.
 
Respond ONLY with a JSON object, no other text:
{{
  "confirmed": true or false,
  "confirmed_count": number of articles that confirm the claim as fact,
  "reasoning": "one sentence explanation",
  "confirmed_titles": ["title1", "title2"]
}}"""
 
    try:
        resp = await client.post(
            LMSTUDIO_URL,
            json={
                "model": LMSTUDIO_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 300,
            },
            timeout=120,
        )
 
        if resp.status_code != 200:
            print(f"  LLM verify failed: {resp.status_code}")
            return {"confirmed": False, "reasoning": "LLM verification unavailable.", "confirmed_titles": []}
 
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"  LLM verify raw response: {raw}")
 
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
 
        result = json.loads(raw)
        print(f"  LLM verify result: confirmed={result.get('confirmed')}, reasoning={result.get('reasoning')}")
        return result
 
    except Exception as e:
        import traceback; print(f"  LLM verify error: {e}"); traceback.print_exc()
        return {"confirmed": False, "reasoning": "LLM verification failed.", "confirmed_titles": []}
 

@app.post("/search")
async def search_news(data: SearchInput):
    if not data.query or len(data.query.strip()) < 5:
        raise HTTPException(status_code=400, detail="Query too short")
 
    try:
        async with httpx.AsyncClient() as client:
            broad, sg_newsapi, rss_sg, rss_global = await asyncio.gather(
                client.get(NEWSAPI_URL, params=make_params(data.query, page_size=5), timeout=10),
                client.get(NEWSAPI_URL, params=make_params(data.query, page_size=3, domains=SG_DOMAINS), timeout=10),
                fetch_google_news_rss(client, data.query, geo="SG"),
                fetch_google_news_rss(client, data.query, geo="US"),
            )
 
        print(f"Query: {data.query} | Broad: {broad.status_code} | SG NewsAPI: {sg_newsapi.status_code}")
 
        if broad.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid NewsAPI key")
 
        broad_articles  = broad.json().get("articles", [])      if broad.status_code == 200      else []
        sg_api_articles = sg_newsapi.json().get("articles", []) if sg_newsapi.status_code == 200 else []
        total_results   = broad.json().get("totalResults", 0)   if broad.status_code == 200      else 0
 
        seen_urls = set()
        all_articles = []
        for a in sg_api_articles + rss_sg + rss_global + broad_articles:
            url = a.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_articles.append(a)
            if len(all_articles) >= 15:
                break
 
        credible_hits = []
        other_hits = []
 
        for article in all_articles:
            source_name = article.get("source", "Unknown")
            if isinstance(source_name, dict):
                source_name = source_name.get("name", "Unknown")
 
            domain   = article.get("real_domain") or get_domain(article.get("url", ""))
            credible = check_credible(domain)
            print(f"  Domain: {domain} | Credible: {credible} | Source: {source_name} | Title: {article.get('title', '')[:60]}")
 
            entry = {
                "title": article.get("title", ""),
                "source": source_name,
                "url": article.get("url", ""),
                "published_at": article.get("published_at", article.get("publishedAt", "")),
                "description": article.get("description", ""),
                "is_credible_source": credible
            }
 
            if credible:
                credible_hits.append(entry)
            else:
                other_hits.append(entry)
 
        # --- LLM verification step ---
        # Only run if we have credible hits, to check they actually confirm the claim
        llm_result = {"confirmed": False, "reasoning": "", "confirmed_titles": []}
        if credible_hits:
            async with httpx.AsyncClient() as client:
                llm_result = await llm_verify_articles(client, data.query, credible_hits[:4])
 
        confirmed        = llm_result.get("confirmed", False)
        llm_reasoning    = llm_result.get("reasoning", "")
        confirmed_count  = llm_result.get("confirmed_count", 0)
 
        # Build final verdict using BOTH source credibility AND LLM confirmation
        if not credible_hits:
            if total_results == 0 and not rss_sg and not rss_global:
                search_verdict = "No Coverage — story not found in any news sources"
            else:
                search_verdict = "Unverified — no credible sources found, treat with caution"
        elif confirmed and confirmed_count >= 2:
            search_verdict = "Widely Reported — multiple credible sources confirm this as fact"
        elif confirmed and confirmed_count == 1:
            search_verdict = "Partially Corroborated — one credible source confirms this as fact"
        else:
            search_verdict = "Unverified — credible sources mention the topic but do NOT confirm this claim as fact"
 
        print(f"Credible hits: {len(credible_hits)} | LLM confirmed: {confirmed} | Verdict: {search_verdict}")
 
        return {
            "query": data.query,
            "total_results": total_results + len(rss_sg) + len(rss_global),
            "search_verdict": search_verdict,
            "llm_reasoning": llm_reasoning,
            "credible_sources_count": len(credible_hits),
            "credible_articles": credible_hits[:5],
            "other_articles": other_hits[:2]
        }
 
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Connection error: {str(e)}")
 



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)







