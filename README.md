# cloud_project

## Fake News Detector Backend

A FastAPI backend that detects fake news using a News Search Tool python script (news_search.py).

## Setup

```bash
pip install fastapi uvicorn httpx pydantic
```

```bash
python news_search.py
```

API will be available at `http://localhost:8001`

## API Endpoints

### 1. Root Endpoint

- **GET** `/`
- Returns a status message confirming the backend is running.
```
{ "message": "News Search Backend is running!" }
```


### 2. Search News

- **POST** `/search`
- Searches for news articles across multiple sources (NewsAPI + Google News RSS) and returns a credibility verdict powered by LLM verification.

**Example Request Body:**

```
{ "query": "Singapore school bullies stricter punishments" }
```

**Example Responses:**

```json
{
  "query": "Singapore school bullies stricter punishments",
  "total_results": 10,
  "search_verdict": "Widely Reported — multiple credible sources confirm this as fact",
  "llm_reasoning": "Both headlines explicitly state that Singapore schools are implementing stricter punishments for bullying, presenting it as a current and factual situation.",
  "credible_sources_count": 2,
  "credible_articles": [
    {
      "title": "Article headline here",
      "source": "Channel NewsAsia",
      "url": "https://channelnewsasia.com/...",
      "published_at": "Fri, 17 Apr 2026 10:00:00 GMT",
      "description": "Brief article summary...",
      "is_credible_source": true
    }
  ],
  "other_articles": [...]
}
```
```json
{
  "query": "humans started living Mars",
  "total_results": 10,
  "search_verdict": "Unverified — no credible sources found, treat with caution",
  "llm_reasoning": "",
  "credible_sources_count": 0,
  "credible_articles": [],
  "other_articles": [...]
}
```

---

# NemoBot Integration Configuration

## LMStudio Configuration

**Endpoint:** `http://127.0.0.1:1234/v1/chat/completions`

**Model:** `google/gemma-3-4b`

---

## Main Response Generator Function

```javascript
async function generateResponse(content, chat, environment) {
  const userMessage = content.text?.trim() || "";
  const stream = chat.createStream("chat");

  const signpost = chat.createSignpost("duration");
  const searchSignpost = chat.createSignpost("search_news");

  signpost.emitEvent("start");

  try {
    await environment.llmChatCompletions.chat.stream({
      onToken: (token) => {
        stream.streamToken(token);
      },
      args: { userMessage },
      options: {
        externalTool: {

          // Tool — News Search 
          search_news: async () => {
            searchSignpost.emitEvent("start");
            const response = await environment.llmFunctions.bot2(
              userMessage,
              environment,
            );
            searchSignpost.emitEvent("finish");
            return response.searchSummary;
          },
        },
      },
    });
  } catch (err) {
    console.error("Streaming failed:", err);
  } finally {
    signpost.emitEvent("end");
  }
}
```

---

## Chat Completion Configuration

**Prompt:**

```
You are a fact-checking AI assistant with tool:

1. search_news — Searches real news sources to check if credible outlets are reporting the same story.

When a user shares a news headline or claim:
- give a final verdict using this logic:

|       Search Result       |  Final Verdict |
|---------------------------|----------------|
| Widely Reported           |    Credible    |
| No Coverage / Unverified  |    Uncertain   |


Always:
- Show which sources were found (if any)
- Only use URLs exactly as provided in the tool response JSON. Never construct, modify, or guess article URLs. If no direct article URL is available, link to the source's homepage instead.
- Recommend the user verify with the linked sources
- Be neutral and factual in tone

```

---

## External Tool Declaration

```json
{
  "name": "search_news",
  "description": "Search real news sources to check if credible outlets are reporting the same story.",
  "parameters": {
    "type": "object",
    "properties": {
      "userMessage": {
        "type": "string"
      }
    },
    "required": ["userMessage"]
}
}
```

---

## News Detector LLM Function

### Preprocess Function

```javascript
async function preprocess(input, environment) {
  return { input };
}
```

### Postprocess Function

```javascript
async function postprocess(llmOutput, environment) {
    let searchQuery = "";

  try {
    let outputStr =
      typeof llmOutput === "string" ? llmOutput : JSON.stringify(llmOutput);

    const jsonMatch = outputStr.match(/\{.*\}/s);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]);
      if (parsed.search_query) searchQuery = parsed.search_query;
    }
  } catch (err) {
    console.warn("Failed to parse search query:", err);
  }

  if (!searchQuery) {
    return {
      searchSummary: JSON.stringify({
        search_verdict: "No query extracted",
        credible_sources_count: 0,
        credible_articles: []
      }),
    };
  }

  const response = await fetch("http://localhost:8001/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: searchQuery }),
  });

  if (!response.ok) {
    throw new Error(`Search API failed: ${response.status}`);
  }

  const data = await response.json();
  console.log("Search result:", data);

  return { searchSummary: JSON.stringify(data) };
}
```

### LLM Function Prompt (Article Extraction)

```
You are an assistant that converts a news claim into a precise search query (4-8 words).

Instructions:
1. Output ONLY a JSON object with a single key: "search_query".
2. Keep ALL key entities — country names, people, organisations, actions.
3. Do NOT simplify or drop words that change the meaning.
4. If no clear claim exists, return an empty string.

Examples:

User: "Is it true the government secretly banned all cars last Tuesday?"
Output: {"search_query": "government secretly banned cars"}

User: "Scientists discover cure for cancer overnight"
Output: {"search_query": "scientists cure cancer discovery"}

User: "Hey what's up"
Output: {"search_query": ""}
```

### Input Format

```json
{ "type": "string" }
```

### Output Format

```json
{
  "type": "object",
  "properties": {
    "searchSummary": { "type": "string" }
  },
  "required": ["searchSummary"]
}
```

---

## Usage Instructions

1. **Start the News Search Tool**: Run `python news_search.py` to start the FastAPI server on `http://localhost:8001`
2. **Configure LMStudio**: Use endpoint `http://127.0.0.1:1234/v1/chat/completions` with model `gemma-3-4b`
3. **Copy configs above** into NemoBot
4. **Test it**: Paste a news headline and ask NemoBot to fact-check it
