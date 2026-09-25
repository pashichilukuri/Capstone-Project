# Zepto Policy RAG Chatbot

A small Retrieval-Augmented Generation (RAG) application built with **LangGraph, ChromaDB, Pydantic, FastAPI, and the OpenAI-compatible Groq API**.

The application classifies an incoming question into:

- `policy_question` → retrieves the top-3 relevant Zepto policy chunks from ChromaDB and answers from the retrieved context.
- `general_question` → does not retrieve policy chunks and returns a direct/general response.

The final answer is validated with a Pydantic schema:

```json
{
  "answer": "string",
  "sources": ["chunk_id"],
  "confidence": 1.0
}
```

## Project requirements

This project is designed around the following assignment requirements:

1. Build a LangGraph router for `policy_question` and `general_question`.
2. Retrieve the top-3 relevant policy chunks for policy questions.
3. Return retrieved chunk/document IDs in `sources`.
4. Return `sources=[]` for general questions.
5. Validate the final response with a Pydantic `FinalAnswer` model.
6. In mock mode, create the Pydantic response deterministically without an LLM.
7. In optional real-LLM mode, validate the raw LLM JSON and retry up to two additional times if validation fails.
8. Expose the graph through a FastAPI `POST /ask` endpoint.
9. Containerize the application with Docker and run Uvicorn on port `7860`.

## Architecture

```text
                    User Query
                        |
                        v
                  FastAPI /ask
                        |
                        v
                 LangGraph State
                        |
                        v
                +----------------+
                | Intent Router  |
                +----------------+
                   /          \
                  /            \
                 v              v
       policy_question     general_question
              |                   |
              v                   v
        ChromaDB Retrieval    Direct Answer
              |
              v
        Top 3 Chunks
              |
              v
       FinalAnswer Pydantic
              |
              v
          JSON Response
```

## Response schema

The final response uses:

```python
class FinalAnswer(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
```

`confidence` is constrained to the range `0.0` to `1.0`.

Example policy response:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": [
    "chunk_0",
    "chunk_4",
    "chunk_7"
  ],
  "confidence": 1.0
}
```

Example general-question response:

```json
{
  "answer": "This is a general question.",
  "sources": [],
  "confidence": 1.0
}
```

## Mock mode

The application is intended to run in mock mode by default.

Use:

```python
MOCK_LLM = os.getenv("MOCK_LLM", "1")
```

or an equivalent implementation.

When `MOCK_LLM` is unset, the default should be mock mode.

For a policy question, mock mode should deterministically construct a `FinalAnswer` using the retrieved chunk IDs.

For a general question, no policy retrieval should occur and:

```json
"sources": []
```

should be returned.

> Keep the environment variable name consistent everywhere. The recommended name is `MOCK_LLM`, not `mock_LLM`.

## Optional real LLM mode

The application uses the OpenAI-compatible client to access the Groq API:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("Groq_API"),
    base_url="https://api.groq.com/openai/v1"
)
```

The API key must **never** be hardcoded in the Python source code, Dockerfile, README, or repository.

Set it as an environment variable when running locally.

The optional real-LLM answer generation validates the raw response with:

```python
FinalAnswer.model_validate_json(raw_output)
```

If validation fails, the implementation retries up to two additional times with a corrective JSON-format instruction. After three total attempts, it returns a clearly marked error response with `confidence=0.0`.

## FastAPI API

Run the application locally with:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### POST /ask

Request:

```json
{
  "query": "What is the order cancellation policy?"
}
```

Example response:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": [
    "chunk_4"
  ],
  "confidence": 1.0
}
```

### General question example

Request:

```json
{
  "query": "What is the capital of France?"
}
```

Expected structure:

```json
{
  "answer": "This is a general question.",
  "sources": [],
  "confidence": 1.0
}
```

The actual answer text depends on the implementation.

## Testing with curl

Policy question:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the order cancellation policy?"}'
```

General question:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the capital of France?"}'
```

On Windows PowerShell, `curl.exe` can be used:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/ask" `
  -H "Content-Type: application/json" `
  -d '{"query":"What is the order cancellation policy?"}'
```

## Docker

The required graded containerization baseline is local Docker build and run.

Expected project structure:

```text
project/
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

Example Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Build:

```bash
docker build -t zepto-policy-rag .
```

Run:

```bash
docker run --rm -p 7860:7860 zepto-policy-rag
```

Then open:

```text
http://localhost:7860/docs
```

The container must listen on:

```text
0.0.0.0:7860
```

so that the endpoint is accessible through the Docker port mapping.

### Real LLM Docker testing

Do not put the API key into the image.

Pass it at runtime instead:

```bash
docker run --rm \
  -p 7860:7860 \
  -e MOCK_LLM=0 \
  -e Groq_API="YOUR_API_KEY" \
  zepto-policy-rag
```

For the graded baseline, leave `MOCK_LLM` at its default.

## Data

The example application contains Zepto policy text covering areas such as:

- Delivery
- Returns and refunds
- Membership tiers
- Order tracking
- Order cancellation
- Damaged or missing items
- Gift cards
- Customer support

The text is split into chunks and stored in ChromaDB with IDs such as:

```text
chunk_0
chunk_1
chunk_2
...
```

The retrieved IDs are returned in the `sources` field.

## Important implementation notes

### 1. State access

LangGraph state should be accessed as:

```python
query = state["query"]
```

not:

```python
query = state[query]
```

### 2. RouterState annotation

Use:

```python
def retrieve_and_answer(state: RouterState):
```

rather than:

```python
def retrieve_and_answer(state, RouterState):
```

The second form treats `RouterState` as a normal function argument.

### 3. Source IDs

Only retrieved IDs should be returned:

```python
source_ids = [
    chunk_id
    for similarity, chunk_id, document in top_chunks
]
```

Do not return every ID in the Chroma collection.

### 4. LLM JSON output

The real LLM prompt must request JSON because the raw response is validated using:

```python
FinalAnswer.model_validate_json(raw_output)
```

The prompt should explicitly say:

```text
Return ONLY valid JSON.
Do not return Markdown.
Do not return text outside the JSON object.
```

### 5. FastAPI response

The endpoint should return the validated final answer:

```python
@app.post("/ask", response_model=FinalAnswer)
def ask(request: AskRequest):
    result = router_graph.invoke({"query": request.query})
    return FinalAnswer.model_validate(result["final_answer"])
```

## Known code cleanup before running

The notebook/code version used during development contains some notebook-specific and draft-code artifacts. Before running it as a standalone FastAPI/Docker application, remove or correct items such as:

- `!pip install ...` notebook commands.
- `IPython.display.Image` / `display(...)` graph visualization if not needed by the API.
- Duplicate imports.
- Inconsistent `mock_LLM` vs `MOCK_LLM` naming.
- Any hardcoded API key.
- Undefined helper functions such as `embed_query()` unless they are defined elsewhere.
- Incorrect `math.sqrt(generator)` expressions in the cosine-similarity implementation.
- Any use of `final_answer` when the actual variable is named `result`.
- Any direct-answer node that returns only a string instead of the required `FinalAnswer` structure.

## Security

Never commit:

```text
.env
.env.*
```

or API keys.

Recommended `.gitignore` / `.dockerignore` entries include:

```text
.env
.env.*
__pycache__/
*.pyc
.venv/
venv/
.git/
*.log
```

## Optional Hugging Face Spaces deployment

Hugging Face Spaces deployment is optional and is not required for the graded Docker baseline.

If deployed, the same Dockerfile can be used with the free community CPU tier. Store the LLM API key as a **Space Secret** and never hardcode or commit it.

The required submission is satisfied by successfully:

```text
docker build
      |
      v
docker run
      |
      v
POST /ask
      |
      v
validated FinalAnswer JSON
```
