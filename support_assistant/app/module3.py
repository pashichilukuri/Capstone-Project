#install dependecies
#pip3 install langgraph openai chromadb
#pip3 install -U --force-reinstall chromadb opentelemetry-api opentelemetry-sdk

#import dependecies
import os
import json
from openai import OpenAI
import math
import os
from typing import TypedDict
from pydantic import BaseModel, Field, ValidationError
from IPython.display import Image, display
from fastapi import FastAPI
from pydantic import BaseModel, Field
import chromadb

def call_llm():
  from openai import OpenAI
  groq_api = os.getenv("Groq_API")
  return OpenAI(
      api_key = groq_api,
      base_url="https://api.groq.com/openai/v1")

#Load the data
doc_01 = "Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes."

doc_02 ="Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened, resalable condition. Approved refunds are credited to the original payment method within 3–5 business days, or instantly to the Zepto wallet if the customer opts for wallet credit. Personal care items that have been opened are non-returnable except in the case of a manufacturing defect. Return pickup, where required, is arranged free of cost by Zepto."

doc_03 = "Zepto offers three account tiers: Basic (free, default tier, standard delivery fees apply), Zepto Pass (INR 49 per month, free standard delivery on all orders and 5% off select categories), and Zepto Pass+ (INR 99 per month, free priority delivery, 10% off select categories, and early access to limited-time deals 24 hours before they go live to Basic and Pass members). Membership can be cancelled at any time from account settings; cancelling stops the next billing cycle but does not refund the current membership period."

doc_04 = "Every Zepto order shows a live rider-tracking map from the moment it is packed until delivery, accessible from the 'Track Order' screen. Estimated delivery time updates automatically as the rider moves. If an order's status shows no movement for more than 20 minutes past its original estimated delivery time, customers should contact support directly rather than continue waiting, since this indicates a likely delivery issue."

doc_05 = "Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be cancelled through the app, since the rider is dispatched immediately after packing given Zepto's quick-delivery model. If a packed order cannot be delivered due to a Zepto-side issue (for example, rider unavailability), the order is auto-cancelled and fully refunded without any cancellation fee."

doc_06 = "If an order arrives with damaged, spoiled, or missing items, customers must report it within 24 hours of delivery through the 'Report an Issue' button on the order page. Zepto ships a free replacement or issues a full refund for damaged, spoiled, or missing items without requiring the customer to return the original item, unless the order value exceeds INR 1000, in which case a photo of the issue must be submitted through the report form before a replacement or refund is processed."

doc_07 = "Zepto gift cards are available in fixed denominations of INR 100, INR 250, INR 500, and INR 1000, and are delivered by email or SMS within minutes of purchase. Gift cards are valid for 1 year from the date of issue and carry no maintenance fees. Gift card balance can be combined with one other payment method at checkout but cannot be combined with another gift card in the same transaction. Gift card balance cannot be redeemed for cash except where required by law."

doc_08 ="Zepto customer support is available via in-app chat 24 hours a day, 7 days a week, given the time-sensitive nature of quick commerce deliveries. Average in-app chat response time is under 2 minutes. Email support is also available for non-urgent queries and is answered within 24 hours on business days. Phone support is not offered."

#Chunk the documents

def chunk_documents(text, source_name):

  paragraphs = text.strip().split("\n\n")
  chunks = []
  for para in paragraphs:
    para = para.strip()
    if len(para) < 50:
      continue

    if para.startswith("====="):
      continue

    chunks.append({
        "text": para,
        "source": source_name
    })

  return chunks
doc_01_chunks=chunk_documents(doc_01, "Delivery Policy")
doc_02_chunks =chunk_documents(doc_02, "Returns & Refunds")
doc_03_chunks=chunk_documents(doc_03, "Membership Tiers")
doc_04_chunks=chunk_documents(doc_04, "Order Tracking")
doc_05_chunks=chunk_documents(doc_05, "Order Cancellation Policy")
doc_06_chunks=chunk_documents(doc_06, "Damaged or Missing Items")
doc_07_chunks=chunk_documents(doc_07, "Gift Cards")
doc_08_chunks=chunk_documents(doc_08, "Customer Support Hours")

zepto_policy_chunks = doc_01_chunks + doc_02_chunks + doc_03_chunks + doc_04_chunks +doc_05_chunks +doc_06_chunks + doc_07_chunks +doc_08_chunks
print(len(zepto_policy_chunks))



#Storing the chunks in Chroma DB

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="Zepto_Policy")

documents = []
ids       = []
metadata  = []

for i, chunk in enumerate (zepto_policy_chunks):
  documents.append(chunk["text"])
  ids.append(f"chunk_{i}")
  metadata.append({"source" : chunk["source"]})

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadata
)

# cosine similarity function

def cosine_similarity(vec1, vec2):
  '''calculatecosine similarity between 2 vectors '''

  dot = sum(a * b for a,b in zip (vec1,vec2))
  magnitude1 = math.sqrt (sum( a * a for a in vec1))
  magnitude2 = math.sqrt (sum( b * b for b in vec2))

  if magnitude1 ==0 or magnitude2==0:
    return 0.0
  return dot/(magnitude1 * magnitude2)

#clasify the state with intent

# MOCK_LLM unset or "1" = mock mode
# MOCK_LLM="0" = real LLM mode

mock_LLM = os.getenv("mock_LLM", "1")

#PYDANTIC OUTPUT SCHEMA
class FinalAnswer(BaseModel):
  answer : str
  sources : list[str] = Field(default_factory=list)
  confidence : float = Field(
      ge= 0.0,
      le = 1.0)
#Langraph state

class RouterState(TypedDict):
  query : str
  intent : str
  retrieved_chunks : list[str]
  answer : str
  final_answer: dict
  
 #Node1 clasify the intent
def clasify_intent(state: RouterState):
  '''clasify the incoming_query in to policy_question or Genearl_question'''
  query = state["query"]

  if mock_LLM != "0":
    keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
    intent = "policy_question" if any(k in query.lower() for k in keywords) else "general_question"
  else:
    response = call_llm().chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "classify the question exactly into one category: policy_question or general_question. reply with only one category"},
            {"role": "user", "content": query}
        ],
        temperature=0.2
    )
    intent = response.choices[0].message.content.strip()

  if intent not in ["policy_question", "general_question"]:
    intent = "general_question"

  return {"intent": intent}


  if state["intent"] == "general_question":
        result = FinalAnswer(
        answer="This is a general question.",
        sources=[],
        confidence=1.0
        )
  return {
        "final_answer": result.model_dump(),
        "retrieved_chunks": []
         }  
  
  if state["intent"] == "policy_question":
    result = FinalAnswer(
    answer=answer,
    sources=source_ids,
    confidence=1.0
     )

# Ensure valid intent
  if mock_LLM != "0":
    keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card","support hours"]
    lower_query = query.lower()
    if any (keyword in lower_query for keyword in keywords):
      intent = "policy_question"
    else:
      intent= "general_question"

#Optional extension (REAL LLM Model)
  else:
    llm_client = call_llm()

    response = call_llm().chat.completions.create(
        model = "openai/gpt-oss-20b",
        messages = [
            {"role":"system",
             "content":(
             "clasify the question exatly into one category policy_question  or general_question"
             "if category is policy_question  go and get the information from zepo policies"
             "if catogory is general_question no need to get information from zepo policy corpus answer directly"
             "reply with only one catogory")},
            {"role":"user","content": query}
        ],
        temperature  = 0.2

    )

    intent=response.choices[0].message.content.strip()

  if intent not in [
        "policy_question",
        "general_question"
    ]:
        intent = "general_question"
  return {"intent" :intent}

#Node2 retrieve_and_answer

def retrieve_and_answer (state: RouterState):
  '''embeds the query and retrieves the top-3 most similar chunks from ChromaDB via cosine similarity'''
  query = state["query"]

# STEP 1: EMBED QUERY
  results = collection.query(query_texts=[query], n_results=3)

  results = collection.get(
    include=[
        "documents",
        "embeddings"
    ])
  documents = results["documents"]
  embeddings = results["embeddings"]
  ids = results["ids"]

#CALCULATE COSINE SIMILARITY
  chunks = []

  for chunk_id, doc_text, doc_emb in zip(
      ids,
      documents,
      embeddings
      ):
    similarity = cosine_similarity(doc_emb,query_embedding)

    chunks.append(
        (similarity,
        chunk_id,
        doc_text
         )
       )
 #SORT BY SIMILARITY
  chunks.sort(
      key=lambda x: x[0],
      reverse=True
     )

#tops 3 chuncks
  top_chunks = chunks[:3]

# SOURCE IDS
  source_ids = [
      chunk_id
      for similarity, chunk_id, document 
      in top_chunks
   ]
#mock mode
  if mock_LLM != "0":
    if top_chunks:
      top_chunk_snippet = top_chunks[0][2][:200]
      
      answer = (
          f"Based on the retrieved context:" 
          f"{top_chunk_snippet} "
      )

# Deterministically create Pydantic object.
      result = FinalAnswer(
          answer = answer,
          sources = source_ids,
          confidence = 1.0
       )
    else:
      result = FinalAnswer(
      answer= ("I could not find relevant information"
                   "in Zepto policies corpus"
                   ),

      sources= [],
      confidence = 1.0
      )

#real llm mode

  else:
    context = "\n\n".join(
        doc_text
        for similarity, chunk_id, doc_text
        in top_chunks
        )

    result = generate_validated_llm_answer(
        query = query,
        context= context,
        source_ids = source_ids
    )
  return {"retrieved_chunks": top_chunks, "final_answer": result.model_dump()}
# REAL LLM + PYDANTIC VALIDATION
def generate_validated_llm_answer(
  query : str,
  context : str,
  source_ids : list[str]
  ):

  llm_client = call_llm()
  prompt = f"""
ROLE:
"You are a helpful and accurate question-answering assistant."
"You answer questions using only the information provided in the retrieval from the Zepto policy corpus context."

CONTEXT:
"The following information was retrieved from the knowledge base:"

{context}

TASK:
"Answer the user's question using the provided context."
"If the answer is not available in the context, clearly say:"
"I can only answer questions about Zepto policies right now"

NEGATIVE CONSTRAINT:
"do not answer using information not present in the provided context"

FEW-SHOT EXAMPLE:

Question:
 "what is the order Cancellation  policy"

Context:
"Orders can be cancelled free of cost any time before the order status changes to 'Packed'."

Answer:
{
    "answer": "Orders can be cancelled free of cost any time before the order status changes to 'Packed'.",
    "sources": ["chunk_1"],
    "confidence": 1.0
}

FORMAT:
Return ONLY valid JSON matching this schema:

{
    "answer": "string",
    "sources": ["chunk_id"],
    "confidence": 0.0
}

Do not return Markdown.
Do not return text outside the JSON object.

LENGTH:
"Keep the answer below 100 lines or 1 or 2 sentenses."

USER QUESTION:
{query}
"""

  for attempt in range(3):

    llm_client = call_llm()

    response =llm_client.chat.completions.create(
        model = "openai/gpt-oss-20b",
        messages = [
        {"role":"user",
         "content": prompt
         }
        ],
        temperature = 0.2
    )

    raw_output = response.choices[0].message.content.strip()

    try:
        result = FinalAnswer.model_validate_json(
          raw_output)

        return result
    
    except ValidationError as error:
      if attempt < 2:
        prompt = f"""
Your previous response failed validation.
Validation error:
{error}

Return ONLY valid JSON matching this schema:

{{
    "answer": "string",
    "sources": ["chunk_id"],
    "confidence": 0.0
}}

The valid source IDs are:

{source_ids}

Do not include Markdown or any text outside the JSON.
"""
    else:
      return FinalAnswer(
          answer=(
                  "ERROR: The LLM failed to produce "
                  "a valid response after 3 attempts."
                    ),
          sources=[],
          confidence=0.0
                  )

#node 3 direct_answer

def direct_answer(state: RouterState):
   query = state["query"]

   if mock_LLM != "0":
       answer_text = "I can only answer questions about Zepto policies right now."
   else:
       response = call_llm().chat.completions.create(
           model="openai/gpt-oss-20b",
           messages=[
               {"role": "system", "content": "You are a helpful assistant. Answer directly and concisely."},
               {"role": "user", "content": query}
           ],
           temperature=0.2
       )
       answer_text = response.choices[0].message.content.strip()

   result = FinalAnswer(answer=answer_text, sources=[], confidence=1.0)
   return {"final_answer": result.model_dump()}
#create the router intent
def router_intent(state: RouterState):
  if state["intent"] == "policy_question":
    return "retrieve_and_answer"
  return "direct_answer"

#adding nodes using langraph
from langgraph.graph import StateGraph, START ,END

router_graph = StateGraph(RouterState)

router_graph.add_node("clasify_intent", clasify_intent)
router_graph.add_node("retrieve_and_answer", retrieve_and_answer)
router_graph.add_node("direct_answer", direct_answer)

#adding edges

router_graph.add_edge(START, "clasify_intent")

#adding conditional_edges
router_graph.add_conditional_edges(
    "clasify_intent",
    router_intent,
    {
      "retrieve_and_answer" : "retrieve_and_answer",
      "direct_answer" :"direct_answer"
    }
    )

router_graph.add_edge("retrieve_and_answer", END)
router_graph.add_edge("direct_answer", END)


zepto_app = router_graph.compile()

if __name__ == "__main__":
    display(Image(zepto_app.get_graph().draw_mermaid_png()))
    results = zepto_app.invoke({"query": "what are the list of Membership Tiers"})
    print(results)

# Request schema

class ask_Request(BaseModel):
  query: str

# FastAPI application
app = FastAPI(
    title="Policy RAG Chatbot API",
    version="1.0"
)

# POST /ask
@app.post(
    "/ask",
    response_model=FinalAnswer
)
def ask(request: ask_Request):

    state = {
        "query": request.query
    }

    result = zepto_app.invoke(state)
    return FinalAnswer.model_validate(result["final_answer"])