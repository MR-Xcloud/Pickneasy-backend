import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------
# ENV & INIT
# -------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

QA_INDEX_NAME = "pickneasy"
WEB_INDEX_NAME = "pickneasy-wb"

app = FastAPI(title="PicknEasy Chatbot API (Pinecone)")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to pickneasy.com later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)

pc = Pinecone(api_key=PINECONE_API_KEY)

qa_index = pc.Index(QA_INDEX_NAME)
web_index = pc.Index(WEB_INDEX_NAME)

# -------------------------------------------------
# REQUEST MODEL
# -------------------------------------------------

class ChatRequest(BaseModel):
    question: str

# -------------------------------------------------
# WEBSITE SCRAPER
# -------------------------------------------------

def scrape_site(base_url: str) -> List[str]:
    visited = set()
    texts = []

    def crawl(url):
        if url in visited or not url.startswith(base_url):
            return
        visited.add(url)

        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")

            page_text = soup.get_text(separator=" ", strip=True)
            texts.append(page_text)

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/"):
                    crawl(base_url + href)
                elif href.startswith(base_url):
                    crawl(href)

        except:
            pass

    crawl(base_url)
    return texts

# -------------------------------------------------
# ONE-TIME INDEX BUILD (SAFE IF RUN MULTIPLE TIMES)
# -------------------------------------------------

def build_qa_index():
    df = pd.read_csv("pickneasy_chatbot_knowledge_base.csv")

    vectors = []
    for i, row in df.iterrows():
        vec = embeddings.embed_query(row["user_question"])
        vectors.append(
            (
                f"qa-{i}",
                vec,
                {
                    "user_question": row["user_question"],  # Store original question for LLM matching
                    "answer": row["answer"],
                    "category": row["category"]
                }
            )
        )

    if vectors:
        qa_index.upsert(vectors)


def build_web_index():
    pages = scrape_site("https://pickneasy.com")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    vectors = []
    idx = 0

    for page in pages:
        chunks = splitter.split_text(page)
        for chunk in chunks:
            vec = embeddings.embed_query(chunk)
            vectors.append(
                (
                    f"web-{idx}",
                    vec,
                    {"text": chunk}
                )
            )
            idx += 1

    if vectors:
        web_index.upsert(vectors)

# -------------------------------------------------
# STARTUP
# -------------------------------------------------

@app.on_event("startup")
def startup():
    # Build indexes only if empty
    try:
        if qa_index.describe_index_stats()["total_vector_count"] == 0:
            print("Building QA index...")
            build_qa_index()
    except Exception as e:
        print(f"Warning: Failed to build QA index: {e}")
        print("Server will start, but QA index may be empty. You can build it later.")

    try:
        if web_index.describe_index_stats()["total_vector_count"] == 0:
            print("Building website index...")
            build_web_index()
    except Exception as e:
        print(f"Warning: Failed to build website index: {e}")
        print("Server will start, but website index may be empty. You can build it later.")

    print("Pinecone chatbot ready.")

# -------------------------------------------------
# CHAT ENDPOINT
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question
    query_vector = embeddings.embed_query(question)

    # 1️⃣ QA FIRST - Get multiple potential matches
    qa_res = qa_index.query(
        vector=query_vector,
        top_k=5,  # Get more candidates for LLM to evaluate
        include_metadata=True
    )

    if qa_res["matches"]:
        # If we have a very high confidence match, still validate it
        best_match = qa_res["matches"][0]
        if best_match["score"] > 0.92:  # Raised threshold for direct match
            # Quick validation for high-confidence matches too
            question_lower = question.lower()
            answer_lower = best_match["metadata"]["answer"].lower()
            
            # Check "why" questions even for high-confidence matches
            if "why" in question_lower:
                answer_start = answer_lower[:60].strip()
                starts_with_description = (
                    answer_start.startswith("pickneasy is a") or
                    answer_start.startswith("pickneasy is ") or
                    answer_start.startswith("it's a") or
                    answer_start.startswith("it is a") or
                    answer_start.startswith("is a") or
                    "is a fun" in answer_start
                )
                if starts_with_description:
                    # Don't use high-confidence match, continue to LLM evaluation
                    pass
                else:
                    # Answer seems to explain why, use it
                    return {
                        "source": "qa",
                        "category": best_match["metadata"]["category"],
                        "answer": best_match["metadata"]["answer"]
                    }
            else:
                # Not a "why" question, use high-confidence match
                return {
                    "source": "qa",
                    "category": best_match["metadata"]["category"],
                    "answer": best_match["metadata"]["answer"]
                }
        
        # Otherwise, use LLM to determine if any of the matches answer the question
        # Filter matches with reasonable similarity (score > 0.75) - raised threshold
        relevant_matches = [m for m in qa_res["matches"] if m["score"] > 0.75]
        
        if relevant_matches:
            # Build context from potential QA matches
            qa_context = "\n\n".join([
                f"Question: {m['metadata'].get('user_question', 'N/A')}\nAnswer: {m['metadata'].get('answer', 'N/A')}"
                for m in relevant_matches[:3]  # Use top 3 matches
            ])
            
            # Use LLM to determine if any QA answer matches the user's question
            qa_prompt = f"""You are PicknEasy's official chatbot assistant.

The user asked: "{question}"

Below are some potential Q&A pairs from our knowledge base. Determine if ANY of these DIRECTLY answers the user's specific question.

Q&A Pairs:
{qa_context}

CRITICAL INSTRUCTIONS:
- Only respond with "MATCH:" if one of the answers above DIRECTLY answers the user's question. The answer must address the SPECIFIC question asked, not just be related to the topic.
- Pay attention to KEYWORDS in the question:
  * If question asks "why" → answer must explain reasons/benefits/purpose (not just what it is)
  * If question asks "who is the founder/creator/inventor" → answer must mention a person's name (founder/creator/inventor)
  * If question asks "who designed/made" → answer must mention who designed/made it
  * If question asks about "vision/purpose/goal/mission" → answer must be about vision/purpose/goal/mission
  * If question asks "what is it?" → answer must explain what it is (not vision, not cleaning, not who made it, not why)
  * If question asks "when/founded/date" → answer must contain time/date information
  * If question asks "how to clean" → answer must be about cleaning
- Examples:
  * User: "why PicknEasy?" Answer: "because it's smarter and reusable" → MATCH (explains reason/benefit)
  * User: "why PicknEasy?" Answer: "it's a fun tool" → NO_MATCH (describes what it is, not why)
  * User: "why PicknEasy?" Answer: "PicknEasy is a fun tool that helps..." → NO_MATCH (starts with description, doesn't explain WHY)
  * User: "why use it?" Answer: "removes learning curve" → MATCH (explains benefit/reason)
  * User: "why use it?" Answer: "it's a fun tool" → NO_MATCH (doesn't explain why)
  * User: "why use it?" Answer: "PicknEasy is a fun tool" → NO_MATCH (describes what it is, not why to use it)
  * User: "who is the founder?" Answer: "John Smith founded it" → MATCH (mentions founder)
  * User: "who is the founder?" Answer: "it's a fun tool" → NO_MATCH (doesn't mention founder/person)
  * User: "who designed it?" Answer: "Tam Tran designed it" → MATCH (mentions designer)
  * User: "who designed it?" Answer: "it's a fun tool" → NO_MATCH (doesn't mention designer)
  * User: "what is the vision?" Answer: "it's smarter and reusable" → MATCH (vision-related)
  * User: "what is the vision?" Answer: "it's a fun tool" → NO_MATCH (doesn't answer vision)
  * User: "what is it?" Answer: "it's a fun tool" → MATCH (explains what it is)
  * User: "what is it?" Answer: "top rack recommended" → NO_MATCH (about cleaning, not what it is)
  * User: "when was it founded?" Answer: "founded in 2020" → MATCH
  * User: "when was it founded?" Answer: "it's a fun tool" → NO_MATCH (wrong topic)
- If the answer matches, respond with "MATCH:" followed by the answer text exactly as shown.
- If NO answer directly addresses the question, respond with "NO_MATCH" - do NOT guess or provide a related answer.

Your response:"""
            
            llm_response = llm.invoke([HumanMessage(content=qa_prompt)]).content.strip()
            
            # Check if LLM found a match
            if llm_response.startswith("MATCH:"):
                # Extract the answer (everything after "MATCH:")
                answer = llm_response.replace("MATCH:", "").strip()
                
                # Find which match this answer came from (try to match the answer text)
                matched_qa = None
                for m in relevant_matches:
                    stored_answer = m["metadata"].get("answer", "")
                    # Check if the LLM's answer matches or is very similar to stored answer
                    if answer == stored_answer or answer in stored_answer or stored_answer in answer:
                        matched_qa = m
                        break
                
                # If no exact match found, use the best match
                if not matched_qa:
                    matched_qa = relevant_matches[0]
                    answer = matched_qa["metadata"].get("answer", answer)
                
                # Additional validation: Check if question keywords suggest the answer might be wrong
                question_lower = question.lower()
                answer_lower = answer.lower()
                
                # Check for keyword mismatches
                validation_failed = False
                
                # "Why" questions - must explain reasons/benefits/purpose, not just describe what it is
                why_keywords = ["why"]
                if any(kw in question_lower for kw in why_keywords):
                    # Check if answer starts with description (these don't answer "why")
                    answer_start = answer_lower[:60].strip()  # Check first 60 chars
                    # More comprehensive check for description patterns
                    starts_with_description = (
                        answer_start.startswith("pickneasy is a") or
                        answer_start.startswith("pickneasy is ") or
                        answer_start.startswith("it's a") or
                        answer_start.startswith("it is a") or
                        answer_start.startswith("is a") or
                        "is a fun" in answer_start or
                        "is a self-grabbing" in answer_start
                    )
                    
                    # Check if answer explains reasons/benefits/purpose (must be strong indicators)
                    reason_indicators = ["because", "removes", "solves", "avoids", "reduces", "better", "easier", "smarter", "reusable", "benefit", "reason", "purpose", "removes the", "solves the"]
                    has_strong_reason = any(indicator in answer_lower for indicator in reason_indicators)
                    
                    # If answer starts with description pattern, ALWAYS reject (even if it has "helps")
                    if starts_with_description:
                        validation_failed = True
                    elif not has_strong_reason:
                        # No strong reason indicators - "helps" alone is not enough for "why" questions
                        validation_failed = True
                
                # "Who" questions (founder/creator/inventor/designer)
                who_keywords = ["who is", "who was", "who are", "founder", "creator", "inventor", "designed", "made by"]
                if any(kw in question_lower for kw in who_keywords):
                    # Check if answer contains a person's name or person-related indicators
                    person_indicators = ["tam tran", "inventor", "founder", "creator", "designer", "created by", "designed by", "made by"]
                    # Also check for common name patterns (capitalized words that might be names)
                    has_person_name = any(indicator in answer_lower for indicator in person_indicators)
                    # Check if answer contains capitalized words that might be names (simple heuristic)
                    words = answer.split()
                    has_capitalized_name = any(len(word) > 2 and word[0].isupper() and word[1:].islower() for word in words)
                    
                    if not (has_person_name or has_capitalized_name):
                        # Answer doesn't mention a person, reject match
                        validation_failed = True
                
                # Time-based questions
                time_keywords = ["when", "founded", "date", "year", "started", "created", "established"]
                if any(kw in question_lower for kw in time_keywords):
                    time_indicators = ["202", "19", "20", "year", "since", "ago", "founded", "started", "created"]
                    if not any(indicator in answer_lower for indicator in time_indicators):
                        validation_failed = True
                
                # Vision/purpose/mission questions
                vision_keywords = ["vision", "purpose", "mission", "goal", "objective", "aim"]
                if any(kw in question_lower for kw in vision_keywords):
                    vision_indicators = ["vision", "purpose", "mission", "goal", "future", "smarter", "reusable", "solve", "problem"]
                    if not any(indicator in answer_lower for indicator in vision_indicators):
                        validation_failed = True
                
                # "What is it?" questions should explain what it is, not other topics
                if "what is" in question_lower and ("vision" not in question_lower and "purpose" not in question_lower and "founder" not in question_lower):
                    # Check if answer actually explains what it is
                    what_is_indicators = ["tool", "eating", "chopstick", "utensil", "device", "product"]
                    if not any(indicator in answer_lower for indicator in what_is_indicators):
                        # Answer might be about something else (cleaning, vision, etc.)
                        validation_failed = True
                
                if validation_failed:
                    # Answer doesn't match question type, reject and continue to website search
                    pass  # Will fall through to website search below
                else:
                    # Answer seems valid, return it
                    return {
                        "source": "qa",
                        "category": matched_qa["metadata"]["category"],
                        "answer": answer
                    }
            
            # If no match or match was rejected, continue to website search

    # 2️⃣ WEBSITE FALLBACK
    web_res = web_index.query(
        vector=query_vector,
        top_k=4,
        include_metadata=True
    )

    if web_res["matches"]:
        context = "\n\n".join(
            m["metadata"]["text"] for m in web_res["matches"]
        )

        # Determine question type for better prompting
        question_lower = question.lower()
        is_why_question = "why" in question_lower
        is_what_question = question_lower.startswith("what")
        is_who_question = question_lower.startswith("who")
        is_when_question = any(kw in question_lower for kw in ["when", "founded", "date", "year"])
        
        # Build appropriate prompt based on question type
        if is_why_question:
            prompt = f"""
You are PicknEasy's official chatbot.

The user asked: "{question}"

Below is context from our website. Extract and synthesize information that explains WHY PicknEasy exists, its benefits, reasons to use it, or its purpose.

INSTRUCTIONS:
- Look for information about benefits, reasons, purposes, problems solved, advantages, or value propositions
- Synthesize the information from the context to answer "why" PicknEasy
- If you find relevant information (even if not worded exactly as the question), provide a helpful answer
- Only say "This information is not available on our website" if there is truly NO relevant information about benefits, reasons, or purpose
- Be helpful and extract the key reasons/benefits from the context

Context:
{context}

Your response:"""
        elif is_who_question:
            prompt = f"""
You are PicknEasy's official chatbot.

The user asked: "{question}"

Below is context from our website. Answer ONLY using this context.

INSTRUCTIONS:
- Look for names of people (founders, creators, designers, inventors, team members)
- If you find a person's name related to the question, provide that information
- Only say "This information is not available on our website" if no person's name is found

Context:
{context}

Your response:"""
        elif is_when_question:
            prompt = f"""
You are PicknEasy's official chatbot.

The user asked: "{question}"

Below is context from our website. Answer ONLY using this context.

INSTRUCTIONS:
- Look for dates, years, time-related information
- If you find time-related information, provide it
- Only say "This information is not available on our website" if no dates/years are found

Context:
{context}

Your response:"""
        else:
            prompt = f"""
You are PicknEasy's official chatbot.

The user asked: "{question}"

Below is context from our website. Answer ONLY using this context.

INSTRUCTIONS:
- If the answer to the question is stated in the context (even if worded differently), provide that answer
- Synthesize information from the context if needed to answer the question
- Only say "This information is not available on our website" if there is truly NO relevant information in the context
- Be helpful and extract relevant information from the context

Context:
{context}

Your response:"""

        answer = llm.invoke([HumanMessage(content=prompt)]).content

        return {
            "source": "website",
            "answer": answer.strip()
        }

    # 3️⃣ FINAL FALLBACK
    return {
        "source": "none",
        "answer": "This information is not available yet."
    }
