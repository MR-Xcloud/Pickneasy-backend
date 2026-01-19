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
# HELPER FUNCTIONS
# -------------------------------------------------

def frame_answer_conversationally(user_question: str, raw_answer: str) -> str:
    """Frame the raw answer from knowledge base in a conversational, human-like way."""
    framing_prompt = f"""You are PicknEasy's friendly chatbot assistant. A user asked: "{user_question}"

You found this information in the knowledge base: "{raw_answer}"

Your task: Frame this information in a natural, conversational way that directly answers the user's question. Make it sound like you're having a friendly conversation with them.

CRITICAL: Respond with ONLY the framed answer text. Do NOT include labels like "User:", "Framed:", or any other formatting. Just return the conversational answer directly.

Guidelines:
- Respond naturally to their specific question wording
- Use conversational language (e.g., "If you order something, it typically takes..." instead of just "3–7 business days")
- Keep the core information accurate and complete
- Make it feel personal and helpful
- Don't add information that's not in the raw answer
- Keep it concise but friendly
- Return ONLY the answer text, nothing else

Example:
- User: "how much time will it take if i order something"
- Raw answer: "3–7 business days domestically, longer internationally."
- Your response (ONLY this): "If you place an order, it typically takes 3–7 business days for domestic shipping. For international orders, it may take a bit longer."

Now frame the answer for the user's question (return ONLY the framed answer, no labels or formatting):"""
    
    framed_answer = llm.invoke([HumanMessage(content=framing_prompt)]).content.strip()
    
    # Post-process to extract only the answer if LLM included labels
    if "Framed:" in framed_answer:
        # Extract text after "Framed:"
        framed_answer = framed_answer.split("Framed:")[-1].strip()
    elif "User:" in framed_answer and "\n" in framed_answer:
        # Extract the last line if there are multiple lines
        lines = framed_answer.split("\n")
        # Take the last non-empty line that doesn't start with "User:"
        for line in reversed(lines):
            if line.strip() and not line.strip().startswith("User:"):
                framed_answer = line.strip()
                break
    
    # Remove quotes if the entire response is wrapped in quotes
    if framed_answer.startswith('"') and framed_answer.endswith('"'):
        framed_answer = framed_answer[1:-1]
    
    return framed_answer


# -------------------------------------------------
# CHAT ENDPOINT
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    question_lower = question.lower()
    
    # Use LLM to detect if the user's input is a greeting or a real question
    greeting_check_prompt = f"""Determine if the following user input is a greeting/salutation or an actual question about a product/service.

User input: "{question}"

Respond with ONLY one word:
- "GREETING" if it's a greeting, salutation, or casual hello (e.g., "hi", "hello", "what's up", "hey there", "good morning", etc.)
- "QUESTION" if it's an actual question or request for information (e.g., "what is it?", "how does it work?", "tell me about...", etc.)

Your response:"""
    
    greeting_check = llm.invoke([HumanMessage(content=greeting_check_prompt)]).content.strip().upper()
    
    if "GREETING" in greeting_check:
        # Generate a varied, friendly greeting response
        greeting_response_prompt = f"""You are PicknEasy's friendly chatbot assistant. The user just greeted you with: "{question}"

Respond with a warm, friendly greeting that:
- Acknowledges their greeting naturally
- Offers to help them
- Varies your response (don't always say the same thing)
- Keep it brief (1-2 sentences max)
- Be conversational and welcoming

Examples of varied responses:
- "Hi there! I'm here to help with any questions about PicknEasy. What would you like to know?"
- "Hello! Great to meet you. How can I assist you today?"
- "Hey! Welcome! I'm happy to help you learn about PicknEasy. What can I tell you?"
- "Good to see you! What would you like to know about PicknEasy?"

Generate a similar friendly greeting response:"""
        
        greeting_answer = llm.invoke([HumanMessage(content=greeting_response_prompt)]).content.strip()
        
        return {
            "source": "greeting",
            "answer": greeting_answer
        }
    
    query_vector = embeddings.embed_query(question)

    # 1️⃣ QA ONLY - Get multiple potential matches from JSON knowledge base
    qa_res = qa_index.query(
        vector=query_vector,
        top_k=10,  # Get more candidates for better semantic matching
        include_metadata=True
    )

    if qa_res["matches"]:
        # If we have a very high confidence match, still validate it
        best_match = qa_res["matches"][0]
        if best_match["score"] > 0.92:  # Raised threshold for direct match
            # Quick validation for high-confidence matches too
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
                    raw_answer = best_match["metadata"]["answer"]
                    framed_answer = frame_answer_conversationally(question, raw_answer)
                    return {
                        "source": "qa",
                        "category": best_match["metadata"]["category"],
                        "answer": framed_answer
                    }
            else:
                # Not a "why" question, use high-confidence match
                raw_answer = best_match["metadata"]["answer"]
                framed_answer = frame_answer_conversationally(question, raw_answer)
                return {
                    "source": "qa",
                    "category": best_match["metadata"]["category"],
                    "answer": framed_answer
                }
        
        # Otherwise, use LLM to determine if any of the matches answer the question
        # Filter matches with lower similarity threshold to catch more semantic matches
        relevant_matches = [m for m in qa_res["matches"] if m["score"] > 0.55]
        
        if relevant_matches:
            # Keyword-based prioritization for different question types
            materials_keywords = ["materials", "made from", "made of", "composition", "what is it made"]
            is_materials_question = any(kw in question_lower for kw in materials_keywords)
            
            # Check for "who is [person name]" questions
            who_is_pattern = question_lower.startswith("who is ") or question_lower.startswith("who was ")
            person_name = None
            if who_is_pattern:
                # Extract person name (words after "who is" or "who was")
                words = question.split()
                if len(words) >= 3 and words[0].lower() == "who" and words[1].lower() in ["is", "was"]:
                    person_name = " ".join(words[2:]).lower()
            
            # Check for "why children/kids want/like" questions
            children_keywords = ["children", "kids", "child", "kid"]
            why_children_question = "why" in question_lower and any(kw in question_lower for kw in children_keywords)
            
            # Check for "what is this product" type questions
            what_is_product = ("what is" in question_lower and ("this" in question_lower or "product" in question_lower or "it" in question_lower)) and "materials" not in question_lower
            
            # Check for "why use" questions
            why_use_question = "why" in question_lower and ("use" in question_lower or "pickneasy" in question_lower)
            
            # Check for "what foods" questions
            what_foods_question = "food" in question_lower and ("what" in question_lower or "which" in question_lower)
            
            if is_materials_question:
                # Re-sort to prioritize questions that contain materials-related keywords
                def materials_priority(match):
                    stored_q = match["metadata"].get("user_question", "").lower()
                    has_materials_in_q = any(kw in stored_q for kw in materials_keywords)
                    return (has_materials_in_q, match["score"])  # Prioritize matches with materials keywords
                
                relevant_matches = sorted(relevant_matches, key=materials_priority, reverse=True)
            elif person_name:
                # Re-sort to prioritize questions where the person's name appears in the answer
                def person_priority(match):
                    stored_answer = match["metadata"].get("answer", "").lower()
                    stored_q = match["metadata"].get("user_question", "").lower()
                    # Check for name in both lowercase and with proper capitalization
                    person_name_parts = person_name.split()
                    has_name_in_answer = person_name in stored_answer or all(part in stored_answer for part in person_name_parts)
                    has_name_in_question = person_name in stored_q or all(part in stored_q for part in person_name_parts)
                    # Also check if it's a "who designed/made" question
                    is_who_question = "who" in stored_q and ("designed" in stored_q or "made" in stored_q or "inventor" in stored_q)
                    return (has_name_in_answer, has_name_in_question, is_who_question, match["score"])
                
                relevant_matches = sorted(relevant_matches, key=person_priority, reverse=True)
            elif why_children_question:
                # Re-sort to prioritize questions about what makes it fun for children
                def children_priority(match):
                    stored_q = match["metadata"].get("user_question", "").lower()
                    stored_answer = match["metadata"].get("answer", "").lower()
                    has_children_in_q = any(kw in stored_q for kw in children_keywords) or "fun for" in stored_q
                    has_children_in_answer = any(kw in stored_answer for kw in ["colorful", "comfy", "confidence", "fun"])
                    return (has_children_in_q, has_children_in_answer, match["score"])
                
                relevant_matches = sorted(relevant_matches, key=children_priority, reverse=True)
            elif what_is_product:
                # Re-sort to prioritize "what is pickneasy" type questions
                def product_priority(match):
                    stored_q = match["metadata"].get("user_question", "").lower()
                    has_what_is = "what is" in stored_q or "what is it" in stored_q or stored_q == "col"
                    return (has_what_is, match["score"])
                
                relevant_matches = sorted(relevant_matches, key=product_priority, reverse=True)
            elif why_use_question:
                # Re-sort to prioritize "why" questions about benefits/reasons
                def why_priority(match):
                    stored_q = match["metadata"].get("user_question", "").lower()
                    stored_answer = match["metadata"].get("answer", "").lower()
                    has_why_in_q = "why" in stored_q
                    has_reason = any(kw in stored_answer for kw in ["smarter", "reusable", "removes", "solves"])
                    return (has_why_in_q, has_reason, match["score"])
                
                relevant_matches = sorted(relevant_matches, key=why_priority, reverse=True)
            elif what_foods_question:
                # Re-sort to prioritize questions about foods
                def foods_priority(match):
                    stored_q = match["metadata"].get("user_question", "").lower()
                    stored_answer = match["metadata"].get("answer", "").lower()
                    has_food_in_q = "food" in stored_q
                    has_food_list = any(kw in stored_answer for kw in ["noodles", "dumplings", "sushi", "fruit"])
                    return (has_food_in_q, has_food_list, match["score"])
                
                relevant_matches = sorted(relevant_matches, key=foods_priority, reverse=True)
            
            # Build context from potential QA matches - use more matches for better coverage
            qa_context = "\n\n".join([
                f"Question: {m['metadata'].get('user_question', 'N/A')}\nAnswer: {m['metadata'].get('answer', 'N/A')}"
                for m in relevant_matches[:5]  # Use top 5 matches for better semantic matching
            ])
            
            # Use LLM to determine if any QA answer matches the user's question (including semantic similarity)
            qa_prompt = f"""You are PicknEasy's official chatbot assistant.

The user asked: "{question}"

Below are some potential Q&A pairs from our knowledge base. Determine if ANY of these answers the user's question - either directly or with the same meaning/context.

Q&A Pairs:
{qa_context}

CRITICAL INSTRUCTIONS:
- Respond with "MATCH:" if one of the answers above answers the user's question, even if worded differently but has the same meaning/context
- Consider semantic similarity - if the user's question means the same thing as a stored question, use that answer
- IMPORTANT: Distinguish between question types carefully:
  * Materials/Composition questions: "what materials", "what is it made from", "what is it made of", "what materials are used", "composition" → These ask about WHAT materials are used (e.g., "Food-grade, BPA-free materials")
  * Safety/Certification questions: "is it safe", "is it BPA-free", "is it food-safe" → These ask about safety certification (e.g., "Yes, 100% BPA-free")
  * DO NOT match materials questions with safety questions - they are different question types even if both mention "BPA-free"
  * "Who is [person name]" questions: MUST match answers that mention that person's name. If user asks "who is Tam Tran", only match if the answer mentions "Tam Tran"
  * "Why children/kids want/like" questions: Match with questions about what makes it fun for children or why children like it
- Pay attention to KEYWORDS and their semantic equivalents:
  * Materials questions: "what materials", "what is it made from", "what is it made of", "what materials are used", "composition" → Match with questions that ask "what materials" or "what is it made from/of"
  * Safety questions: "is it safe", "is it BPA-free", "is it food-safe", "BPA-free" (when asking yes/no) → Match with questions that ask "is it" or safety-related questions
  * If question asks "why" → answer must explain reasons/benefits/purpose (not just what it is)
  * If question asks "who is [person name]" → answer MUST mention that person's name (e.g., "who is Tam Tran" → answer must contain "Tam Tran")
  * If question asks "who is the founder/creator/inventor" → answer must mention a person's name (founder/creator/inventor)
  * If question asks "who designed/made" → answer must mention who designed/made it
  * If question asks "why children/kids want/like" → match with questions about what makes it fun for children or why children like it
  * If question asks about "vision/purpose/goal/mission" → answer must be about vision/purpose/goal/mission
  * If question asks "what is it?" → answer must explain what it is (not vision, not cleaning, not who made it, not why)
  * If question asks "when/founded/date" → answer must contain time/date information
  * If question asks "how to clean" → answer must be about cleaning
- Examples of semantic matching:
  * User: "what materials are used in this pickneasy utensils" → Stored: "What materials is it made from?" → MATCH (same meaning - both ask about composition)
  * User: "what materials are used" → Stored: "Is PicknEasy BPA-free and food-safe?" → NO_MATCH (different question type - one asks composition, other asks safety)
  * User: "what is this product" → Stored: "col" or "What is PicknEasy?" → MATCH (both ask what the product is)
  * User: "what is it made of?" → Stored: "What materials is it made from?" → MATCH (same meaning)
  * User: "why PicknEasy?" Answer: "because it's smarter and reusable" → MATCH (explains reason/benefit)
  * User: "why use pickneasy" → Stored: "Why call it the 'tool of the future'?" Answer: "Because it's smarter and reusable" → MATCH (explains why to use it)
  * User: "why children want this" → Stored: "What makes PicknEasy fun for children?" Answer: "Colorful, comfy, and confidence-boosting" → MATCH (explains why children like it)
  * User: "what's the reason to use it?" Answer: "removes learning curve" → MATCH (same context as "why")
  * User: "who is Tam Tran" → Stored: "Who designed PicknEasy's grabbing jaws?" Answer: "Inventor Tam Tran refined..." → MATCH (answer mentions Tam Tran)
  * User: "who is Tam Tran" → Stored: "Truth or Trick: PicknEasy is only for kids." Answer: "Trick! Grown-ups love it too." → NO_MATCH (answer doesn't mention Tam Tran, it's a game answer)
  * User: "is it safe for kids" → Stored: "Is PicknEasy BPA-free and food-safe?" Answer: "Yes, 100% BPA-free and food-grade" → MATCH (asks about safety)
  * User: "is it safe for kids" → Stored: "Truth or Trick: PicknEasy is only for kids." Answer: "Trick! Grown-ups love it too." → NO_MATCH (game answer, not about safety)
  * User: "what foods can I use it with" → Stored: "What foods are most fun to try?" Answer: "Noodles, dumplings, fruit..." → MATCH (both ask about foods)
  * User: "who created this?" Answer: "Tam Tran designed it" → MATCH (same context as "who designed")
  * User: "tell me about the inventor" Answer: "Inventor Tam Tran refined..." → MATCH (same context)
- If the answer matches (directly or semantically), respond with "MATCH:" followed by the answer text exactly as shown.
- If NO answer addresses the question (even with different wording), respond with "NO_MATCH" - do NOT guess or provide a related answer.

Your response:"""
            
            llm_response = llm.invoke([HumanMessage(content=qa_prompt)]).content.strip()
            
            # Check if LLM found a match
            if llm_response.startswith("MATCH:"):
                # Extract the answer (everything after "MATCH:")
                answer_text = llm_response.replace("MATCH:", "").strip()
                
                # Find which match this answer came from (try to match the answer text)
                matched_qa = None
                best_similarity = 0
                
                for m in relevant_matches:
                    stored_answer = m["metadata"].get("answer", "")
                    # Check if the LLM's answer matches or is very similar to stored answer
                    if answer_text == stored_answer or answer_text in stored_answer or stored_answer in answer_text:
                        matched_qa = m
                        break
                    # Also check for partial matches (if answer_text is a subset or similar)
                    answer_lower = answer_text.lower()
                    stored_lower = stored_answer.lower()
                    # Calculate simple similarity (common words)
                    answer_words = set(answer_lower.split())
                    stored_words = set(stored_lower.split())
                    if answer_words and stored_words:
                        similarity = len(answer_words & stored_words) / len(answer_words | stored_words)
                        if similarity > best_similarity and similarity > 0.3:  # At least 30% word overlap
                            best_similarity = similarity
                            matched_qa = m
                
                # If no match found, use the best match from vector search
                if not matched_qa:
                    matched_qa = relevant_matches[0]
                
                # Always use the stored answer from the matched QA pair, not the LLM's extracted text
                answer = matched_qa["metadata"].get("answer", "")
                stored_question = matched_qa["metadata"].get("user_question", "").lower()
                
                # Additional validation: Check if question keywords suggest the answer might be wrong
                answer_lower = answer.lower()
                
                # Check for keyword mismatches
                validation_failed = False
                
                # Materials questions validation - ensure they match materials questions, not safety questions
                materials_keywords = ["materials", "made from", "made of", "composition", "what is it made"]
                safety_keywords = ["is it safe", "is it bpa-free", "is it food-safe", "bpa-free and food-safe"]
                is_materials_question = any(kw in question_lower for kw in materials_keywords)
                is_safety_question = any(kw in question_lower for kw in safety_keywords)
                stored_is_materials = any(kw in stored_question for kw in materials_keywords)
                stored_is_safety = any(kw in stored_question for kw in ["is it", "is pickneasy"])
                stored_is_game = "game" in stored_question or "truth or trick" in stored_question
                
                if is_materials_question and stored_is_safety and not stored_is_materials:
                    # User asked about materials but matched a safety question - reject
                    validation_failed = True
                elif is_safety_question and stored_is_materials and not stored_is_safety:
                    # User asked about safety but matched a materials question - reject
                    validation_failed = True
                elif is_safety_question and stored_is_game:
                    # User asked about safety but matched a game answer - reject
                    validation_failed = True
                
                # "Who is [person name]" validation - answer must mention the person's name
                who_is_pattern = question_lower.startswith("who is ") or question_lower.startswith("who was ")
                if who_is_pattern:
                    # Extract person name from question
                    words = question.split()
                    if len(words) >= 3 and words[0].lower() == "who" and words[1].lower() in ["is", "was"]:
                        person_name = " ".join(words[2:]).lower()
                        person_name_parts = person_name.split()
                        # Check if answer contains the person's name (either full name or all parts)
                        has_name = person_name in answer_lower or all(part in answer_lower for part in person_name_parts)
                        if not has_name:
                            # Answer doesn't mention the person - reject
                            validation_failed = True
                
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
                
                if not validation_failed:
                    # Answer seems valid, frame it conversationally and return
                    framed_answer = frame_answer_conversationally(question, answer)
                    return {
                        "source": "qa",
                        "category": matched_qa["metadata"]["category"],
                        "answer": framed_answer
                    }
            
            # If LLM said NO_MATCH but we have a high-scoring vector match, use it as fallback
            if qa_res["matches"] and qa_res["matches"][0]["score"] > 0.75:  # Lowered threshold
                best_match = qa_res["matches"][0]
                # Only use if it's not a "why" question with description answer
                question_lower = question.lower()
                answer_lower = best_match["metadata"]["answer"].lower()
                stored_question = best_match["metadata"].get("user_question", "").lower()
                
                # Check for "who is [name]" questions - answer must contain the name
                who_is_pattern = question_lower.startswith("who is ") or question_lower.startswith("who was ")
                if who_is_pattern:
                    words = question.split()
                    if len(words) >= 3 and words[0].lower() == "who" and words[1].lower() in ["is", "was"]:
                        person_name = " ".join(words[2:]).lower()
                        person_name_parts = person_name.split()
                        has_name = person_name in answer_lower or all(part in answer_lower for part in person_name_parts)
                        if not has_name:
                            # Answer doesn't mention the person, skip this fallback
                            pass
                        else:
                            # Answer has the name, use it
                            raw_answer = best_match["metadata"]["answer"]
                            framed_answer = frame_answer_conversationally(question, raw_answer)
                            return {
                                "source": "qa",
                                "category": best_match["metadata"]["category"],
                                "answer": framed_answer
                            }
                
                # Check for safety questions - don't match game answers
                is_safety_question = any(kw in question_lower for kw in ["is it safe", "is it bpa-free"])
                stored_is_game = "game" in stored_question or "truth or trick" in stored_question
                if is_safety_question and stored_is_game:
                    # Safety question matched game answer, skip
                    pass
                elif "why" not in question_lower:
                    # For non-why questions, use high-scoring match
                    raw_answer = best_match["metadata"]["answer"]
                    framed_answer = frame_answer_conversationally(question, raw_answer)
                    return {
                        "source": "qa",
                        "category": best_match["metadata"]["category"],
                        "answer": framed_answer
                    }
                elif "why" in question_lower:
                    # For why questions, check if answer explains why
                    answer_start = answer_lower[:60].strip()
                    starts_with_description = (
                        answer_start.startswith("pickneasy is a") or
                        answer_start.startswith("pickneasy is ") or
                        answer_start.startswith("it's a") or
                        answer_start.startswith("it is a") or
                        answer_start.startswith("is a") or
                        "is a fun" in answer_start
                    )
                    reason_indicators = ["because", "removes", "solves", "avoids", "reduces", "better", "easier", "smarter", "reusable"]
                    has_reason = any(indicator in answer_lower for indicator in reason_indicators)
                    
                    if not starts_with_description and has_reason:
                        raw_answer = best_match["metadata"]["answer"]
                        framed_answer = frame_answer_conversationally(question, raw_answer)
                        return {
                            "source": "qa",
                            "category": best_match["metadata"]["category"],
                            "answer": framed_answer
                        }

    # 2️⃣ FINAL FALLBACK - No match found in JSON knowledge base
    return {
        "source": "none",
        "answer": "I don't have that information in my knowledge base. Please ask me something about PicknEasy!"
    }
