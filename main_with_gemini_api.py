# main.py
# The complete Python backend for The Living Ledger, using FastAPI and the Gemini API.

import os
import io
import json
import httpx
import pandas as pd
from typing import List
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Environment & App Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(
    title="The Living Ledger Storyteller API (Python)",
    description="An API that converts financial transactions into Minecraft-themed storylines using the Gemini API."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Data Validation ---
class Transaction(BaseModel):
    description: str
    amount: float
    type: str  # "credit" or "debit"

class TransactionWithStory(Transaction):
    id: int
    story: str
    timestamp: datetime

class QuestProgress(BaseModel):
    goal_name: str
    target_amount: float
    current_progress: float

# --- In-Memory Storage ---
transactions_log: List[TransactionWithStory] = []
next_transaction_id = 0

# --- 🚀 GEMINI Story Generation Logic ---
GEMINI_PROMPT = """
You are a financial storyteller for an app called 'The Living Ledger', which has a Minecraft theme. Your task is to take a raw bank transaction and convert it into a creative, short story statement. First, you must identify the category of the transaction. Then, map the currency to 'diamonds' and the purchase to a relevant Minecraft item or event.
--- EXAMPLES ---
Input: `Description: "UPI/AMAZON_IN/", Type: debit, Amount: 1200` -> Output: Traded 1200 diamonds with a wandering villager for rare crafting supplies.
Input: `Description: "SALARY CREDIT SEPTEMBER", Type: credit, Amount: 75000` -> Output: A massive vein of 75000 diamonds was discovered in the great mine!
Input: `Description: "Zomato", Type: debit, Amount: 350` -> Output: Spent 350 diamonds to craft a few enchanted golden apples for a quick feast.
--- YOUR TASK ---
Input: `Description: "{desc}", Type: {type}, Amount: {amount}` -> Output:
"""


INSIGHT_PROMPT = """
You are the "Oracle" for an app called 'The Living Ledger', which has a Minecraft theme. Your task is to take a user's financial quest data and provide a single, short, and encouraging insight. The insight should be based on the user's progress toward their goal. Use a creative, Minecraft-themed analogy.

--- EXAMPLES ---
Input: `Quest: "Emergency Fund", Target: 5000, Progress: 2500` -> Output: Your emergency fund is half-full! Keep mining that gold, and you'll have a fortress for a rainy day in no time.
Input: `Quest: "New PC", Target: 1500, Progress: 1450` -> Output: The final boss is in sight! Only 50 more diamonds to go before you can craft that powerful new enchanted computer.
Input: `Quest: "Travel Fund", Target: 3000, Progress: 200` -> Output: Every journey begins with a single step. Keep crafting small savings, and your travel portal will open before you know it.
--- YOUR TASK ---
Input: `Quest: "{goal}", Target: {target}, Progress: {progress}` -> Output:
"""

async def get_story_from_gemini(transaction: Transaction) -> str:
    if not GEMINI_API_KEY:
        return "API Key not configured. The Oracle is silent."
    
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={'AIzaSyADhPPJfDD2Z3QqesYXn36CRjprtdV90q0'}"
    full_prompt = GEMINI_PROMPT.format(
        desc=transaction.description, type=transaction.type.lower(), amount=transaction.amount
    )
    
    payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
    
    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            response = await client.post(gemini_api_url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            response_data = response.json()
            return response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (httpx.RequestError, KeyError, IndexError) as e:
            print(f"Error calling Gemini API: {e}")
            return "The connection to the celestial Oracle was lost..."

# --- API Endpoints ---
@app.get("/transactions", response_model=List[TransactionWithStory])
async def get_all_transactions():
    return transactions_log

@app.post("/transaction", response_model=TransactionWithStory)
async def create_transaction(transaction: Transaction):
    global next_transaction_id
    story = await get_story_from_gemini(transaction)
    
    new_transaction = TransactionWithStory(
        id=next_transaction_id,
        description=transaction.description,
        amount=transaction.amount,
        type=transaction.type,
        story=story,
        timestamp=datetime.utcnow()
    )
    
    transactions_log.insert(0, new_transaction)
    next_transaction_id += 1
    
    return new_transaction

@app.post("/api/quest/insight")
async def get_quest_insight(quest_progress: QuestProgress):
    full_prompt = INSIGHT_PROMPT.format(
        goal=quest_progress.goal_name,
        target=quest_progress.target_amount,
        progress=quest_progress.current_progress
    )
    
    # Use the same get_story_from_gemini function, but pass in the new prompt
    # Note: You may want to rename get_story_from_gemini to a more general name
    story = await get_story_from_gemini(full_prompt)

    return {"insight": story}

@app.post("/upload_csv", response_model=List[TransactionWithStory])
async def upload_transactions_csv(file: UploadFile = File(...)):
    global next_transaction_id
    newly_created_transactions: List[TransactionWithStory] = []
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        required_columns = {'Description', 'Amount', 'Type'}
        if not required_columns.issubset(df.columns):
            raise HTTPException(status_code=400, detail=f"CSV must contain {required_columns} columns.")

        # Process all API calls in parallel for efficiency
        story_tasks = []
        for _, row in df.iterrows():
            transaction_data = Transaction(
                description=row['Description'],
                amount=float(row['Amount']),
                type=row['Type'].lower()
            )
            story_tasks.append(get_story_from_gemini(transaction_data))
        
        stories = await asyncio.gather(*story_tasks)

        for (index, row), story in zip(df.iterrows(), stories):
            # Use timestamp from CSV if available, otherwise use current time
            timestamp = pd.to_datetime(row.get('Timestamp', datetime.utcnow())).to_pydatetime()
            
            new_transaction = TransactionWithStory(
                id=next_transaction_id,
                description=row['Description'],
                amount=float(row['Amount']),
                type=row['Type'].lower(),
                story=story,
                timestamp=timestamp
            )
            newly_created_transactions.append(new_transaction)
            next_transaction_id += 1
        
        transactions_log.extend(newly_created_transactions)
        transactions_log.sort(key=lambda tx: tx.timestamp, reverse=True)

        return newly_created_transactions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")

# Add asyncio import for gather
import asyncio

