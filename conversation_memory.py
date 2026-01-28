"""
Conversation Memory Module
Manages conversation history and generates context-aware queries.
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class Message(BaseModel):
    """A single message in the conversation."""
    role: str = Field(description="Either 'user' or 'assistant'")
    content: str = Field(description="The message content")


class ConversationMemory:
    """
    Manages conversation history and provides context-aware query rewriting.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize the conversation memory.
        
        Args:
            max_history: Maximum number of messages to keep in history
        """
        self.messages: List[Message] = []
        self.max_history = max_history
        
        # Initialize LLM for query rewriting
        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        llm_base_url = os.getenv("LLM_BASE_URL")
        
        llm_kwargs = {"model": llm_model, "temperature": 0}
        if llm_base_url:
            llm_kwargs["api_base"] = llm_base_url
        
        self.llm = OpenAI(**llm_kwargs)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append(Message(role=role, content=content))
        
        # Trim history if needed
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_history(self) -> List[Message]:
        """Get the full conversation history."""
        return self.messages
    
    def get_history_text(self, last_n: Optional[int] = None) -> str:
        """
        Get conversation history as formatted text.
        
        Args:
            last_n: Only include last N messages (None for all)
        """
        messages = self.messages[-last_n:] if last_n else self.messages
        
        if not messages:
            return ""
        
        lines = []
        for msg in messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role_label}: {msg.content}")
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear the conversation history."""
        self.messages = []
    
    def condense_query(self, current_query: str) -> str:
        """
        Condense the current query with conversation history to create
        a standalone, context-aware query.
        
        Args:
            current_query: The user's current question
            
        Returns:
            A standalone query that incorporates relevant context
        """
        # If no history or very short history, return as is
        if len(self.messages) < 2:
            return current_query
        
        # Get recent history
        history_text = self.get_history_text(last_n=6)
        
        prompt = f"""Given the following conversation history and a new question, 
rewrite the question to be a standalone question that includes all necessary context.

If the new question doesn't reference anything from the history, return it unchanged.
If it uses pronouns or references to previous topics, expand them.

Conversation History:
{history_text}

New Question: {current_query}

Standalone Question (just output the rewritten question, nothing else):"""

        try:
            response = self.llm.complete(prompt)
            condensed = response.text.strip()
            
            # Sanity check: if the condensed query is too different or too long, use original
            if len(condensed) > len(current_query) * 3 or len(condensed) < 5:
                return current_query
            
            return condensed
        except Exception as e:
            print(f"Query condensation error: {e}")
            return current_query
    
    def get_context_summary(self) -> Optional[str]:
        """
        Generate a brief summary of the conversation context.
        Useful for including in prompts.
        """
        if len(self.messages) < 2:
            return None
        
        history_text = self.get_history_text(last_n=6)
        
        prompt = f"""Summarize the key topics and context from this conversation in 1-2 sentences:

{history_text}

Summary:"""

        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception:
            return None


class PaperConversationManager:
    """
    Manages separate conversation histories for different papers.
    """
    
    def __init__(self):
        self.conversations: dict[str, ConversationMemory] = {}
    
    def get_memory(self, paper_id: str) -> ConversationMemory:
        """Get or create conversation memory for a paper."""
        if paper_id not in self.conversations:
            self.conversations[paper_id] = ConversationMemory()
        return self.conversations[paper_id]
    
    def clear_paper(self, paper_id: str):
        """Clear conversation history for a specific paper."""
        if paper_id in self.conversations:
            self.conversations[paper_id].clear()
    
    def clear_all(self):
        """Clear all conversation histories."""
        self.conversations = {}
