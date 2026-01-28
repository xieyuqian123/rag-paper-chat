"""
Query Classifier Module
Classifies user queries into different types and selects optimal answering strategies.
"""

import os
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class QueryType(str, Enum):
    """Types of queries that can be asked about papers."""
    FACTUAL = "factual"           # Direct facts from paper (pure RAG)
    ANALYTICAL = "analytical"      # Analysis/interpretation (RAG + LLM reasoning)
    COMPARATIVE = "comparative"    # Compare with external methods (RAG + LLM knowledge)
    BACKGROUND = "background"      # General background knowledge (pure LLM)
    SUMMARY = "summary"            # Summarization request (RAG + synthesis)


class QueryClassification(BaseModel):
    """Classification result for a query."""
    query_type: QueryType = Field(description="The type of query")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the classification")
    rewritten_query: Optional[str] = Field(
        default=None, 
        description="Optionally rewritten query for better retrieval"
    )


CLASSIFICATION_PROMPT = """You are a query classifier for an academic paper Q&A system.

Classify the following query into one of these categories:
- FACTUAL: Questions asking for specific facts, data, methods, or details directly stated in the paper.
  Examples: "What dataset was used?", "What is the model architecture?", "What were the results?"

- ANALYTICAL: Questions requiring interpretation, evaluation, or deeper analysis of the paper content.
  Examples: "What are the strengths of this approach?", "Why did they choose this method?", "What are the limitations?"

- COMPARATIVE: Questions comparing the paper's content with external knowledge or other methods.
  Examples: "How does this compare to BERT?", "What's different from traditional approaches?"

- BACKGROUND: Questions about general concepts that don't require paper-specific information.
  Examples: "What is attention mechanism?", "Explain gradient descent", "What is a transformer?"

- SUMMARY: Requests for summarization or overview of paper sections.
  Examples: "Summarize the introduction", "Give me an overview", "What is this paper about?"

User Query: {query}

Respond with a JSON object containing:
- query_type: one of [factual, analytical, comparative, background, summary]
- confidence: float between 0 and 1
- reasoning: brief explanation (1-2 sentences)
- rewritten_query: (optional) a clearer version of the query for better retrieval

JSON Response:"""


class QueryClassifier:
    """Classifies queries to determine optimal answering strategy."""
    
    def __init__(self):
        """Initialize the classifier with LLM."""
        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        llm_base_url = os.getenv("LLM_BASE_URL")
        
        llm_kwargs = {"model": llm_model, "temperature": 0}
        if llm_base_url:
            llm_kwargs["api_base"] = llm_base_url
        
        self.llm = OpenAI(**llm_kwargs)
    
    def classify(self, query: str) -> QueryClassification:
        """
        Classify a query into a type.
        
        Args:
            query: The user's question
            
        Returns:
            QueryClassification with type, confidence, and reasoning
        """
        try:
            prompt = CLASSIFICATION_PROMPT.format(query=query)
            response = self.llm.complete(prompt)
            
            # Parse JSON response
            import json
            result = json.loads(response.text.strip())
            
            return QueryClassification(
                query_type=QueryType(result.get("query_type", "factual")),
                confidence=float(result.get("confidence", 0.8)),
                reasoning=result.get("reasoning", ""),
                rewritten_query=result.get("rewritten_query")
            )
        except Exception as e:
            # Fallback to factual if classification fails
            print(f"Classification error: {e}")
            return QueryClassification(
                query_type=QueryType.FACTUAL,
                confidence=0.5,
                reasoning="Fallback classification due to error"
            )
    
    def get_strategy(self, query_type: QueryType) -> dict:
        """
        Get the answering strategy for a query type.
        
        Returns:
            Dictionary with strategy configuration
        """
        strategies = {
            QueryType.FACTUAL: {
                "use_rag": True,
                "use_llm_knowledge": False,
                "retrieval_k": 5,
                "prompt_style": "precise",
                "description": "Direct retrieval from paper"
            },
            QueryType.ANALYTICAL: {
                "use_rag": True,
                "use_llm_knowledge": True,
                "retrieval_k": 8,
                "prompt_style": "analytical",
                "description": "RAG + LLM reasoning"
            },
            QueryType.COMPARATIVE: {
                "use_rag": True,
                "use_llm_knowledge": True,
                "retrieval_k": 6,
                "prompt_style": "comparative",
                "description": "RAG + external knowledge"
            },
            QueryType.BACKGROUND: {
                "use_rag": False,
                "use_llm_knowledge": True,
                "retrieval_k": 0,
                "prompt_style": "educational",
                "description": "Pure LLM knowledge"
            },
            QueryType.SUMMARY: {
                "use_rag": True,
                "use_llm_knowledge": False,
                "retrieval_k": 10,
                "prompt_style": "summary",
                "description": "RAG with synthesis"
            }
        }
        return strategies.get(query_type, strategies[QueryType.FACTUAL])
