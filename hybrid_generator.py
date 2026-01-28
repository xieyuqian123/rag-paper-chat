"""
Hybrid Answer Generator Module
Combines RAG retrieval with LLM reasoning for comprehensive answers.
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import NodeWithScore
from dotenv import load_dotenv

from query_classifier import QueryType, QueryClassification

load_dotenv()


class AnswerResult(BaseModel):
    """Result of answer generation."""
    answer: str = Field(description="The generated answer")
    sources: List[dict] = Field(default_factory=list, description="Source citations")
    query_type: QueryType = Field(description="The classified query type")
    strategy_used: str = Field(description="Description of the strategy used")


# Prompt templates for different query types
PROMPTS = {
    QueryType.FACTUAL: """You are an academic paper assistant. Answer the following question based ONLY on the provided context from the paper.
Be precise and cite specific sections when possible. If the information is not in the context, say so.

Context from the paper:
{context}

Question: {question}

Provide a clear, factual answer:""",

    QueryType.ANALYTICAL: """You are an academic paper analyst. Analyze and interpret the following question based on the paper context.
Provide insights, identify implications, and offer your analytical perspective while staying grounded in the paper's content.

Context from the paper:
{context}

Question: {question}

Provide an analytical answer with interpretation:""",

    QueryType.COMPARATIVE: """You are an academic expert. Answer the following comparison question using both the paper context AND your broader knowledge of the field.
When using external knowledge, clearly indicate what comes from the paper vs. general knowledge.

Context from the paper:
{context}

Question: {question}

Note: Feel free to draw on your knowledge of related methods, but clearly distinguish between what's in the paper and external knowledge.

Provide a comparative analysis:""",

    QueryType.BACKGROUND: """You are an academic tutor. Explain the following concept or background question using your knowledge.
This question is about general background knowledge, not specific paper content.

Question: {question}

Provide a clear, educational explanation:""",

    QueryType.SUMMARY: """You are an academic paper summarizer. Synthesize the following context from the paper to answer the summarization request.
Focus on key points, main contributions, and important findings.

Context from the paper:
{context}

Question: {question}

Provide a comprehensive summary:"""
}


class HybridAnswerGenerator:
    """
    Generates answers using a hybrid approach combining RAG retrieval with LLM reasoning.
    """
    
    def __init__(self):
        """Initialize the generator with LLM."""
        llm_model = os.getenv("LLM_MODEL", "gpt-4o")
        llm_base_url = os.getenv("LLM_BASE_URL")
        llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        
        llm_kwargs = {"model": llm_model, "temperature": llm_temperature}
        if llm_base_url:
            llm_kwargs["api_base"] = llm_base_url
        
        self.llm = OpenAI(**llm_kwargs)
    
    def format_context(self, source_nodes: List[NodeWithScore]) -> str:
        """Format retrieved nodes into context string."""
        if not source_nodes:
            return "No relevant context found."
        
        context_parts = []
        for i, node in enumerate(source_nodes, 1):
            # Get metadata
            if hasattr(node, 'node'):
                metadata = node.node.metadata
                text = node.node.text
            else:
                metadata = node.metadata
                text = node.text
            
            page = metadata.get('page_number', metadata.get('page_label', '?'))
            section = metadata.get('section', 'Unknown')
            
            context_parts.append(f"[Source {i} - Page {page}, Section: {section}]\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def format_sources(self, source_nodes: List[NodeWithScore]) -> List[dict]:
        """Extract source information for citations."""
        sources = []
        seen = set()
        
        for node in source_nodes:
            if hasattr(node, 'node'):
                metadata = node.node.metadata
            else:
                metadata = node.metadata
            
            page = metadata.get('page_number', metadata.get('page_label', '?'))
            section = metadata.get('section', 'Unknown')
            file_name = metadata.get('file_name', 'Document')
            
            key = f"{page}-{section}"
            if key in seen:
                continue
            seen.add(key)
            
            sources.append({
                "page": page,
                "section": section,
                "file_name": file_name
            })
        
        return sources[:5]  # Limit to 5 sources
    
    def generate(
        self,
        question: str,
        classification: QueryClassification,
        source_nodes: Optional[List[NodeWithScore]] = None,
        conversation_context: Optional[str] = None
    ) -> AnswerResult:
        """
        Generate an answer using the appropriate strategy.
        
        Args:
            question: The user's question
            classification: Query classification result
            source_nodes: Retrieved source nodes (optional for background queries)
            conversation_context: Summary of conversation history (optional)
        """
        query_type = classification.query_type
        
        # Use rewritten query if available
        effective_question = classification.rewritten_query or question
        
        # Format context
        context = self.format_context(source_nodes) if source_nodes else ""
        
        # Get prompt template
        prompt_template = PROMPTS.get(query_type, PROMPTS[QueryType.FACTUAL])
        
        # Build the prompt
        if query_type == QueryType.BACKGROUND:
            # No context needed for background questions
            prompt = prompt_template.format(question=effective_question)
        else:
            prompt = prompt_template.format(
                context=context,
                question=effective_question
            )
        
        # Add conversation context if available
        if conversation_context:
            prompt = f"Conversation context: {conversation_context}\n\n{prompt}"
        
        try:
            response = self.llm.complete(prompt)
            answer = response.text.strip()
        except Exception as e:
            answer = f"Error generating answer: {e}"
        
        # Determine strategy description
        strategy_descriptions = {
            QueryType.FACTUAL: "📄 Direct retrieval from paper",
            QueryType.ANALYTICAL: "🔍 RAG + analytical reasoning",
            QueryType.COMPARATIVE: "⚖️ RAG + external knowledge comparison",
            QueryType.BACKGROUND: "📚 LLM background knowledge",
            QueryType.SUMMARY: "📝 RAG with synthesis"
        }
        
        return AnswerResult(
            answer=answer,
            sources=self.format_sources(source_nodes) if source_nodes else [],
            query_type=query_type,
            strategy_used=strategy_descriptions.get(query_type, "Unknown strategy")
        )
    
    def format_answer_with_sources(self, result: AnswerResult) -> str:
        """Format the answer with sources for display."""
        output = result.answer
        
        # Add strategy indicator
        output = f"*{result.strategy_used}*\n\n{output}"
        
        # Add sources if available
        if result.sources:
            source_lines = []
            for src in result.sources:
                source_lines.append(f"- **Page {src['page']}** | Section: *{src['section']}*")
            
            output += "\n\n---\n📚 **Sources:**\n" + "\n".join(source_lines)
        
        return output
