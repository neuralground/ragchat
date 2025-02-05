from typing import List, Dict, Any
from pydantic import Field, BaseModel
from langchain_core.memory import BaseMemory
from langchain_core.messages import AIMessage, HumanMessage

class SimpleMemory(BaseMemory, BaseModel):
    """A simple memory implementation for chat history tracking."""
    
    chat_memory: List = Field(default_factory=list)
    max_tokens: int = Field(default=5)
    return_messages: bool = Field(default=True)
    
    @property
    def memory_variables(self) -> List[str]:
        """Define memory variables."""
        return ["chat_history"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables."""
        return {
            "chat_history": self.chat_memory[-self.max_tokens * 2:] if self.chat_memory else []
        }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        if question := inputs.get("question"):
            self.chat_memory.append(HumanMessage(content=question))
        if answer := outputs.get("answer"):
            self.chat_memory.append(AIMessage(content=answer))
        
        # Keep only the last n exchanges
        if len(self.chat_memory) > self.max_tokens * 2:
            self.chat_memory = self.chat_memory[-self.max_tokens * 2:]

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory = []

    def get_recent_messages(self, n: int = None) -> List:
        """Get the n most recent messages."""
        if n is None:
            n = self.max_tokens
        return self.chat_memory[-n * 2:] if self.chat_memory else []

