"""Base agent class for the multi-agent system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    recipient: str
    content: dict[str, Any]
    round_id: int = 0


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    role: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    temperature: float = 0.0
    max_retries: int = 3


class BaseAgent(ABC):
    """Abstract base class for all specialized agents.

    Each agent has a specific role (Designer, Critic, Theorist, etc.),
    a set of tools, and a system prompt defining its behavior.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.role = config.role
        self._history: list[AgentMessage] = []

    @abstractmethod
    def act(self, input_message: AgentMessage) -> AgentMessage:
        """Process an incoming message and produce a response.

        Args:
            input_message: Message from another agent or the orchestrator.

        Returns:
            AgentMessage with this agent's response.
        """
        ...

    def receive(self, message: AgentMessage) -> None:
        """Record an incoming message in history."""
        self._history.append(message)

    def clear_history(self) -> None:
        """Reset agent memory."""
        self._history.clear()

    @property
    def history(self) -> list[AgentMessage]:
        return self._history.copy()


class LLMAgent(BaseAgent):
    """Agent backed by an LLM (via Ollama or API).

    This agent uses LangChain's ReAct pattern to reason and invoke tools.
    """

    def __init__(self, config: AgentConfig, llm: Any, tools: list, prompt_template: str):
        super().__init__(config)
        self.llm = llm
        self.tools = tools
        self.prompt_template = prompt_template

    def act(self, input_message: AgentMessage) -> AgentMessage:
        """Use LLM ReAct loop to process input and produce output."""
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_core.prompts import PromptTemplate

        prompt = PromptTemplate.from_template(self.prompt_template)
        agent = create_react_agent(self.llm, self.tools, prompt)
        executor = AgentExecutor(
            agent=agent, tools=self.tools, verbose=False, handle_parsing_errors=True
        )

        user_input = input_message.content.get("task", "")
        context = input_message.content.get("context", {})

        # Inject role context into the prompt
        enriched_input = f"[Role: {self.role}]\n[Context from {input_message.sender}]: {context}\n\nTask: {user_input}"

        result = executor.invoke({"input": enriched_input})

        response_content = {
            "result": result.get("output", ""),
            "raw": result,
        }

        return AgentMessage(
            sender=self.name,
            recipient=input_message.sender,
            content=response_content,
            round_id=input_message.round_id,
        )
