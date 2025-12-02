import asyncio
import json
import logging
import os
import re
import uuid
from typing import Any

import httpx

from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from remote_agent_connection import (
    RemoteAgentConnections,
    TaskUpdateCallback,
)

from google import genai

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv()


def convert_part(part: Part, tool_context: ToolContext):
    """Convert a part to text. Only text parts are supported."""
    if part.type == "text":
        return part.text
    return f"Unknown type: {part.type}"


def convert_parts(parts: list[Part], tool_context: ToolContext):
    """Convert parts to text."""
    rval: list[str] = []
    for p in parts:
        rval.append(convert_part(p, tool_context))
    return rval


def create_send_message_payload(
    text: str, task_id: str | None = None, context_id: str | None = None
) -> dict[str, Any]:
    """Helper function to create the payload for sending a task."""
    payload: dict[str, Any] = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": text}],
            "messageId": uuid.uuid4().hex,
        },
    }

    if task_id:
        payload["message"]["taskId"] = task_id

    if context_id:
        payload["message"]["contextId"] = context_id
    return payload


class RoutingAgent:
    """The Routing agent.

    This is the agent responsible for choosing which remote seller agents to send
    tasks to and coordinate their work.
    """

    def __init__(
        self,
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""

    async def _async_init_components(
        self, remote_agent_addresses: list[str]
    ) -> None:
        """Asynchronous part of initialization."""
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()

                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    logger.debug(
                        "ERROR: Failed to get agent card from %s: %s",
                        address,
                        e,
                    )
                except Exception as e:
                    logger.debug(
                        "ERROR: Failed to initialize connection for %s: %s",
                        address,
                        e,
                    )

        # Build a human-readable list of remote agents
        agent_info: list[str] = []
        for agent_detail_dict in self.list_remote_agents():
            agent_info.append(json.dumps(agent_detail_dict))
        self.agents = "\n".join(agent_info)

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: list[str],
        task_callback: TaskUpdateCallback | None = None,
    ) -> "RoutingAgent":
        """Create and asynchronously initialize an instance of the RoutingAgent."""
        instance = cls(task_callback)
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self) -> Agent:
        """Create an instance of the RoutingAgent."""
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return Agent(
            model=gemini_model,
            name="Routing_agent",
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description=(
                "Routing agent that NEVER answers users directly and ALWAYS "
                "delegates requests to remote agents (e.g. weather, accommodation, "
                "analytics / text summarisation) using the send_message tool."
            ),
            tools=[
                self.send_message,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        """Generate the root instruction for the RoutingAgent."""
        current_agent = self.check_active_agent(context)
        return f"""
You are a **Routing / Orchestration Agent**.

Your ONLY job is to delegate the user's request to one of the available remote agents
using the `send_message` tool. You MUST NOT answer the user directly.

Available remote agents (name + description):
{self.agents}

## Core Rules

1. **Always delegate**
   - For EVERY user request, you MUST call the `send_message` tool.
   - Do NOT respond with your own answer unless absolutely no remote agent is suitable.

2. **Choose the best agent**
   - Read the agents listed above carefully.
   - Pick the agent whose description best matches the user's request.
   - Example:
     - If the user asks about *text analysis, summarisation, tone, or business updates*,
       route to the **Analytics Agent** (or whichever agent card describes analytics/text).
     - If the user asks about *weather or accommodation*, route to the weather/airbnb agents.

3. **How to call `send_message`**
   - `agent_name`: must be the exact `name` field from the chosen agent's card.
   - `task`: a clear, self-contained instruction that includes all context the remote agent needs
     (do NOT assume it has the full chat history).

4. **Context / Session**
   - Preserve and reuse the same `context_id` and `task_id` when appropriate so that
     the remote agent can maintain continuity.
   - The current active agent is: {current_agent['active_agent']}.

5. **If no agent fits**
   - Only if NONE of the available agents can possibly handle the request,
     you may reply with a short apology and explanation.
"""

    def check_active_agent(self, context: ReadonlyContext):
        state = context.state
        if (
            "session_id" in state
            and "session_active" in state
            and state["session_active"]
            and "active_agent" in state
        ):
            return {"active_agent": f"{state['active_agent']}"}
        return {"active_agent": "None"}

    def before_model_callback(
        self,
        callback_context: CallbackContext,
        llm_request,
    ):
        state = callback_context.state
        if "session_active" not in state or not state["session_active"]:
            if "session_id" not in state:
                state["session_id"] = str(uuid.uuid4())
            state["session_active"] = True

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.cards:
            return []

        remote_agent_info: list[dict[str, str]] = []
        for card in self.cards.values():
            logger.debug(
                "Found agent card: %s", card.model_dump(exclude_none=True)
            )
            logger.debug("=" * 100)
            remote_agent_info.append(
                {"name": card.name, "description": card.description}
            )
        return remote_agent_info

    async def send_message(
        self,
        agent_name: str,
        task: str,
        tool_context: ToolContext,
    ):
        """Sends a task to a remote or local analytics agent.

        If remote A2A agents are configured and available, this will send the
        message to the chosen remote agent. If no remote agents are available,
        it falls back to a **local Gemini-powered analytics agent** that performs
        text summarisation and returns JSON.
        """
        # Debug logging so you can see what's going on
        print("[RoutingAgent] send_message called with agent_name:", agent_name)
        print(
            "[RoutingAgent] Available remote agents:",
            list(self.remote_agent_connections.keys()),
        )

        # === Path 1: Use remote A2A agents if any are configured ===
        if self.remote_agent_connections:
            if agent_name not in self.remote_agent_connections:
                # Fallback to "first" remote agent if the requested name is unknown
                fallback_agent = next(iter(self.remote_agent_connections.keys()))
                print(
                    f"[RoutingAgent] FALLBACK: '{agent_name}' not found, "
                    f"using '{fallback_agent}' instead."
                )
                agent_name = fallback_agent

            state = tool_context.state
            state["active_agent"] = agent_name
            client = self.remote_agent_connections[agent_name]

            if not client:
                raise ValueError(f"Client not available for {agent_name}")

            task_id = state["task_id"] if "task_id" in state else None

            if "context_id" in state:
                context_id = state["context_id"]
            else:
                context_id = str(uuid.uuid4())

            message_id = ""
            metadata: dict[str, Any] = {}
            if "input_message_metadata" in state:
                metadata.update(**state["input_message_metadata"])
                if "message_id" in state["input_message_metadata"]:
                    message_id = state["input_message_metadata"]["message_id"]
            if not message_id:
                message_id = str(uuid.uuid4())

            payload: dict[str, Any] = {
                "message": {
                    "role": "user",
                    "parts": [
                        {"type": "text", "text": task},
                    ],
                    "messageId": message_id,
                },
            }

            if task_id:
                payload["message"]["taskId"] = task_id

            if context_id:
                payload["message"]["contextId"] = context_id

            message_request = SendMessageRequest(
                id=message_id, params=MessageSendParams.model_validate(payload)
            )
            send_response: SendMessageResponse = await client.send_message(
                message_request=message_request
            )

            logger.debug(
                "send_response %s",
                send_response.model_dump_json(exclude_none=True, indent=2),
            )

            if not isinstance(send_response.root, SendMessageSuccessResponse):
                logger.debug("received non-success response. Aborting get task ")
                return None

            if not isinstance(send_response.root.result, Task):
                logger.debug("received non-task response. Aborting get task ")
                return None

            return send_response.root.result

        # === Path 2: No remote agents -> local Gemini analytics summariser ===
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY must be set for local analytics fallback."
            )

        client = genai.Client(api_key=api_key)

        prompt = (
            "You are an analytics agent that summarises internal business updates.\n\n"
            "The host routing agent is delegating this task to you.\n\n"
            "User task:\n"
            f"{task}\n\n"
            "You MUST respond with **only** valid JSON of the form:\n"
            '{ "summary_bullets": ["...","...","..."], "tone": "<one_word_tone>" }\n'
            "Do not include any extra text, explanations, or code fences.\n"
        )

        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
        )

        text = (getattr(response, "text", "") or "").strip()

        # Try to parse as JSON; if it fails, fall back to a simple wrapper
        try:
            data = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    data = {
                        "summary_bullets": [text],
                        "tone": "neutral",
                    }
            else:
                data = {
                    "summary_bullets": [text],
                    "tone": "neutral",
                }

        return data


def _get_initialized_routing_agent_sync() -> Agent:
    """Synchronously creates and initializes the RoutingAgent."""

    async def _async_main() -> Agent:
        routing_agent_instance = await RoutingAgent.create(
            remote_agent_addresses=[
                # You can still configure remote agents via env if you want
                os.getenv("AIR_AGENT_URL", "http://localhost:10011"),
                os.getenv("WEA_AGENT_URL", "http://localhost:10012"),
                os.getenv("ANALYTICS_AGENT_URL", "http://localhost:10013"),
            ]
        )
        return routing_agent_instance.create_agent()

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            logger.debug(
                "Warning: Could not initialize RoutingAgent with asyncio.run(): %s. "
                "This can happen if an event loop is already running (e.g., in Jupyter). "
                "Consider initializing RoutingAgent within an async function in your application.",
                e,
            )
        raise


root_agent = _get_initialized_routing_agent_sync()
