import asyncio
import time
import uuid

from a2a.client import ClientConfig, ClientFactory
from a2a.client.client_task_manager import ClientTaskManager
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    Message,
    Role,
    TextPart,
    TransportProtocol,
)


factory = ClientFactory(ClientConfig())


async def send_text(agent_card: AgentCard, text: str) -> str:
    """Send a plain-text message to an A2A agent and return its final text."""
    client = factory.create(agent_card)

    msg = Message(
        kind="message",
        role=Role.user,
        message_id=uuid.uuid4().hex,
        parts=[TextPart(text=text)],
    )

    task_mgr = ClientTaskManager()
    last_message: Message | None = None

    async for event in client.send_message(msg):  # type: ignore[attr-defined]
        # Some transports wrap the event in a tuple
        if isinstance(event, tuple):
            event = event[0]
        await task_mgr.process(event)
        if isinstance(event, Message):
            last_message = event

    # Prefer Task artifacts (normal path)
    task = task_mgr.get_task()
    if task and task.artifacts:
        for artifact in reversed(task.artifacts):
            if artifact.parts:
                for part in reversed(artifact.parts):
                    if hasattr(part, "root") and hasattr(part.root, "text"):
                        return part.root.text  # type: ignore[attr-defined]

    # Fallback: plain Message
    if last_message and last_message.parts:
        pieces = []
        for part in last_message.parts:
            if hasattr(part, "text") and part.text:
                pieces.append(part.text)
        return "\n".join(pieces)

    return ""


# --- AgentCards matching your running servers --------------------------------

# This mirrors dice_agent_rest/__main__.py (HTTP JSON / REST)
dice_rest_card = AgentCard(
    name="Dice Agent",
    description="An agent that can roll arbitrary dice and answer if numbers are prime",
    url="http://localhost:10101/",  # base URL, REST transport
    version="1.0.0",
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=True),
    skills=[],  # skills not needed for the client to call
    preferred_transport=TransportProtocol.http_json,
)

# This mirrors dice_comment_agent using JSON-RPC at /a2a/v1
comment_card = AgentCard(
    name="Dice Comment Agent",
    description="Comments on dice outcomes in a friendly way.",
    url="http://localhost:10102/a2a/v1",  # JSON-RPC endpoint
    version="1.0.0",
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[],
    preferred_transport=TransportProtocol.jsonrpc,
)


async def main() -> None:
    # 1) Ask DiceAgent (REST) to roll and check primeness
    start = time.perf_counter()
    dice_response = await send_text(
        dice_rest_card,
        "Roll a 10-sided die and tell me if the result is prime.",
    )
    mid = time.perf_counter()

    print("=== DiceAgent (REST) response ===")
    print(dice_response, "\n")

    # 2) Send that result to the Dice Comment Agent
    comment_prompt = (
        "Here is the dice result and analysis:\n"
        f"{dice_response}\n\n"
        "Give a short, friendly comment for the player."
    )

    comment_response = await send_text(
        comment_card,
        comment_prompt,
    )
    end = time.perf_counter()

    print("=== DiceCommentAgent response ===")
    print(comment_response, "\n")

    # 3) Simple latency numbers (REST + JSON-RPC combo)
    print(f"DiceAgent latency:     {(mid - start) * 1000:.1f} ms")
    print(f"CommentAgent latency:  {(end - mid) * 1000:.1f} ms")
    print(f"End-to-end latency:    {(end - start) * 1000:.1f} ms")


if __name__ == "__main__":
    asyncio.run(main())
