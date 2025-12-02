# dice_comment_agent/agent_executor.py

from __future__ import annotations

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_task
from a2a.utils.errors import ServerError

from .agent import build_comment


class DiceCommentAgentExecutor(AgentExecutor):
    """AgentExecutor that adds deterministic commentary for dice rolls."""

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # Extract plain text from the incoming A2A message.
        query_text = context.get_user_input()

        # Ensure we have a Task to attach artifacts to.
        task: Task | None = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        # (Optional) mark as working – not strictly needed, but nice for traces.
        await updater.update_status(TaskState.working)

        # Build deterministic comment.
        comment = build_comment(query_text or '')

        # Attach the comment as an artifact and mark Task complete.
        await updater.add_artifact(
            [Part(root=TextPart(text=comment))],
            name='comment',
        )
        await updater.complete()

    async def cancel(
        self,
        request: RequestContext,
        event_queue: EventQueue,
    ) -> Task | None:
        # This simple agent does not support cancellation.
        raise ServerError(error=UnsupportedOperationError())
