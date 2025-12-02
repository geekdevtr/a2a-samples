# dice_comment_agent/__main__.py

from __future__ import annotations

import logging

import click
import uvicorn

from a2a.server.apps import A2ARESTFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TransportProtocol,
)

from .agent_executor import DiceCommentAgentExecutor  # type: ignore[import-untyped]


logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10102)
def main(host: str, port: int) -> None:
    """Start the deterministic dice comment agent over HTTP/JSON (REST)."""

    skills = [
        AgentSkill(
            id='c816ad3e-6e5a-4e68-b153-ac8f0e6a9d11',
            name='Dice Commentator',
            description='Generates deterministic commentary for dice roll results.',
            tags=['dice', 'comment'],
            examples=[
                '{"roll": 19, "sides": 20, "is_prime": true}',
                '{"roll": 3, "sides": 6, "is_prime": false}',
            ],
        ),
    ]

    agent_card = AgentCard(
        name='Dice Comment Agent',
        description='Adds deterministic commentary to dice roll results.',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=False),
        skills=skills,
        preferred_transport=TransportProtocol.http_json,
    )

    executor = DiceCommentAgentExecutor()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    app = A2ARESTFastAPIApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    ).build()

    logging.info(f'Dice Comment Agent listening on http://{host}:{port}')
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
