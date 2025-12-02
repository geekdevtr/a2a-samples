# dice_comment_agent/agent.py

from __future__ import annotations

import json
from typing import Any, Tuple


def _safe_load_json(text: str) -> Tuple[bool, Any]:
    """Try to parse JSON; return (ok, obj_or_None)."""
    try:
        return True, json.loads(text)
    except json.JSONDecodeError:
        return False, None


def _comment_for_roll(roll: int, sides: int | None, is_prime: bool | None) -> str:
    """Deterministic, rule-based comment for a single dice roll."""
    # Base description
    if sides and sides > 0:
        base = f'You rolled {roll} on a d{sides}.'
    else:
        base = f'You rolled {roll}.'

    # Quality of the roll
    quality = ''
    if sides and sides > 0:
        if roll == sides:
            quality = ' That is the maximum possible roll – a critical success!'
        elif roll == 1:
            quality = ' That is the minimum roll – quite unlucky this time.'
        elif roll >= int(0.9 * sides):
            quality = ' That is a very high roll – luck seems to be on your side.'
        elif roll <= max(1, int(0.2 * sides)):
            quality = ' That is a low roll – maybe the next one will be better.'
        else:
            quality = ' That is a solid mid-range result.'
    else:
        quality = ' That looks like a reasonable roll.'

    # Prime flavour
    prime_note = ''
    if is_prime is True:
        prime_note = ' It is also a prime number, which feels a bit special.'
    elif is_prime is False:
        prime_note = ' It is not prime, but it still counts for the game.'

    return base + quality + prime_note


def build_comment(raw_text: str) -> str:
    """Entry point for the comment agent.

    Expected primary format (JSON):

        {
          "roll": 17,
          "sides": 20,
          "is_prime": true,
          "history": [17, 3, 19]   # optional
        }

    Any non-JSON input falls back to a generic, deterministic comment.
    """
    ok, payload = _safe_load_json(raw_text)

    if not ok or not isinstance(payload, dict):
        # Fallback: deterministic but generic.
        snippet = raw_text.strip()
        if len(snippet) > 40:
            snippet = snippet[:37] + '...'
        return (
            "I received a dice message"
            + (f' (“{snippet}”)' if snippet else '')
            + ". Looks like a roll result; keep going!"
        )

    roll = payload.get('roll')
    sides = payload.get('sides')
    is_prime = payload.get('is_prime')

    # Ensure roll is an int we can reason about.
    try:
        roll_int = int(roll)
    except (TypeError, ValueError):
        return 'I could not find a valid numeric roll in the payload.'

    sides_int: int | None = None
    if sides is not None:
        try:
            sides_int = int(sides)
        except (TypeError, ValueError):
            sides_int = None

    # Normalise is_prime to bool | None
    if isinstance(is_prime, bool):
        is_prime_bool: bool | None = is_prime
    else:
        is_prime_bool = None

    comment = _comment_for_roll(roll_int, sides_int, is_prime_bool)

    # Very light use of history (optional)
    history = payload.get('history')
    if isinstance(history, list) and len(history) >= 3:
        try:
            nums = [int(x) for x in history[-3:]]
        except (TypeError, ValueError):
            nums = []
        if nums and all(n >= 0.8 * (sides_int or max(nums)) for n in nums):
            comment += ' Your recent streak has been consistently high – impressive!'

    return comment
