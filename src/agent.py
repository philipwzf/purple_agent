import json
from typing import Any

import json
import os
from typing import Any

import httpx
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = 60


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here

    @staticmethod
    def _fallback_actions(trials: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        actions: dict[str, list[dict[str, Any]]] = {}
        for trial in trials:
            trial_id = str(trial.get("trial_id") or "").strip()
            if not trial_id:
                continue
            actions[trial_id] = [{"action": "Done"}]
        return actions

    async def _plan_actions(self, trials: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        if not OPENROUTER_API_KEY:
            return self._fallback_actions(trials)

        user_payload = {
            "trials": [
                {
                    "trial_id": trial.get("trial_id"),
                    "goal_instruction": trial.get("goal_instruction"),
                    "metadata": trial.get("metadata"),
                }
                for trial in trials
            ]
        }
        system_prompt = (
            "You are an AI2-THOR action planner. "
            "Return ONLY a JSON object mapping trial_id to a list of action dicts. "
            "Each action dict must include 'action' and may include 'object_id', "
            "'receptacle_id', or 'agentId'."
        )
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, indent=2)},
            ],
            "temperature": 0.5,
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return self._fallback_actions(trials)

        if not isinstance(parsed, dict):
            return self._fallback_actions(trials)

        actions: dict[str, list[dict[str, Any]]] = {}
        for trial in trials:
            trial_id = str(trial.get("trial_id") or "").strip()
            if not trial_id:
                continue
            value = parsed.get(trial_id)
            actions[trial_id] = value if isinstance(value, list) else [{"action": "Done"}]

        return actions

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        input_text = get_message_text(message)

        payload = None
        try:
            payload = json.loads(input_text)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict) and isinstance(payload.get("trials"), list):
            trials = [trial for trial in payload["trials"] if isinstance(trial, dict)]
            await updater.update_status(
                TaskState.working, new_agent_text_message("Planning actions...")
            )
            actions_payload = {"actions": await self._plan_actions(trials)}
            response_text = json.dumps(actions_payload)
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=response_text))],
                name="Actions",
            )
            return

        # Replace this example code with your agent logic

        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=input_text))],
            name="Echo",
        )
