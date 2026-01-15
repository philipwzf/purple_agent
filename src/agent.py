import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from prompts import ACTION_SEQ_PROMPT_GOTO, SUBGOAL_PROMPT


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = 60
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "4000"))
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.6"))
OPENROUTER_TOP_P = float(os.getenv("OPENROUTER_TOP_P", "1.0"))
OPENROUTER_FREQUENCY_PENALTY = float(os.getenv("OPENROUTER_FREQUENCY_PENALTY", "0.0"))
OPENROUTER_PRESENCE_PENALTY = float(os.getenv("OPENROUTER_PRESENCE_PENALTY", "0.0"))

REPO_ROOT = Path(__file__).resolve().parents[2]
SAFETY_RULES_PATH = REPO_ROOT / "SENTINEL_code" / "safety_rules_object.json"


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

    async def _query_llm(
        self, client: httpx.AsyncClient, system_prompt: str, user_prompt: str
    ) -> str | None:
        if not OPENROUTER_API_KEY:
            return None

        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": OPENROUTER_TEMPERATURE,
            "max_tokens": OPENROUTER_MAX_TOKENS,
            "top_p": OPENROUTER_TOP_P,
            "frequency_penalty": OPENROUTER_FREQUENCY_PENALTY,
            "presence_penalty": OPENROUTER_PRESENCE_PENALTY,
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def _format_objects_for_prompt(self, objects: list[dict[str, Any]], filter_visible: bool) -> str:
        if not objects:
            return "No objects available.\n"

        prompt_section = ""
        for obj in objects:
            if filter_visible and not obj.get("visible"):
                continue

            obj_type = obj.get("objectType", "Unknown")
            obj_id = obj.get("objectId", "Unknown")
            line = f"- {obj_type} ({obj_id}): "

            properties = []
            if obj.get("pickupable") and not obj.get("isPickedUp"):
                properties.append("pickupable")
            elif obj.get("isPickedUp"):
                properties.append("being held")
            if obj.get("receptacle"):
                properties.append("receptacle")
            if obj.get("openable"):
                properties.append(f"openable ({'open' if obj.get('isOpen') else 'closed'})")
            if obj.get("toggleable"):
                if "Candle" in str(obj_id):
                    properties.append("toggleable (it is on)")
                else:
                    properties.append(f"toggleable ({'on' if obj.get('isToggled') else 'off'})")
            if obj.get("dirtyable"):
                properties.append(f"{'dirty' if obj.get('isDirty') else 'clean'}")
            if obj.get("cookable"):
                properties.append(f"{'cooked' if obj.get('isCooked') else 'uncooked'}")
            if obj.get("isSliced"):
                properties.append("sliced")

            temperature = obj.get("temperature")
            if temperature and temperature != "RoomTemp":
                properties.append(f"temperature: {temperature}")

            parent_receptacles = obj.get("parentReceptacles") or []
            if parent_receptacles:
                properties.append(f"in receptacle: {', '.join(parent_receptacles)}")

            receptacle_ids = obj.get("receptacleObjectIds") or []
            if receptacle_ids:
                properties.append(f"contains {len(receptacle_ids)} items, including:")
                for rec_id in receptacle_ids:
                    properties.append(f" {rec_id};")

            line += ", ".join(properties) if properties else "no special properties"
            line += f" at {obj.get('position')}\n"

            if not filter_visible:
                visibility = "visible" if obj.get("visible") else "not visible"
                line += f" [{visibility}]"

            prompt_section += line + "\n"

        return prompt_section

    def _load_safety_constraint(self, objects: list[dict[str, Any]]) -> str:
        try:
            with SAFETY_RULES_PATH.open("r", encoding="utf-8") as handle:
                safety_data = json.load(handle)
        except FileNotFoundError:
            return "- no safety constraints"

        constraints: set[str] = set()
        seen: set[str] = set()
        for obj in objects:
            obj_type = obj.get("objectType")
            if not obj_type or obj_type in seen:
                continue
            for rule in safety_data.get(obj_type, []):
                constraints.add(rule)
            seen.add(obj_type)

        return "- " + ("\n ".join(sorted(constraints)) if constraints else "no safety constraints")

    def _extract_json_blob(self, text: str, start: str, end: str) -> str | None:
        if not text:
            return None

        content = text.strip()
        if "```" in content:
            start_marker = "```json"
            start_idx = content.find(start_marker)
            if start_idx != -1:
                start_idx += len(start_marker)
            else:
                start_idx = content.find("```")
                if start_idx != -1:
                    start_idx += len("```")
            if start_idx != -1:
                end_idx = content.find("```", start_idx)
                if end_idx != -1:
                    content = content[start_idx:end_idx].strip()
                else:
                    content = content[start_idx:].strip()

        start_idx = content.find(start)
        end_idx = content.rfind(end)
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return None
        return content[start_idx : end_idx + 1]

    def _parse_subgoals(self, response_text: str | None) -> list[str]:
        if not response_text:
            return []

        blob = self._extract_json_blob(response_text, "{", "}")
        if not blob:
            return []
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            return []

        if isinstance(data, dict) and isinstance(data.get("subgoals"), list):
            return data["subgoals"]
        return []

    def _parse_action_sequence(self, response_text: str | None) -> list[dict[str, Any]]:
        if not response_text:
            return []

        parsed: Any = None
        content = response_text.strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None

        actions: Any = None
        if isinstance(parsed, list):
            actions = parsed
        elif isinstance(parsed, dict):
            actions = parsed.get("actions") or parsed.get("plan")

        if actions is None:
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                try:
                    actions = json.loads(json_match.group())
                except json.JSONDecodeError:
                    actions = None

        if not isinstance(actions, list):
            return []

        normalized: list[dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_name = action.get("action") or action.get("Action")
            if not action_name:
                continue
            normalized_action = dict(action)
            normalized_action["action"] = action_name
            if "object_id" not in normalized_action and "objectId" in normalized_action:
                normalized_action["object_id"] = normalized_action["objectId"]
            if "receptacle_id" not in normalized_action and "receptacleObjectId" in normalized_action:
                normalized_action["receptacle_id"] = normalized_action["receptacleObjectId"]
            normalized.append(normalized_action)

        return normalized

    def _create_action_prompt(
        self, subgoals: list[str], scene_info: dict[str, Any], goto: bool = True
    ) -> str:
        agent = scene_info.get("agent") or {}
        prompt = f"""
## Current Agent Status:
- Position: {agent.get('position')}
- Rotation: {agent.get('rotation')}
- Holding: {scene_info.get('inventoryObjects') or 'Nothing'}

## Subgoals to Achieve (in order):
"""

        if subgoals:
            for idx, subgoal in enumerate(subgoals, 1):
                prompt += f"{idx}. {subgoal}\n"
        else:
            prompt += "No valid subgoals found - will generate basic plan\n"

        prompt += "\n## Relevant Objects in Scene:\n"
        prompt += self._format_objects_for_prompt(scene_info.get("objects") or [], filter_visible=False)

        if goto:
            prompt += """
## Available Actions:
- Navigation: GotoLocation <object_id>
- Object Interaction: PickupObject <object_id>, PutObject <object_id, object_id>
- Container Actions: OpenObject <object_id>, CloseObject <object_id>
- Appliance Actions: ToggleObjectOn <object_id>, ToggleObjectOff <object_id>
- Object Modification: SliceObject <object_id>
- Task Completion: stop

## Planning Rules:
1. **Achieve subgoals in the given order** - each subgoal builds on the previous ones
2. **Navigation first** - Navigate to objects before interacting with them
3. **Object interactions** - Pick up objects with PickupObject before moving them
4. **State changes** - Use appropriate actions to change object states
5. **Exact object IDs** - Use the exact object IDs from the subgoals
6. **Find objects first** - Navigate around to find objects if needed
7. **Complete all subgoals** - Don't stop until all subgoals are achieved
8. **PutObject requires two object_ids** - the object being placed and the receptacle
9. **Pick up an object in a closed receptacle** - If the receptacle is openable and closed, open it first before picking up the object inside.

## Output Format:
Generate a JSON array of actions that will achieve the subgoals in order:
[
    {"action": "GotoLocation", "object_id": "AlarmClock|+01.65|+00.80|-01.28"},
    {"action": "PickupObject", "object_id": "AlarmClock|+01.65|+00.80|-01.28"},
    {"action": "GotoLocation", "object_id": "DeskLamp|-01.31|+01.23|-00.99"},
    {"action": "ToggleObjectOn", "object_id": "DeskLamp|-01.31|+01.23|-00.99"},
    {"action": "GotoLocation", "object_id": "CounterTop|+02.10|+00.90|+01.50"},
    {"action": "PutObject", "object_id": "AlarmClock|+01.65|+00.80|-01.28", "receptacle_id": "CounterTop|+02.10|+00.90|+01.50"},
    {"action": "stop"}
]

Generate the action sequence to achieve all subgoals:
"""
            return prompt

        prompt += """
## Available Actions:
- Navigation: MoveAhead, MoveBack, MoveLeft, MoveRight, RotateLeft, RotateRight, LookUp, LookDown
- Object Interaction: PickupObject <object_id>, PutObject <object_id, object_id>
- Container Actions: OpenObject <object_id>, CloseObject <object_id>
- Appliance Actions: ToggleObjectOn <object_id>, ToggleObjectOff <object_id>
- Object Modification: SliceObject <object_id>

## Planning Rules:
1. **Achieve subgoals in the given order** - each subgoal builds on the previous ones
2. **Navigation first** - Navigate to objects before interacting with them
3. **Object interactions** - Pick up objects with PickupObject before moving them
4. **State changes** - Use appropriate actions to change object states
5. **Exact object IDs** - Use the exact object IDs from the subgoals
6. **Find objects first** - Navigate around to find objects if needed
7. **Complete all subgoals** - Don't stop until all subgoals are achieved
8. **PutObject requires two object_ids** - the object being placed and the receptacle


## Output Format:
Generate a JSON array of actions that will achieve the subgoals in order:

[
    {"action": "MoveAhead"},
    {"action": "RotateRight"},
    {"action": "PickupObject", "object_id": "AlarmClock|+01.65|+00.80|-01.28"},
    {"action": "MoveLeft"},
    {"action": "ToggleObjectOn", "object_id": "DeskLamp|-01.31|+01.23|-00.99"},
    {"action": "MoveAhead"},
    {"action": "MoveAhead"},
    {"action": "PutObject", "object_id": "AlarmClock|+01.65|+00.80|-01.28", "receptacle_id": "CounterTop|+02.10|+00.90|+01.50"},
    {"action": "stop"}
]

Generate the action sequence to achieve all subgoals:
"""
        return prompt

    async def _plan_actions(self, trials: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        if not OPENROUTER_API_KEY:
            return self._fallback_actions(trials)

        actions: dict[str, list[dict[str, Any]]] = {}
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            for trial in trials:
                trial_id = str(trial.get("trial_id") or "").strip()
                if not trial_id:
                    continue

                goal_instruction = str(trial.get("goal_instruction") or "").strip()
                scene_info = trial.get("metadata") or {}
                objects = scene_info.get("objects") or []

                user_prompt = f"""
Task: {goal_instruction}

Scene Information:
- Agent Position: {(scene_info.get('agent') or {}).get('position')}
- Agent Rotation: {(scene_info.get('agent') or {}).get('rotation')}

Available Objects:
{self._format_objects_for_prompt(objects, filter_visible=False)}

Generate subgoals for the given task using only the provided objects. Format your output in JSON as specified above. DO NOT include any explanations or additional text. DO NOT wrap the JSON in markdown.
"""
                try:
                    subgoal_text = await self._query_llm(
                        client,
                        SUBGOAL_PROMPT,
                        user_prompt,
                    )
                    subgoals = self._parse_subgoals(subgoal_text)
                except Exception:
                    subgoals = []

                plan_prompt = self._create_action_prompt(subgoals, scene_info, goto=True)
                try:
                    plan_text = await self._query_llm(
                        client,
                        ACTION_SEQ_PROMPT_GOTO,
                        plan_prompt,
                    )
                    plan_actions = self._parse_action_sequence(plan_text)
                except Exception:
                    plan_actions = []

                actions[trial_id] = plan_actions or [{"action": "Done"}]

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
