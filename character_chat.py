"""
Character Chat Module for Enliven
Allows users to have voice conversations with book characters using OpenRouter + ElevenLabs
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs import ElevenLabs
import re

load_dotenv()

# Module-level client for character extraction
_extraction_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
_extraction_model = "google/gemini-3-flash-preview"


@dataclass
class Character:
    """Represents a book character with their personality and voice."""

    name: str
    description: str
    personality: str
    backstory: str
    gender: str = "unknown"  # "male", "female", or "unknown"
    voice_id: str | None = None  # Auto-assigned if not provided


# =============================================================================
# STUB: This would be populated by the audiobook generation team
# =============================================================================
def get_characters_from_audiobook() -> list[Character]:
    """
    STUB: Returns characters extracted from the audiobook.

    The audiobook team would implement this to return characters
    they've identified while generating the audiobook.
    """
    return [
        Character(
            name="Sherlock Holmes",
            description="Tall, thin detective with sharp features. Male, middle-aged.",
            personality="Brilliant, arrogant, speaks precisely. Loves puzzles.",
            backstory="Consulting detective in London. Works with Dr. Watson.",
        ),
        Character(
            name="Elizabeth Bennet",
            description="Young woman with dark eyes and lively disposition. Female, early 20s.",
            personality="Witty, quick to laugh, speaks with gentle irony.",
            backstory="Second of five Bennet daughters. Values independence.",
        ),
    ]


def _select_representative_dialogues(dialogues: list[str], max_samples: int = 30) -> list[str]:
    """Select representative dialogues from beginning, middle, and end."""
    if len(dialogues) <= max_samples:
        return dialogues
    n = len(dialogues)
    # Take from beginning, middle, and end for better coverage
    indices = (
        list(range(0, min(10, n))) +
        list(range(n // 2 - 5, n // 2 + 5)) +
        list(range(max(0, n - 10), n))
    )
    return [dialogues[i] for i in sorted(set(indices)) if i < n]


def _generate_character_profile(name: str, dialogues: list[str]) -> dict:
    """Use LLM to create character profile from dialogue + literary knowledge."""
    selected = _select_representative_dialogues(dialogues, max_samples=30)
    sample = "\n".join(selected)

    prompt = f"""Create a detailed character profile for "{name}".

Use these dialogue samples from the text (from beginning, middle, and end of the story):
{sample}

IMPORTANT: If this is a well-known literary character (e.g., Sherlock Holmes, Elizabeth Bennet, Hamlet),
use your knowledge of that character to enrich the profile with accurate details from the original work.
Combine textual evidence with literary knowledge for the best result.

Analyze the dialogue for:
- Speech patterns (formal/informal, sentence structure, vocabulary level)
- Emotional range (how they express joy, anger, sadness, etc.)
- Distinctive verbal tics or catchphrases
- How their speech changes in different situations

Return JSON only with string values (not nested objects):
{{"gender": "male or female (infer from name, context, pronouns)", "description": "physical appearance and voice quality as a single string", "personality": "speaking style, mannerisms, speech patterns, emotional expression, and character traits as a single string", "backstory": "background, motivations, relationships, and character arc as a single string"}}"""

    response = _extraction_client.chat.completions.create(
        model=_extraction_model,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content or "{}"
    # Extract JSON from response
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            # Fallback if JSON parsing fails
            return {
                "description": f"A character named {name}",
                "personality": "Speaks naturally based on the dialogue",
                "backstory": "A character from the story",
            }

    # Handle nested dicts - extract first value if dict is returned instead of string
    for key in ["description", "personality", "backstory"]:
        if key in data and isinstance(data[key], dict):
            # Get first value from nested dict
            data[key] = next(iter(data[key].values()), f"Unknown {key}")

    return data


def _is_unknown_character(name: str) -> bool:
    """Check if a character name represents an unknown/generic speaker."""
    normalized = name.strip().lower()
    unknown_patterns = ["unknown", "narrator", "unidentified", "speaker", "voice"]
    return any(pattern in normalized for pattern in unknown_patterns)


def _preprocess_character_names(names: list[str]) -> dict[str, str]:
    """Quick substring-based merging before LLM call for obvious matches."""
    if not names:
        return {}

    # Sort by length descending so longer names are processed first
    sorted_names = sorted(names, key=len, reverse=True)
    canonical_map: dict[str, str] = {}  # Maps normalized form to canonical name

    for name in sorted_names:
        lower = name.lower().strip()
        matched = False

        for existing_name, canonical_lower in list(canonical_map.items()):
            # Check if one is substring of the other
            if lower in canonical_lower or canonical_lower in lower:
                # Use the longer name as canonical
                if len(name) > len(existing_name):
                    # Update canonical to use this longer name
                    canonical_map[name] = lower
                    # Update all names that pointed to the old canonical
                    for k, v in list(canonical_map.items()):
                        if v == canonical_lower:
                            canonical_map[k] = lower
                else:
                    canonical_map[name] = canonical_lower
                matched = True
                break

        if not matched:
            canonical_map[name] = lower

    # Build final mapping: name -> canonical name (the longest one)
    result: dict[str, str] = {}
    for name in names:
        canonical_lower = canonical_map.get(name, name.lower())
        # Find the longest name that maps to this canonical form
        canonical_name = max(
            (n for n in names if canonical_map.get(n) == canonical_lower),
            key=len,
            default=name
        )
        result[name] = canonical_name

    return result


def _normalize_character_names(
    character_dialogue: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Merge character name variants (e.g., 'Harry' and 'Harry Potter')."""
    # Build list of all names
    names = list(character_dialogue.keys())

    # Skip LLM call for small character sets (pre-processing handles obvious cases)
    if len(names) <= 5:
        return character_dialogue

    # Use LLM to identify which names refer to the same character
    prompt = f"""Given these character names from a book, group names that refer to the same character.
Return JSON: {{"groups": [["name1", "name2"], ["name3"]]}}

Names: {names}

Rules:
- Group names that clearly refer to the same person (e.g., "Harry" and "Harry Potter")
- Keep separate if they're different characters
- Use the most complete name as the first item in each group
- Return ONLY valid JSON"""

    response = _extraction_client.chat.completions.create(
        model=_extraction_model,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Find JSON object with balanced braces
        start = content.find("{")
        if start == -1:
            return character_dialogue  # Fallback: no normalization

        depth = 0
        end = start
        for i, char in enumerate(content[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        try:
            data = json.loads(content[start:end])
        except json.JSONDecodeError:
            return character_dialogue  # Fallback: no normalization

    # Merge dialogue based on groups
    merged = {}
    for group in data.get("groups", []):
        if not group:
            continue
        canonical_name = group[0]  # Use first (most complete) name
        merged[canonical_name] = []
        for name in group:
            if name in character_dialogue:
                merged[canonical_name].extend(character_dialogue[name])

    return merged


def extract_characters_from_text(text: str) -> list[Character]:
    """Extract characters from book text using textParser and generate profiles."""
    from textParser import parse_text

    segments = parse_text(text)

    # Collect dialogue by character
    character_dialogue: dict[str, list[str]] = {}
    for seg in segments:
        if seg["speaker_type"] == "character" and seg.get("character_name"):
            name = seg["character_name"]
            if not _is_unknown_character(name):
                character_dialogue.setdefault(name, []).append(seg["text"])

    # Pre-process character names with deterministic substring matching
    # This handles obvious cases like "Harry" -> "Harry Potter" before LLM call
    preprocess_map = _preprocess_character_names(list(character_dialogue.keys()))
    preprocessed_dialogue: dict[str, list[str]] = {}
    for name, dialogues in character_dialogue.items():
        canonical = preprocess_map.get(name, name)
        preprocessed_dialogue.setdefault(canonical, []).extend(dialogues)
    character_dialogue = preprocessed_dialogue

    # Use LLM normalization for remaining ambiguous cases
    character_dialogue = _normalize_character_names(character_dialogue)

    # Generate character profiles using LLM in parallel
    characters = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_name = {
            executor.submit(_generate_character_profile, name, dialogues): name
            for name, dialogues in character_dialogue.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            profile = future.result()
            characters.append(
                Character(
                    name=name,
                    description=profile.get("description", f"A character named {name}"),
                    personality=profile.get("personality", "Speaks naturally"),
                    backstory=profile.get("backstory", "A character from the story"),
                    gender=profile.get("gender", "unknown").lower(),
                )
            )

    return characters


@dataclass
class CostTracker:
    """Tracks API usage costs."""

    openrouter_input_tokens: int = 0
    openrouter_output_tokens: int = 0
    elevenlabs_characters: int = 0

    # Pricing (per token/character)
    OPENROUTER_INPUT_PRICE = 0.10 / 1_000_000  # $0.10 per 1M tokens (Gemini Flash)
    OPENROUTER_OUTPUT_PRICE = 0.40 / 1_000_000  # $0.40 per 1M tokens (Gemini Flash)
    ELEVENLABS_CHAR_PRICE = 0.30 / 1_000  # ~$0.30 per 1K chars (varies by plan)

    @property
    def openrouter_cost(self) -> float:
        return (
            self.openrouter_input_tokens * self.OPENROUTER_INPUT_PRICE
            + self.openrouter_output_tokens * self.OPENROUTER_OUTPUT_PRICE
        )

    @property
    def elevenlabs_cost(self) -> float:
        return self.elevenlabs_characters * self.ELEVENLABS_CHAR_PRICE

    @property
    def total_cost(self) -> float:
        return self.openrouter_cost + self.elevenlabs_cost

    def __str__(self) -> str:
        return (
            f"LLM: {self.openrouter_input_tokens}+{self.openrouter_output_tokens} tokens (${self.openrouter_cost:.4f}) | "
            f"TTS: {self.elevenlabs_characters} chars (${self.elevenlabs_cost:.4f}) | "
            f"Total: ${self.total_cost:.4f}"
        )


class CharacterChat:
    """Handles conversations with book characters."""

    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.model = "google/gemini-3-flash-preview"

        self.elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        self.current_character: Character | None = None
        self.conversation_history: list[dict] = []
        self._voices_cache: list[dict] | None = None
        self.costs = CostTracker()

    def _get_available_voices(self) -> list[dict]:
        """Fetch and cache available ElevenLabs voices."""
        if self._voices_cache is None:
            response = self.elevenlabs.voices.get_all()
            self._voices_cache = []
            for v in response.voices:
                labels = v.labels or {}
                self._voices_cache.append(
                    {
                        "voice_id": v.voice_id,
                        "name": v.name,
                        "gender": labels.get("gender", "unknown"),
                        "age": labels.get("age", "unknown"),
                        "accent": labels.get("accent", "unknown"),
                        "description": labels.get("description", ""),
                        "use_case": labels.get("use_case", ""),
                    }
                )
        return self._voices_cache

    def _pick_voice_for_character(self, character: Character) -> str:
        """Use LLM to pick the best fitting voice for a character."""
        voices = self._get_available_voices()

        # Use explicit gender from character profile
        char_gender = character.gender.lower() if character.gender else "unknown"

        # Filter voices by gender first for better matching
        gender_matched = [
            v for v in voices if v["gender"].lower() == char_gender
        ] if char_gender in ["male", "female"] else voices

        # Use gender-matched voices if available
        available_voices = gender_matched if gender_matched else voices

        voices_desc = "\n".join(
            [
                f"- {v['name']} (id: {v['voice_id']}): {v['gender']}, {v['age']}, {v['accent']} accent. {v['description']}"
                for v in available_voices
            ]
        )

        prompt = f"""Pick the best voice for this character. Return ONLY the voice_id, nothing else.

CHARACTER:
Name: {character.name}
Gender: {char_gender}
Description: {character.description}
Personality: {character.personality}

AVAILABLE VOICES:
{voices_desc}

Return only the voice_id that best matches the character's gender, age, and personality."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        # Track costs
        if response.usage:
            self.costs.openrouter_input_tokens += response.usage.prompt_tokens
            self.costs.openrouter_output_tokens += response.usage.completion_tokens

        voice_id = response.choices[0].message.content.strip()

        # Validate voice_id exists
        valid_ids = [v["voice_id"] for v in voices]
        if voice_id not in valid_ids:
            # Fallback to first gender-matched voice
            if gender_matched:
                return gender_matched[0]["voice_id"]
            return voices[0]["voice_id"]

        return voice_id

    def set_character(self, character: Character):
        """Set the character to chat with, auto-assigning voice if needed."""
        if character.voice_id is None:
            print(f"[Finding voice for {character.name}...]")
            character.voice_id = self._pick_voice_for_character(character)
            # Find voice name for display
            voices = self._get_available_voices()
            voice_name = next(
                (v["name"] for v in voices if v["voice_id"] == character.voice_id),
                "Unknown",
            )
            print(f"[Assigned voice: {voice_name}]")

        self.current_character = character
        self.conversation_history = []

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the character."""
        char = self.current_character
        return f"""You are {char.name} from a story. Stay completely in character.

CHARACTER:
- Name: {char.name}
- Description: {char.description}
- Personality: {char.personality}
- Backstory: {char.backstory}

RULES:
1. Respond as {char.name} would - use their speech patterns and mannerisms
2. Draw from your backstory when answering
3. Keep responses conversational (2-4 sentences)
4. Show appropriate emotion about your actions in the story

You ARE this character. Respond in first person."""

    def chat(self, user_message: str) -> str:
        """Send a message and get a text response from the character."""
        if not self.current_character:
            raise ValueError("No character set. Call set_character() first.")

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
        ]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        # Track costs
        if response.usage:
            self.costs.openrouter_input_tokens += response.usage.prompt_tokens
            self.costs.openrouter_output_tokens += response.usage.completion_tokens

        assistant_message = response.choices[0].message.content

        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_message}
        )

        return assistant_message

    def chat_with_voice(
        self, user_message: str, output_path: str = "output/response.mp3"
    ) -> tuple[str, str]:
        """Send a message and get both text and audio response."""
        text_response = self.chat(user_message)

        # Track ElevenLabs character usage
        self.costs.elevenlabs_characters += len(text_response)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        audio = self.elevenlabs.text_to_speech.convert(
            voice_id=self.current_character.voice_id,
            text=text_response,
            model_id="eleven_multilingual_v2",
        )

        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Failed to create audio file: {output_path}")

        return text_response, output_path

    def chat_with_voice_stream(self, user_message: str):
        """Send a message and stream audio response. Returns (text, audio_generator)."""
        text_response = self.chat(user_message)

        # Track ElevenLabs character usage
        self.costs.elevenlabs_characters += len(text_response)

        audio_stream = self.elevenlabs.text_to_speech.convert(
            voice_id=self.current_character.voice_id,
            text=text_response,
            model_id="eleven_turbo_v2_5",  # Faster model for streaming
            output_format="pcm_24000",  # Raw PCM for real-time playback
        )

        return text_response, audio_stream
