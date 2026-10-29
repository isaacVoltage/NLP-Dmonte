"""
Experiment 5B: Named Entity Recognition (NER) for Marathi Recipe Text
Extracts culinary entities:
- INGREDIENT: घटकांचे नाव (उदा. तेल, कांदा, मोहरी, तूप, रवा, बटाटा)
- QUANTITY: प्रमाण किंवा संख्या (उदा. दोन, एक, अर्धा, पाव, ५०, 1/2)
- UNIT: मोजमापाचे एकक (उदा. चमचे, कप, वाटी, ग्रॅम, लिटर)
- ACTION: स्वयंपाक कृती (उदा. घाला, परता, शिजवा, तळा, उकळा)
- DURATION: लागणारा वेळ (उदा. पाच मिनिटे, अर्धा तास)
- TEMPERATURE: उष्णता पातळी (उदा. मंद आच, मध्यम आचेवर, कडकडीत)
- UTENSIL: स्वयंपाकाची भांडी (उदा. कढई, पातेले, तवा, कुकर)
"""
import re
from typing import List, Dict, Any, Tuple
from src.regex_extractor import (
    MarathiRegexExtractor,
    DEV_BOUNDARY_START,
    DEV_BOUNDARY_END
)

class MarathiRecipeNER:
    def __init__(self):
        self.extractor = MarathiRegexExtractor()
        
        # Color mapping for entity visualization
        self.entity_colors = {
            "INGREDIENT": "#10b981",    # Emerald green
            "QUANTITY": "#f59e0b",      # Amber
            "UNIT": "#3b82f6",          # Blue
            "ACTION": "#ef4444",        # Red
            "DURATION": "#8b5cf6",      # Purple
            "TEMPERATURE": "#ec4899",   # Pink
            "UTENSIL": "#6366f1"        # Indigo
        }

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts all recognized named entities with their start/end offsets,
        entity label, and token text.
        """
        entities = []
        occupied_spans = []

        # 1. Durations (multi-token, high priority)
        for d in self.extractor.duration_regex.finditer(text):
            entities.append({
                "text": d.group(0),
                "label": "DURATION",
                "start": d.start(),
                "end": d.end()
            })
            occupied_spans.append((d.start(), d.end()))

        # 2. Temperatures (multi-token)
        for t in self.extractor.temperature_regex.finditer(text):
            if not any(s <= t.start() < e or s < t.end() <= e for s, e in occupied_spans):
                entities.append({
                    "text": t.group(0),
                    "label": "TEMPERATURE",
                    "start": t.start(),
                    "end": t.end()
                })
                occupied_spans.append((t.start(), t.end()))

        # 3. Ingredients (from ingredient dictionary)
        for term in self.extractor.ingredient_terms:
            pattern = re.compile(rf"{DEV_BOUNDARY_START}{re.escape(term)}{DEV_BOUNDARY_END}", re.IGNORECASE)
            for m in pattern.finditer(text):
                if not any(s <= m.start() < e or s < m.end() <= e for s, e in occupied_spans):
                    entities.append({
                        "text": m.group(0),
                        "label": "INGREDIENT",
                        "start": m.start(),
                        "end": m.end()
                    })
                    occupied_spans.append((m.start(), m.end()))

        # 4. Units
        for unit in self.extractor.units_list:
            pattern = re.compile(rf"{DEV_BOUNDARY_START}{re.escape(unit)}{DEV_BOUNDARY_END}", re.IGNORECASE)
            for m in pattern.finditer(text):
                if not any(s <= m.start() < e or s < m.end() <= e for s, e in occupied_spans):
                    entities.append({
                        "text": m.group(0),
                        "label": "UNIT",
                        "start": m.start(),
                        "end": m.end()
                    })
                    occupied_spans.append((m.start(), m.end()))

        # 5. Quantities
        q_pattern = re.compile(rf"{DEV_BOUNDARY_START}(?:{self.extractor.quantity_regex}){DEV_BOUNDARY_END}", re.IGNORECASE)
        for m in q_pattern.finditer(text):
            if not any(s <= m.start() < e or s < m.end() <= e for s, e in occupied_spans):
                entities.append({
                    "text": m.group(0),
                    "label": "QUANTITY",
                    "start": m.start(),
                    "end": m.end()
                })
                occupied_spans.append((m.start(), m.end()))

        # 6. Utensils
        for term in self.extractor.utensil_terms:
            pattern = re.compile(rf"{DEV_BOUNDARY_START}{re.escape(term)}{DEV_BOUNDARY_END}", re.IGNORECASE)
            for m in pattern.finditer(text):
                if not any(s <= m.start() < e or s < m.end() <= e for s, e in occupied_spans):
                    entities.append({
                        "text": m.group(0),
                        "label": "UTENSIL",
                        "start": m.start(),
                        "end": m.end()
                    })
                    occupied_spans.append((m.start(), m.end()))

        # 7. Actions
        for action in self.extractor.action_terms:
            pattern = re.compile(rf"{DEV_BOUNDARY_START}{re.escape(action)}{DEV_BOUNDARY_END}", re.IGNORECASE)
            for m in pattern.finditer(text):
                if not any(s <= m.start() < e or s < m.end() <= e for s, e in occupied_spans):
                    entities.append({
                        "text": m.group(0),
                        "label": "ACTION",
                        "start": m.start(),
                        "end": m.end()
                    })
                    occupied_spans.append((m.start(), m.end()))

        # Sort entities by start index
        entities.sort(key=lambda x: x["start"])
        return entities

    def render_annotated_html(self, text: str) -> str:
        """
        Renders beautiful inline highlighted HTML with styled colored badges for each recognized entity.
        """
        entities = self.extract_entities(text)
        if not entities:
            return f"<span>{text}</span>"

        html_parts = []
        last_idx = 0

        for ent in entities:
            start = ent["start"]
            end = ent["end"]
            label = ent["label"]
            color = self.entity_colors.get(label, "#94a3b8")
            
            # Text before entity
            if start > last_idx:
                html_parts.append(text[last_idx:start])

            # Entity with styled tag
            token_text = text[start:end]
            badge_html = (
                f'<span style="background-color: {color}22; border: 1px solid {color}; '
                f'color: {color}; padding: 2px 6px; border-radius: 6px; font-weight: 600; margin: 0 2px;">'
                f'{token_text} '
                f'<span style="font-size: 0.72em; opacity: 0.85; text-transform: uppercase;">[{label}]</span>'
                f'</span>'
            )
            html_parts.append(badge_html)
            last_idx = end

        if last_idx < len(text):
            html_parts.append(text[last_idx:])

        return "".join(html_parts)
