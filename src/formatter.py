"""
Output Formatter - Structured coaching output

Formats the coaching output as YAML for display.
"""

from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any
import yaml

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax


@dataclass
class CoachingOutput:
    turn: int
    timestamp: str
    customer_said: str
    detected_situation: str
    matched_signal: str
    stage: str
    recommendation: dict


def format_coaching_output(
    turn: int,
    transcript: str,
    situation_id: str,
    matched_signal: str,
    stage: str,
    principle_name: str,
    principle_source: str,
    response_text: str,
    why_it_works: str
) -> CoachingOutput:
    """
    Format the coaching output.

    Returns:
        CoachingOutput dataclass
    """
    return CoachingOutput(
        turn=turn,
        timestamp=datetime.now().isoformat(),
        customer_said=transcript,
        detected_situation=situation_id,
        matched_signal=matched_signal,
        stage=stage,
        recommendation={
            "principle": principle_name,
            "source": principle_source,
            "response": response_text,
            "why_it_works": why_it_works
        }
    )


def display_coaching_output(output: CoachingOutput) -> None:
    """Display coaching output in terminal with rich formatting."""
    console = Console()

    # Convert to YAML
    yaml_str = yaml.dump(asdict(output), default_flow_style=False, sort_keys=False)

    # Display with syntax highlighting
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)

    console.print(Panel(
        syntax,
        title=f"[bold green]Turn {output.turn} - Coaching Output[/bold green]",
        border_style="green"
    ))
