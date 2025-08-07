# deepseek_agent/deepseek_llm_agent.py
"""
DeepSeekLLMAgent
────────────────
A minimal LLM-powered Gomoku agent that calls the 7 B-parameter
DeepSeek-R1-Distill-Qwen model via HuggingFace Transformers.

Prerequisites
-------------
• transformers >= 4.40
• accelerate / bitsandbytes optional (for quantisation)
Environment
-----------
The agent honours the usual HuggingFace caching rules.
Set HF_HOME or TRANSFORMERS_CACHE if you need a custom cache path.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Tuple, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from gomoku.agents import Agent  # base abstract class


_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# System prompt: keep it short to save context tokens
_SYSTEM_PROMPT = (
    "You are a strong Gomoku engine. The board is 8×8. "
    "Return the best legal move as a JSON array [row, col] (0-indexed)."
)


def _board_to_ascii(board) -> str:
    """Render gomoku.Board (or numpy / list) to grid of . X O characters."""
    symbols = {0: ".", 1: "X", 2: "O"}
    rows: List[str] = []
    size = board.size if hasattr(board, "size") else len(board)
    for r in range(size):
        if hasattr(board, "__getitem__") and not isinstance(board[r], list):
            # numpy-like access
            row = " ".join(symbols[board[r, c]] for c in range(size))
        else:
            row = " ".join(symbols[board[r][c]] for c in range(size))
        rows.append(row)
    return "\n".join(rows)


class DeepSeekLLMAgent(Agent):
    """LLM-based Gomoku agent using DeepSeek-R1-Distill-Qwen-7B."""

    # Metadata exposed in `gomoku list --detailed`
    display_name = "DeepSeekLLMAgent"
    author = ["Kevin"]
    version = "0.2.0"
    description = "LLM-powered agent using DeepSeek-R1-Distill-Qwen-7B"

    def __init__(self, name: str, color: int, **kwargs):
        """
        Parameters
        ----------
        name   : the agent’s display name (passed by the framework)
        color  : 1 = black (X), 2 = white (O)
        kwargs : ignored but accepted for compatibility
        """
        super().__init__(name, color, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_ID, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    # --------------------------------------------------------------------- #
    # Required by gomoku.agents.Agent
    # --------------------------------------------------------------------- #
    def get_move(self, board, color: int, time_limit: float) -> Tuple[int, int]:
        """
        Return a legal (row, col) tuple within `time_limit` seconds.
        Falls back to a random legal move if anything goes wrong.
        """
        deadline = time.time() + max(0.05, time_limit - 0.25)

        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"{_board_to_ascii(board)}\n"
            f"Your colour: {'X' if color == 1 else 'O'}\n"
            "Move?"
        )

        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(_DEVICE)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=8,
                    temperature=0.1,
                    top_p=0.95,
                    do_sample=False,
                )
            completion = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            move = json.loads(completion.splitlines()[-1])
            row, col = int(move[0]), int(move[1])
            if hasattr(board, "is_legal_move"):
                if board.is_legal_move((row, col)):
                    return row, col
            elif (row, col) in board.get_legal_moves():  # type: ignore[arg-type]
                return row, col
        except Exception:
            # any parsing / generation / OOM error falls through to random
            pass

        # ---------------------------------------------------------------- #
        # Fallback strategy: choose a random legal move before time runs out
        # ---------------------------------------------------------------- #
        legal_moves = (
            board.get_legal_moves()
            if hasattr(board, "get_legal_moves")
            else [(r, c) for r in range(board.size) for c in range(board.size) if board[r, c] == 0]
        )
        return random.choice(legal_moves)
