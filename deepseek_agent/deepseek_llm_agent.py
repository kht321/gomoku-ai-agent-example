# deepseek_agent/deepseek_llm_agent.py
# ---- DeepSeekLLMAgent (v0.2.0) ----
from __future__ import annotations

import json
import random
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from gomoku.agents import Agent

_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_SYS = (
    "You are a strong Gomoku engine. The board is 8×8. "
    "Return the best legal move as a JSON array [row, col] (0-indexed)."
)


def _ascii(board) -> str:
    """Convert board to ASCII grid of . X O"""
    sym = {0: ".", 1: "X", 2: "O"}
    s = board.size
    return "\n".join(" ".join(sym[board[r, c]] for c in range(s)) for r in range(s))


class DeepSeekLLMAgent(Agent):
    display_name = "DeepSeekLLMAgent"
    author = ["Kevin"]
    version = "0.2.0"
    description = "LLM-powered agent using DeepSeek-R1-Distill-Qwen-7B"

    def __init__(self, name: str, color: int, **kwargs):
        super().__init__(name, color, **kwargs)
        self.tok = AutoTokenizer.from_pretrained(_MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    # ------------------------------------------------------------------ #
    # Required by gomoku.agents.Agent
    # ------------------------------------------------------------------ #
    def get_move(self, board, color: int, time_limit: float) -> Tuple[int, int]:
        prompt = (
            f"{_SYS}\n\n{_ascii(board)}\n"
            f"Your colour: {'X' if color == 1 else 'O'}\nMove?"
        )

        try:
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(_DEVICE)
            out = self.model.generate(
                ids,
                max_new_tokens=8,
                temperature=0.1,
                top_p=0.95,
                do_sample=False,
            )
            move = json.loads(
                self.tok.decode(
                    out[0][ids.shape[1] :], skip_special_tokens=True
                ).strip().splitlines()[-1]
            )
            if board.is_legal_move(tuple(move)):  # type: ignore[arg-type]
                return tuple(move)  # type: ignore[return-value]
        except Exception:
            pass  # fall through to random fallback

        return random.choice(board.get_legal_moves())
# ---- end ----
