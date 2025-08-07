# deepseek_llm_agent.py

from gomoku.agents import Agent
from typing import Tuple, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class DeepSeekLLMAgent(Agent):
    def __init__(self, name="DeepSeek", device="cuda"):
        super().__init__(name)
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.device = device

    def generate_prompt(self, board: List[List[str]], color: str) -> str:
        board_str = "\n".join(" ".join(row) for row in board)
        return f"""
You are playing Gomoku as {'Black (X)' if color == 'X' else 'White (O)'}. Here's the board:

{board_str}

Reply with your next move as a pair of integers in the format: (row, col)
"""

    def parse_response(self, output: str) -> Tuple[int, int]:
        try:
            move = eval(output.strip().splitlines()[-1])
            if isinstance(move, tuple) and len(move) == 2:
                return int(move[0]), int(move[1])
        except:
            pass
        return (0, 0)  # fallback

    def move(self, board: List[List[str]], color: str) -> Tuple[int, int]:
        prompt = self.generate_prompt(board, color)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_new_tokens=30)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return self.parse_response(output_text)
