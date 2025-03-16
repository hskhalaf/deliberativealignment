To run r1gen.py, open a separate terminal first and do: 

CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B   --enable-reasoning   --reasoning-parser deepseek_r1   --tensor-parallel-size 4
