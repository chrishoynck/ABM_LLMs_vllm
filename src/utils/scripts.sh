
# run testing and automatic evaluation for optimized post prompt 
PYTHONPATH=src python -m utils.prompt_optimizer     --mode tweets-rerun-test     --instruction-file data/test_post/optimized_tweets/Qwen3.5-27B_seed53/optimized_instruction_tweet.txt     --persona-phq9-file data/personas_eval_1000_phq9.csv     --num-agents  100     --sample-seed 999     --seeds 42     --model Qwen/Qwen3.5-27B