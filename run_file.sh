python src/llama_activate.py sf --rounds 10 --num_agents 5 --p 0.3
python src/llama_activate.py r --rounds 600 --num_agents 8 --p 0.3 --use_saved_network
python src/llama_activate.py sda --rounds 10 --num_agents 5 --alpha 1.5 --m 2
python src/llama_activate.py r --use_saved_network --rounds 300 --num_agents 8 --p 0.3

# 1. Generate and Save (Basis state)
python src/llama_activate.py sda --num_agents 20 --rounds 3 --save --seeds 101
python src/llama_activate.py sda --num_agents 100 --rounds 3 --seeds 101


# 2. Load the previous network and run 2 NEW rounds
# (Ensure args.rounds is set, otherwise it just analyzes the loaded net)
python src/llama_activate.py sda --num_agents 20 --rounds 2 --save --seeds 101
python src/llama_activate.py sda --use_saved_network --num_agents 20 --rounds 2 --seeds 101


python src/llama_activate.py sda --num_agents 1000 --rounds 3 --seeds 101 --degree 30