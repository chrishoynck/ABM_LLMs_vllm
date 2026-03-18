python src/llama_activate.py sf --rounds 10 --num_agents 5 --p 0.3
python src/llama_activate.py r --rounds 600 --num_agents 8 --p 0.3 --use_saved_network
python src/llama_activate.py sda --rounds 10 --num_agents 5 --alpha 1.5 
python src/llama_activate.py r --use_saved_network --rounds 300 --num_agents 8 --p 0.3


python src/llama_activate.py sda --num_agents 20 --rounds 10 --seeds 101 --degree 6
python src/llama_activate.py sda --num_agents 30 --rounds 30 --seeds 101 --degree 6  --save

python src/llama_activate.py sda --num_agents 20 --rounds 2 --save --seeds 101 101
python src/llama_activate.py sda --use_saved_network --num_agents 20 --rounds 20 --seeds 103 104 --save


python src/llama_activate.py sda --num_agents 100 --rounds 600 --seeds 33 34 --save --degree 18

python src/llama_activate.py sda --num_agents 20 --rounds 20 --seeds 33 --save --degree 0
