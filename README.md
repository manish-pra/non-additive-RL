# Submodular Reinforcement Learning (SubRL)

The repository contains all code and experiments for submodular policy optimization (SubPO). Currently algorithm/experiments are focused to optimize Non-additive rewards modelled using monotone submodular function. However, the algorithm is general and can be used with any monotone non-Additive reward functions. 

## Dependencies
1. The code is tested on Python 3.8.5 and 3.8.10
2. On the cluster, you can load  gcc/8.2.0 python/3.8.5 ffmpeg/5.0
3. Install packages from requirements.txt

## To Run
1. Set the following params in main.py; by default, it is set to name "subrl". Rest all the things can be changed in the config file
```  
workspace = "subrl"
env_load_path = workspace + \
    "/environments/" + params["env"]["node_weight"]+ "/env_" + \
    str(args.env)
```
args.env is environment number, params["env"]["node_weight"] takes a value from $\{ "constant", "constant", "linear", "bimodal", "gp", "entropy", "steiner\_covering"\}$. It loads the appropriate environment.

2. The following commands run the experiments:

```
python3 subrl/main.py -i $i -env $env_i -param $param_i (finite state action spaces)
python3 subrl/gym_env/gym_subrl.py -i $i -param $param_i (Mujoco-Ant)
python3 subrl/car_racing/subrl.py -i $i -param $param_i (Car-Racing)
```

```
$param_i = Name of the param file (see params folder) to pick an algorithm and the environment type
$env_i = an integer to pick an instance of the environment
$i = an integer to run multiple instances
```

3. Following example forllowing scripts runs subrl in different environments

 ``` 
 python3 subrl/main.py  -i 1 -env 1 -param "GP/subrl_M"
 python3 subrl/main.py  -i 1 -env 1 -param "steiner/subrl_M"
 python3 subrl/main.py  -i 1 -env 1 -param "entropy/subrl_M"
 python3 subrl/main.py  -i 1 -env 1 -param "gorilla/subrl_M"
 python3 subrl/main.py  -i 1 -env 1 -param "two_rooms/subrl_M"
 python3 subrl/main.py  -i 1 -env 1 -param "coverage/subrl_M" (set node_weight = constant or bimodal)
 python3 subrl/main.py  -i 1 -env 1 -param "bimodal/subrl_M"
 python3 subrl/car_racing/subrl.py -i 1 -param "car_subrl" 
 python3 subrl/gym_env/gym_subrl.py -i 1 -param "gym_subRL" 
```

4. Also you can create more config files that can be passed on similar to subrl_M, eg subrl_NM, subrl_SRL, etc. You can visualize results on directly wandb.
