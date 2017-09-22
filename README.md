THOR Challenge Environment compatible with OpenAI-gym

### Step 1: Install OpenAI-gym
    
    pip install gym

### Step 2: Install gym-thor

change directory into `gym-thor/gym_thor/envs/`, then

#### OSX Install:
    pip install https://s3-us-west-2.amazonaws.com/ai2-vision-robosims/builds/robosims-0.0.9-py2.py3-none-any.whl

    wget https://s3-us-west-2.amazonaws.com/ai2-vision-robosims/builds/thor-201705011400-OSXIntel64.zip
    wget https://s3-us-west-2.amazonaws.com/ai2-vision-robosims/builds/thor-challenge-targets.zip
    unzip thor-challenge-targets.zip
    unzip thor-201705011400-OSXIntel64.zip

#### Ubuntu Install:
    
    pip install https://s3-us-west-2.amazonaws.com/ai2-vision-robosims/builds/robosims-0.0.9-py2.py3-none-any.whl
    wget https://s3-us-west-2.amazonaws.com/ai2-vision-robosims/builds/thor-201705011400-Linux64.zip
    wget https://s3-us-west-2.amazonaws.com/ai2-vision-robosims/builds/thor-challenge-targets.zip
    unzip thor-challenge-targets.zip
    unzip thor-201705011400-Linux64.zip

change directory into `gym-thor`, then
    
    pip install -e .

### Step 3: Use gym-thor
    
    import gym
    import gym_thor
    env = gym.make('thor-v0')
    path = 'relative/path/to/gym-thor/gym_thor/envs'
    env.start(path)
    
