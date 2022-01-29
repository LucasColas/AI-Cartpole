import gym

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback

env = gym.make('CartPole-v1')

#See our Environment
obs = env.reset()
for i in range(1000):
    obs, reward, done, info = env.step(env.action_space.sample())
    if done: break

#Save Models
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1): #check_freq : how often we save our model
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

Save_Dir = './train/'
Logs_Dir = './logs/'

callback = TrainAndLoggingCallback(check_freq=10000, save_path=Save_Dir) #Every 10000 steps we save our model


#Model

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=90000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
