from PPO import Params
from IPPO import IPPO 
from sim import armsim
import optuna
from collections import deque

import multiprocessing
def objective(trial):
    env=armsim()
    params=Params(n_agents=1)
    params.action_dim=8
    params.state_dim=39
    params.N_steps=1e6
    params.beta_ent=trial.suggest_float("entropy", .00001, 0.1, log=True)
    params.lr_actor=trial.suggest_float("learn_rate", 1e-6, 1e-3, log=True)
    params.N_batch=trial.suggest_int("batch_size", 4,10)
    params.K_epochs=trial.suggest_int("number_epochs", 10,30)
    params.lr_critic=params.lr_actor*10.0
    learner=IPPO(params)

    step=0
    r_hist=deque(maxlen=10)
    while step<params.N_steps:
        
        
        running_reward=0
        for j in range(params.N_batch):
            state=env.reset()
            done=False
            while not done:
                step+=1
                action=learner.act([state[0]])
                state,G,reward,done=env.step([action]*3)
                learner.add_reward_terminal([reward[0]],done)
                running_reward+=reward[0]
        r_hist.append(running_reward/float(params.N_batch))
        trial.report(max(r_hist), step)
        if trial.should_prune() and PRUNE:
            raise optuna.TrialPruned()
        learner.train(step)
    return max(r_hist)

def run_study(worker_id):
    print(f"Worker {worker_id} starting optimization...")
    study = optuna.load_study(
        study_name="multiagent_opt",  # Load the existing study
        storage='sqlite:///optimize.db'  # Shared SQLite storage
    )
    study.optimize(objective, n_trials=15)  

if __name__ == "__main__":
    n_workers=6
    sampler=optuna.samplers.GPSampler()
    direction="maximize"
    name="multiagent_opt"
    location='sqlite:///optimize.db'

    study = optuna.create_study(sampler=sampler,study_name=name,direction=direction,storage=location)
    
    with multiprocessing.Pool(n_workers) as pool:
        pool.map(run_study, range(n_workers))