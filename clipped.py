import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
import matplotlib.pyplot as plt
from statistics import mean
from environment import CustomEnv
import csv

from torch.distributions import MultivariateNormal
from torch.optim import Adam

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

class FeeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(FeeedForwardNN, self).__init__()
        
        self.layer1=nn.Linear(in_dim,64)
        self.layer2=nn.Linear(64,64)
        self.layer3=nn.Linear(64,out_dim)

        # Initialize weights with orthogonal initialization
        nn.init.orthogonal_(self.layer1.weight)
        nn.init.orthogonal_(self.layer2.weight)
        nn.init.orthogonal_(self.layer3.weight)

        
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs=torch.tensor(obs, dtype=torch.float)
        
        activation1=F.relu(self.layer1(obs))
        activation2=F.relu(self.layer2(activation1))
        output=F.sigmoid(self.layer3(activation2))
        return output

class PPO:
    def __init__(self,env):
        # Initialize hyperparameters
        self._init_hyperparameters()
        
        # Extrace environment information
        self.env=env
        self.obs_dim=env.observation_space.shape[0]
        self.act_dim=env.action_space.shape[0]
        
        # ALG STEP 1
        # Initialize actor and critic networks
        self.actor=FeeedForwardNN(self.obs_dim, self.act_dim)
        self.critic=FeeedForwardNN(self.obs_dim,1)
        
        # Initialize actor and critic networks
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        # Create the covariance matrix for get_action
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        self.average_episodic_returns=[]
        self.average_tot_ts_sofar=[]
        
        self.action=np.zeros(512*2,dtype=np.float64)

    
    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 30            # timesteps per batch
        self.max_timesteps_per_episode = 10      # timesteps per episode
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2 # As recommended by the paper
        self.lr = 0.005 # learning rate of optimizers
        self.max_grad_norm=0.5
        
    def learn(self, total_timesteps):
        t_so_far = 0 # Timesteps simulated so far
        file=open('file.csv', 'w', newline='')
        writer = csv.writer(file)
        
        while t_so_far < total_timesteps:              # ALG STEP 2
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs,batch_rtgs, batch_lens = self.rollout()
            
            # Calculate how many timesteps we collected this batch
            t_so_far+=np.sum(batch_lens)
            
            # Calculate V_{phi, k}
            # V, _ = self.evaluate(batch_obs, batch_acts)   
            
            V, _=self.evaluate(batch_obs, batch_acts)
                     
            # ALG STEP 5
            # Calculate advantage
            A_k = batch_rtgs-V.detach()
            
            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            
            # ALG STEP 6 & 7
            for _ in range(self.n_updates_per_iteration):
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                # Calculate V_phi and pi_theta(a_t | s_t)
                V , curr_log_probs = self.evaluate(batch_obs, batch_acts)
                
                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                
                # Calculate actor losses
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                
                
                # Calculate gradients and perform backward propagation for actor 
                # network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

                self.actor_optim.step()
                
                # Calculate gradients and perform backward propagation for critic network    
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.critic_optim.step()
            print(t_so_far,actor_loss,critic_loss)
            writer.writerow([t_so_far,actor_loss,critic_loss])
        file.close()
        

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        # Return predicted values V and log probs log_probs
        return V, log_probs
    
    def rollout(self):
        print("Rollout started")
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch
        
        # Number of timesteps run so far this batch
        t = 0 
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews=[]
            obs = self.env.reset()
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                for o in obs:
                    batch_obs.append(o)
                
                for i in range(512):
                    action, log_prob = self.get_action(obs)
                
                print(action)
                obs, rew, done, _ = self.env.step(action)

                # Collect reward, action, and log prob
                for r in rew:
                    ep_rews.append(r)
                for a in action:
                    batch_acts.append(action)
                for l in log_prob:
                    batch_log_probs.append(l)
                # If the environment tells us the episode is terminated, break
                #if done:
                #    break
            
            self.average_episodic_returns.append(mean(ep_rews))
            # Track episodic length and rewards
            for i in range(512):
                batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)
            
        # Reshape data as tensors in the shape specified before returning
        batch_obs=torch.tensor(np.array(batch_obs),dtype=torch.float)
        batch_acts=torch.tensor(np.array(batch_acts),dtype=torch.float)
        batch_log_probs=torch.tensor(np.array(batch_log_probs),dtype=torch.float).flatten()
        
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)
        
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs,batch_rtgs, batch_lens
    
    def get_action(self, obs):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        # Extract the array from the tuple
        #print(obs_array)
        
        log_prob=[]
        for i in range(512):
            # Convert the array to
            mean= self.actor.forward(obs[i:i+4])
            # Create our Multivariate Normal Distribution
            
            dist = MultivariateNormal(mean, self.cov_mat)
            # Sample an action from the distribution and get its log prob
            actionh = dist.sample()
            log_prob.append(dist.log_prob(actionh).detach())
            actionn=actionh.detach().numpy()
            
            # action0
            actionn[0]=actionn[0]*0.7
            actionn[0]+=0.3
            if actionn[0]>1:
                self.action[i*2]=1*100000
            elif actionn[0]<0.3:
                self.action[i*2]=0.3*100000
            else:
                self.action[i*2]=actionn[0]*100000
            
            #action1
            actionn[1]=actionn[1]*np.pi/2
            if actionn[1]<0:
                self.action[i*2+1]=0
            elif actionn[1]>np.pi/2:
                self.action[i*2+1]=np.pi/2
            else:
                self.action[i*2+1]=actionn[1]
            
        
        # Return the sampled action and the log prob of that action
        return self.action, log_prob
    
    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    def show(self):
        plt.style.use('_mpl-gallery')

        # make data
        x = np.linspace(0,1, len(self.average_episodic_returns))
        y = self.average_episodic_returns
        # plot
        fig, ax = plt.subplots()

        ax.plot(x, y, linewidth=1.0)

        #ax.set(xlim=(0, 8), xticks=np.arange(1, 8),ylim=(0, 8), yticks=np.arange(1, 8))
        plt.show()

def main():
    
    env=CustomEnv()
    model = PPO(env)
    model.learn(50)
    model.show()
    
    

if __name__=='__main__':
    main()