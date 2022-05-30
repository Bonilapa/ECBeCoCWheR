
import numpy as np
import torch as T
from torch.ppo_memory import PPOMemory
from torch.actor_network import ActorNetwork
from torch.critic_network import CriticNetwork
class AI:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor_turn = ActorNetwork(n_actions, input_dims, alpha)
        self.critic_turn = CriticNetwork(input_dims, alpha)
        self.memory_turn = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor_turn.save_checkpoint()
        self.critic_turn.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor_turn.load_checkpoint()
        self.critic_turn.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist_turn = self.actor_turn(state)
        value_turn = self.critic_turn(state)
        action_turn = dist_turn.sample()

        probs_turn = T.squeeze(dist_turn.log_prob(action_turn)).item()
        action_turn = T.squeeze(action_turn).item()
        value_turn = T.squeeze(value_turn).item()

        return [action_turn,\
                probs_turn,\
                value_turn]

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor_turn.optimizer.zero_grad()
                self.critic_turn.optimizer.zero_grad()
                total_loss.backward()
                self.actor_turn.optimizer.step()
                self.critic_turn.optimizer.step()

        self.memory.clear_memory()  
    