from maze_env import Maze
from RL_brain import SarsaLambdaTable


def update():
    for episode in range(100):
        observation = env.reset()

        action = RL.choose_action(str(observation))
        # difference between sarsa and qlearning

        RL.eligibility_trace *= 0

        while True:
            # fresh env
            env.render()

            # Rl take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # Rl learn from this transition (s,a,r,s,a)==> sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print("game over")
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
