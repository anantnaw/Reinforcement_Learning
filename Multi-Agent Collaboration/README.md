
# Multi-agent Collaboration: Project Details

In this project two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play, and thus the agents must collaborate with each other.


## Environment

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

  After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.This yields a single score for each episode. The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Videos
Here is an example of two untrained agents, sampling their actions from a random policy ( uniform random distribution between -1 to 1). They both manage an average score of no more than 0.

[![Random Agent](https://img.youtube.com/vi/3xn_RkPcfQI/0.jpg)](https://youtu.be/3xn_RkPcfQI "Random Agent")

Here is a video recording of the fully trained agent (where each agent managed a score of 2.6).

[![Trained Agent](https://img.youtube.com/vi/q8LJp9XtcvA/0.jpg)](https://youtu.be/q8LJp9XtcvA "Trained Agent")


## Getting Started

1. Install [Unity](https://unity3d.com/get-unity/download)
2. Install ml agents v 0.4.0 with (`pip3 install --upgrade mlagents==0.4.0`)
3. (Optional) If there are issues with the installation of tensorflow while installing mlagents, install tensorflow with this command `pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.7.1-py3-none-any.whl)`
4. Download the single agent environment from one of the links below. You need only select the environment that matches your operating system:
   * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
   * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
   * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
   
    (For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the environment.

5. Place the file in root of the folder and unzip the file.

6. Run the cells in MA_Collaboration-training.ipynb to test the environment and train the agent. For testing the trained agent, ensure that the model_weights are in the same folder as the notebook, then run the cells of MA_Collaboration_Inference.ipynb.

### Instructions

To train the agent run  the cells in *MA_Collaboration-training.ipynb notebook**. For testing a pretrained agent, ensure that the model_weights are in the same folder as the notebook, then run the cells of **MA_Collaboration-inference.ipynb**.

Description of the implementation is provided in **report.md**. For technical details see the code in the notebook.

Model weights for the actor 1 and are stored in  **checkpoint_agent0_actor.pth** and **checkpoint_agent1_actor.pth** respectively











