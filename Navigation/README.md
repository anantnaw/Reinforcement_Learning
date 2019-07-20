
# Navigation:

In this project an agent uses deep reinforcement learning agent with the goal to navigate a virtual world while collecting yellow or purple bananas. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Hence, the goal of the agent is to collect the maximum possible amount of yellow bananas while avoiding purple bananas.

## Environment

The state space consists of 37 dimensions consiting of the agent's velocity as well as a ray-based perception of the agent's forward direction. With this information the agent has to maximize its score by choosing one of the following actions in the optimal manner:

* `0` - move forward
* `1` - move backward
* `2` - turn left
* `3` - turn right


The task is episodic, and in order to know if the agent has "solved" the task, it must get an average score of +13 over 100 consecutive episodes.

## Videos
Here is an example of an untrained agent, which chooses one of the 4 actions randomly ( equiprobably) at each step. It manages an average score between -2 to 2.<br>
[![Random agent](https://img.youtube.com/vi/Du7vpSd0JeY/0.jpg)](https://youtu.be/Du7vpSd0JeY "Random Agent")
<br>
Here is a video recording of the fully trained agent (which manages a score of 16.00).<br>
[![Trained agent](https://img.youtube.com/vi/tfKJGH8lEMY/0.jpg)](https://youtu.be/tfKJGH8lEMY "Trained Agent")


## Techniques used for training

Several experiments were tried and the best-performing Deep Reinforcement Learning agent was the one trained with Deep Q Learning with fixed targets, double DQN, dueling DQN, and experience replay. More details can be found in the report.md file.

## Getting Started

1. Install [Unity](https://unity3d.com/get-unity/download)
2. Install ml agents v 0.4.0 with (`pip3 install --upgrade mlagents==0.4.0`)
3. (Optional) If there are issues with the installation of tensorflow while isntalling mlagents, install tensorflow with this command `pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.7.1-py3-none-any.whl)`
4. Download the environment from one of the links below. You need only select the environment that matches your operating system:
        Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
        Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
        Windows (32-bit): [click here]
        Windows (64-bit): click here

    (For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the environment.

5. Place the file in root of the folder and unzip the file.

6. Run the cells in Navigation_Train.ipynb to test the environment and train the agent. For testing the trained agent, ensure that the model_weights are in the same folder as the notebook, then run the cells of Navigation_Inference.ipynb.

### Instructions

To train the agent run  the cells in **Navigation_Train.ipynb notebook**. For testing a pretrained agent, ensure that the model_weights are in the same folder as the notebook, then run the cells of **Navigation_Inference.ipynb**.

Description of the implementation is provided in **report.pdf**. For technical details see the code in the notebook.

Model weights are stored in **dqn_model.pth**










