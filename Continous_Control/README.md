
# Navigation: Project Details

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible. 

## Environment

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


The task is episodic, and in order to know if the agent has "solved" the task, it must get an average score of +30 over 100 consecutive episodes.

## Videos
Here is an example of an untrained agent, samples from a uniform random  distribution between -1 to 1. It manages an average score of 0.

[![Random agent]( "Random Agent")
[![Random agent](https://youtu.be/aVyJsIQ_Qbk/0.jpg)](https://youtu.be/aVyJsIQ_Qbk"Random Agent")

Here is a video recording of the fully trained agent (which manages a score of 30-50).

[![Trained Agent](https://youtu.be/sWK1l8EXcEs/0.jpg)](https://youtu.be/sWK1l8EXcEs "Random Agent")



## Getting Started

1. Install [Unity](https://unity3d.com/get-unity/download)
2. Install ml agents v 0.4.0 with (`pip3 install --upgrade mlagents==0.4.0`)
3. (Optional) If there are issues with the installation of tensorflow while installing mlagents, install tensorflow with this command `pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.7.1-py3-none-any.whl)`
4. Download the environment from one of the links below. You need only select the environment that matches your operating system:
   * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
   * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
   * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
   
    (For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip ) to obtain the environment.

5. Place the file in root of the folder and unzip the file.

6. Run the cells in Continous_Control_Train.ipynb to test the environment and train the agent. For testing the trained agent, ensure that the model_weights are in the same folder as the notebook, then run the cells of Continuos_Control_Inference.ipynb.

### Instructions

To train the agent run  the cells in ** Continous_Control_Train.ipynb notebook**. For testing a pretrained agent, ensure that the model_weights are in the same folder as the notebook, then run the cells of ** Continous_Control_Train.ipynb_Inference.ipynb**.

Description of the implementation is provided in **report.md**. For technical details see the code in the notebook.

Model weights are stored in **reacher_ddpg.pth**











