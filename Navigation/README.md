
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
Here is an example of an untrained agent, which chooses one of the 4 actions randomly ( equiprobably) at each step.
[![Random agent](https://img.youtube.com/vi/Du7vpSd0JeY/0.jpg)](https://youtu.be/Du7vpSd0JeY "Random Agent")

## Prerequisites

What things you need to install the software and how to install them

1. Install Unity
2. ML agents v0.4.0 ( you can install this with `pip install ml`

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc


