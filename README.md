# Polymorphic NeuroEvolution of Augmenting Topologies with Memory (PNEATM)
C++ library for evolving neural networks with a modified NEAT that allows any kind of data and that is able to access any previous data.

⚠️ WARNING: This projetc is currently being developed. For now, it cannot be used for profuction as there are several issues.

<p align="center">
	<img src="https://github.com/titofra/PNEATM/blob/main/resources/network.png">
</p>

## Goals
[Neuroevolution of augmenting topologies (NEAT)](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) is a machine learning technique used to evolve artificial neural networks. NEAT employs a genetic algorithm to optimize the topology, weights, and activation functions of neural networks in order to solve a given problem. By dynamically adding or removing neurons and connections, NEAT enables networks to evolve and adapt over time. It is commonly applied to complex tasks such as game playing and robotics, where manually designed neural networks may prove ineffective. However, NEAT does have certain limitations, including a lack of memory and difficulty in handling multiple types of data.

To address these limitations, the primary goal of **PNEATM** is to provide a substantial memory capacity and the ability to handle various forms of data. This enhancement is intended to allow for the utilization of larger neuroevolution networks and to enhance their capabilities through the implementation of a memory system.

## Dependencies
This code depends on both SFML (for graphics) and spdlog (for logs) libraries:
- [SFML install](https://www.sfml-dev.org/download.php)
- [spdlog install](https://github.com/gabime/spdlog)

## Examples
A Snake AI powered by PNEATM is available on [/examples/snake/](https://github.com/titofra/PNEATM/tree/main/examples/snake).

![](https://github.com/titofra/PNEATM/tree/main/examples/snake/resources/snakeGameplay.gif)

## TODO
- [x] NEAT
- [x] generic kind of obj (float, vectors...)
- [x] use of multiple kind over one network
- [x] allow multiple activation function
- [x] allow recurent connection from any level
- [ ] ~~memory arg on activation functions~~
- [ ] mem management: ptr -> unique_ptr, ...
