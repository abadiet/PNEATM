# Polymorphic NeuroEvolution of Augmenting Topologies with Memory (PNEATM)
Pure C++ library for evolving neural networks with a modified NEAT that allows any kind of data and that is able to access any previous data.

[![Lines of Code](https://tokei.rs/b1/github/titofra/PNEATM?category=code)](https://github.com/XAMPPRocky/tokei)

<p align="center">
	<img src="https://github.com/titofra/PNEATM/blob/main/resources/network.png">
</p>

## Goals
[Neuroevolution of augmenting topologies (NEAT)](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) is a machine learning technique used to evolve artificial neural networks. NEAT employs a genetic algorithm to optimize the topology, weights, and activation functions of neural networks in order to solve a given problem. By dynamically adding or removing neurons and connections, NEAT enables networks to evolve and adapt over time. It is commonly applied to complex tasks such as game playing and robotics, where manually designed neural networks may prove ineffective. However, NEAT does have certain limitations, including a lack of memory and difficulty in handling multiple types of data.

To address these limitations, the primary goal of **PNEATM** is to provide a substantial memory capacity and the ability to handle various forms of data. This enhancement is intended to enable its utilization for more complex tasks.

### Pros of PNEATM
* **Handling various data types**: This allows us to use more inputs/outputs as the network can *distinguish* different objects. For example, two 5-element vectors are not considered as 10 inputs but only 2 inputs. This helps the neural network better understand what it is processing.
* **Support for multiple data types simultaneously**: PNEATM widens its capacities by accommodating different types of data in a single network.
* **Multiple activation functions**: PNEATM can employ various activation functions, expanding its capabilities to determine the best-suited functions. This flexibility is particularly valuable in tasks like image processing where traditional activation functions like sigmoid may not be ideal (utilizing convolutions, blurring functions, etc.).
* **Dynamic activation functions evolution**: As connection weights are mutated, activation function parameters are also evolved. PNEATM selects the most suitable function with its optimal parameters for each specific application.
* **Access to previous processed data**: The network can retrieve any previously processed data from memory, allowing it to consider past inputs or outputs in its decision-making process.
* **Diversity and exploration**: PNEATM's lack of strong restrictions allows it to explore a wide range of networks in the network's space, leading to improved diversity.

### Cons of PNEATM
* **Complexity and execution time**: Due to the network's unrestricted nature, it explores a wide range of networks in the network's space, leading to increased complexity and execution time.

## Dependencies
This code depends on both SFML (for graphics) and spdlog (for logs) libraries:
- [SFML install](https://www.sfml-dev.org/download.php)
- [spdlog install](https://github.com/gabime/spdlog)

⚠️ Please note that these dependencies introduce impurity to PNEATM. However, they are not necessary, so you can easily remove them to have a fully pure C++ library.

## Examples & Documentations
Documentation is available at [https://titofra.github.io/PNEATM/](https://titofra.github.io/PNEATM/). Moreover, a Snake AI powered by PNEATM is available on [/examples/snake/](https://github.com/titofra/PNEATM/tree/main/examples/snake) as POC.

<p align="center">
	<img src="https://github.com/titofra/PNEATM/blob/main/examples/snake/resources/snakeGameplay.gif">
</p>

## Warning

This project is currently under development and is provided as-is, without any guarantees. There are several issues that still persist, such as slow convergence, too few optimizations, and the potential for an over-representation problem of a species.
<p align="center">
	<img src="https://github.com/titofra/PNEATM/blob/main/examples/snake/resources/over-representation_problem.png" alt="over-representation problem of the red species">
</p>
<p align="center">
	<em> over-representation problem of the red species </em>
</p>

## TODO
- [x] MultiThreading
- [ ] GPU acceleration
- [ ] Vectorization, SIMD?
- [x] Memory Management optimization
- [x] Algo optimization: sort connections by calls and keep it in mind...
- [x] vector, unordered_map and map
- [ ] connections key can be innovation id: better for distance processing, ...
- [ ] Compiler optimizations (PGO, LTO ...)
- [ ] Node innovation tracker need improvements 
- [ ] 1-declaration var + memory pool
- [ ] std::vector explicit sz constructor
