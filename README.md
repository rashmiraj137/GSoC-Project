## GSoC Final Report
### Implement JAX based automatic differentiation to Stingray

The project involved study of the modern statistical modeling to augment the accuracy, speed, and robustness of the likelihood function, into a software package called Stingray. This report demonstrates the experiment done for a combination of different optimizers to fit the scipy.optimize function. Another emphasis is to investigate the gradient calculation using JAX and compare it with scipy.optimize. 
The proposed milestone was to investigate the room for improvement to enhance the overall performance of modeling to Stingray, using JAX. However, the current stage of model is still a sandbox model. Stingray is astrophysical spectral timing software, a library in python built to perform time series analysis and related tasks on astronomical light curves. JAX is a python library designed for high-performance numerical computing. Its API for numerical functions is based on NumPy, a collection of functions used in scientific computing. Both Python and NumPy are widely used and familiar, making JAX simple, flexible, and easy to adopt. It can differentiate through a large subset of python’s features, including loops, ifs, recursion, and closures, and it can even take derivatives of derivatives. Such modern differentiation packages deploy a broad range of computational techniques to improve the applicability, run time, and memory management.
JAX utilizes the grad function transformation to convert a function into a function that returns the original function’s gradient, just like Autograd. Beyond that, JAX offers a function transformation jit for just-in-time compilation of existing functions and vmap and pmap for vectorization and parallelization, respectively.

## Experiment:

The powerlaw and lorentzian function are the most used to describe periodograms in astronomy. In practice, we use the sum of these components to design a realistic model. For the analysis here we consider a quasi-periodic oscillation and a constant and try to fail the algorithm by, (i) reduce the amplitude, (ii) start the optimization process with parameters very far away from the true parameters, (iii) try different optimizers to experiment on different sensitive aspect of the current likelihood calculation. The current ongoing milestone is to try alternatives of scipy.optimize but this requires series of tests for the same. 
The above tests can be visualized in the notebook added on Github: https://github.com/rashmiraj137/GSoC-Project

## Impact:

JAX-based automatic differentiation offers following advantages over traditional methods:

*Efficiency*: JAX utilizes just-in-time (JIT) compilation and hardware acceleration to compute gradients efficiently, leading to faster training and inference times compared to manual differentiation.

*Flexibility*: JAX allows for dynamic computation graphs, enabling more flexible and complex models to be built and trained, which may be challenging with static computational graphs used in traditional differentiation methods.

*Numerical Stability*: JAX employs advanced numerical techniques to ensure stability during gradient computation, reducing the likelihood of numerical errors such as vanishing or exploding gradients.

*Compatibility*: JAX seamlessly integrates with other libraries like NumPy and TensorFlow, allowing for easy interoperability and leveraging existing codebases.

*Parallelism*: JAX can automatically parallelize operations across multiple devices or processors, maximizing resource utilization and speeding up computation for large-scale datasets or models.

## Repositories:
https://github.com/StingraySoftware/stingray
https://github.com/StingraySoftware/notebooks
BlogPost:

## Profiles:
GitHub: https://github.com/rashmiraj137
LinkedIn: https://www.linkedin.com/in/rashmi-raj-4b8a2b106/ 
Medium: https://medium.com/@rashmi_13737
![image](https://user-images.githubusercontent.com/42755704/130421857-ec70a240-0707-4227-884b-57594628a462.png)

