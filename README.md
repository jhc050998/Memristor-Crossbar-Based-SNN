# Memristor-Crossbar-Based-SNN

Dataset.py --> Read out the dataset we do experiments on, including three UCI benchmarks as Wisconsin Diagnostic Breast Cancer (WDBC), BUPA liver disorder (BUPA-LD), and Johns Hopkins University Ionosphere database (JHUI) and two image classification challenge as MNIST and Fashion-MNIST;

SNN.py --> The novel temporal-coded SNN model and its quantized version trained by gradient descent (GD).

memristor_vi_crossbar.py --> Simulation of the memristor devices' conductance modulation.

MNIST.py --> Our SNN's software baseline used on MNIST dataset;
MNIST_M.py --> Software simulation of the complete version of our SNN with quantized synaptic weights training, weight state pruning, and simulated annealing optimization.

FMNIST.py & FMNIST_M.py --> Same with above on Fashion-MNIST dataset.

WDBC.py & WDBC_M.py & BUPA.py & BUPA_M.py & JHUI.py & JHUI_M.py --> Same with above on the three UCI datasets.

Draw.py --> Use the experiment data to draw the images for the paper.
