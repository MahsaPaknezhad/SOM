from typing import Tuple 
import matplotlib.pyplot as plt
import numpy as np

class SOM():

    def __init__(self, 
             input_data: np.ndarray, 
             n_max_iterations: int, 
             width: int, 
             height: int,
             initial_lr: float) -> None:
        """
        Initialize the SOM (Self-Organizing Map) with given parameters.

        Args:
            input_data (np.ndarray): The input data for training the SOM.
            n_max_iterations (int): The maximum number of iterations for training.
            width (int): The width of the SOM grid.
            height (int): The height of the SOM grid.
            initial_lr (float): The initial learning rate.
            
        Returns:
            None
        """
        assert isinstance(input_data, np.ndarray) and len(input_data.shape)==2, "Error: input_data should be a numpy 2D array"
        assert isinstance(width,int) and isinstance(height,int), "Error: width and height should be of type integer"
        assert width>0 and height>0, "Error: width and height should be greater than zero"
        assert isinstance(n_max_iterations,int) and n_max_iterations>0, "Error: n_max_iterations should be a positive integer"
        assert isinstance(initial_lr, (int,float)), "Error: learning rate should be of type integer or loat"

        # Store the input data for training
        self._data = input_data
        # Set the maximum number of iterations for training
        self._n_iters = n_max_iterations
        # Set the width of the SOM grid
        self._w = width
        # Set the height of the SOM grid
        self._h = height
        # Set the initial learning rate
        self._lr_0 = initial_lr
        # Calculate the initial neighborhood radius (half the maximum dimension)
        self._neigh_rad_0 = max(width, height) / 2
        # Calculate the time constant for learning rate decay
        self._lmbda = n_max_iterations / np.log(self._neigh_rad_0)
        # Initialize the weights randomly in a 3D array (width x height x 3)
        self._weights = np.random.random((width, height, input_data.shape[1]))
        # Initialize the mesh grid for the SOM
        self._xx, self._yy = self._init_mesh(width, height)

    
    def _init_mesh(self, 
               width: int,
               height: int
               ) -> np.ndarray:
        """
        Initialize a mesh grid for the given width and height.
        
        Args:
            width (int): The width of the grid.
            height (int): The height of the grid.
            
        Returns:
            np.ndarray: Two 2D arrays representing the x and y coordinates of the grid points.
        """
        # Create a range of values for the width
        rx = np.arange(width)
        # Create a range of values for the height
        ry = np.arange(height) 

        # Create a mesh grid from the ranges
        xx, yy = np.meshgrid(rx, ry)

        # Convert the mesh grid to float type
        xx = xx.astype(float)
        yy = yy.astype(float)

        # Return the mesh grid as two 2D arrays for x and y coordinates
        return xx, yy
    
    def _update_neighborhood_radius(self, 
                                t: int
                                ) -> float:
        """
        Update the neighborhood radius based on the current iteration.
        
        Args:
            t (int): The current iteration number.
            
        Returns:
            float: The updated neighborhood radius.
        """
        # Calculate the updated neighborhood radius using the exponential decay formula
        # self.neigh_rad_0 is the initial neighborhood radius
        # t is the current iteration
        # self.lmbda is the constant that controls the decay rate
        return self._neigh_rad_0 * np.exp(-t / self._lmbda)
    
    def _update_learning_rate(self,
                          t: int
                          ) -> float:
        """
        Update the learning rate based on the current iteration.
        
        Args:
            t (int): The current iteration number.
            
        Returns:
            float: The updated learning rate.
        """
        # Calculate the updated learning rate using the exponential decay formula
        # self.lr_0 is the initial learning rate
        # t is the current iteration
        # self.lmbda is the constant that controls the decay rate
        return self._lr_0 * np.exp(-t / self._lmbda)
    
    def _find_bmu(self,
              input: np.ndarray
              ) -> Tuple[int, int]:
        """
        Find the Best Matching Unit (BMU) for a given input vector.

        Args:
            input (np.ndarray): The input vector for which the BMU is to be found.
            
        Returns:
            Tuple[int, int]: The coordinates (x, y) of the BMU in the weight matrix.
        """
        # Calculate the Euclidean distance between the input vector and all weight vectors
        bmu = np.linalg.norm(np.subtract(self._weights, input), axis=-1).argmin()
        
        # Find the index of the minimum distance (BMU) and convert it to 2D coordinates
        return np.unravel_index(bmu, (self._w, self._h))
    
    def _calc_influence(self,
                    lr_t: float,
                    rad_t: float,
                    bmu_x: int,
                    bmu_y: int
                    ) -> np.ndarray:
        """
        Calculate the influence of the Best Matching Unit (BMU) on its neighboring nodes.
        
        Args:
            lr_t (float): The learning rate at iteration t.
            rad_t (float): The neighborhood radius at iteration t.
            bmu_x (int): The x-coordinate of the BMU.
            bmu_y (int): The y-coordinate of the BMU.
            
        Returns:
            np.ndarray: The influence of the BMU on each node.
        """
        # Calculate the squared radius for the Gaussian function
        d = 2 * rad_t * rad_t

        # Compute the squared distance from the BMU for all nodes
        ax = -np.square(self._xx - bmu_x)
        ay = -np.square(self._yy - bmu_y)

        # Calculate the influence of the BMU using a Gaussian function
        influence = lr_t * np.exp((ax + ay).T / d)

        return influence
    
    def _update_weights(self,
                   influence: np.ndarray,
                   input: np.ndarray
                   ) -> None:
        """
        Update the weights of the nodes in the network based on the input data and the influence of the BMU.
        
        Args:
            influence (np.ndarray): The influence of the BMU on each node in the network.
            input (np.ndarray): The input data vector.
            
        Returns:
            None
        """
        # Update the weights by applying the influence to the difference between the input and the current weights
        # np.einsum is used for element-wise multiplication and summation over specified axes
        self._weights += np.einsum('ij, ijk->ijk', influence, input-self._weights)
        return 


    @property
    def weights(self) -> np.ndarray:
        """
        Getter for the weights attribute.
        
        Returns:
            np.ndarray: The current weights of the network.
        """
        return self._weights
    
    @weights.setter
    def weights(self, value: np.ndarray):
        """
        Setter for the weights attribute.

        Args:
            value (np.ndarray): The new weights to set.
        """
        self._weights = value

    
    def train(self) -> None:
        """
        Train the SOM using the given data and update the weights accordingly.
        
        Returns:
            np.ndarray: The updated weights after training.
        """
        # Iterate over the number of training iterations
        for t in range(self._n_iters):
            # Update the learning rate based on the current iteration
            lr_t = self._update_learning_rate(t)
            # Update the neighborhood radius based on the current iteration
            rad_t = self._update_neighborhood_radius(t)
            
            # Iterate over each data point in the dataset
            for vt in self._data:
                # Find the Best Matching Unit (BMU) for the current data point
                bmu_x, bmu_y = self._find_bmu(vt)
                # Calculate the influence of the BMU on the neighboring nodes
                influence = self._calc_influence(lr_t, rad_t, bmu_x, bmu_y)
                # Update the weights based on the influence and the difference between the data point and the weights
                self._update_weights(influence, vt)
        
        return 

if __name__ == '__main__':

    # set seed 
    np.random.seed(99)
    # Generate data
    input_data = np.random.random((10,3))
    som_ins1 = SOM(input_data, 100, 10, 10, 0.1)
    som_ins1.train()
    image_data = som_ins1.weights

    fig = plt.subplots()
    plt.imsave('100.png', image_data)
    plt.imshow(image_data)

    # Generate data
    input_data = np.random.random((10,3))
    som_ins2 = SOM(input_data, 1000, 100, 100, 0.1)
    som_ins2.train()
    image_data = som_ins2.weights

    fig = plt.subplots()
    plt.imsave('1000.png', image_data)
    plt.imshow(image_data)
