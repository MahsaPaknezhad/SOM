import pytest
import numpy as np
from som import SOM  # Adjust the import according to your project structure

@pytest.fixture
def som_instance():
    np.random.seed(42)
    # Setup a sample SOM instance for testing
    input_data = np.array([[0.24689827, 0.93347329],
       [0.43796128, 0.97143973],
       [0.18799005, 0.29907733],
       [0.43021339, 0.84576234],
       [0.09981293, 0.56649221]])
    n_max_iterations = 100
    width = 5
    height = 5
    initial_lr = 0.1
    return SOM(input_data, n_max_iterations, width, height, initial_lr)

def test_som_initialization(som_instance):
    # Test initialization of SOM
    assert isinstance(som_instance._data, np.ndarray)
    assert som_instance._data.shape == (5, 2)
    assert isinstance(som_instance._w, int)
    assert isinstance(som_instance._h, int)
    assert som_instance._w > 0
    assert som_instance._h > 0
    assert isinstance(som_instance._n_iters, int)
    assert som_instance._n_iters > 0
    assert isinstance(som_instance._lr_0, (int, float))
    assert isinstance(som_instance._neigh_rad_0, float)
    assert isinstance(som_instance._lmbda, float)
    assert som_instance._weights.shape == (som_instance._w, som_instance._h, som_instance._data.shape[1])
    assert som_instance._xx.shape == (som_instance._w, som_instance._h)
    assert som_instance._yy.shape == (som_instance._w, som_instance._h)

def test_init_mesh(som_instance):
    # Test the _init_mesh method
    xx, yy = som_instance._init_mesh(5, 5)
    assert xx.shape == (5, 5)
    assert yy.shape == (5, 5)
    assert xx.dtype == float
    assert yy.dtype == float

def test_update_neighborhood_radius(som_instance):
    # Test the _update_neighborhood_radius method
    t = 10
    rad_t = som_instance._update_neighborhood_radius(t)
    expected_rad_t = som_instance._neigh_rad_0 * np.exp(-t / som_instance._lmbda)
    assert np.isclose(rad_t, expected_rad_t)

def test_update_learning_rate(som_instance):
    # Test the _update_learning_rate method
    t = 10
    lr_t = som_instance._update_learning_rate(t)
    expected_lr_t = som_instance._lr_0 * np.exp(-t / som_instance._lmbda)
    assert np.isclose(lr_t, expected_lr_t)

def test_find_bmu(som_instance):
    # Test the _find_bmu method
    input_vec = np.array([0.5, 0.2])
    bmu_x, bmu_y = som_instance._find_bmu(input_vec)
    print(bmu_x)
    print(type(bmu_x))
    assert isinstance(bmu_x, np.int64)
    assert isinstance(bmu_y, np.int64)
    assert 0 <= bmu_x < som_instance._w
    assert 0 <= bmu_y < som_instance._h

def test_calc_influence(som_instance):
    # Test the _calc_influence method
    lr_t = 0.1
    rad_t = 1.0
    bmu_x = 2
    bmu_y = 2
    influence = som_instance._calc_influence(lr_t, rad_t, bmu_x, bmu_y)
    assert influence.shape == (som_instance._w, som_instance._h)
    assert np.all(influence >= 0)

def test_update_weights(som_instance):
    # Test the _update_weights method
    influence = np.ones((som_instance._w, som_instance._h))
    input_vec = np.array([0.5, 0.2])
    old_weights = np.copy(som_instance._weights)
    som_instance._update_weights(influence, input_vec)
    assert not np.array_equal(som_instance._weights, old_weights)

def test_weights_property(som_instance):
    # Test the weights property getter and setter
    weights = som_instance.weights
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (som_instance._w, som_instance._h, 2)
    
    new_weights = np.random.random((som_instance._w, som_instance._h, 2))
    som_instance.weights = new_weights
    assert np.array_equal(som_instance.weights, new_weights)

def test_train(som_instance):
    # Test the train method 
    som_instance.train()
    # Check that weights are updated
    assert isinstance(som_instance.weights, np.ndarray)
    assert som_instance.weights.shape == (som_instance._w, som_instance._h, 2)
    expected_weights = np.array([
        [[0.40429235, 0.95965226],
         [0.40661285, 0.93593981],
         [0.41744331, 0.87715781],
         [0.4242152,  0.84797352],
         [0.41949295, 0.83417401]],
        
        [[0.34382321, 0.94712732],
         [0.34897659, 0.92439551],
         [0.37410096, 0.86098658],
         [0.39394581, 0.81595908],
         [0.38227981, 0.77218808]],
        
        [[0.27570074, 0.92184422],
         [0.26307643, 0.86989224],
         [0.25315176, 0.74946694],
         [0.26310897, 0.63728832],
         [0.25958704, 0.54679026]],
        
        [[0.23010387, 0.85958503],
         [0.18519415, 0.74117356],
         [0.14998448, 0.59470591],
         [0.16024338, 0.4703797],
         [0.18129015, 0.38001625]],
        
        [[0.18067855, 0.74038303],
         [0.1353638,  0.62320028],
         [0.1225076,  0.53815184],
         [0.14507387, 0.44048138],
         [0.17230694, 0.35283815]]
    ])
    assert np.allclose(som_instance.weights, expected_weights), "The SOM weights do not match the expected values."
