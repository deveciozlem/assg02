import numpy as np
import pandas as pd
import sklearn
#import unittest
from twisted.trial import unittest
from assg_tasks import load_dataset
from assg_tasks import flatten_image_dataset
from assg_tasks import standardize_image_dataset
from assg_tasks import sigmoid
from assg_tasks import initialize_parameters
from assg_tasks import forward_pass
from assg_tasks import backward_pass
from assg_tasks import optimize
from assg_tasks import predict


class test_flatten_image_dataset(unittest.TestCase):

    def setUp(self):
        self.train_set_x_orig, _, self.test_set_x_orig, _, _ = load_dataset()

    def test_random_matrix(self):
        image_dataset = np.random.randint(low=0, high=256, size=(256, 128, 64, 3), dtype=np.uint8)
        image_dataset_flattened = flatten_image_dataset(image_dataset)
        self.assertTrue(image_dataset_flattened.shape == (256, (128 * 64 * 3)))
        self.assertAlmostEqual(image_dataset_flattened[0,0], image_dataset[0,0,0,0])
        self.assertAlmostEqual(image_dataset_flattened[255, (128 * 64 * 3) - 1], image_dataset[255, 127, 63, 2])
        self.assertIsInstance(image_dataset_flattened.dtype, np.dtypes.UInt8DType)

    def test_train_set_images(self):
        image_dataset = self.train_set_x_orig
        image_dataset_flattened = flatten_image_dataset(image_dataset)
        num_images, width, height, channels = image_dataset.shape
        self.assertTrue(image_dataset_flattened.shape == (num_images, (width * height * channels)))
        self.assertAlmostEqual(image_dataset_flattened[0,0], image_dataset[0,0,0,0])
        self.assertAlmostEqual(image_dataset_flattened[num_images - 1, (width * height * channels) - 1], 
                               image_dataset[num_images - 1, width - 1, height - 1, channels - 1])
        self.assertIsInstance(image_dataset_flattened.dtype, np.dtypes.UInt8DType)

    def test_test_set_images(self):
        image_dataset = self.test_set_x_orig
        image_dataset_flattened = flatten_image_dataset(image_dataset)
        num_images, width, height, channels = image_dataset.shape
        self.assertTrue(image_dataset_flattened.shape == (num_images, (width * height * channels)))
        self.assertAlmostEqual(image_dataset_flattened[0,0], image_dataset[0,0,0,0])
        self.assertAlmostEqual(image_dataset_flattened[num_images - 1, (width * height * channels) - 1], 
                               image_dataset[num_images - 1, width - 1, height - 1, channels - 1])
        self.assertIsInstance(image_dataset_flattened.dtype, np.dtypes.UInt8DType)


class test_standardize_image_dataset(unittest.TestCase):

    def setUp(self):
        self.train_set_x_orig, _, self.test_set_x_orig, _, _ = load_dataset()

    def test_random_matrix(self):
        image_dataset = np.random.randint(low=0, high=256, size=(256, 128, 64, 3), dtype=np.uint8)
        image_dataset_standardized = standardize_image_dataset(image_dataset)
        self.assertTrue(image_dataset_standardized.shape == (256, (128 * 64 * 3)))
        self.assertAlmostEqual(image_dataset_standardized[0,0], image_dataset[0,0,0,0]/255.0)
        self.assertAlmostEqual(image_dataset_standardized[255, (128 * 64 * 3) - 1], image_dataset[255, 127, 63, 2]/255.0)
        self.assertAlmostEqual(image_dataset_standardized.max(), 1.0)
        self.assertAlmostEqual(image_dataset_standardized.min(), 0.0)
        self.assertIsInstance(image_dataset_standardized.dtype, np.dtypes.Float64DType)
        

    def test_train_set_images(self):
        image_dataset = self.train_set_x_orig
        image_dataset_standardized = standardize_image_dataset(image_dataset)
        num_images, width, height, channels = image_dataset.shape
        self.assertTrue(image_dataset_standardized.shape == (num_images, (width * height * channels)))
        self.assertAlmostEqual(image_dataset_standardized[0,0], image_dataset[0,0,0,0]/255.0)
        self.assertAlmostEqual(image_dataset_standardized[num_images - 1, (width * height * channels) - 1], 
                               image_dataset[num_images - 1, width - 1, height - 1, channels - 1]/255.0)
        self.assertAlmostEqual(image_dataset_standardized.max(), 1.0)
        self.assertAlmostEqual(image_dataset_standardized.min(), 0.0)
        self.assertIsInstance(image_dataset_standardized.dtype, np.dtypes.Float64DType)

    def test_test_set_images(self):
        image_dataset = self.test_set_x_orig
        image_dataset_standardized = standardize_image_dataset(image_dataset)
        num_images, width, height, channels = image_dataset.shape
        self.assertTrue(image_dataset_standardized.shape == (num_images, (width * height * channels)))
        self.assertAlmostEqual(image_dataset_standardized[0,0], image_dataset[0,0,0,0]/255.0)
        self.assertAlmostEqual(image_dataset_standardized[num_images - 1, (width * height * channels) - 1], 
                               image_dataset[num_images - 1, width - 1, height - 1, channels - 1]/255.0)
        self.assertAlmostEqual(image_dataset_standardized.max(), 1.0)
        self.assertAlmostEqual(image_dataset_standardized.min(), 0.0)
        self.assertIsInstance(image_dataset_standardized.dtype, np.dtypes.Float64DType)


class test_sigmoid(unittest.TestCase):

    def setUp(self):
        pass

    def test_input_scalar(self):
        s = sigmoid(3)
        self.assertAlmostEqual(s, 0.9525741268224334)

    def test_input_vector(self):
        # test a 1-d vector
        x = np.array([-5, 0, 3])
        s = sigmoid(x)
        self.assertTrue(np.allclose(s, np.array([0.00669285, 0.5, 0.95257413])))

    def test_input_matrix(self):
        # test a 3-d tensor
        x = np.linspace(-5, 5, 27).reshape((3,3,3))
        s = sigmoid(x)
        expected_s = np.array(
            [[[0.00669285, 0.00980136, 0.01433278],
              [0.02091496, 0.03042661, 0.04406926],
              [0.06342879, 0.09048789, 0.12751884]],
            
             [[0.17675903, 0.23978727, 0.31664553],
              [0.40501421, 0.5,        0.59498579],
              [0.68335447, 0.76021273, 0.82324097]],
            
             [[0.87248116, 0.90951211, 0.93657121],
              [0.95593074, 0.96957339, 0.97908504],
              [0.98566722, 0.99019864, 0.99330715]]]
        )
        self.assertTrue(np.allclose(s, expected_s))

    def test_input_list(self):
        # a regular list still does not work for a vectorized function
        x = [-5, 0, 3]
        with self.assertRaises(TypeError):
            s = sigmoid(x)


class test_initialize_parameters(unittest.TestCase):

    def setUp(self):
        pass

    def test_input_dim5(self):
        w, b = initialize_parameters(5)
        self.assertTrue(w.shape == (5,))
        self.assertAlmostEqual(w.sum(), 0.0)
        self.assertAlmostEqual(b, 0.0)
        self.assertIsInstance(w.dtype, np.dtypes.Float64DType)

    def test_input_dim12288(self):
        w, b = initialize_parameters(12288)
        self.assertTrue(w.shape == (12288,))
        self.assertAlmostEqual(w.sum(), 0.0)
        self.assertAlmostEqual(b, 0.0)
        self.assertIsInstance(w.dtype, np.dtypes.Float64DType)


class test_forward_pass(unittest.TestCase):

    def setUp(self):
        pass

    def test_input_dim5(self):
        w = np.linspace(1.0, 2.0, 5)
        x = np.linspace(-1.0, 1.0, 10).reshape(2, 5)
        b = 0.5
        a = forward_pass(x, w, b)
        expected_a = np.array([0.04265125, 0.99463363])
        self.assertTrue(np.allclose(a, expected_a))

    def test_input_dim100(self):
        w = np.linspace(-0.7, 0.5, 100)
        x = np.linspace(0.0, 1.0, 500).reshape(5, 100)
        b = 0.5
        a = forward_pass(x, w, b)
        expected_a = np.array([0.82230812, 0.38415625, 0.07756133, 0.01120685, 0.00152541])
        self.assertTrue(np.allclose(a, expected_a))

    def test_expected_case(self):
        w = np.array([1.0, 2.0])
        b = 2.0
        x = np.array([[ 1.0,  3.0],
                    [ 2.0,  4.0],
                    [-1.0, -3.2]])
        y =  np.array([1, 0, 1])
        a = forward_pass(x, w, b)
        expected_a = np.array([0.99987661, 0.99999386, 0.00449627])
        self.assertTrue(np.allclose(a, expected_a))       


class test_backward_pass(unittest.TestCase):

    def setUp(self):
        pass

    def test_input_dim5(self):
        w = np.linspace(1.0, 2.0, 5)
        b = 0.5
        x = np.linspace(-1.0, 1.0, 10).reshape(2, 5)
        y = np.array([0, 1])
        a = forward_pass(x, w, b)
        expected_a = np.array([0.04265125, 0.99463363])
        self.assertTrue(np.allclose(a, expected_a))
        cost, dw, db = backward_pass(x, y, a)
        expected_cost = 0.024484179813433496
        expected_dw = np.array([-0.02162376, -0.01748099, -0.01333823, -0.00919546, -0.0050527 ])
        expected_db = 0.01864243964091697

    def test_input_dim100(self):
        w = np.linspace(-0.7, 0.5, 100)
        b = 0.5
        x = np.linspace(0.0, 1.0, 500).reshape(5, 100)
        y = np.array([0, 1, 0, 1, 0])
        a = forward_pass(x, w, b)
        expected_a = np.array([0.82230812, 0.38415625, 0.07756133, 0.01120685, 0.00152541])
        self.assertTrue(np.allclose(a, expected_a))
        cost, dw, db = backward_pass(x, y, a)
        expected_cost = 1.4515802271220486
        expected_dw = np.array(
            [-0.13711418, -0.13739605, -0.13767791, -0.13795977, -0.13824163,
             -0.13852349, -0.13880535, -0.13908721, -0.13936907, -0.13965093,
             -0.13993279, -0.14021465, -0.14049651, -0.14077837, -0.14106023,
             -0.14134209, -0.14162395, -0.14190581, -0.14218767, -0.14246953,
             -0.1427514 , -0.14303326, -0.14331512, -0.14359698, -0.14387884,
             -0.1441607 , -0.14444256, -0.14472442, -0.14500628, -0.14528814,
             -0.14557   , -0.14585186, -0.14613372, -0.14641558, -0.14669744,
             -0.1469793 , -0.14726116, -0.14754302, -0.14782489, -0.14810675,
             -0.14838861, -0.14867047, -0.14895233, -0.14923419, -0.14951605,
             -0.14979791, -0.15007977, -0.15036163, -0.15064349, -0.15092535,
             -0.15120721, -0.15148907, -0.15177093, -0.15205279, -0.15233465,
             -0.15261651, -0.15289837, -0.15318024, -0.1534621 , -0.15374396,
             -0.15402582, -0.15430768, -0.15458954, -0.1548714 , -0.15515326,
             -0.15543512, -0.15571698, -0.15599884, -0.1562807 , -0.15656256,
             -0.15684442, -0.15712628, -0.15740814, -0.15769   , -0.15797186,
             -0.15825372, -0.15853559, -0.15881745, -0.15909931, -0.15938117,
             -0.15966303, -0.15994489, -0.16022675, -0.16050861, -0.16079047,
             -0.16107233, -0.16135419, -0.16163605, -0.16191791, -0.16219977,
             -0.16248163, -0.16276349, -0.16304535, -0.16332721, -0.16360908,
             -0.16389094, -0.1641728 , -0.16445466, -0.16473652, -0.16501838]            
        )
        expected_db = -0.1406484075530334

    def test_expected_case(self):
        w = np.array([1.0, 2.0])
        b = 2.0
        x = np.array([[ 1.0,  3.0],
                    [ 2.0,  4.0],
                    [-1.0, -3.2]])
        y =  np.array([1, 0, 1])
        a = forward_pass(x, w, b)
        expected_a = np.array([0.99987661, 0.99999386, 0.00449627])
        self.assertTrue(np.allclose(a, expected_a))
        cost, dw, db = backward_pass(x, y, a)
        expected_cost = 5.801545319394553
        expected_dw = np.array([0.99845601, 2.39507239])
        expected_db = 0.001455578136784208
        self.assertAlmostEqual(cost, expected_cost)
        self.assertTrue(np.allclose(dw, expected_dw))
        self.assertAlmostEqual(db, expected_db)


class test_optimize(unittest.TestCase):

    def setUp(self):
        pass

    def test_expected_case(self):
        w = np.array([1.0, 2.0])
        b = 2.0
        x = np.array([[ 1.0,  3.0],
                    [ 2.0,  4.0],
                    [-1.0, -3.2]])
        y =  np.array([1, 0, 1])

        w, b, costs = optimize(x, y, w, b, num_iterations=100, learning_rate=0.009, print_cost=False)
        expected_w = np.array([0.19033591, 0.12259159])
        self.assertTrue(np.allclose(w, expected_w))
        expected_b = 1.9253598300845747
        self.assertAlmostEqual(b, expected_b)
        self.assertEqual(len(costs), 100)
        expected_cost_0 = 5.801545319394553
        self.assertAlmostEqual(costs[0], expected_cost_0)
        expected_cost_99 = 1.0784313398164709
        self.assertAlmostEqual(costs[99], expected_cost_99)


class test_predict(unittest.TestCase):

    def setUp(self):
        pass

    def test_expected_case(self):
        w = np.array([0.1124579, 0.23106775])
        b = -0.3
        x = np.array([[ 1.0, 1.2],
                    [-1.1, 2.0],
                    [-3.2, 0.1]])
        y_pred = predict(x, w, b)
        expected_y_pred = np.array([1.0, 1.0, 0.0])
        self.assertTrue(np.allclose(y_pred, expected_y_pred))
