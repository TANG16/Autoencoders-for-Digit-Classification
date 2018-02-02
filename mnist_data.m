function [mnist_train_images,mnist_train_labels,mnist_test_images,mnist_test_labels] = mnist_data()


mnist_train_images = load_images('mnist_data/train-images.idx3-ubyte');
mnist_train_labels = load_labels('mnist_data/train-labels.idx1-ubyte');

mnist_test_images = load_images('mnist_data/t10k-images.idx3-ubyte');
mnist_test_labels = load_labels('mnist_data/t10k-labels.idx1-ubyte');

end