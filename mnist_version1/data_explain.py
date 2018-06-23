from tensorflow.examples.tutorials.mnist import input_data

# 载入MNIST数据集，如果指定目录没有数据集，则会从网络上下载
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# 打印 Training data size
print("Training data size is : ", mnist.train.num_examples)

# 打印validating data size
print("Validating data size is :", mnist.validation.num_examples)

# 打印 Testing data size
print("Testing data size is : ", mnist.test.num_examples)

# 打印 Example training data
print("Example training data: ", mnist.train.images[0])

# 打印 Example training data lable
print("Example training data lable is :", mnist.train.labels[0])
