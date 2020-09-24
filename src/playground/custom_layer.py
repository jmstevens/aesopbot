#__init__ , where you can do all input-independent initialization
#build, where you know the shapes of the input tensors and can do the rest of the initialization
#call, where you do the forward computation
import tensorflow as tf
print(tf.test.is_gpu_available())

# In the tf.keras.layers package, layers are objects. To construct a layer,
# simply construct the object. Most layers take as a first argument the number
# of output dimensions / channels.
layer = tf.keras.layers.Dense(100)
# The number of input dimensions is often unnecessary, as it can be inferred
# the first time the layer is used, but it can be provided if you want to
# specify it manually, which is useful in some complex models.
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

print(layer(tf.zeros([10,5])))
# Layers have many useful methods. For example, you can inspect all variables
# in a layer using `layer.variables` and trainable variables using
# `layer.trainable_variables`. In this case a fully-connected layer
# will have variables for weights and biases.
print(layer.variables)
# Layers have many useful methods. For example, you can inspect all variables
# in a layer using `layer.variables` and trainable variables using
# `layer.trainable_variables`. In this case a fully-connected layer
# will have variables for weights and biases.
print(layer.variables)


## Custom layer
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", 
                                      shape=[int(input_shape[-1]),
                                             self.num_outputs])
    def call(self, input):
        return tf.matmul(input, self.kernel)

layer = MyDenseLayer(10)
_ = layer(tf.zeros([10, 5]))
print([var.name for var in layer.trainable_variables])
