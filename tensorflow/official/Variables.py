import tensorflow as tf

my_tensor = tf.constant([[1.0, 2.0],[3.0, 4.0]])
my_variable= tf.Variable(my_tensor)

bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5 + 4j, 6 + 1j])

print("Shape: ", my_variable.shape)
print("DType: ", my_variable.dtype)
print("As NumPy: ", my_variable.numpy())
print()
print("Most tensor operations work on variables as expected, although variables cannot be reshaped.")
print()

print("A variable:", my_variable)
print("\nViewed as a tensor:", tf.convert_to_tensor(my_variable))
print("\nIndex of highest value:", tf.math.argmax(my_variable))

# This creates a new tensor; it does not reshape the variable.
print("\nCopying and reshaping: ", tf.reshape(my_variable, [1,4]))
print()
print("variables are backed by tensors. You can reassign the tensor using tf.Variable.assign.")
a= tf.Variable([2.0, 3.0])
a.assign([1,2])
print(a)
print()
print("Not allowed as it resizes the variable: ")
try:
    a.assign([1.0, 2.0, 3.0])
except Exception as e:
    print(f"{type(e).__name__}: {e}")
print()
print("Two variables will not share the same memory.")
print()
a = tf.Variable([2.0, 3.0])
# Create b based on the value of a
b = tf.Variable(a)
a.assign([5, 6])

# a and b are different
print(a.numpy())
print(b.numpy())

# There are other versions of assign
print(a.assign_add([2,3]).numpy())  # [7. 9.]
print(a.assign_sub([7,9]).numpy())  # [0. 0.]
print()
print("-----Lifecycles, naming, and watching-----")
print(" tf.Variable instance have the same lifecycle as other Python objects. When there are no references to a variable it is automatically deallocated.")
print()
a = tf.Variable(my_tensor, name="Mark")
b = tf.Variable(my_tensor + 1, name="Mark")
print("a==b: \n",a == b)
print()
print("You can turn off gradients for a variable by setting trainable to false at creation.")
step_counter = tf.Variable(1, trainable=False)
print()
print("-----Placing variables and tensors-----")
print()
with tf.device('CPU:0'):
    a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print("tf.matmul(a, b):\n",c)

print()
print("---It's possible to set the location of a variable or tensor on one device and do the computation on another device. ---")
print()
with tf.device('CPU:0'):
    a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.Variable([[1.0, 2.0, 3.0]])

with tf.device('GPU:0'):
    k = a * b
    print("k:", k)