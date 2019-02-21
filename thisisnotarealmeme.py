import tensorflow as tf

def generator(noise):
    first = tf.layers.dense(noise, 100)
    second = tf.layers.dense(first, 100)
    out = tf.layers.dense(second, 10000)
    return out

def discrimator(sample):
    first = tf.layers.dense(sample, 100)
    second = tf.layers.dense(first, 100)
    third = tf.layers.dense(second, 10000)
    out = tf.layers.dense(third, 1)
    return out

#Eli- noise function should go here
noise = 0.3

#Eli - data plz
sample = tf.placeholder(tf.float32,[None,2])


generated = generator(noise)
nonscaled_sample_identification, probability_sample_identification = discrimator(sample)
nonscaled_generated_identification, probability_generated_identification = discrimator(generated)