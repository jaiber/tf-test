#!/usr/bin/python3

import sys
import tensorflow as tf

def main():
    t1 = tf.constant([[[1, 2, 3], [4, 5, 6]]])
    t2 = tf.constant([[[7, 8, 9], [10, 11, 12]]])

    tf.print("t1: ", t1)
    tf.print("t2: ", t2)

    t3 = tf.concat([t1, t2], axis=2)
    tf.print("t3: ", t3)

if __name__ == "__main__":
    main()