#!/usr/bin/env python3

# This code does the following:
# 1. Load training images, continuous and discrete values from JSON files

import os
import sys
import json
import logging
import tensorflow as tf


class DataLoader:
    """Load data from JSON files"""

    def __init__(self, masterJsonFile):
        self.masterJsonFile = masterJsonFile
        self.config = None
        self.num_observations = 0
        self.num_episodes = 0
        self.num_patches = 0
        self.input_dim = 768
        self.continuous_dim = 0
        self.discrete_dim = 0
        self.x_size = 80  # Target image size
        self.y_size = 64
        self.x_scale = 0
        self.y_scale = 0

    def image_to_patches(self, image_file) -> tf.Tensor:
        """Load and extract image patches"""

        # Read PNG file
        logging.info("Creating image tokens, read image: %s", image_file)
        image = tf.io.read_file(image_file)

        image = tf.image.decode_png(image, channels=3)  # decode PNG
        image = tf.cast(image, dtype=tf.float32)  # cast to float32
        image = image / 255.0  # normalize to [0, 1]

        # Resize image to x_size, y_size
        image = tf.image.resize(image, (self.x_size, self.y_size))
        logging.debug("  image shape: %s", image.shape)

        # Split image into num_patches of size 16x16
        image = tf.image.extract_patches(
            images=tf.expand_dims(image, axis=0),
            sizes=[1, 16, 16, 1],
            strides=[1, 16, 16, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        logging.debug("  patch extracted image: %s", image.shape)

        # Reshape to (1, num_patches, input_dim)
        return tf.reshape(image, (1, self.num_patches, self.input_dim))

    def create_encoding(self):
        """Create sequence encoding tensor"""
        arr = []

        # 0 - image patch embedding
        # 1 - continuous value embedding
        # 2 - discrete embedding (actions, texts)
        for i in range(self.num_observations):
            arr.extend([0] * self.num_patches + [1, 2])
        # logging.debug("encoding: %s", arr)

        logging.info("Creating encoding..")
        return tf.constant([arr])

    def encode_continuous_value(self, value):
        """Encode continuous value"""

        logging.info("Creating continuous value..")
        # resize value to input_dim
        value = value + [0.0] * (self.input_dim - len(value))
        value = tf.cast(value, dtype=tf.float32)
        logging.debug("  value: %s", value)
        return tf.reshape(value, (1, 1, self.input_dim))

    def encode_discrete_value(self, value: int):
        """Encode discrete value"""
        logging.info("Creating discrete value..")
        arr = [0.0] * self.input_dim
        arr[0] = value
        logging.debug("  arr: %s", arr)
        return tf.reshape(arr, (1, 1, self.input_dim))

    def encode_row_pos(self):
        """Create row_pos tensor"""

        logging.info("Creating row_pos..")
        # row_pos from
        arr1 = [i / self.y_scale for i in range(self.y_scale)]
        arr1 *= self.x_scale
        arr1.extend([0, 0])
        # Repeat this array num_observations times
        arr1 *= self.num_observations
        logging.debug(" arr1: %s", arr1)

        # row_pos to
        arr2 = [(i + 1) / self.y_scale for i in range(self.y_scale)]
        arr2 *= self.x_scale
        arr2.extend([0, 0])
        # Repeat this array num_observationss times
        arr2 *= self.num_observations
        logging.debug(" arr2: %s", arr2)

        return (
            tf.constant([arr1]),  # pos_from
            tf.constant([arr2]),  # pos_to
        )

    def encode_col_pos(self):
        """Create col_pos tensor"""
        logging.info("Creating col_pos..")
        arr1 = []
        arr2 = []

        # col_pos from
        for i in range(self.x_scale):
            arr1.extend([i / self.x_scale] * self.y_scale)

        arr1.extend([0, 0])
        # Repeat this array num_observations times
        arr1 *= self.num_observations
        logging.debug(" arr1: %s", arr1)

        for i in range(self.x_scale):
            arr2.extend([(i + 1) / self.x_scale] * self.y_scale)

        arr2.extend([0, 0])
        # Repeat this array num_observations times
        arr2 *= self.num_observations
        logging.debug(" arr2: %s", arr2)

        return (
            tf.constant([arr1]),  # pos_from
            tf.constant([arr2]),  # pos_to
        )

    def encode_obs(self):
        """Create obs tensor"""

        logging.info("Creating obs..")
        # obs token
        arr1 = [i for i in range(self.num_patches + 2)]
        arr1 = arr1 * self.num_observations
        logging.debug("arr1: %s", arr1)

        arr2 = [1] * (self.num_patches + 1) + [0]
        arr2 = arr2 * self.num_observations
        logging.debug("arr2: %s", arr2)

        return (
            tf.constant([arr1]),  # obs token
            tf.constant([arr2]),  # obs token masking (for action tokens)
        )

    def process_episode(self, episode_config_file, prefix):
        """Process episode config file and create input_ids tensor"""

        input_ids = None
        input_array = []

        logging.debug("Loading episode_config: %s", episode_config_file)
        # Load episode config
        with open(episode_config_file) as f:
            episode_config = json.load(f)

            # For each step in episode config, create input_ids
            for key in episode_config["steps"]:
                logging.debug("  jointAngles: %s", key["jointAngles"])
                logging.debug("   action: %s", key["action"])
                logging.debug("    snapshot: %s", key["snapshot"])

                img_file = prefix + key["snapshot"]
                logging.debug("    img_file: %s", img_file)

                image = self.image_to_patches(img_file)
                continuous_value = self.encode_continuous_value(key["jointAngles"])
                discrete_value = self.encode_discrete_value(key["action"])

                input_array.append(image)
                input_array.append(continuous_value)
                input_array.append(discrete_value)

            logging.debug("input_array size: %s", len(input_array))
            input_ids = tf.concat(
                input_array,  # repeat num_observations times
                axis=1,
            )
            logging.info("input_ids shape: %s", input_ids.shape)
            return input_ids

    def load(self):
        """Load data from JSON files"""

        try:
            self.config = json.load(open(self.masterJsonFile))
        except Exception as e:
            logging.error("Error loading JSON file: %s", e)
            sys.exit(1)

        exp_info = self.config["experiment"]
        self.num_observations = exp_info["stepsPerEpisode"]
        self.num_episodes = exp_info["totalEpisodes"]
        logging.debug("num_observations: %s", self.num_observations)

        prefix = os.path.dirname(self.masterJsonFile) + "/"
        logging.debug("#### prefix: %s", prefix)
        self.num_patches = (self.x_size // 16) * (self.y_size // 16)
        logging.debug("num_patches: %s", self.num_patches)

        self.x_scale = self.x_size // 16
        self.y_scale = self.y_size // 16

        all_ids = None
        all_encoding = None
        all_row_pos = None
        all_col_pos = None
        all_obs = None

        # Loop through each episode config file
        # for episode_config_file in self.config["episodes"]:
        for i in range(self.num_episodes):
            episode_config_file = prefix + self.config["episodes"][i]
            logging.debug("episode_config_file: %s", episode_config_file)

            input_ids = self.process_episode(episode_config_file, prefix=prefix)
            logging.debug("input_ids shape: %s", input_ids.shape)
            # Append to all_ods
            if all_ids is None:
                all_ids = input_ids
            else:
                all_ids = tf.concat([all_ids, input_ids], axis=1)

            encoding = self.create_encoding()
            logging.debug(" encoding shape: %s", encoding)
            # Append to all_encoding
            if all_encoding is None:
                all_encoding = encoding
            else:
                all_encoding = tf.concat([all_encoding, encoding], axis=1)
            logging.debug("  encoding shape: %s", encoding.shape)

            row_pos = self.encode_row_pos()
            # Append to all_row_pos
            if all_row_pos is None:
                all_row_pos = row_pos
            else:
                all_row_pos = (
                    tf.concat([all_row_pos[0], row_pos[0]], axis=1),
                    tf.concat([all_row_pos[1], row_pos[1]], axis=1),
                )
            logging.debug("  row_pos shape: %s, %s", row_pos[0].shape, row_pos[1].shape)

            col_pos = self.encode_col_pos()
            # Append to all_col_pos
            if all_col_pos is None:
                all_col_pos = col_pos
            else:
                all_col_pos = (
                    tf.concat([all_col_pos[0], col_pos[0]], axis=1),
                    tf.concat([all_col_pos[1], col_pos[1]], axis=1),
                )
            logging.debug("  col_pos shape: %s, %s", col_pos[0].shape, col_pos[1].shape)

            obs = self.encode_obs()
            # Append to all_obs
            if all_obs is None:
                all_obs = obs
            else:
                all_obs = (
                    tf.concat([all_obs[0], obs[0]], axis=1),
                    tf.concat([all_obs[1], obs[1]], axis=1),
                )
            logging.debug("  obs shape: %s, %s", obs[0].shape, obs[1].shape)

        # Print shapes
        logging.info("all_ids shape: %s", all_ids.shape)
        logging.info("all_encoding shape: %s", all_encoding.shape)
        logging.info(
            "all_row_pos shape: %s, %s", all_row_pos[0].shape, all_row_pos[1].shape
        )
        logging.info(
            "all_col_pos shape: %s, %s", all_col_pos[0].shape, all_col_pos[1].shape
        )
        logging.info("all_obs shape: %s, %s", all_obs[0].shape, all_obs[1].shape)

        return (all_ids, all_encoding, all_row_pos, all_col_pos, all_obs)
