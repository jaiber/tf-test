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
        logging.debug("Creating image tokens, read image: %s", image_file)
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
        
        """
        # Code to save each patch as png image, for debugging
        image = tf.reshape(image, (1, self.num_patches, self.input_dim))
        logging.info("  patch extracted image: %s", image.shape)

        # Save each patch as png images
        for i in range(self.num_patches):
            patch = image[0, i, :]
            patch = tf.reshape(patch, (16, 16, 3))
            patch = tf.image.convert_image_dtype(patch, dtype=tf.uint8)
            patch = tf.image.encode_png(patch)
            patch_file = "patch-" + str(i) + ".png"
            tf.io.write_file(patch_file, patch)
            logging.debug("  patch: %s", patch_file)

        sys.exit(0)
        """

        # Reshape to (1, num_patches, input_dim)
        return tf.reshape(image, (1, self.num_patches, self.input_dim))

    def create_encoding(self, continuous_dim=True, discrete_dim=True):
        """Create sequence encoding tensor"""
        arr = []

        # 0 - image patch embedding
        # 1 - continuous value embedding
        # 2 - discrete embedding (actions, texts)
        value = [0] * self.num_patches
        if continuous_dim:
            value += [1]
        if discrete_dim:
            value += [2]
        for i in range(self.num_observations):
            arr.extend(value)
        # logging.debug("encoding: %s", arr)

        logging.debug("Creating encoding..")
        return tf.constant([arr])

    def encode_continuous_value(self, value):
        """Encode continuous value"""

        logging.debug("Creating continuous value..")
        # resize value to input_dim
        value = value + [0.0] * (self.input_dim - len(value))
        value = tf.cast(value, dtype=tf.float32)
        logging.debug("  value: %s", value)
        return tf.reshape(value, (1, 1, self.input_dim))

    def encode_discrete_value(self, value: int):
        """Encode discrete value"""
        logging.debug("Creating discrete value..")
        arr = [0.0] * self.input_dim
        arr[0] = value
        logging.debug("  arr: %s", arr)
        return tf.reshape(arr, (1, 1, self.input_dim))

    def encode_row_pos(self):
        """Create row_pos tensor"""

        logging.debug("Creating row_pos..")
        # row_pos from
        arr1 = [i / self.y_scale for i in range(self.y_scale)]
        arr1 *= self.x_scale
        # arr1.extend([0, 0]) # For continuous and discrete values
        # Repeat this array num_observations times
        arr1 *= self.num_observations
        logging.debug(" arr1: %s", arr1)

        # row_pos to
        arr2 = [(i + 1) / self.y_scale for i in range(self.y_scale)]
        arr2 *= self.x_scale
        # arr2.extend([0, 0]) # For continuous and discrete values
        # Repeat this array num_observationss times
        arr2 *= self.num_observations
        logging.debug(" arr2: %s", arr2)

        return (
            tf.constant([arr1]),  # pos_from
            tf.constant([arr2]),  # pos_to
        )

    def encode_col_pos(self):
        """Create col_pos tensor"""
        logging.debug("Creating col_pos..")
        arr1 = []
        arr2 = []

        # col_pos from
        for i in range(self.x_scale):
            arr1.extend([i / self.x_scale] * self.y_scale)

        # arr1.extend([0, 0])
        # Repeat this array num_observations times
        arr1 *= self.num_observations
        logging.debug(" arr1: %s", arr1)

        for i in range(self.x_scale):
            arr2.extend([(i + 1) / self.x_scale] * self.y_scale)

        # arr2.extend([0, 0])
        # Repeat this array num_observations times
        arr2 *= self.num_observations
        logging.debug(" arr2: %s", arr2)

        return (
            tf.constant([arr1]),  # pos_from
            tf.constant([arr2]),  # pos_to
        )

    def encode_obs(self, mask_image=False, mask_continuous=False, mask_discrete=False):
        """Create obs tensor"""

        logging.debug("Creating obs..")
        # obs token
        #arr1 = [i for i in range(self.num_patches + 2)]
        arr1 = [i for i in range(self.num_patches)]
        arr1 = arr1 * self.num_observations
        logging.debug("arr1: %s", arr1)

        imask = [0] if mask_image else [1]
        cmask = [0] if mask_continuous else [1]
        dmask = [0] if mask_discrete else [1]

        arr2 = imask * (self.num_patches)  # + cmask + dmask  # Don't mask discrete tokens
        arr2 = arr2 * self.num_observations
        logging.debug("arr2: %s", arr2)

        return (
            tf.constant([arr1]),  # obs token
            tf.constant([arr2]),  # obs token masking (for action tokens)
        )

    def process_episode(self, episode_config_file, prefix):
        """Process episode config file and create input_ids tensor"""

        image_array = []
        continuous_array = []
        discrete_array = []
        action_array = []
        axis = 0

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

                #x = 0 if key == episode_config["steps"][-1] else key["action"]
                action_array.append(key["action"])

                # Append to tensors
                image_array.append(tf.constant(image, dtype=tf.float32))
                continuous_array.append(tf.constant(continuous_value, dtype=tf.float32))
                discrete_array.append(tf.constant(discrete_value, dtype=tf.float32))

            # Concatenate tensors
            image_tensor = tf.concat(image_array, axis=axis)
            continuous_tensor = tf.concat(continuous_array, axis=axis)
            discrete_tensor = tf.concat(discrete_array, axis=axis)
            logging.info(" image_tensor shape: %s", image_tensor.shape)
            logging.info("  continuous_tensor shape: %s", continuous_tensor.shape)
            logging.info("   discrete_tensor shape: %s", discrete_tensor.shape)

            return image_tensor, continuous_tensor, discrete_tensor, action_array

    def re_encode(self, input, row_len, col_dups):
        """Re-encode input to sliding window"""
        encoding = input[:row_len, :]
        columns = tf.tile(encoding[:, tf.newaxis, :], multiples=[1, col_dups, 1])
        return tf.reshape(columns, (row_len, -1))

    def load(
        self,
        mask_image=False,
        mask_continuous=False,
        mask_discrete=False,
        sliding_batch_size=4,
    ):
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

        image_tensors = None
        continuous_tensors = None
        discrete_tensors = None
        action_tensors = None

        all_encoding = None
        all_row_pos = None
        all_col_pos = None
        all_obs = None
        all_actions = None
        axis = 0

        # Loop through each episode config file
        # for episode_config_file in self.config["episodes"]:
        for i in range(self.num_episodes):
            episode_config_file = prefix + self.config["episodes"][i]
            logging.debug("episode_config_file: %s", episode_config_file)

            # input_ids, discrete_array = self.process_episode(episode_config_file, prefix=prefix)
            image_tensor, continuous_tensor, discrete_tensor, action_array = self.process_episode(
                episode_config_file, prefix=prefix
            )

            # Convert discrete_array to tensor
            action_array = tf.constant([action_array], dtype=tf.int32)
            #print("action_array: ", action_array.numpy())
            # one hot encoding
            action_array = tf.one_hot(action_array, depth=3, dtype=tf.int32)
            #print("action_array: ", action_array.numpy())
            action_array = tf.transpose(action_array, perm=[1, 0, 2])
            #print("action_array: ", action_array.numpy())
            #print("action array shape: ", action_array.shape)
            # tf.print("discrete_array: ", discrete_array)

            # Append to all_ods
            if image_tensors is None:
                image_tensors = image_tensor
                continuous_tensors = continuous_tensor
                discrete_tensors = discrete_tensor
                all_actions = action_array
            else:
                image_tensors = tf.concat([image_tensors, image_tensor], axis=axis)
                continuous_tensors = tf.concat([continuous_tensors, continuous_tensor], axis=axis)
                discrete_tensors = tf.concat([discrete_tensors, discrete_tensor], axis=axis)
                all_actions = tf.concat([all_actions, action_array], axis=axis)

            encoding = self.create_encoding(continuous_dim=False, discrete_dim=False)
            logging.debug(" encoding shape: %s", encoding)
            # Append to all_encoding
            if all_encoding is None:
                all_encoding = encoding
            else:
                all_encoding = tf.concat([all_encoding, encoding], axis=axis)
            logging.debug("  encoding shape: %s", encoding.shape)

            row_pos = self.encode_row_pos()
            # Append to all_row_pos
            if all_row_pos is None:
                all_row_pos = row_pos
            else:
                all_row_pos = (
                    tf.concat([all_row_pos[0], row_pos[0]], axis=axis),
                    tf.concat([all_row_pos[1], row_pos[1]], axis=axis),
                )
            logging.debug("  row_pos shape: %s, %s", row_pos[0].shape, row_pos[1].shape)

            col_pos = self.encode_col_pos()
            # Append to all_col_pos
            if all_col_pos is None:
                all_col_pos = col_pos
            else:
                all_col_pos = (
                    tf.concat([all_col_pos[0], col_pos[0]], axis=axis),
                    tf.concat([all_col_pos[1], col_pos[1]], axis=axis),
                )
            logging.debug("  col_pos shape: %s, %s", col_pos[0].shape, col_pos[1].shape)

            obs = self.encode_obs(
                mask_image=mask_image,
                mask_continuous=mask_continuous,
                mask_discrete=mask_discrete,
            )
            # Append to all_obs
            if all_obs is None:
                all_obs = obs
            else:
                all_obs = (
                    tf.concat([all_obs[0], obs[0]], axis=axis),
                    tf.concat([all_obs[1], obs[1]], axis=axis),
                )
            logging.debug("  obs shape: %s, %s", obs[0].shape, obs[1].shape)

        # Print shapes
        logging.info("image_tensors shape: %s", image_tensors.shape)
        logging.info("continuous_tensors shape: %s", continuous_tensors.shape)
        logging.info("discrete_tensors shape: %s", discrete_tensors.shape)
        logging.info("all_actions shape: %s", all_actions.shape)
        logging.info("all_encoding shape: %s", all_encoding.shape)
        logging.info("all_row_pos shape: %s, %s", all_row_pos[0].shape, all_row_pos[1].shape)
        logging.info("all_col_pos shape: %s, %s", all_col_pos[0].shape, all_col_pos[1].shape)
        logging.info("all_obs shape: %s, %s", all_obs[0].shape, all_obs[1].shape)

        # Shift actions by 1 step, and append [1, 0, 0] (0) - no action at the end
        actions = tf.concat([all_actions[1:], [[[1, 0, 0]]]], axis=0)
        print("all actions: ", all_actions.shape)
        print("actions: ", actions.shape)

        # Reshape factor for vertical stacking
        reshape_factor = self.num_observations * self.num_episodes
        print("reshape_factor: ", reshape_factor)
        
        # Reshape all_encoding, example: from (1, 120) to (6, 20)
        all_encoding = tf.reshape(all_encoding, (reshape_factor, -1))
        logging.info("all_encoding shape: %s", all_encoding.shape)
        all_row_pos = (
            tf.reshape(all_row_pos[0], (reshape_factor, -1)),
            tf.reshape(all_row_pos[1], (reshape_factor, -1)),
        )
        logging.info("all_row_pos shape: %s, %s", all_row_pos[0].shape, all_row_pos[1].shape)
        all_col_pos = (
            tf.reshape(all_col_pos[0], (reshape_factor, -1)),
            tf.reshape(all_col_pos[1], (reshape_factor, -1)),
        )
        logging.info("all_col_pos shape: %s, %s", all_col_pos[0].shape, all_col_pos[1].shape)
        all_obs = (
            tf.reshape(all_obs[0], (reshape_factor, -1)),
            tf.reshape(all_obs[1], (reshape_factor, -1)),
        )
        logging.info("all_obs shape: %s, %s", all_obs[0].shape, all_obs[1].shape)

        return (
            image_tensors,
            continuous_tensors,
            discrete_tensors,
            all_encoding,
            all_row_pos,
            all_col_pos,
            all_obs,
            actions,
        )


        print("==========Re-arranging to 4 sliding windows ======================")

        num_batches = image_tensors.shape[0]
        multiplier = sliding_batch_size * num_batches

        # There are 132 tokens for 6 observations, each observation has 22 tokens
        image = image_tensors[:, :60, :]
        image = tf.concat([image, image_tensors[:, 20:80, :]], axis=0)
        image = tf.concat([image, image_tensors[:, 40:100, :]], axis=0)
        image = tf.concat([image, image_tensors[:, 60:, :]], axis=0)

        # Discrete values advaced by 3 steps
        # tf.print("all_actions: ", all_actions, summarize=-1)
        actions = tf.gather(all_actions, [2, 3, 4, 5], axis=1)
        actions = tf.reshape(actions, (multiplier, 1, 3))
        # tf.print("actions: ", actions, summarize=-1)

        encoding = self.re_encode(all_encoding, multiplier, 2)

        row_pos = (
            self.re_encode(all_row_pos[0], multiplier, 2),
            self.re_encode(all_row_pos[1], multiplier, 2),
        )

        col_pos = (
            self.re_encode(all_col_pos[0], multiplier, 2),
            self.re_encode(all_col_pos[1], multiplier, 2),
        )

        obs = (
            self.re_encode(all_obs[0], multiplier, 2),
            self.re_encode(all_obs[1], multiplier, 2),
        )

        logging.info(" image shape: %s", image.shape)
        logging.info(" actions shape: %s", actions.shape)
        logging.info(" encoding shape: %s", encoding.shape)
        logging.info(" row_pos shape: %s, %s", row_pos[0].shape, row_pos[1].shape)
        logging.info(" col_pos shape: %s, %s", col_pos[0].shape, col_pos[1].shape)
        logging.info(" obs shape: %s, %s", obs[0].shape, obs[1].shape)
        return (
            image,
            continuous_tensors,
            discrete_tensors,
            encoding,
            row_pos,
            col_pos,
            obs,
            actions,
        )
