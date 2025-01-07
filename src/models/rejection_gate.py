import random


class RandomRejector:
    def __init__(self, rejection_rate=0.1):
        self.rejection_rate = rejection_rate

    def filter(self, images, labels):
        """
        Apply rejection mechanism to a batch of images.

        Args:
            images (list or tensor): Batch of image tensors.
            labels (list): Corresponding labels for the images.

        Returns:
            tuple: Kept images, kept labels, rejected indices, rejected images, rejected labels.
        """
        batch_size = len(images)
        reject_indices = random.sample(
            range(batch_size), int(batch_size * self.rejection_rate)
        )
        keep_indices = [i for i in range(batch_size) if i not in reject_indices]

        rejected_images = images[reject_indices]
        rejected_labels = labels[reject_indices]

        return (
            images[keep_indices],
            labels[keep_indices],
            reject_indices,
            rejected_images,
            rejected_labels,
        )

    def filter_single(self):
        """
        Apply rejection mechanism to a single image.

        Returns:
            bool: True if rejected, False otherwise.
        """
        return random.random() < self.rejection_rate
