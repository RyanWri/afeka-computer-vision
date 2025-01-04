import random


class RandomRejector:
    def __init__(self, rejection_rate=0.1):
        self.rejection_rate = rejection_rate

    def filter(self, images, labels):
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
