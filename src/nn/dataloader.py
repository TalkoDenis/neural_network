class DataLoader:
    def __init__(self, x_data, y_data, batch_size=32):
        self.x = x_data
        self.y = y_data
        self.batch_size = batch_size
        self.total_samples = len(x_data)

    def get_batches(self):
        num_batches = self.total_samples // self.batch_size

        if self.total_samples % self.batch_size != 0:
            num_batches += 1

        batches = []
        for i in range(num_batches):
            start_index = i * self.batch_size
            end_index = start_index + self.batch_size

            x_batch = self.x[start_index:end_index]
            y_batch = self.y[start_index:end_index]

            batches.append((x_batch, y_batch))

        return batches
