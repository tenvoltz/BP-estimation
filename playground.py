import torch


class ExampleModel:
    def forward(self, centers, labels, classes):
        same_class_mask = (classes.unsqueeze(1) == classes.unsqueeze(0))
        diff_class_mask = ~same_class_mask
        score_matrix = (centers[labels.unsqueeze(1)] - centers[labels.unsqueeze(0)]) ** 2
        print(diff_class_mask)
        score_matrix = score_matrix.mean(dim=-1)
        print(same_class_mask)
        loss = 0
        if same_class_mask.any():
            loss += score_matrix[same_class_mask].mean()
        if diff_class_mask.any():
            loss += (1 / (1 + score_matrix[diff_class_mask])).mean()
        return loss


# Example usage
model = ExampleModel()

# Define some class centers (e.g., 3 classes, each with 2-dimensional features)
centers = torch.FloatTensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])
labels = torch.LongTensor([0, 1, 2])
classes = torch.LongTensor([0, 1, 1])
# Calculate the loss
loss = model.forward(centers, labels, classes)
print("Calculated Loss:", loss.item())
