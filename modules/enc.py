import torchvision

encoder = torchvision.models.resnet50(pretrained=True)

# Access the 0th layer using named_children()
layer_0 = next(encoder.named_children())[-1]
print(layer_0)
