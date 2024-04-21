import torch
import torchvision.transforms as transforms
import torchvision.models as models
from dataloader import Data_utils  
import torch.nn as nn
from dataloader import Data_utils
import os



os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


saved_model_path = 'saved_model.pt'
saved_model_state_dict = torch.load(saved_model_path)


model = models.googlenet(pretrained=True)


num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)  


model.load_state_dict(saved_model_state_dict)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
model.eval()


selected_modules = ["maxpool1","conv3.conv","inception3a.branch1","inception4c.branch4.1.bn","inception5b.branch4.1.conv"]


feature_maps = {name: None for name in selected_modules}
non_positive_percentages = {name: [] for name in selected_modules}




def getActivation(name):
  
  def hook(model, input, output):
    feature_maps[name] = output.cpu()
  return hook


hooks = []
for name, module in model.named_modules():
    if name in selected_modules:
       
        
        hook = module.register_forward_hook(getActivation(name))
        hooks.append(hook)


def compute_non_positive_percentage(tensor):
    num_non_positive = torch.sum(tensor <= 0)
    total_elements = tensor.numel()
    return (num_non_positive.item() / total_elements) * 100


data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


batch_size = 1
datasets = Data_utils(testing=False, Test=True, num=True, transform=data_transforms)
dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=False)


for batch in dataloader:
    input_batch = batch["image"].to(device)
    _ = model(input_batch)
    for name, feature_map in feature_maps.items():
        
        percentage = compute_non_positive_percentage(feature_map)
        non_positive_percentages[name].append(percentage)
            



average_percentages = {}
for name, percentages in non_positive_percentages.items():
    if percentages:
        average_percentages[name] = sum(percentages) / len(percentages)


print("Average percentage of non-positive values:")
for name, percentage in average_percentages.items():
    print(f"{name}: {percentage}")

# Remove the hooks
for hook in hooks:
    hook.remove()
