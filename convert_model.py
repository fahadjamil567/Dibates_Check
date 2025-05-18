import torch
import torch.nn as nn

class DiseaseNet(nn.Module):
    def __init__(self):
        super(DiseaseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def convert_to_onnx():
    # Load PyTorch model
    model = DiseaseNet()
    model.load_state_dict(torch.load('disease_model.pth', map_location=torch.device('cpu')))
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 64, 64)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        'disease_model.onnx',
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}}
    )

if __name__ == '__main__':
    convert_to_onnx()
    print("Model converted to ONNX format successfully!") 