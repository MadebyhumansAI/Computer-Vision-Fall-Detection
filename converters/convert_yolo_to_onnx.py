import torch
import argparse

def load_model(model_path):
    """
    Load a YOLOv5 PyTorch model from a file.

    Args:
    - model_path (str): Path to the model file.

    Returns:
    - model (torch.nn.Module): A PyTorch model object.
    """
    model = torch.load(model_path, map_location=torch.device('cpu'))["model"].float()
    model.eval()
    return model

def convert_to_onnx(model, output_path):
    """
    Convert a PyTorch model to ONNX format.

    Args:
    - model (torch.nn.Module): A PyTorch model object.
    - output_path (str): Path to save the ONNX file.

    Returns:
    - None
    """
    x = torch.randn(1, 3, 640, 640)
    torch_out = model(x)
    torch.onnx.export(model, x, output_path, opset_version=11, verbose=True, input_names=['input'], output_names=['output'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a YOLOv5 PyTorch model to ONNX format.')
    parser.add_argument('model_path', type=str, help='Path to the model file.')
    parser.add_argument('output_path', type=str, help='Path to save the ONNX file.')
    args = parser.parse_args()

    model = load_model(args.model_path)
    convert_to_onnx(model, args.output_path)

