from .plain_cnn import PlainCNN
from .resnet import ResNet


def get_model_configs():
    
    configs = {
        "Small": {
            "depths": [2, 2],      # 2 blocchi per stage
            "channels": [16, 32],
            "initial_channels": 16
        },
        "Medium": {
            "depths": [5, 5],      # 5 blocchi per stage  
            "channels": [16, 32],
            "initial_channels": 16
        },
        "Large": {
            "depths": [7, 7],      # 7 blocchi per stage
            "channels": [16, 32], 
            "initial_channels": 16
        }
    }
    return configs


def create_model(model_type: str, size: str, num_classes: int = 10):
   
    configs = get_model_configs()
    
    if size not in configs:
        raise ValueError(f"Size '{size}' not supported. Available: {list(configs.keys())}")
    
    config = configs[size]
    
    if model_type == "PlainCNN":
        return PlainCNN(
            num_classes=num_classes,
            depths=config["depths"],
            channels=config["channels"],
            initial_channels=config["initial_channels"],
            input_rgb=True,
            name=f"PlainCNN_{size}"
        )
    elif model_type == "ResNet":
        return ResNet(
            num_classes=num_classes,
            depths=config["depths"],
            channels=config["channels"],
            initial_channels=config["initial_channels"],
            input_rgb=True,
            name=f"ResNet_{size}"
        )
    else:
        raise ValueError(f"Model type '{model_type}' not supported. Available: ['PlainCNN', 'ResNet']")


def get_model_info(model_type: str, size: str):
    
    configs = get_model_configs()
    config = configs[size]
    
    temp_model = create_model(model_type, size)
    total_params = sum(p.numel() for p in temp_model.parameters())
    trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
    
    return {
        "name": f"{model_type}_{size}",
        "type": model_type,
        "size": size,
        "depths": config["depths"],
        "channels": config["channels"],
        "total_params": total_params,
        "trainable_params": trainable_params
    }


def create_all_models(num_classes: int = 10):
   
    models = {}
    sizes = ["Small", "Medium", "Large"]
    model_types = ["PlainCNN", "ResNet"]
    
    for size in sizes:
        for model_type in model_types:
            model_name = f"{model_type}_{size}"
            models[model_name] = create_model(model_type, size, num_classes)
    
    return models


def print_all_model_info():
    sizes = ["Small", "Medium", "Large"]
    model_types = ["PlainCNN", "ResNet"]
    
    
    print("Available Model Information")
  
    print(f"{'Model':<20} {'Size':<10} {'Depths':<15} {'Channels':<15} {'Parameters':<15}")
    print("-" * 80)
    
    for size in sizes:
        for model_type in model_types:
            info = get_model_info(model_type, size)
            depths_str = str(info["depths"])
            channels_str = str(info["channels"])
            
            print(f"{model_type:<20} {size:<10} {depths_str:<15} {channels_str:<15} {info['total_params']:<15,}")


if __name__ == "__main__":
    
    print_all_model_info()
    
    model = create_model("PlainCNN", "Small")
    print(f"\nCreated model: {model.name}")
    print(f"Model class: {model.__class__.__name__}")