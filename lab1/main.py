import os
import torch
import yaml
import argparse
from pathlib import Path

from models.model_factor import create_model
from utils.data_loader import create_data_loaders
from utils.training import train_model_simple, create_optimizer_and_scheduler, calculate_model_stats
from utils.visualization import create_simple_report


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config):
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def main():
    parser = argparse.ArgumentParser(description='CNN vs ResNet Comparison')
    parser.add_argument('--config', type=str, default='/data01/dl24dorgio/dla/lab1/my_config.yaml')
    parser.add_argument('--models', nargs='+', default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = setup_device(config)
    
    train_loader, val_loader, _ = create_data_loaders(config)
    
    if args.models:
        models_to_train = [tuple(m.split('_')) for m in args.models]
    else:
        model_types = config['models']['types']
        sizes = config['models']['sizes']
        models_to_train = [(mt, s) for mt in model_types for s in sizes]
    
    print(f"Training models: {[f'{mt}_{s}' for mt, s in models_to_train]}")
    
    all_results = {}
    
    for model_type, size in models_to_train:
        model_name = f"{model_type}_{size}"
 
        print(f"Training {model_name}")
    
        
        try:
            model = create_model(model_type, size, 10).to(device)
            stats = calculate_model_stats(model)
            print(f"Parameters: {stats['total_params']:,}")
            
            optimizer, scheduler = create_optimizer_and_scheduler(model, config['training'])
            
            results = train_model_simple(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=config['training']['epochs']
            )
            
            all_results[model_name] = results
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
    
    if all_results:
        create_simple_report(all_results, config['experiment']['results_dir'])
        print("\nTraining completed!")
    else:
        print("No models were successfully trained.")


if __name__ == "__main__":
    main()