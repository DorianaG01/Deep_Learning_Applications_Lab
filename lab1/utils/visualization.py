import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_results(results, save_path="./results/training_plot.png"):
    
    try:
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
    
        ax1.set_title("Training Loss per Epoch", fontsize=14)
        for i, (model_name, metrics) in enumerate(results.items()):
            epochs = list(range(1, len(metrics['train_losses']) + 1))
            ax1.plot(epochs, metrics['train_losses'], 
                    label=model_name, color=colors[i % len(colors)], 
                    linewidth=2, marker='o')
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
    
        ax2.set_title("Validation Accuracy per Epoch", fontsize=14)
        for i, (model_name, metrics) in enumerate(results.items()):
            epochs = list(range(1, len(metrics['val_accuracies']) + 1))
            ax2.plot(epochs, metrics['val_accuracies'], 
                    label=model_name, color=colors[i % len(colors)], 
                    linewidth=2, marker='s')
        
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
       
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {save_path}")
        
    except Exception as e:
        print(f"Error creating plot: {e}")


def print_results_table(results):
   
    print("Training Results")
  
    print(f"{'Model':<20} {'Best Val Acc':<12} {'Final Loss':<12}")
   
    
    for model_name, metrics in results.items():
        best_acc = max(metrics['val_accuracies'])
        final_loss = metrics['train_losses'][-1]
        print(f"{model_name:<20} {best_acc:<12.4f} {final_loss:<12.4f}")
    



def create_simple_report(results, save_dir="./results"):
    
    
    print_results_table(results)

    plot_path = f"{save_dir}/training_results.png"
    plot_results(results, plot_path)
    
    print(f"Report completed! Check {save_dir}/")