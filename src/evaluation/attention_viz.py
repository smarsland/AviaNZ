"""
Attention visualization for bird audio classification models.
Generates model-specific heatmaps showing what regions the model focuses on.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from pathlib import Path
from matplotlib.axes import Axes
from src.core import config


class GradCAM:
    """Model-specific relevance visualization for CNN and transformer models."""
    
    def __init__(self, model, model_type='ast'):
        self.model = model
        self.model_type = model_type
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self.register_hooks()
    
    def register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        if self.model_type != 'regnet':
            return

        target_layer = self.get_regnet_target_layer()
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))
    
    def get_regnet_target_layer(self):
        """Get a spatially meaningful convolutional layer for RegNet.

        Using the deepest stage can be too coarse for spectrograms, so prefer the
        last bottleneck of stage s3 when available. This usually gives a better
        time-frequency map while remaining semantically useful.
        """
        backbone = getattr(self.model, 'backbone', None)

        if backbone is not None and hasattr(backbone, 's3'):
            stage = backbone.s3
            stage_blocks = list(stage.children())
            if stage_blocks:
                last_block = stage_blocks[-1]
                if hasattr(last_block, 'conv3'):
                    conv3 = getattr(last_block, 'conv3')
                    if hasattr(conv3, 'conv'):
                        return conv3.conv
                    return conv3

        if backbone is not None and hasattr(backbone, 's4'):
            stage = backbone.s4
            stage_blocks = list(stage.children())
            if stage_blocks:
                last_block = stage_blocks[-1]
                if hasattr(last_block, 'conv3'):
                    conv3 = getattr(last_block, 'conv3')
                    if hasattr(conv3, 'conv'):
                        return conv3.conv
                    return conv3

        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise ValueError("No Conv2d layer found in RegNet model")
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for input.
        
        Args:
            input_tensor: Input spectrogram (1, 1, H, W) or (1, H, W)
            target_class: Class index to visualize (or None for highest prediction)
        
        Returns:
            cam: Normalized CAM heatmap (H, W)
            prediction: Model prediction logits
        """
        self.model.eval()

        if self.model_type == 'ast':
            return self.generate_ast_cam(input_tensor, target_class)
        
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(1)
        
        input_tensor = input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        
        class_score = output[0, target_class]
        class_score.backward()
        
        if self.model_type == 'regnet':
            cam = self.compute_cam_regnet(input_tensor.shape[-2:])
        else:
            cam = self.compute_cam_ast(input_tensor.shape[-2:])
        
        self.model.zero_grad()
        
        return cam, output.detach(), target_class

    def generate_ast_cam(self, input_tensor, target_class=None):
        """Generate a class-specific AST token relevance map.

        AST in this project pools transformer patch tokens with a learned
        attention module before a linear classifier. For visualization, use the
        model's actual token weights and per-token class evidence instead of a
        Grad-CAM approximation on the patch projection.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(1)

        normalized_input = self.prepare_ast_input(input_tensor)

        with torch.no_grad():
            hidden_states = self.model.ast(normalized_input).last_hidden_state

            if self.model.use_adapters:
                for i in range(12):
                    adapter_output = self.model.adapters[i](hidden_states)
                    hidden_states = hidden_states + adapter_output

            patch_tokens = hidden_states[:, 2:, :]
            token_scores = self.model.pool.attn(patch_tokens).squeeze(-1)
            token_weights = torch.softmax(token_scores, dim=1)
            token_logits = F.linear(
                patch_tokens,
                self.model.classifier.weight,
                self.model.classifier.bias
            )
            prediction = torch.sum(token_weights.unsqueeze(-1) * token_logits, dim=1)

            if target_class is None:
                target_class = prediction.argmax(dim=1).item()

            token_relevance = token_weights[0] * token_logits[0, :, target_class]
            token_relevance = torch.relu(token_relevance)

            grid_height, grid_width = self.infer_ast_patch_grid(
                normalized_input.shape[-2:],
                token_relevance.numel()
            )
            cam = token_relevance.view(grid_height, grid_width)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), prediction.detach(), target_class

    def prepare_ast_input(self, input_tensor):
        """Match AST input normalization used during inference."""
        x = input_tensor.float()

        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        if self.model.per_chunk_norm:
            batch_size, _, width = x.shape
            chunk_width = width // self.model.num_chunks
            chunks = []
            for i in range(self.model.num_chunks):
                start = i * chunk_width
                end = start + chunk_width if i < self.model.num_chunks - 1 else width
                chunk = x[:, :, start:end]
                chunk_min = chunk.reshape(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(2)
                chunk_max = chunk.reshape(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(2)
                chunks.append((chunk - chunk_min) / (chunk_max - chunk_min + 1e-6))
            return torch.cat(chunks, dim=2)

        return (x - config.AST_MEAN) / config.AST_STD

    def infer_ast_patch_grid(self, input_size, num_tokens):
        """Infer the AST patch grid for the current input spectrogram."""
        projection = self.model.ast.embeddings.patch_embeddings.projection
        patch_size = projection.kernel_size
        stride = projection.stride

        grid_height = (input_size[0] - patch_size[0]) // stride[0] + 1
        grid_width = (input_size[1] - patch_size[1]) // stride[1] + 1

        if grid_height * grid_width == num_tokens:
            return grid_height, grid_width

        for height in range(int(num_tokens ** 0.5), 0, -1):
            if num_tokens % height == 0:
                width = num_tokens // height
                return height, width

        raise ValueError(f'Cannot infer AST patch grid for {num_tokens} tokens')
    
    def compute_cam_regnet(self, target_size):
        """Compute CAM for RegNet (standard spatial gradients)."""
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        
        cam = F.relu(cam)
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(
            cam,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )[0, 0]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()
    
    def compute_cam_ast(self, target_size):
        """Compute CAM for AST via the patch embedding Conv2d.

        The patch embedding output is (B, hidden_dim, H', W') — spatially
        organized before any self-attention mixing — so we can apply the same
        channel-weighted spatial Grad-CAM used for RegNet.
        """
        return self.compute_cam_regnet(target_size)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()


def visualize_attention(model, dataloader, output_folder, model_type='ast', 
                       num_samples=10, device='cuda', class_names=None):
    """Generate attention visualizations for test samples.
    
    Args:
        model: Trained model
        dataloader: Test data loader
        output_folder: Where to save visualizations
        model_type: 'ast' or 'regnet'
        num_samples: Number of samples to visualize
        device: Device to run on
        class_names: List of class names for labeling
    """
    os.makedirs(output_folder, exist_ok=True)
    
    model.eval()
    grad_cam = GradCAM(model, model_type=model_type)
    
    samples_processed = 0
    
    print(f"\n{'='*60}")
    print(f"ATTENTION VISUALIZATION (Post-Training Analysis)")
    print(f"{'='*60}")
    print(f"Generating relevance heatmaps for {num_samples} samples...")
    print(f"RegNet uses Grad-CAM; AST uses token relevance from attention pooling")
    print(f"Output folder: {output_folder}")
    print(f"{'='*60}\n")
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if samples_processed >= num_samples:
            break
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        batch_size = inputs.shape[0]
        
        for i in range(batch_size):
            if samples_processed >= num_samples:
                break
            
            input_sample = inputs[i:i+1]
            target = targets[i]

            with torch.no_grad():
                prediction = model(input_sample)[0].detach().cpu()

            predicted_classes, prediction_source = select_predicted_classes(prediction)

            cams = []
            for class_idx in predicted_classes:
                cam, _, _ = grad_cam.generate_cam(
                    input_sample,
                    target_class=class_idx
                )
                cams.append(cam)

            save_multiclass_plot(
                input_sample[0].detach().cpu(),
                cams,
                predicted_classes,
                torch.sigmoid(prediction),
                target.detach().cpu(),
                output_folder,
                sample_idx=samples_processed,
                class_names=class_names,
                prediction_source=prediction_source
            )
            
            samples_processed += 1
            
            if samples_processed % 1 == 0:
                print(f"  Processed {samples_processed}/{num_samples} samples")
    
    grad_cam.remove_hooks()
    print(f"\n✓ Saved {samples_processed} attention visualizations to {output_folder}")


def save_attention_plot(input_spec, cam, prediction, target, output_folder, 
                       sample_idx=0, class_names=None, explained_class=None,
                       explained_class_source=None):
    """Save a visualization with original spectrogram and attention heatmap.
    
    Args:
        input_spec: Input spectrogram (1, H, W) or (H, W)
        cam: CAM heatmap (H, W)
        prediction: Model prediction logits (num_classes,)
        target: Ground truth labels (num_classes,)
        output_folder: Where to save
        sample_idx: Sample index for filename
        class_names: List of class names
    """
    if input_spec.dim() == 3:
        input_spec = input_spec[0]
    
    input_spec = input_spec.detach().numpy() if input_spec.requires_grad else input_spec.numpy()
    
    prediction_probs = torch.sigmoid(prediction).numpy()
    target = target.numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    
    ax = axes[0]
    im = ax.imshow(input_spec, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    ax.set_title('Original Spectrogram', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.colorbar(im, ax=ax, label='Magnitude')
    
    ax = axes[1]
    ax.imshow(input_spec, aspect='auto', origin='lower', cmap='gray', alpha=0.55, interpolation='nearest')
    cam_kwargs = {
        'aspect': 'auto',
        'origin': 'lower',
        'cmap': 'jet',
        'alpha': 0.6
    }
    if cam.shape != input_spec.shape:
        cam_overlay = ax.imshow(
            cam,
            interpolation='nearest',
            extent=(0, input_spec.shape[1], 0, input_spec.shape[0]),
            **cam_kwargs
        )
    else:
        cam_overlay = ax.imshow(cam, interpolation='bilinear', **cam_kwargs)
    ax.set_title('Attention / Relevance Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.colorbar(cam_overlay, ax=ax, label='Relevance')
    
    predicted_classes = np.where(prediction_probs > 0.5)[0]
    true_classes = np.where(target > 0.5)[0]
    
    pred_str = ", ".join([
        f"{class_names[i] if class_names else i} ({prediction_probs[i]:.2f})"
        for i in predicted_classes[:5]
    ]) if len(predicted_classes) > 0 else "None"
    
    true_str = ", ".join([
        class_names[i] if class_names else str(i)
        for i in true_classes[:5]
    ]) if len(true_classes) > 0 else "None"

    explained_str = "None"
    if explained_class is not None:
        explained_name = class_names[explained_class] if class_names else str(explained_class)
        explained_str = f"{explained_name} ({prediction_probs[explained_class]:.2f})"
        if explained_class_source:
            explained_str = f"{explained_str} via {explained_class_source}"
    
    fig.suptitle(
        f"Sample {sample_idx}\n"
        f"Explained class: {explained_str}\n"
        f"Predicted: {pred_str}\n"
        f"True: {true_str}",
        fontsize=12,
        y=0.98
    )

    output_path = os.path.join(output_folder, f'attention_sample_{sample_idx:04d}.png')
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close()


def select_visualization_class(model, input_sample, target, device):
    """Choose a class to explain for multilabel Grad-CAM.

    Prefer the highest-scoring true label when labels are available so the heatmap
    corresponds to a relevant bird class instead of an arbitrary argmax.
    """
    with torch.no_grad():
        logits = model(input_sample)
        probs = torch.sigmoid(logits[0])

    true_classes = torch.where(target > 0.5)[0]
    if len(true_classes) > 0:
        best_true_idx = probs[true_classes].argmax()
        return true_classes[best_true_idx].item(), 'highest-scoring true label'

    return probs.argmax().item(), 'top prediction'


def select_predicted_classes(prediction, threshold=0.5, max_classes=4):
    """Choose the predicted classes to explain.

    Uses the model's actual positive predictions when available. If there are no
    positive predictions at the chosen threshold, falls back to the top logit so
    the user can still inspect what drove the nearest prediction.
    """
    prediction_probs = torch.sigmoid(prediction)
    predicted_classes = torch.where(prediction_probs > threshold)[0]

    if len(predicted_classes) > 0:
        sorted_probs, sorted_indices = torch.sort(prediction_probs[predicted_classes], descending=True)
        del sorted_probs
        predicted_classes = predicted_classes[sorted_indices][:max_classes]
        return predicted_classes.tolist(), f'predicted positives > {threshold:.2f}'

    top_class = prediction_probs.argmax().item()
    return [top_class], f'no predicted positives > {threshold:.2f}; showing top logit fallback'


def visualize_top_predictions(model, dataloader, output_folder, model_type='ast',
                             num_samples=10, device='cuda', class_names=None):
    """Generate visualizations for top K predicted classes per sample.
    
    Shows separate heatmaps for each of the top predicted classes.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    model.eval()
    
    samples_processed = 0
    
    print(f"\nGenerating per-class attention visualizations...")
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if samples_processed >= num_samples:
            break
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        for i in range(inputs.shape[0]):
            if samples_processed >= num_samples:
                break
            
            input_sample = inputs[i:i+1]
            target = targets[i]
            
            output = model(input_sample)
            probs = torch.sigmoid(output[0]).cpu()
            
            top_k = min(3, len(probs))
            top_classes = torch.topk(probs, top_k).indices
            
            grad_cam = GradCAM(model, model_type=model_type)
            
            cams = []
            for class_idx in top_classes:
                cam, _, _ = grad_cam.generate_cam(input_sample, target_class=class_idx.item())
                cams.append(cam)
            
            grad_cam.remove_hooks()
            
            save_multiclass_plot(
                input_sample[0].cpu(),
                cams,
                top_classes.tolist(),
                probs,
                target.cpu(),
                output_folder,
                sample_idx=samples_processed,
                class_names=class_names,
                prediction_source='top predicted classes'
            )
            
            samples_processed += 1
    
    print(f"✓ Saved {samples_processed} multi-class visualizations")


def save_multiclass_plot(input_spec, cams, top_classes, probs, target,
                        output_folder, sample_idx=0, class_names=None,
                        prediction_source=None):
    """Save visualization with heatmaps for multiple predicted classes."""
    if input_spec.dim() == 3:
        input_spec = input_spec[0]
    
    input_spec = input_spec.detach().numpy() if input_spec.requires_grad else input_spec.numpy()
    target = target.numpy() if torch.is_tensor(target) else target
    
    num_classes_viz = len(cams)
    fig, axes = plt.subplots(1 + num_classes_viz, 1, figsize=(14, 4 * (1 + num_classes_viz)), constrained_layout=True)

    if isinstance(axes, Axes):
        axes = [axes]
    
    true_classes = set(np.where(target > 0.5)[0].tolist())
    predicted_classes = list(top_classes)

    pred_parts = []
    for class_idx in predicted_classes[:5]:
        class_name = class_names[class_idx] if class_names else str(class_idx)
        pred_parts.append(f"{class_name} ({probs[class_idx]:.2f})")
    pred_str = ', '.join(pred_parts) if pred_parts else 'None'

    true_parts = []
    for class_idx in sorted(true_classes)[:5]:
        class_name = class_names[class_idx] if class_names else str(class_idx)
        true_parts.append(class_name)
    true_str = ', '.join(true_parts) if true_parts else 'None'

    ax = axes[0]
    im = ax.imshow(input_spec, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    ax.set_title(f'True: {true_str}  |  Predicted: {pred_str}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.colorbar(im, ax=ax, label='Magnitude')
    
    for idx, (cam, class_idx) in enumerate(zip(cams, top_classes)):
        ax = axes[idx + 1]
        ax.imshow(input_spec, aspect='auto', origin='lower', cmap='gray', alpha=0.5, interpolation='nearest')
        cam_kwargs = {
            'aspect': 'auto',
            'origin': 'lower',
            'cmap': 'jet',
            'alpha': 0.7
        }
        if cam.shape != input_spec.shape:
            cam_overlay = ax.imshow(
                cam,
                interpolation='nearest',
                extent=(0, input_spec.shape[1], 0, input_spec.shape[0]),
                **cam_kwargs
            )
        else:
            cam_overlay = ax.imshow(cam, interpolation='bilinear', **cam_kwargs)
        
        class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
        prob = probs[class_idx].item()
        correctness = 'correct' if class_idx in true_classes else 'incorrect'
        if cam.shape != input_spec.shape:
            grid_label = f', AST grid {cam.shape[0]}x{cam.shape[1]}'
        else:
            grid_label = ''
        ax.set_title(f'{class_name} (p={prob:.3f}, {correctness}{grid_label})', fontsize=12)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        plt.colorbar(cam_overlay, ax=ax, label='Relevance')
    
    output_path = os.path.join(output_folder, f'multiclass_attention_{sample_idx:04d}.png')
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close()
