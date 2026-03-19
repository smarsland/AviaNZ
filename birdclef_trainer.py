"""
BirdClef fine-tuning trainer using BaseTrainer.
Demonstrates the refactored approach - just model creation + saving logic.
"""

from base_trainer import BaseTrainer
from models import BirdClefModel
import torch
import os
import json
import config


class BirdClefTrainer(BaseTrainer):
    """BirdClef fine-tuning trainer with DANN support."""
    
    def __init__(self, data_folder, output_folder, pretrained_path, 
                 freeze_backbone=False, freeze_stages=0, 
                 use_dann=False, target_folder=None, lambda_domain=0.1,
                 **kwargs):
        self.freeze_backbone = freeze_backbone
        self.freeze_stages = freeze_stages
        self.birdclef_pretrained_path = pretrained_path
        self.use_dann = use_dann
        self.target_folder = target_folder
        self.lambda_domain = lambda_domain
        
        super().__init__(
            data_folder=data_folder,
            output_folder=output_folder,
            pretrained_path=None,  # BirdClef handles its own pretrained loading
            **kwargs
        )
        
        # Load target domain if using DANN
        if self.use_dann and self.target_folder:
            from data_utils import DataLoader as DL
            target_loader = DL(target_folder, noise_folder=self.noise_folder)
            target_data = target_loader.load_data(self.multilabel, validation_share=0.0)
            
            from data_utils import create_data_loaders
            loaders = create_data_loaders(
                target_data['train'], None,
                batch_size=self.batch_size,
                img_height=self.img_height,
                img_width=self.img_width,
                multilabel=self.multilabel,
                noise_ratio=0.0,
                mixup_alpha=0.0,
                remove_baseline=self.remove_baseline,
                normalize=self.normalize,
                use_temporal_roll=self.use_temporal_roll
            )
            self.target_loader = loaders['train']
        else:
            self.target_loader = None
            
        # Setup DANN components if needed
        if self.use_dann:
            from models import GradientReversalLayer, DomainDiscriminator
            self.grl = GradientReversalLayer().to(self.device)
            feature_dim = 1280  # BirdClef RegNetY-008 output
            self.domain_classifier = DomainDiscriminator(input_dim=feature_dim).to(self.device)
            self.domain_criterion = torch.nn.BCEWithLogitsLoss()
    
    def create_model(self):
        """Create BirdClef model with pretrained weights."""
        model = BirdClefModel(
            num_classes=self.num_classes,
            pretrained_path=self.birdclef_pretrained_path,
            model_name='regnety_008',
            freeze_backbone=self.freeze_backbone,
            freeze_stages=self.freeze_stages
        ).to(self.device)
        
        return model
    
    def save_model(self, model, best=False):
        """Save BirdClef model and config."""
        filename = 'birdclef_finetuned_best.pt' if best else 'birdclef_finetuned.pt'
        torch.save(model.state_dict(), os.path.join(self.output_folder, filename))
        
        model_config = config.get_model_config()
        model_config['freq_bins'] = self.img_height
        model_config['time_bins'] = self.img_width
        model_config['model_type'] = 'birdclef_finetuned'
        model_config['num_classes'] = model.num_classes
        model_config['multilabel'] = self.multilabel
        model_config['class_names'] = self.data['class_names']
        model_config['normalize'] = self.normalize
        model_config['remove_baseline'] = self.remove_baseline
        
        config_path = os.path.join(self.output_folder, f"{filename.replace('.pt', '_config.json')}")
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
    
    def create_optimizer(self, model):
        """Override to include domain classifier params if using DANN."""
        params = list(model.parameters())
        if self.use_dann:
            params += list(self.domain_classifier.parameters())
        return torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
