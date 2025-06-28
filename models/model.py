import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
from collections import defaultdict, deque
import math
warnings.filterwarnings('ignore')

class HierarchicalMemoryModel(nn.Module):
    def __init__(self, 
                 object_feature_size=20, 
                 chunk_size=60, 
                 short_hidden_size=128,
                 long_hidden_size=256,
                 memory_size=512,
                 num_memory_slots=64):
        super(HierarchicalMemoryModel, self).__init__()
        
        self.chunk_size = chunk_size
        self.short_hidden_size = short_hidden_size
        self.long_hidden_size = long_hidden_size
        self.memory_size = memory_size
        self.num_memory_slots = num_memory_slots
        
        self.short_term_lstm = nn.LSTM(
            input_size=object_feature_size,
            hidden_size=short_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Chunk encoder (summarizes each chunk)
        self.chunk_encoder = nn.Sequential(
            nn.Linear(short_hidden_size, short_hidden_size // 2),
            nn.LayerNorm(short_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(short_hidden_size // 2, memory_size)
        )
        
        # External memory bank (stores long-term patterns)
        self.memory_bank = nn.Parameter(
            torch.randn(num_memory_slots, memory_size) * 0.1
        )
        
        # Memory attention mechanism
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=memory_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Long-term context processor
        self.long_term_lstm = nn.LSTM(
            input_size=memory_size,
            hidden_size=long_hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.2
        )
        
        # Decoder components
        self.context_combiner = nn.Sequential(
            nn.Linear(long_hidden_size + memory_size, long_hidden_size),
            nn.LayerNorm(long_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.decoder_lstm = nn.LSTM(
            input_size=long_hidden_size,
            hidden_size=short_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(short_hidden_size, short_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(short_hidden_size // 2, object_feature_size)
        )
        
        # Memory update mechanism
        self.memory_updater = nn.Linear(memory_size, memory_size)
        self.memory_gate = nn.Sequential(
            nn.Linear(memory_size * 2, memory_size),
            nn.Sigmoid()
        )
    
    def process_chunk(self, chunk, hidden_state=None):
        output, hidden = self.short_term_lstm(chunk, hidden_state)
        
        chunk_summary = self.chunk_encoder(output[:, -1, :])  # Use last timestep
        
        return chunk_summary, hidden
    
    def update_memory(self, chunk_summary):
        """Update external memory bank with new information"""
        batch_size = chunk_summary.size(0)
        
        # Compute attention weights between chunk summary and memory slots
        chunk_expanded = chunk_summary.unsqueeze(1)  # [batch, 1, memory_size]
        memory_expanded = self.memory_bank.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Attention mechanism
        attended_memory, attention_weights = self.memory_attention(
            chunk_expanded, memory_expanded, memory_expanded
        )
        
        # Update memory bank (running average)
        with torch.no_grad():
            # Find most attended memory slot
            max_attention_idx = attention_weights.squeeze(1).argmax(dim=1)
            
            for i, idx in enumerate(max_attention_idx):
                # Update memory with exponential moving average
                alpha = 0.1  # Learning rate for memory update
                self.memory_bank[idx] = (1 - alpha) * self.memory_bank[idx] + alpha * chunk_summary[i]
        
        return attended_memory.squeeze(1), attention_weights.squeeze(1)
    
    def forward(self, sequence_chunks, return_intermediate=False):
        """
        Forward pass for chunked sequence
        sequence_chunks: [batch, num_chunks, chunk_size, features]
        """
        batch_size, num_chunks, chunk_size, features = sequence_chunks.shape
        

        short_hidden = None
        long_hidden = None
        chunk_summaries = []
        
        # Process each chunk sequentially
        for i in range(num_chunks):
            chunk = sequence_chunks[:, i ,:,:]  # [batch, chunk_size, features]
            
            
            chunk_summary, short_hidden = self.process_chunk(chunk, short_hidden)
            
            memory_context, _ = self.update_memory(chunk_summary)
            
            enhanced_summary = torch.cat([chunk_summary, memory_context], dim=1)
            chunk_summaries.append(enhanced_summary)
        
        # Stack chunk summaries for long-term processing
        chunk_sequence = torch.stack(chunk_summaries, dim=1)  # [batch, num_chunks, memory_size * 2]
        
        # Long-term processing
        long_output, long_hidden = self.long_term_lstm(chunk_sequence)
        
        # Decode back to original sequence
        # Use the long-term context to reconstruct each chunk
        reconstructed_chunks = []
        
        for i in range(num_chunks):
            context = self.context_combiner(long_output[:, i, :])
            
            context_expanded = context.unsqueeze(1).repeat(1, chunk_size, 1)
            
            decoded_chunk, _ = self.decoder_lstm(context_expanded)
            reconstructed_chunk = self.output_layer(decoded_chunk)
            
            reconstructed_chunks.append(reconstructed_chunk)
        
        reconstructed = torch.stack(reconstructed_chunks, dim=1)
        
        if return_intermediate:
            return reconstructed, {
                'chunk_summaries': chunk_summaries,
                'long_term_context': long_output,
                'memory_bank': self.memory_bank.clone()
            }
        
        return reconstructed


class TemporalDownsamplingModel(nn.Module):
    """
    Alternative approach: Smart frame sampling with temporal attention
    """
    def __init__(self, 
                 object_feature_size=20,
                 max_sequence_length=300,
                 sampled_length=50,
                 hidden_size=256):
        super(TemporalDownsamplingModel, self).__init__()
        
        self.max_sequence_length = max_sequence_length
        self.sampled_length = sampled_length
        self.hidden_size = hidden_size
        
        # Temporal importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Conv1d(object_feature_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 2, hidden_size // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Multi-scale temporal convolutions
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(object_feature_size, hidden_size // 4, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 11]  # Different temporal scales
        ])
        
        # Main sequence processor
        self.main_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Reconstruction layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, object_feature_size)
        )
    
    def smart_downsample(self, sequence):
        """Intelligently sample frames based on importance"""
        batch_size, seq_len, features = sequence.shape
        
        # Transpose for conv1d: [batch, features, seq_len]
        sequence_t = sequence.transpose(1, 2)
        
        # Calculate importance scores
        importance = self.importance_scorer(sequence_t).squeeze(1)  # [batch, seq_len]
        
        # Multi-scale feature extraction
        multi_scale_features = []
        for conv in self.multi_scale_conv:
            feat = torch.relu(conv(sequence_t))
            multi_scale_features.append(feat)
        
        # Combine multi-scale features
        combined_features = torch.cat(multi_scale_features, dim=1)  # [batch, hidden_size, seq_len]
        combined_features = combined_features.transpose(1, 2)  # [batch, seq_len, hidden_size]
        
        # Sample based on importance
        if seq_len <= self.sampled_length:
            return combined_features, torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        
        # Top-k sampling based on importance
        _, top_indices = torch.topk(importance, self.sampled_length, dim=1)
        top_indices_sorted, _ = torch.sort(top_indices, dim=1)
        
        # Gather sampled features
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.sampled_length)
        sampled_features = combined_features[batch_indices, top_indices_sorted]
        
        return sampled_features, top_indices_sorted
    
    def forward(self, sequence):
        """Forward pass with smart downsampling"""
        # Smart downsampling
        sampled_sequence, sample_indices = self.smart_downsample(sequence)
        
        # Process sampled sequence
        lstm_output, _ = self.main_lstm(sampled_sequence)
        
        # Reconstruct original sequence length
        batch_size, orig_seq_len, features = sequence.shape
        
        # Create full-length reconstruction
        full_reconstruction = torch.zeros(batch_size, orig_seq_len, sequence.size(2), 
                                        device=sequence.device)
        
        # Decode sampled positions
        decoded_samples = self.decoder(lstm_output)
        
        # Place decoded samples back
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.sampled_length)
        full_reconstruction[batch_indices, sample_indices] = decoded_samples
        
        # Interpolate missing frames
        for b in range(batch_size):
            for f in range(sequence.size(2)):  # For each feature
                # Find sampled positions for this batch and feature
                sampled_pos = sample_indices[b].cpu().numpy()
                sampled_vals = decoded_samples[b, :, f].cpu().numpy()
                
                # Interpolate for all positions
                all_positions = np.arange(orig_seq_len)
                interpolated = np.interp(all_positions, sampled_pos, sampled_vals)
                full_reconstruction[b, :, f] = torch.tensor(interpolated, device=sequence.device)
        
        return full_reconstruction


class LongTermDataset(Dataset):
    """Dataset that handles long sequences by chunking or sampling"""
    def __init__(self, long_trajectories, chunk_size=60, max_length=300, mode='chunking'):
        self.trajectories = long_trajectories
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.mode = mode  # 'chunking' or 'sampling'
        
        self.samples = self._prepare_samples()
    
    def _prepare_samples(self):
        samples = []
        
        for traj in self.trajectories:
            if len(traj) < self.chunk_size:
                continue
                
            if self.mode == 'chunking':
                # Create overlapping chunks
                for start in range(0, len(traj) - self.max_length + 1, self.chunk_size // 2):
                    end = min(start + self.max_length, len(traj))
                    if end - start >= self.chunk_size:
                        sequence = traj[start:end]
                        # Pad to max_length if necessary
                        if len(sequence) < self.max_length:
                            padding = [sequence[-1]] * (self.max_length - len(sequence))
                            sequence.extend(padding)
                        samples.append(sequence)
            
            elif self.mode == 'sampling':
                # Use full trajectory (will be downsampled in model)
                if len(traj) > self.max_length:
                    # Random sampling for training variety
                    start = np.random.randint(0, len(traj) - self.max_length + 1)
                    sequence = traj[start:start + self.max_length]
                else:
                    sequence = traj
                samples.append(sequence)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sequence = self.samples[idx]
        
        if self.mode == 'chunking':
            # Reshape for chunked processing
            seq_array = np.array(sequence)
            num_chunks = len(sequence) // self.chunk_size
            
            # Truncate to fit chunks evenly
            truncated_length = num_chunks * self.chunk_size
            seq_array = seq_array[:truncated_length]
            
            # Reshape to chunks
            chunked = seq_array.reshape(num_chunks, self.chunk_size, -1)
            tensor_seq = torch.FloatTensor(chunked)
            
            return tensor_seq, tensor_seq
        
        else:  # sampling mode
            tensor_seq = torch.FloatTensor(sequence)
            return tensor_seq, tensor_seq


class LongTermTrainer:
    """Trainer optimized for long sequences"""
    def __init__(self, model, model_type='hierarchical', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.0005,  # Lower learning rate for stability
            weight_decay=1e-4
        )
        
        self.accumulation_steps = 4
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.001,
            epochs=100,
            steps_per_epoch=100,  # Adjust based on your data
            pct_start=0.3
        )
        
        self.reconstruction_errors = []
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        epoch_errors = []
        
        self.optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            try:
                if self.model_type == 'hierarchical':
                    reconstructed = self.model(data)
                    # Reshape for loss calculation
                    batch_size, num_chunks, chunk_size, features = reconstructed.shape
                    reconstructed = reconstructed.view(batch_size, -1, features)
                    target = target.view(batch_size, -1, features)
                else:
                    reconstructed = self.model(data)
                
                loss = self.criterion(reconstructed, target)
                
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps
                loss.backward()
                
                # Store errors for threshold calculation
                with torch.no_grad():
                    if self.model_type == 'hierarchical':
                        sample_errors = torch.mean((reconstructed.view(batch_size, -1, features) - 
                                                  target.view(batch_size, -1, features)) ** 2, dim=(1, 2))
                    else:
                        sample_errors = torch.mean((reconstructed - target) ** 2, dim=(1, 2))
                    epoch_errors.extend(sample_errors.cpu().numpy())
                
                total_loss += loss.item() * self.accumulation_steps
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    continue
                else:
                    raise e
        
        # Final gradient step if needed
        if len(dataloader) % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        self.reconstruction_errors.extend(epoch_errors)
        return total_loss / len(dataloader)
    
    def train(self, train_loader, epochs=50):
        print(f"Training long-term anomaly detection model...")
        print(f"Model type: {self.model_type}")
        print(f"Using gradient accumulation: {self.accumulation_steps} steps")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch}: Loss: {train_loss:.6f}')
                torch.cuda.empty_cache()
        
        threshold = np.percentile(self.reconstruction_errors, 90)
        print(f"Training complete! Threshold: {threshold:.6f}")
        return threshold


def generate_long_trajectory_data(num_objects=20, trajectory_length=400):
    """Generate synthetic long trajectory data"""
    trajectories = []
    
    for obj_id in range(num_objects):
        trajectory = []
        
        behavior_type = np.random.choice(['normal_walk', 'loitering', 'patrol', 'erratic'])
        
        x, y = np.random.uniform(50, 590, 2)  # Starting position
        vx, vy = np.random.uniform(-2, 2, 2)  # Initial velocity
        
        for frame in range(trajectory_length):
            if behavior_type == 'normal_walk':
                # Smooth walking with occasional direction changes
                if frame % 80 == 0:  # Change direction occasionally
                    vx += np.random.uniform(-1, 1)
                    vy += np.random.uniform(-1, 1)
                    vx = np.clip(vx, -3, 3)
                    vy = np.clip(vy, -3, 3)
                
            elif behavior_type == 'loitering':
                # Mostly stationary with small movements
                if frame > 100 and frame < 300:  # Loitering period
                    vx = np.random.uniform(-0.5, 0.5)
                    vy = np.random.uniform(-0.5, 0.5)
                
            elif behavior_type == 'patrol':
                # Back and forth movement
                if frame % 120 == 0:
                    vx = -vx  # Reverse direction
                    vy = -vy
                
            elif behavior_type == 'erratic':
                # Random direction changes
                if frame % 20 == 0:
                    vx = np.random.uniform(-4, 4)
                    vy = np.random.uniform(-4, 4)
            
            # Update position
            x += vx + np.random.normal(0, 0.5)
            y += vy + np.random.normal(0, 0.5)
            
            # Boundary conditions
            x = np.clip(x, 0, 640)
            y = np.clip(y, 0, 480)
            
            # Create feature vector (20 features)
            features = [
                x/640, y/480,  # Position
                np.random.uniform(0.05, 0.15),  # Area
                np.random.uniform(1.5, 2.5),  # Aspect ratio
                np.random.uniform(0.8, 0.95),  # Confidence
                0.0,  # Class (person)
                vx/640, vy/480,  # Velocity
                np.sqrt(vx**2 + vy**2)/640,  # Speed
                np.arctan2(vy, vx)/np.pi,  # Direction
                np.random.uniform(0.01, 0.05),  # Smoothness
                np.random.uniform(0.0, 0.2),  # Direction changes
                frame/1000,  # Age
                min(frame, 50)/50,  # Trajectory length
                abs(x/640 - 0.5),  # Distance from center X
                abs(y/480 - 0.5),  # Distance from center Y
                # Zone features
                1.0 if x < 320 and y < 240 else 0.0,
                1.0 if x >= 320 and y < 240 else 0.0,
                1.0 if x < 320 and y >= 240 else 0.0,
                1.0 if x >= 320 and y >= 240 else 0.0,
            ]
            
            trajectory.append(features)
        
        trajectories.append(trajectory)
    
    return trajectories


# Example usage
if __name__ == "__main__":
    print("🚀 Long-Term Memory Anomaly Detection")
    print("=" * 50)
    
    # Generate long trajectory data
    print("Generating long trajectory data...")
    long_trajectories = generate_long_trajectory_data(num_objects=30, trajectory_length=360)
    print(f"Generated {len(long_trajectories)} trajectories of {len(long_trajectories[0])} frames each")
    
    # Method 1: Hierarchical chunking approach
    print("\n📊 Method 1: Hierarchical Memory Model")
    hierarchical_model = HierarchicalMemoryModel(
        object_feature_size=20,
        chunk_size=60,  # 60 frames per chunk (0.5-1 second at 60-120fps)
        short_hidden_size=128,
        long_hidden_size=256
    )
    
    chunk_dataset = LongTermDataset(
        long_trajectories, 
        chunk_size=60, 
        max_length=300, 
        mode='chunking'
    )
    
    chunk_loader = DataLoader(chunk_dataset, batch_size=2, shuffle=True)
    
    print(f"   • Processes {60}-frame chunks")
    print(f"   • Maintains context across {300//60} chunks")
    print(f"   • External memory bank with {64} slots")
    print(f"   • Hierarchical LSTM processing")
    
    # Method 2: Temporal downsampling approach  
    print("\n📊 Method 2: Smart Temporal Downsampling")
    downsampling_model = TemporalDownsamplingModel(
        object_feature_size=20,
        max_sequence_length=300,
        sampled_length=50
    )
    
    sampling_dataset = LongTermDataset(
        long_trajectories,
        max_length=300,
        mode='sampling'
    )
    
    sampling_loader = DataLoader(sampling_dataset, batch_size=4, shuffle=True)
    
    print(f"   • Intelligently samples {50} frames from {300}")
    print(f"   • Multi-scale temporal convolutions")
    print(f"   • Importance-based frame selection")
    print(f"   • Interpolation for full reconstruction")
    
    # Training examples (commented out for demonstration)
    """
    # Train hierarchical model
    hierarchical_trainer = LongTermTrainer(hierarchical_model, 'hierarchical')
    hierarchical_threshold = hierarchical_trainer.train(chunk_loader, epochs=20)
    
    # Train downsampling model  
    downsampling_trainer = LongTermTrainer(downsampling_model, 'downsampling')
    downsampling_threshold = downsampling_trainer.train(sampling_loader, epochs=20)
    """
    
    print(f"\n🎯 Long-Term Anomaly Detection Capabilities:")
    capabilities = [
        "Loitering detection (stationary for extended periods)",
        "Patrol pattern recognition",
        "Long-term trajectory analysis", 
        "Context-aware behavior modeling",
        "Multi-scale temporal pattern detection",
        "Memory-augmented pattern learning",
        "Efficient processing of 300+ frame sequences",
        "Gradual behavior change detection"
    ]
    
    for capability in capabilities:
        print(f"   • {capability}")
    
    print(f"\n🔧 Technical Solutions for Long Sequences:")
    solutions = [
        "Chunked processing with context transfer",
        "External memory banks for pattern storage", 
        "Smart frame sampling based on importance",
        "Gradient accumulation for memory efficiency",
        "Hierarchical temporal modeling",
        "Multi-scale convolutions for different time scales",
        "Interpolation for missing frame reconstruction",
        "Memory-efficient training strategies"
    ]
    
    for solution in solutions:
        print(f"   • {solution}")
    
    print(f"\n💡 Usage Recommendations:")
    print(f"   • Use chunking for real-time processing")
    print(f"   • Use downsampling for batch analysis") 
    print(f"   • Combine both for maximum flexibility")
    print(f"   • Adjust chunk size based on behavior timescales")
    print(f"   • Monitor memory usage and adjust batch sizes")