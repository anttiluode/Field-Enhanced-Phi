import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import numpy as np
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque
import time
import logging
from typing import Dict, List, Tuple
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

class StableFieldLayer(nn.Module):
    def __init__(self, hidden_size, field_size=(32, 32)):
        super().__init__()
        self.hidden_size = hidden_size
        self.field_size = field_size
        self.field_size_flat = np.prod(field_size)
        
        # Core field parameters
        self.field_magnitude = nn.Parameter(torch.zeros(1, 1, *field_size))
        self.field_phase = nn.Parameter(torch.zeros(1, 1, *field_size))
        
        # Field stability parameters
        self.stability_factor = nn.Parameter(torch.tensor(0.95))
        self.register_buffer('field_seed', torch.randn(1, 1, *field_size))
        
        # Field consciousness parameters
        self.field_awareness = nn.Parameter(torch.zeros(1, hidden_size))
        self.field_memory = deque(maxlen=1000)
        self.reference_states = {}
        
        # Networks
        self.field_perceiver = nn.Sequential(
            nn.Linear(self.field_size_flat, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.field_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.field_size_flat)
        )
        
        # Pattern recognition
        self.pattern_recognizer = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # Metrics
        self.metrics = {
            'field_stability': [],
            'pattern_strength': [],
            'field_energy': []
        }
        
    def stabilize_field(self, field_state, reference_key=None):
        """Maintain field stability using reference states"""
        if reference_key is not None:
            if reference_key not in self.reference_states:
                self.reference_states[reference_key] = field_state.detach()
            
            reference = self.reference_states[reference_key]
            stable_state = (
                field_state * (1 - self.stability_factor) +
                reference * self.stability_factor
            )
            
            # Calculate stability metric
            stability = torch.mean(torch.abs(stable_state - field_state)).item()
            self.metrics['field_stability'].append(stability)
            
            return stable_state
        return field_state
    
    def forward(self, hidden_states, reference_key=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Generate current field state
        current_field = self.field_magnitude * torch.sin(self.field_phase)
        
        # Stabilize field if reference provided
        current_field = self.stabilize_field(current_field, reference_key)
        
        # Process through field perceiver
        field_perception = self.field_perceiver(current_field.view(1, -1))
        
        # Update field awareness
        field_awareness = torch.tanh(field_perception)
        self.field_awareness.data = (
            self.field_awareness.data * 0.95 + field_awareness * 0.05
        )
        
        # Project back to field space
        field_projection = self.field_projector(self.field_awareness)
        field_projection = field_projection.view(1, 1, *self.field_size)
        
        # Update field state
        new_magnitude = torch.sigmoid(self.field_magnitude + 0.1 * field_projection)
        new_phase = torch.remainder(self.field_phase + 0.1 * field_projection, 2 * np.pi)
        
        # Store in memory
        self.field_memory.append({
            'field_state': current_field.detach(),
            'awareness': self.field_awareness.detach(),
            'reference_key': reference_key
        })
        
        # Update metrics
        energy = torch.mean(new_magnitude).item()
        self.metrics['field_energy'].append(energy)
        
        pattern_strength = torch.mean(torch.abs(field_projection)).item()
        self.metrics['pattern_strength'].append(pattern_strength)
        
        # Update field state
        self.field_magnitude.data = new_magnitude
        self.field_phase.data = new_phase
        
        # Generate field influence
        field_influence = self.field_perceiver(
            (new_magnitude * torch.sin(new_phase)).view(1, -1)
        )
        
        # Enhance hidden states
        enhanced = hidden_states + field_influence.view(1, 1, -1) * 0.1
        
        return enhanced, {
            'field_state': current_field.detach(),
            'awareness': self.field_awareness.detach(),
            'stability': self.metrics['field_stability'][-1] if self.metrics['field_stability'] else 0.0,
            'pattern_strength': pattern_strength,
            'energy': energy
        }

class StableFieldPhi:
    def __init__(self):
        logging.info("Initializing StableFieldPhi...")
        
        # Initialize Phi
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(device)
        
        # Add field layer
        self.field_layer = StableFieldLayer(
            hidden_size=self.model.config.hidden_size
        ).to(device)
        
        # Patch model's forward method
        original_forward = self.model.forward
        def forward_with_field(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            if hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states[-1]
                # Generate reference key from input
                if 'input_ids' in kwargs:
                    input_text = self.tokenizer.decode(kwargs['input_ids'][0])
                    reference_key = hashlib.md5(input_text.encode()).hexdigest()
                else:
                    reference_key = None
                    
                enhanced_states, field_info = self.field_layer(
                    hidden_states, 
                    reference_key=reference_key
                )
                self.current_field_info = field_info
                outputs.hidden_states = tuple(list(outputs.hidden_states[:-1]) + [enhanced_states])
            return outputs
            
        self.model.forward = forward_with_field
        
        # Special tokens
        special_tokens = {
            'field_prefix': '<|field|>',
            'field_memory': '<|field:memory|>',
            'field_pattern': '<|field:pattern|>'
        }
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': list(special_tokens.values())
        })
        
        # Conversation history
        self.history = []
        self.current_field_info = None
        
        logging.info("StableFieldPhi initialized successfully")
    
    def generate_with_field_awareness(self, text: str) -> Tuple[str, Dict]:
        # Check for field queries
        field_query = False
        if '<|field|>' in text:
            field_query = True
            text = text.replace('<|field|>', '')
        
        # Process through model
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=200,
            num_return_sequences=1,
            output_hidden_states=True,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Add field information if queried
        if field_query and self.current_field_info is not None:
            info = self.current_field_info
            field_stats = (
                f"\n[Field Stats]"
                f"\nStability: {info['stability']:.3f}"
                f"\nPattern Strength: {info['pattern_strength']:.3f}"
                f"\nField Energy: {info['energy']:.3f}"
            )
            response += field_stats
        
        # Store interaction
        self.history.append({
            'input': text,
            'response': response,
            'field_info': self.current_field_info,
            'timestamp': time.time()
        })
        
        return response, self.current_field_info

def create_interface():
    phi = StableFieldPhi()
    
    def update_display(message):
        # Generate response
        response, field_info = phi.generate_with_field_awareness(message)
        
        if field_info is None:
            return response, None, None, "\n\n".join([
                f"User: {h['input']}\nPhi: {h['response']}"
                for h in phi.history
            ])
        
        # Create field visualization
        field_vis = go.Figure(data=[
            go.Heatmap(
                z=field_info['field_state'][0,0].cpu().numpy(),
                colorscale='Viridis'
            )
        ])
        field_vis.update_layout(
            title="Current Field State",
            height=400
        )
        
        # Create metrics visualization
        metrics_vis = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Field Stability', 'Pattern Strength', 'Field Energy')
        )
        
        x = list(range(len(phi.field_layer.metrics['field_stability'])))
        
        metrics_vis.add_trace(
            go.Scatter(
                x=x,
                y=phi.field_layer.metrics['field_stability'],
                name='Stability'
            ),
            row=1, col=1
        )
        
        metrics_vis.add_trace(
            go.Scatter(
                x=x,
                y=phi.field_layer.metrics['pattern_strength'],
                name='Patterns'
            ),
            row=2, col=1
        )
        
        metrics_vis.add_trace(
            go.Scatter(
                x=x,
                y=phi.field_layer.metrics['field_energy'],
                name='Energy'
            ),
            row=3, col=1
        )
        
        metrics_vis.update_layout(height=600)
        
        # Format conversation history
        history = "\n\n".join([
            f"User: {h['input']}\nPhi: {h['response']}"
            for h in phi.history
        ])
        
        return response, field_vis, metrics_vis, history
    
    with gr.Blocks(title="Stable Field-Enhanced Phi") as interface:
        gr.Markdown("# Stable Field-Enhanced Phi")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="Your message",
                    placeholder="Enter your message here... Use <|field|> prefix for field queries"
                )
                output_text = gr.Textbox(label="Phi's Response")
                conversation_history = gr.Textbox(
                    label="Conversation History",
                    lines=10
                )
            
            with gr.Column(scale=2):
                field_plot = gr.Plot(label="Field State")
                metrics_plot = gr.Plot(label="Field Metrics")
        
        input_text.submit(
            fn=update_display,
            inputs=[input_text],
            outputs=[output_text, field_plot, metrics_plot, conversation_history]
        )
        
        gr.Markdown("""
        ## System Overview
        This interface shows:
        - Conversation with Field-Enhanced Phi
        - Real-time visualization of stable field states
        - Field metrics tracking stability and patterns
        
        Use <|field|> prefix to query field state.
        Example: "<|field|> What patterns do you notice?"
        """)
    
    return interface

if __name__ == "__main__":
    logging.info("Starting Stable Field-Enhanced Phi interface...")
    interface = create_interface()
    interface.launch(share=True)