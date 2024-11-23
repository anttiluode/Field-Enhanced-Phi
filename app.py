import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FieldLayer(nn.Module):
    def __init__(self, hidden_size, field_size=(32, 32), coupling_strength=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.field_size = field_size
        self.coupling_strength = coupling_strength
        
        # Field parameters
        self.field_magnitude = nn.Parameter(torch.zeros(1, 1, *field_size))
        self.field_phase = nn.Parameter(torch.zeros(1, 1, *field_size))
        
        # Projection matrices
        self.to_field = nn.Linear(hidden_size, field_size[0] * field_size[1])
        self.from_field = nn.Linear(field_size[0] * field_size[1], hidden_size)
        
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to field space
        field_proj = self.to_field(hidden_states)
        field_proj = field_proj.view(batch_size, seq_len, *self.field_size)
        
        # Compute field influence
        field_effect = field_proj.mean(dim=(0, 1), keepdim=True)
        
        # Update field states
        new_magnitude = torch.sigmoid(
            self.field_magnitude + self.coupling_strength * field_effect
        )
        phase_delta = 0.1 * torch.tanh(field_effect)
        new_phase = torch.remainder(self.field_phase + phase_delta, 2 * np.pi)
        
        # Compute field influence
        field = new_magnitude * torch.sin(new_phase)
        field_flat = field.view(1, -1)
        field_influence = self.from_field(field_flat)
        
        # Apply field effect
        enhanced = hidden_states + field_influence.view(1, 1, -1) * self.coupling_strength
        
        return enhanced, {
            'magnitude': new_magnitude.detach(),
            'phase': new_phase.detach(),
            'field': field.detach()
        }

class FieldEnhancedPhi:
    def __init__(self, field_config=None):
        self.status_message = "Starting initialization..."
        
        try:
            print("Step 1: Importing required modules...")
            import transformers
            print(f"Using transformers version: {transformers.__version__}")
            
            # Load tokenizer
            print("\nStep 2: Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/Phi-3.5-mini-instruct",
                trust_remote_code=True
            )
            
            # Load model
            print("\nStep 3: Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3.5-mini-instruct",
                trust_remote_code=True,
                return_dict=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print(f"Model type: {type(self.model)}")
            print(f"Model config: {self.model.config}")
            
            # Initialize field layers
            print("\nStep 4: Setting up field layers...")
            if field_config is None:
                field_config = {
                    'sizes': [(32, 32)],
                    'coupling_strengths': [0.1],
                    'injection_points': [-1]
                }
            
            self.field_layers = {}
            self.setup_field_layers(field_config)
            print("Field layers initialized successfully")
            
            self.status_message = "Initialization complete!"
            
        except Exception as e:
            error_msg = f"Error during initialization: {str(e)}"
            print(error_msg)
            print("Full error:")
            print(traceback.format_exc())
            self.status_message = error_msg
            raise e  # Correctly re-raise the exception
                
    def setup_field_layers(self, config):
        """Initialize field layers at specified points"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Model hidden size: {self.model.config.hidden_size}")
        print(f"Model dtype: {self.model.dtype}")
        
        for idx, size, strength in zip(
            config['injection_points'],
            config['sizes'],
            config['coupling_strengths']
        ):
            layer = FieldLayer(
                hidden_size=self.model.config.hidden_size,
                field_size=size,
                coupling_strength=strength
            )
            # Move to device and ensure float16
            layer = layer.to(device).to(dtype=torch.float16)
            self.field_layers[idx] = layer
            print(f"Initialized field layer {idx} with size {size}")

    def generate_with_fields(self, 
                           prompt, 
                           max_tokens=100,
                           temperature=0.7):
        """Generate text with field effects"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        field_states_history = []
        
        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(
                    input_ids=input_ids,
                    output_hidden_states=True
                )
                
                hidden_states = list(outputs.hidden_states)
                current_field_states = {}
                
                # Apply field effects
                for idx, field_layer in self.field_layers.items():
                    enhanced_states, field_info = field_layer(hidden_states[idx].to(device))
                    hidden_states[idx] = enhanced_states
                    current_field_states[idx] = field_info
                
                field_states_history.append(current_field_states)
                
                # Get next token
                logits = outputs.logits[:, -1, :] / temperature
                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1),
                    num_samples=1
                )
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return text, field_states_history

def create_interface():
    model = None
    status_output = gr.Textbox(label="Status", value="No model loaded")
    
    def initialize_model(field_sizes, coupling_strengths, injection_points):
        nonlocal model, status_output
        status_messages = []
        
        try:
            # Update status
            status_messages.append("Starting model initialization...")
            yield "\n".join(status_messages)
            
            # Parse configuration
            status_messages.append("Parsing field configuration...")
            sizes = [tuple(map(int, size.strip().split('x'))) for size in field_sizes.split(',')]
            strengths = [float(s.strip()) for s in coupling_strengths.split(',')]
            points = [int(p.strip()) for p in injection_points.split(',')]
            
            if len(sizes) != len(strengths) or len(sizes) != len(points):
                raise ValueError("Number of sizes, strengths, and points must match")
            
            config = {
                'sizes': sizes,
                'coupling_strengths': strengths,
                'injection_points': points
            }
            
            status_messages.append("Creating model instance...")
            yield "\n".join(status_messages)
            
            # Initialize model
            model = FieldEnhancedPhi(field_config=config)
            
            # Update status messages
            status_messages.append(model.status_message)
            status_messages.append("Ready for generation!")
            
            yield "\n".join(status_messages)
            
        except Exception as e:
            status_messages.append(f"Error: {str(e)}")
            yield "\n".join(status_messages)
    
    def generate_text(prompt, temperature, max_tokens):
        if model is None:
            return "Please initialize model first", None
            
        try:
            text, field_states = model.generate_with_fields(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Visualize last field state
            fig = make_subplots(
                rows=len(model.field_layers),
                cols=3,
                subplot_titles=['Magnitude', 'Phase', 'Field']
            )
            
            for i, (layer_idx, states) in enumerate(field_states[-1].items()):
                row = i + 1
                
                # Convert to float32 for visualization
                magnitude = states['magnitude'][0,0].cpu().numpy().astype(np.float32)
                phase = states['phase'][0,0].cpu().numpy().astype(np.float32)
                field = states['field'][0,0].cpu().numpy().astype(np.float32)
                
                # Add traces with explicit data type conversion
                fig.add_trace(
                    go.Heatmap(
                        z=magnitude,
                        colorscale='Viridis',
                        showscale=True,
                        name=f'Layer {layer_idx} Magnitude'
                    ),
                    row=row, col=1
                )
                
                fig.add_trace(
                    go.Heatmap(
                        z=phase,
                        colorscale='HSV',
                        showscale=True,
                        name=f'Layer {layer_idx} Phase'
                    ),
                    row=row, col=2
                )
                
                fig.add_trace(
                    go.Heatmap(
                        z=field,
                        colorscale='RdBu',
                        showscale=True,
                        name=f'Layer {layer_idx} Field'
                    ),
                    row=row, col=3
                )
            
            # Update layout with better titles and sizing
            fig.update_layout(
                height=300*len(model.field_layers),
                width=900,
                title_text="Field Layer Patterns",
                title_x=0.5,
                showlegend=False
            )
            
            # Add axis labels
            for i in range(len(model.field_layers)):
                fig.update_xaxes(title_text="Field Width", row=i+1)
                fig.update_yaxes(title_text="Field Height", row=i+1)
            
            return text, fig
            
        except Exception as e:
            print(f"Error in generate_text: {e}")
            import traceback
            print(traceback.format_exc())
            return f"Error generating text: {str(e)}", None
    
    # Create interface
    with gr.Blocks(title="Field-Enhanced Phi") as interface:
        gr.Markdown("# Field-Enhanced Phi Model")
        
        with gr.Tab("Setup"):
            field_sizes = gr.Textbox(
                label="Field Sizes",
                value="32x32",
                info="Comma-separated sizes (e.g., '32x32,16x16')"
            )
            coupling_strengths = gr.Textbox(
                label="Coupling Strengths",
                value="0.1",
                info="Comma-separated strengths (e.g., '0.1,0.2')"
            )
            injection_points = gr.Textbox(
                label="Injection Points",
                value="-1",
                info="Comma-separated layer indices (e.g., '-1,-4')"
            )
            init_btn = gr.Button("Initialize Model")
            status_output = gr.Textbox(
                label="Status",
                value="Model not loaded",
                lines=10
            )
            
            # Use the streaming output for initialization
            init_btn.click(
                initialize_model,
                inputs=[field_sizes, coupling_strengths, injection_points],
                outputs=status_output
            )
        
        with gr.Tab("Generate"):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=3
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                label="Temperature"
            )
            max_tokens = gr.Slider(
                minimum=10,
                maximum=500,
                value=100,
                step=10,
                label="Max Tokens"
            )
            generate_btn = gr.Button("Generate")
            output_text = gr.Textbox(label="Generated Text")
            field_plot = gr.Plot(label="Field Patterns")
            
            generate_btn.click(
                generate_text,
                inputs=[prompt, temperature, max_tokens],
                outputs=[output_text, field_plot]
            )
            
        gr.Markdown("""
        ## Instructions
        
        1. **Initialize the Model:**
           - Navigate to the **Setup** tab.
           - Enter field sizes (e.g., `"32x32"` for one layer).
           - Enter coupling strengths (e.g., `"0.1"`).
           - Enter injection points (e.g., `"-1"` for the last layer).
           - Click **Initialize Model** and monitor the status.
           
        2. **Generate Text:**
           - Switch to the **Generate** tab.
           - Enter your prompt.
           - Adjust the temperature (higher values = more creativity).
           - Set the maximum number of tokens to generate.
           - Click **Generate** to produce text and visualize field patterns.
           
        3. **Visualization:**
           - The generated visualization shows:
             - **Magnitude:** Strength of the field.
             - **Phase:** Phase patterns of the field.
             - **Field:** Combined field effects.
        """)

    return interface

if __name__ == "__main__":
    print("Starting Field-Enhanced Phi interface...")
    try:
        interface = create_interface()
        interface.queue()
        interface.launch(
            share=True,
            show_error=True,
            server_name="0.0.0.0",
            server_port=7861
        )
    except Exception as e:
        logging.error(f"Error launching interface: {str(e)}")
        logging.error(traceback.format_exc())
        raise e  # Correctly re-raise the exception
