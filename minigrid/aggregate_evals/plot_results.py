import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# Color palette for different model types
MODEL_COLORS = {
    'ppo': '#E74C3C',           # Red
    'h_ppo_product': '#3498DB',  # Blue
    'h_ppo_symloss': '#F39C12',  # Orange
    'h_ppo_symloss_eps': '#2ECC71',  # Green
    'h_ppo_symloss_theta': '#9B59B6',  # Purple
}

# Define ordering for consistent plot arrangement
MODEL_ORDER = [
    ('ppo_', 0),  # ppo_ to avoid matching h_ppo
    ('h_ppo_product', 1),
    ('h_ppo_symloss_eps', 3),  # eps before plain symloss in check order
    ('theta_0.25', 4),
    ('theta_0.5', 5),
    ('theta_0.75', 6),
    ('h_ppo_symloss', 2),  # plain symloss last in checks (after theta/eps)
]

def get_model_order_key(full_name):
    """Return a sort key based on desired model ordering."""
    name_lower = full_name.lower()
    
    # Check for ppo without h_ppo
    if 'ppo_' in name_lower and 'h_ppo' not in name_lower:
        return 0
    
    # Check for theta variants first (most specific)
    if 'theta_0.25' in name_lower:
        return 4
    if 'theta_0.5' in name_lower:
        return 5  
    if 'theta_0.75' in name_lower:
        return 6
    
    # Then other h_ppo variants
    if 'h_ppo_product' in name_lower:
        return 1
    if 'h_ppo_symloss_eps' in name_lower:
        return 3
    if 'h_ppo_symloss' in name_lower:
        return 2
    
    return 100  # Unknown models go last

def get_short_label(full_name):
    """Convert long model names to shorter, cleaner labels."""
    name_lower = full_name.lower()
    
    # Check for theta variants first (most specific)
    if 'theta_0.25' in name_lower:
        return 'H-PPO\nSymLoss θ=0.25'
    if 'theta_0.5' in name_lower:
        return 'H-PPO\nSymLoss θ=0.5'
    if 'theta_0.75' in name_lower:
        return 'H-PPO\nSymLoss θ=0.75'
    
    # Then other variants
    if 'h_ppo_symloss_eps' in name_lower:
        return 'H-PPO\nSymLoss ϵ'
    if 'h_ppo_symloss' in name_lower:
        return 'H-PPO\nSymLoss'
    if 'h_ppo_product' in name_lower:
        return 'H-PPO\nProduct'
    if 'ppo' in name_lower and 'h_ppo' not in name_lower:
        return 'PPO'
    
    return full_name

def get_color(label):
    """Get color based on model type."""
    # Check in order - more specific patterns first
    if 'h_ppo_symloss_theta' in label.lower() or 'theta_0.' in label.lower():
        return MODEL_COLORS['h_ppo_symloss_theta']
    elif 'h_ppo_symloss_eps' in label.lower():
        return MODEL_COLORS['h_ppo_symloss_eps']
    elif 'h_ppo_symloss' in label.lower():
        return MODEL_COLORS['h_ppo_symloss']
    elif 'h_ppo_product' in label.lower():
        return MODEL_COLORS['h_ppo_product']
    elif 'ppo' in label.lower() and 'h_ppo' not in label.lower():
        return MODEL_COLORS['ppo']
    return '#7F8C8D'  # Default gray

def generate_boxplots():
    base_dir = "aggregate_evals"
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return
    
    # Set global style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titleweight'] = 'bold'

    # Loop through env_id_nkeys subdirectories in aggregate_evals
    for env_entry in os.scandir(base_dir):
        if env_entry.is_dir():
            env_folder_name = env_entry.name  # e.g., "MiniGrid-DoorKey-8x8-v0_2keys"
            env_folder_path = env_entry.path
            
            # Data for the combined plot across all eval types
            combined_data = []
            combined_labels = []
            combined_full_names = []
            
            # Data for per-model plots (model_name -> {eval_type -> returns})
            per_model_data = {}
            
            # Loop through eval_type subdirectories (standard, h_ppo_eps_X, random_rules)
            eval_type_dirs = [d for d in os.scandir(env_folder_path) if d.is_dir() and d.name != "per_model"]
            
            # Sort eval_type directories for consistent ordering
            eval_order = ['standard', 'random_rules', 'h_ppo_eps_0.5', 'h_ppo_eps_1.0']
            eval_type_dirs.sort(key=lambda d: eval_order.index(d.name) if d.name in eval_order else 100)
            
            if not eval_type_dirs:
                # Fallback: no subdirectories, check for CSVs directly (old structure)
                csv_files = [f for f in os.listdir(env_folder_path) if f.endswith(".csv")]
                if csv_files:
                    _generate_single_boxplot(env_folder_path, env_folder_name, csv_files)
                continue
            
            for eval_type_entry in eval_type_dirs:
                eval_type_name = eval_type_entry.name  # e.g., "standard", "h_ppo_eps_1.0", "random_rules"
                eval_type_path = eval_type_entry.path
                
                all_data = []
                labels = []
                full_names = []
                
                # Look for CSV files in this eval_type subfolder
                csv_files = [f for f in os.listdir(eval_type_path) if f.endswith(".csv")]
                if not csv_files:
                    continue
                
                # Sort by model order
                csv_files.sort(key=lambda x: get_model_order_key(x.replace(".csv", "")))
                
                for csv_file in csv_files:
                    file_path = os.path.join(eval_type_path, csv_file)
                    returns = []
                    with open(file_path, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # Handle both plain numbers and bracket-wrapped numbers like "[0.5]"
                            val = row["Return"].strip("[]")
                            returns.append(float(val))
                    
                    if returns:
                        all_data.append(returns)
                        full_name = csv_file.replace(".csv", "")
                        full_names.append(full_name)
                        labels.append(get_short_label(full_name))
                        
                        # Also add to combined data with eval_type prefix
                        combined_data.append(returns)
                        combined_labels.append(f"{get_short_label(full_name)}\n[{_format_eval_type(eval_type_name)}]")
                        combined_full_names.append(full_name)
                        
                        # Track data per model for cross-eval-type plots
                        if full_name not in per_model_data:
                            per_model_data[full_name] = {}
                        per_model_data[full_name][eval_type_name] = returns
                
                # Generate boxplot for this eval_type
                if all_data:
                    _create_styled_boxplot(
                        all_data, labels, full_names,
                        title=_format_env_name(env_folder_name, eval_type_name),
                        save_path=os.path.join(eval_type_path, f"boxplot_{eval_type_name}.png")
                    )
                    print(f"Generated boxplot for {env_folder_name}/{eval_type_name}")
            
            # Generate combined boxplot for this env_id_nkeys (all eval types)
            if combined_data:
                _create_styled_boxplot(
                    combined_data, combined_labels, combined_full_names,
                    title=_format_env_name(env_folder_name, None),  # None = General
                    save_path=os.path.join(env_folder_path, f"boxplot_all_eval_types_{env_folder_name}.png"),
                    figsize=(16, 8)
                )
                print(f"Generated combined boxplot for {env_folder_name}")
            
            # Generate per-model boxplots (same model across different eval types)
            _generate_per_model_boxplots(env_folder_path, env_folder_name, per_model_data)


def _generate_per_model_boxplots(env_folder_path, env_folder_name, per_model_data):
    """Generate boxplots showing the same model across different eval types."""
    per_model_dir = os.path.join(env_folder_path, "per_model")
    os.makedirs(per_model_dir, exist_ok=True)
    
    for model_name, eval_type_returns in per_model_data.items():
        if len(eval_type_returns) < 2:
            # Skip if model only has one eval type
            continue
        
        # Sort eval types in a consistent order
        eval_order = ['standard', 'random_rules', 'h_ppo_eps_0.5', 'h_ppo_eps_1.0']
        sorted_eval_types = sorted(
            eval_type_returns.keys(),
            key=lambda x: eval_order.index(x) if x in eval_order else 100
        )
        
        data = [eval_type_returns[et] for et in sorted_eval_types]
        labels = [_format_eval_type(et) for et in sorted_eval_types]
        full_names = [model_name] * len(sorted_eval_types)  # Same model, so same color
        
        short_model_name = get_short_label(model_name).replace('\n', ' ')
        
        _create_styled_boxplot(
            data, labels, full_names,
            title=f"{short_model_name} — {_format_env_name_short(env_folder_name)}",
            save_path=os.path.join(per_model_dir, f"boxplot_{model_name}_across_eval_types.png"),
            figsize=(10, 6)
        )
        print(f"Generated per-model boxplot for {model_name}")


def _format_env_name_short(name):
    """Format environment name for display (short version for per-model plots)."""
    name = name.replace("MiniGrid-", "").replace("-v0", "")
    parts = name.split("_")
    env_part = parts[0] if parts else name
    keys_part = parts[1] if len(parts) > 1 else ""
    
    env_part = env_part.replace("-", " ").replace("x", "×")
    n_keys = keys_part.replace("keys", "")
    key_word = "key" if n_keys == "1" else "keys"
    
    return f"{env_part} ({n_keys} {key_word})"


def _format_env_name(name, eval_type=None):
    """Format environment name for display."""
    # MiniGrid-DoorKey-16x16-v0_1keys -> DoorKey 16×16 (1 key)
    name = name.replace("MiniGrid-", "").replace("-v0", "")
    parts = name.split("_")
    env_part = parts[0] if parts else name
    keys_part = parts[1] if len(parts) > 1 else ""
    
    env_part = env_part.replace("-", " ").replace("x", "×")
    n_keys = keys_part.replace("keys", "")
    key_word = "key" if n_keys == "1" else "keys"
    
    base_title = f"{env_part} ({n_keys} {key_word})"
    
    if eval_type:
        return f"{base_title} — {_format_eval_type(eval_type)}"
    else:
        return f"{base_title} — General"
    
    return base_title


def _format_eval_type(eval_type):
    """Format evaluation type for display."""
    mapping = {
        'standard': 'Standard',
        'random_rules': 'Random Rules',
        'h_ppo_eps_1.0': 'H-PPO (ε=1.0)',
        'h_ppo_eps_0.5': 'H-PPO (ε=0.5)',
    }
    return mapping.get(eval_type, eval_type)


def _create_styled_boxplot(data, labels, full_names, title, save_path, figsize=(14, 7)):
    """Create a clean, styled boxplot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colors based on model types
    colors = [get_color(name) for name in full_names]
    
    # Create boxplot with custom styling
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        flierprops=dict(marker='o', markersize=4, alpha=0.5),
        medianprops=dict(color='#2C3E50', linewidth=2.5),  # Dark blue-gray median line
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    
    # Apply colors to boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Add mean ± std text inside each box
    for i, d in enumerate(data):
        mean_val = np.mean(d)
        std_val = np.std(d)
        # Position text at the center of the box
        ax.text(i + 1, mean_val, f'{mean_val:.2f}\n±{std_val:.2f}',
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='black', bbox=dict(boxstyle='round,pad=0.2', 
                facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Style the plot
    ax.set_title(title, pad=15)
    ax.set_ylabel("Return", fontsize=12)
    ax.set_xlabel("")
    
    # Rotate x labels for readability
    plt.xticks(rotation=30, ha='right', fontsize=10)
    
    # Add subtle horizontal grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # Set y-axis limits with some padding
    all_values = [v for sublist in data for v in sublist]
    y_min = min(all_values)
    y_max = max(all_values)
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(max(0, y_min - y_padding), min(1.1, y_max + y_padding))
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()


def _generate_single_boxplot(folder_path, folder_name, csv_files):
    """Helper function for old structure compatibility (CSVs directly in env folder)."""
    all_data = []
    labels = []
    full_names = []
    # Sort by model order
    csv_files.sort(key=lambda x: get_model_order_key(x.replace(".csv", "")))
    
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        returns = []
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle both plain numbers and bracket-wrapped numbers like "[0.5]"
                val = row["Return"].strip("[]")
                returns.append(float(val))
        
        if returns:
            all_data.append(returns)
            full_name = csv_file.replace(".csv", "")
            full_names.append(full_name)
            labels.append(get_short_label(full_name))
    
    if all_data:
        _create_styled_boxplot(
            all_data, labels, full_names,
            title=_format_env_name(folder_name, "standard"),
            save_path=os.path.join(folder_path, f"boxplot_{folder_name}.png")
        )
        print(f"Generated boxplot for {folder_name}")


if __name__ == "__main__":
    generate_boxplots()
