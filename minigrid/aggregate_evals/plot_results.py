import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Color palette for different model types
MODEL_COLORS = {
    'ppo': '#E74C3C',           # Red
    'ppo_reward_machine': '#1ABC9C',  # Teal
    'h_ppo_product': '#3498DB',  # Blue
    'h_ppo_symloss': '#F39C12',  # Orange
    'h_ppo_symloss_eps': '#2ECC71',  # Green
    'h_ppo_symloss_theta': '#9B59B6',  # Purple
}

# Define ordering for consistent plot arrangement
MODEL_ORDER = [
    ('ppo_', 0),  # ppo_ to avoid matching h_ppo
    ('ppo_reward_machine', 1),
    ('h_ppo_product', 2),
    ('h_ppo_symloss_eps', 4),  # eps before plain symloss in check order
    ('theta_0.25', 5),
    ('theta_0.5', 6),
    ('theta_0.75', 7),
    ('h_ppo_symloss', 3),  # plain symloss last in checks (after theta/eps)
]

def get_model_order_key(full_name):
    """Return a sort key based on desired model ordering."""
    name_lower = full_name.lower()
    
    # Check for ppo without h_ppo first (plain PPO should be first)
    if 'ppo' in name_lower and 'h_ppo' not in name_lower and 'ppo_reward_machine' not in name_lower and 'ppo_rm' not in name_lower:
        return 0
    
    # Check for ppo_reward_machine or ppo_rm (second)
    if 'ppo_reward_machine' in name_lower or 'ppo_rm' in name_lower:
        return 1
    
    # Check for theta variants first (most specific)
    if 'theta_0.25' in name_lower:
        return 5
    if 'theta_0.5' in name_lower:
        return 6  
    if 'theta_0.75' in name_lower:
        return 7
    
    # Then other h_ppo variants
    if 'h_ppo_product' in name_lower:
        return 2
    if 'h_ppo_symloss_eps' in name_lower:
        return 4
    if 'h_ppo_symloss' in name_lower:
        return 3
    
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
    if 'ppo_reward_machine' in name_lower or 'ppo_rm' in name_lower:
        return 'PPO\nReward Machine'
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
    elif 'ppo_reward_machine' in label.lower() or 'ppo_rm' in label.lower() or 'reward machine' in label.lower():
        return MODEL_COLORS['ppo_reward_machine']
    elif 'ppo' in label.lower() and 'h_ppo' not in label.lower():
        return MODEL_COLORS['ppo']
    return '#7F8C8D'  # Default gray


def get_model_type(full_model_name):
    """Extract the model type from full filename (e.g., 'ppo_MiniGrid-DoorKey-8x8' -> 'ppo')."""
    name_lower = full_model_name.lower()
    
    # Check for theta variants first (most specific)
    if 'theta_0.25' in name_lower:
        return 'h_ppo_symloss_theta_0.25'
    if 'theta_0.5' in name_lower:
        return 'h_ppo_symloss_theta_0.5'
    if 'theta_0.75' in name_lower:
        return 'h_ppo_symloss_theta_0.75'
    
    # Then other variants
    if 'h_ppo_symloss_eps' in name_lower:
        return 'h_ppo_symloss_eps'
    if 'h_ppo_symloss' in name_lower:
        return 'h_ppo_symloss'
    if 'h_ppo_product' in name_lower:
        return 'h_ppo_product'
    if 'ppo_reward_machine' in name_lower or 'ppo_rm' in name_lower:
        return 'ppo_reward_machine'
    if 'ppo' in name_lower and 'h_ppo' not in name_lower:
        return 'ppo'
    
    return full_model_name  # Return original if no match

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

    # Global data structure for success rate tables
    # Structure: {eval_type: {map_name: {model_name: success_rate}}}
    global_success_data = {}

    # Loop through env_id_nkeys subdirectories in aggregate_evals
    for env_entry in os.scandir(base_dir):
        if env_entry.is_dir() and env_entry.name != "tables":
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
            eval_order = ['standard', 'random_rules', 'h_ppo_eps_0.25', 'h_ppo_eps_0.5', 'h_ppo_eps_1.0']
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
                        
                        # Calculate and store success rate for global tables
                        success_rate = (sum(1 for r in returns if r != 0) / len(returns)) * 100
                        if eval_type_name not in global_success_data:
                            global_success_data[eval_type_name] = {}
                        if env_folder_name not in global_success_data[eval_type_name]:
                            global_success_data[eval_type_name][env_folder_name] = {}
                        global_success_data[eval_type_name][env_folder_name][full_name] = success_rate
                
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
    
    # Generate success rate tables
    _generate_success_rate_tables(base_dir, global_success_data)


def _generate_success_rate_tables(base_dir, global_success_data):
    """Generate success rate tables for each eval type with models as columns and maps as rows."""
    if not global_success_data:
        print("No success rate data to generate tables.")
        return
    
    tables_dir = os.path.join(base_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    # Create subdirectories for better organization
    per_eval_dir = os.path.join(tables_dir, "per_eval_type")
    per_model_table_dir = os.path.join(tables_dir, "per_model")
    combined_dir = os.path.join(tables_dir, "combined")
    os.makedirs(per_eval_dir, exist_ok=True)
    os.makedirs(per_model_table_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    
    # Convert data to use model types instead of full filenames
    # Structure: {eval_type: {map_name: {model_type: success_rate}}}
    typed_success_data = {}
    all_model_types = set()
    all_maps = set()
    
    for eval_type, maps_data in global_success_data.items():
        if eval_type not in typed_success_data:
            typed_success_data[eval_type] = {}
        for map_name, models_data in maps_data.items():
            all_maps.add(map_name)
            if map_name not in typed_success_data[eval_type]:
                typed_success_data[eval_type][map_name] = {}
            for model_name, success_rate in models_data.items():
                model_type = get_model_type(model_name)
                all_model_types.add(model_type)
                typed_success_data[eval_type][map_name][model_type] = success_rate
    
    # Sort model types by order key
    sorted_model_types = sorted(all_model_types, key=get_model_order_key)
    sorted_maps = sorted(all_maps)
    
    # Generate table for each eval type
    for eval_type in typed_success_data.keys():
        maps_data = typed_success_data[eval_type]
        # Create DataFrame
        table_data = []
        for map_name in sorted_maps:
            if map_name in maps_data:
                row = {'Map': _format_env_name_short(map_name)}
                for model_type in sorted_model_types:
                    short_label = get_short_label(model_type).replace('\n', ' ')
                    if model_type in maps_data[map_name]:
                        row[short_label] = f"{maps_data[map_name][model_type]:.1f}%"
                    else:
                        row[short_label] = "-"
                table_data.append(row)
        
        if not table_data:
            continue
            
        df = pd.DataFrame(table_data)
        
        # Save as CSV in per_eval_type folder
        csv_path = os.path.join(per_eval_dir, f"success_rate_{eval_type}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Generated success rate table: {csv_path}")
        
        # Also create a styled image of the table
        _create_table_image(df, f"Success Rate — {_format_eval_type(eval_type)}", 
                           os.path.join(per_eval_dir, f"success_rate_{eval_type}.png"))
    
    # Generate per-model tables (model across eval types and maps)
    _generate_per_model_tables(per_model_table_dir, typed_success_data, sorted_model_types, sorted_maps)
    
    # Generate a combined table across all eval types
    _generate_combined_success_table(combined_dir, typed_success_data, sorted_model_types, sorted_maps)


def _generate_per_model_tables(per_model_dir, typed_success_data, sorted_model_types, sorted_maps):
    """Generate tables for each model showing success rate across maps and eval types."""
    eval_order = ['standard', 'random_rules', 'h_ppo_eps_0.25', 'h_ppo_eps_0.5', 'h_ppo_eps_1.0']
    sorted_eval_types = sorted(typed_success_data.keys(), 
                               key=lambda x: eval_order.index(x) if x in eval_order else 100)
    
    for model_type in sorted_model_types:
        table_data = []
        for map_name in sorted_maps:
            row = {'Map': _format_env_name_short(map_name)}
            for eval_type in sorted_eval_types:
                if (eval_type in typed_success_data and 
                    map_name in typed_success_data[eval_type] and 
                    model_type in typed_success_data[eval_type][map_name]):
                    row[_format_eval_type(eval_type)] = f"{typed_success_data[eval_type][map_name][model_type]:.1f}%"
                else:
                    row[_format_eval_type(eval_type)] = "-"
            table_data.append(row)
        
        if not table_data or all(all(v == "-" for k, v in row.items() if k != 'Map') for row in table_data):
            continue
            
        df = pd.DataFrame(table_data)
        short_model_name = get_short_label(model_type).replace('\n', ' ')
        safe_filename = model_type.replace('/', '_').replace(' ', '_')
        
        csv_path = os.path.join(per_model_dir, f"success_rate_{safe_filename}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Generated per-model table: {csv_path}")
        
        _create_table_image(df, f"Success Rate — {short_model_name}", 
                           os.path.join(per_model_dir, f"success_rate_{safe_filename}.png"))


def _generate_combined_success_table(tables_dir, typed_success_data, sorted_model_types, sorted_maps):
    """Generate a combined success rate table with eval types as additional column grouping."""
    eval_order = ['standard', 'random_rules', 'h_ppo_eps_0.25', 'h_ppo_eps_0.5', 'h_ppo_eps_1.0']
    sorted_eval_types = sorted(typed_success_data.keys(), 
                               key=lambda x: eval_order.index(x) if x in eval_order else 100)
    
    # Create combined table data
    table_data = []
    for map_name in sorted_maps:
        for eval_type in sorted_eval_types:
            if eval_type in typed_success_data and map_name in typed_success_data[eval_type]:
                row = {
                    'Map': _format_env_name_short(map_name),
                    'Eval Type': _format_eval_type(eval_type)
                }
                for model_type in sorted_model_types:
                    short_label = get_short_label(model_type).replace('\n', ' ')
                    if model_type in typed_success_data[eval_type][map_name]:
                        row[short_label] = f"{typed_success_data[eval_type][map_name][model_type]:.1f}%"
                    else:
                        row[short_label] = "-"
                table_data.append(row)
    
    if table_data:
        df = pd.DataFrame(table_data)
        csv_path = os.path.join(tables_dir, "success_rate_combined.csv")
        df.to_csv(csv_path, index=False)
        print(f"Generated combined success rate table: {csv_path}")
        
        # Also create a PNG image of the combined table
        _create_table_image(df, "Success Rate — Combined", 
                           os.path.join(tables_dir, "success_rate_combined.png"),
                           eval_type_col='Eval Type')


def _create_table_image(df, title, save_path, eval_type_col=None):
    """Create a styled image of a DataFrame table."""
    # Calculate figure size based on content - wider columns for long headers
    n_cols = len(df.columns)
    n_rows = len(df)
    col_width = 2.0  # Wider columns to fit long headers
    fig_width = max(14, n_cols * col_width)
    fig_height = max(4, n_rows * 0.5 + 2)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)  # Slightly smaller font for better fit
    table.auto_set_column_width(col=list(range(n_cols)))  # Auto-size columns
    table.scale(1.2, 1.8)  # Taller rows for better readability
    
    # Style header row with smaller font for long labels
    for j, col in enumerate(df.columns):
        cell = table[(0, j)]
        cell.set_facecolor('#3498DB')
        cell.set_text_props(color='white', fontweight='bold', fontsize=8)
        cell.set_height(0.15)  # Taller header cells
    
    # Define colors for different eval types (light pastel colors for readability)
    eval_type_colors = {
        'Standard': '#E8F6F3',        # Light teal
        'Random Rules': '#FEF9E7',     # Light yellow
        'H-PPO (ε=0.25)': '#F4ECF7',   # Light purple
        'H-PPO (ε=0.5)': '#EBF5FB',    # Light blue
        'H-PPO (ε=1.0)': '#FDEDEC',    # Light red/pink
    }
    default_colors = ['#F8F9FA', '#FFFFFF']  # Alternating gray/white
    
    # Determine row colors based on eval type column if provided
    if eval_type_col and eval_type_col in df.columns:
        eval_col_idx = df.columns.get_loc(eval_type_col)
        for i in range(len(df)):
            eval_type_value = df.iloc[i][eval_type_col]
            row_color = eval_type_colors.get(eval_type_value, default_colors[i % 2])
            for j in range(len(df.columns)):
                cell = table[(i + 1, j)]
                cell.set_facecolor(row_color)
    else:
        # Alternate row colors (fallback)
        for i in range(len(df)):
            for j in range(len(df.columns)):
                cell = table[(i + 1, j)]
                cell.set_facecolor(default_colors[i % 2])
    
    # Bold the highest success rate in each row
    # Find columns that contain percentage values (success rates)
    for i in range(len(df)):
        max_val = -1
        max_col_idx = -1
        for j, col in enumerate(df.columns):
            val_str = str(df.iloc[i, j])
            if '%' in val_str and val_str != '-':
                try:
                    val = float(val_str.replace('%', ''))
                    if val > max_val:
                        max_val = val
                        max_col_idx = j
                except ValueError:
                    pass
        # Bold the cell with max value
        if max_col_idx >= 0:
            cell = table[(i + 1, max_col_idx)]
            cell.set_text_props(fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Generated table image: {save_path}")


def _generate_per_model_boxplots(env_folder_path, env_folder_name, per_model_data):
    """Generate boxplots showing the same model across different eval types."""
    per_model_dir = os.path.join(env_folder_path, "per_model")
    os.makedirs(per_model_dir, exist_ok=True)
    
    for model_name, eval_type_returns in per_model_data.items():
        if len(eval_type_returns) < 2:
            # Skip if model only has one eval type
            continue
        
        # Sort eval types in a consistent order
        eval_order = ['standard', 'random_rules', 'h_ppo_eps_0.25', 'h_ppo_eps_0.5', 'h_ppo_eps_1.0']
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
        'h_ppo_eps_0.25': 'H-PPO (ε=0.25)',
        'h_ppo_eps_0.5': 'H-PPO (ε=0.5)',
        'h_ppo_eps_1.0': 'H-PPO (ε=1.0)',
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
