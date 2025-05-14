from flask import Flask, render_template, request, jsonify, send_file, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os
import io
import base64
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Create directory for static files if it doesn't exist
os.makedirs('static', exist_ok=True)

# Generate synthetic data for nuclear winter simulation
def generate_simulation_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate features
    num_nukes = np.random.randint(10, 10000, n_samples)
    weapon_yield = np.random.uniform(50, 1000, n_samples)  # in kilotons
    detonation_altitude = np.random.uniform(0, 500, n_samples)  # in meters
    atmospheric_conditions = np.random.uniform(0, 1, n_samples)  # 0=clear, 1=cloudy
    population_density = np.random.uniform(10, 5000, n_samples)  # people per sq km
    
    # Generate targets with some noise and relationships
    base_soot = num_nukes * weapon_yield * 0.01
    soot_emission = base_soot * (1 + 0.2 * atmospheric_conditions) * np.random.normal(1, 0.1, n_samples)
    
    base_temp_drop = 0.01 * soot_emission
    temp_drop = base_temp_drop * np.random.normal(1, 0.05, n_samples)
    
    base_agri_impact = temp_drop * 2
    agricultural_impact = base_agri_impact * np.random.normal(1, 0.1, n_samples)  # % reduction
    
    base_gdp_impact = agricultural_impact * 0.5 + (num_nukes * 0.001)
    gdp_impact = base_gdp_impact * np.random.normal(1, 0.08, n_samples)  # % reduction
    
    base_casualties = num_nukes * weapon_yield * population_density * 0.00001
    casualties = base_casualties * np.random.normal(1, 0.2, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'num_nukes': num_nukes,
        'weapon_yield': weapon_yield,
        'detonation_altitude': detonation_altitude,
        'atmospheric_conditions': atmospheric_conditions,
        'population_density': population_density,
        'soot_emission': soot_emission,
        'temp_drop': temp_drop,
        'agricultural_impact': agricultural_impact,
        'gdp_impact': gdp_impact,
        'casualties': casualties
    })
    
    return data

# Generate data
simulation_data = generate_simulation_data(1000)

# Train models
def train_models(data):
    features = ['num_nukes', 'weapon_yield', 'detonation_altitude', 'atmospheric_conditions', 'population_density']
    targets = ['soot_emission', 'temp_drop', 'agricultural_impact', 'gdp_impact', 'casualties']
    
    X = data[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {}
    
    for target in targets:
        y = data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train model (RandomForest for better accuracy)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        models[target] = {
            'model': model,
            'accuracy': model.score(X_test, y_test)
        }
    
    return models, scaler

# Train the models and get the scaler
models, feature_scaler = train_models(simulation_data)

# Generate visualizations
def generate_visualizations():
    # 1. Create pairplot
    plt.figure(figsize=(10, 8))
    sns.pairplot(simulation_data[['num_nukes', 'weapon_yield', 'temp_drop', 'gdp_impact']], 
                 diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.tight_layout()
    plt.savefig('static/pairplot.png', dpi=100)
    plt.close()
    
    # 2. Soot Emission vs. Global Temperature Drop
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='soot_emission', y='temp_drop', data=simulation_data, alpha=0.6, hue='num_nukes', palette='viridis')
    plt.title('Soot Emission vs. Global Temperature Drop')
    plt.xlabel('Soot Emission (Tg)')
    plt.ylabel('Temperature Drop (°C)')
    plt.colorbar(label='Number of Nuclear Weapons')
    plt.tight_layout()
    plt.savefig('static/soot_temp.png', dpi=100)
    plt.close()
    
    # 3. Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sns.histplot(simulation_data['temp_drop'], kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Temperature Drop Distribution')
    
    sns.histplot(simulation_data['agricultural_impact'], kde=True, ax=axes[0, 1], color='green')
    axes[0, 1].set_title('Agricultural Impact Distribution')
    
    sns.histplot(simulation_data['gdp_impact'], kde=True, ax=axes[1, 0], color='orange')
    axes[1, 0].set_title('GDP Impact Distribution')
    
    sns.histplot(simulation_data['casualties'], kde=True, ax=axes[1, 1], color='red')
    axes[1, 1].set_title('Casualties Distribution')
    
    plt.tight_layout()
    plt.savefig('static/distribution.png', dpi=100)
    plt.close()
    
    # 4. GDP Impact by Nukes Used (boxplot)
    plt.figure(figsize=(12, 6))
    
    # Create categories for num_nukes
    simulation_data['nuke_category'] = pd.qcut(simulation_data['num_nukes'], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    sns.boxplot(x='nuke_category', y='gdp_impact', data=simulation_data, palette='RdYlBu_r')
    plt.title('GDP Impact by Number of Nuclear Weapons')
    plt.xlabel('Number of Nuclear Weapons Category')
    plt.ylabel('GDP Impact (%)')
    plt.tight_layout()
    plt.savefig('static/boxplot.png', dpi=100)
    plt.close()

# Generate visualizations on startup
generate_visualizations()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download')
def download():
    # Create a CSV file from the simulation data
    return send_file(
        io.BytesIO(simulation_data.to_csv(index=False).encode()),
        mimetype='text/csv',
        download_name='nuclear_winter_simulation.csv',
        as_attachment=True
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    try:
        num_nukes = int(request.form.get('num_nukes', 100))
        weapon_yield = float(request.form.get('weapon_yield', 100))
        detonation_altitude = float(request.form.get('detonation_altitude', 100))
        atmospheric_conditions = float(request.form.get('atmospheric_conditions', 0.5))
        population_density = float(request.form.get('population_density', 1000))
        
        # Prepare input data
        input_data = np.array([[num_nukes, weapon_yield, detonation_altitude, 
                               atmospheric_conditions, population_density]])
        
        # Scale the input data
        input_scaled = feature_scaler.transform(input_data)
        
        # Make predictions
        predictions = {}
        for target, model_info in models.items():
            model = model_info['model']
            prediction = model.predict(input_scaled)[0]
            predictions[target] = prediction
        
        # Generate visualization for the prediction
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar chart for impacts
        impact_labels = ['Temperature\nDrop (°C)', 'Agricultural\nImpact (%)', 'GDP\nImpact (%)', 'Casualties']
        impact_values = [predictions['temp_drop'], predictions['agricultural_impact'], 
                         predictions['gdp_impact'], predictions['casualties']]
        
        bars = ax.bar(impact_labels, impact_values, color=['skyblue', 'green', 'orange', 'red'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'Predicted Impact of {num_nukes} Nuclear Weapons (Yield: {weapon_yield} kt)')
        plt.tight_layout()
        
        # Save to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        
        # Convert to base64 for embedding in HTML
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Return predictions and visualization
        return jsonify({
            'success': True,
            'predictions': {
                'soot_emission': float(predictions['soot_emission']),
                'temp_drop': float(predictions['temp_drop']),
                'agricultural_impact': float(predictions['agricultural_impact']),
                'gdp_impact': float(predictions['gdp_impact']),
                'casualties': float(predictions['casualties'])
            },
            'visualization': img_str
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)