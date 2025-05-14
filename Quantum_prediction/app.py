from flask import Flask, render_template, send_file, request, jsonify, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import io
import base64


app = Flask(__name__)

# Step 1: Data Simulation (Synthetic)
def generate_simulation():
    import numpy as np
    np.random.seed(42)
    n_simulations = 10000

    data = {
        "nukes_used": np.random.randint(1, 50, size=n_simulations),
        "avg_yield_kt": np.random.uniform(15, 1000, size=n_simulations),
        "target_city_population": np.random.uniform(1e5, 2e7, size=n_simulations),
        "urbanization_level": np.random.uniform(0.4, 1.0, size=n_simulations),
        "soot_emission_Tg": np.random.uniform(0.1, 10, size=n_simulations),
    }
    
    data["total_targeted_population"] = data["target_city_population"] * data["nukes_used"] * data["urbanization_level"]
    data["human_lives_lost_millions"] = data["total_targeted_population"] * np.random.uniform(0.2, 0.8, size=n_simulations) / 1e6
    data["gdp_impact_pct"] = np.clip(data["nukes_used"] * np.random.uniform(0.5, 1.5, size=n_simulations), 5, 100)
    data["expected_global_temp_drop_C"] = data["soot_emission_Tg"] * np.random.uniform(0.05, 0.2, size=n_simulations)
    data["estimated_famine_risk_millions"] = data["expected_global_temp_drop_C"] * np.random.uniform(50, 500, size=n_simulations)

    return pd.DataFrame(data)

# Global variable
df_sim = generate_simulation()

# Feature and target setup
features = ['nukes_used', 'avg_yield_kt', 'target_city_population', 'urbanization_level', 'soot_emission_Tg']
targets = ['human_lives_lost_millions', 'gdp_impact_pct', 'expected_global_temp_drop_C', 'estimated_famine_risk_millions']

# Scaling and splitting
X = df_sim[features]
y = df_sim[targets]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a model for each target
models = {}
for i, target in enumerate(targets):
    y_target = y[target]
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_target, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    models[target] = model


# Save to CSV for download
@app.route('/download')
def download():
    csv_path = 'nuclear_winter_simulations.csv'
    df_sim.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True)

# Create visualizations on startup
def create_visualizations():
    sns.set(style="whitegrid")

    # Pairplot
    pairplot_path = 'static/pairplot.png'
    sns.pairplot(df_sim[["nukes_used", "avg_yield_kt", "human_lives_lost_millions", "gdp_impact_pct"]])
    plt.savefig(pairplot_path)
    plt.clf()

    # Soot vs Temperature Drop
    soot_temp_path = 'static/soot_temp.png'
    plt.figure(figsize=(10, 6))
    sns.regplot(x='soot_emission_Tg', y='expected_global_temp_drop_C', data=df_sim)
    plt.title('Soot Emission vs. Global Temperature Drop')
    plt.xlabel('Soot Emission (Tg)')
    plt.ylabel('Expected Global Temperature Drop (°C)')
    plt.savefig(soot_temp_path)
    plt.clf()

    # Histogram plots
    dist_path = 'static/distribution.png'
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df_sim['human_lives_lost_millions'], kde=True)
    plt.title('Distribution of Human Lives Lost (Millions)')
    plt.subplot(1, 2, 2)
    sns.histplot(df_sim['gdp_impact_pct'], kde=True)
    plt.title('Distribution of GDP Impact (%)')
    plt.tight_layout()
    plt.savefig(dist_path)
    plt.clf()

    # Boxplot
    boxplot_path = 'static/boxplot.png'
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='nukes_used', y='gdp_impact_pct', data=df_sim)
    plt.title('GDP Impact vs. Number of Nukes Used')
    plt.savefig(boxplot_path)
    plt.clf()

    

@app.route('/')
def home():
    return render_template('index.html')

from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor
import joblib

# === Data Loading and Model Training ===
# Load sample data from CSV
# Assume CSV has columns: nukes_used, avg_yield_kt, target_city_population,
# urbanization_level, soot_emission_Tg,
# human_lives_lost_millions, gdp_impact_pct,
# expected_global_temp_drop_C, estimated_famine_risk_millions
df = pd.read_csv('nuclear_winter_simulations.csv')

# Define feature matrix X and target matrix y
features = ['nukes_used', 'avg_yield_kt', 'target_city_population', 
            'urbanization_level', 'soot_emission_Tg']
targets = ['human_lives_lost_millions', 'gdp_impact_pct', 
           'expected_global_temp_drop_C', 'estimated_famine_risk_millions']
X = df[features].values
y = df[targets].values

# Train Random Forest Regressor on sample data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)  # scikit-learn usage as in official docs:contentReference[oaicite:5]{index=5}

# Save the trained model to disk for later use
joblib.dump(model, 'nuclear_winter_model.joblib')  # serialize model:contentReference[oaicite:6]{index=6}

# Load the serialized model (so we don't retrain on each run)
model = joblib.load('nuclear_winter_model.joblib')  # load saved model:contentReference[oaicite:7]{index=7}

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            print("Processing POST request to /predict")
            
            # Get form values
            nukes_used = int(request.form.get('nukes_used'))
            avg_yield_kt = float(request.form.get('avg_yield_kt'))
            target_population = int(request.form.get('target_city_population'))
            urbanization = float(request.form.get('urbanization_level'))
            soot_emission = float(request.form.get('soot_emission_Tg'))
            
            print(f"Input values: nukes={nukes_used}, yield={avg_yield_kt}, pop={target_population}, urban={urbanization}, soot={soot_emission}")
            
            # Create input feature array
            input_features = np.array([[nukes_used, avg_yield_kt, target_population, 
                                    urbanization, soot_emission]])
            
            # Use the global models dictionary to make predictions
            predictions = {}
            for target in targets:
                model = models[target]
                scaled_input = scaler.transform(input_features)
                predictions[target] = float(model.predict(scaled_input)[0])
            
            print(f"Generated predictions: {predictions}")

            # === Visualization ===
            print("Generating visualization...")
            
            # Create visualization using BytesIO to send the image directly
            plt.figure(figsize=(10, 6))
            categories = list(predictions.keys())
            values = list(predictions.values())
            
            # Make more readable labels
            display_categories = [
                'Lives Lost (M)', 
                'GDP Impact (%)', 
                'Temp Drop (°C)', 
                'Famine Risk (M)'
            ]
            
            plt.bar(display_categories, values, color='skyblue')
            plt.title('Nuclear Winter Impact Predictions')
            plt.ylabel('Estimated Value')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save the figure to a BytesIO object
            img_io = io.BytesIO()
            plt.savefig(img_io, format='png')
            img_io.seek(0)
            plt.close()
            
            # Convert to base64 for embedding directly in HTML
            img_data = base64.b64encode(img_io.getvalue()).decode('utf-8')
            img_src = f"data:image/png;base64,{img_data}"
            
            print("Image generated and encoded successfully")
            print(f"Image data starts with: {img_src[:50]}...")

            return render_template('predict.html', 
                               predictions=predictions, 
                               img_src=img_src)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in predict route: {error_details}")
            return render_template('predict.html', error=str(e))

    return render_template('predict.html')

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    create_visualizations()
    app.run(debug=True)
