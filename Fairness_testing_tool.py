import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import time
from datetime import datetime
import multiprocessing as mp
from joblib import Parallel, delayed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def silence_tensorflow_warnings():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    import logging
    logging.getLogger('absl').setLevel(logging.ERROR)
    
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_and_preprocess_data(file_path, target_column):
    print(f"Loading dataset from {file_path} with target column '{target_column}'")
    df = pd.read_csv(file_path)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()}")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test



def load_dnn_model(model_path):
    if hasattr(load_dnn_model, 'cached_model') and hasattr(load_dnn_model, 'cached_path'):
        if load_dnn_model.cached_path == model_path:
            return load_dnn_model.cached_model
    
    # Load the model
    model = load_model(model_path)
    
    # Cache the model
    load_dnn_model.cached_model = model
    load_dnn_model.cached_path = model_path
    
    return model


def get_sample_hash(sample):
    values_tuple = tuple(sample.values)
    return hash(values_tuple)


def generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns):
    idx = np.random.choice(len(X_test))
    sample_a = X_test.iloc[idx].copy()
    sample_b = sample_a.copy()

    # Apply perturbation on sensitive features (random flipping)
    for col in sensitive_columns:
        if col in X_test.columns: 
            unique_values = X_test[col].unique()
            current_value = sample_b[col]
            other_values = [v for v in unique_values if v != current_value]
            if other_values: 
                sample_b[col] = np.random.choice(other_values)

    # Apply perturbation on non-sensitive features
    for col in non_sensitive_columns:
        if col in X_test.columns:  # Ensure the column exists
            col_dtype = X_test[col].dtype
            
            # Only apply perturbation to numeric columns
            if np.issubdtype(col_dtype, np.number):
                min_val = X_test[col].min()
                max_val = X_test[col].max()
                perturbation = np.random.uniform(-0.1 * (max_val - min_val), 0.1 * (max_val - min_val))
                
                # Check if the column is integer type, and cast appropriately
                if np.issubdtype(col_dtype, np.integer):
                    # For integer columns, round the result and convert back to integer
                    sample_a[col] = int(np.clip(sample_a[col] + perturbation, min_val, max_val))
                    sample_b[col] = int(np.clip(sample_b[col] + perturbation, min_val, max_val))
                else:
                    # For float columns, keep as float
                    sample_a[col] = np.clip(sample_a[col] + perturbation, min_val, max_val)
                    sample_b[col] = np.clip(sample_b[col] + perturbation, min_val, max_val)

    return sample_a, sample_b

def make_predictions_batch(model, samples, batch_size=32):
    if not samples:
        return []
    
    # Convert samples to DataFrame and then to numpy
    if isinstance(samples[0], pd.Series):
        samples_df = pd.DataFrame(samples)
    else:
        samples_df = pd.DataFrame(samples)
    
    # Get numpy array
    samples_array = samples_df.values
    
    # Make predictions in batches
    predictions = []
    for i in range(0, len(samples_array), batch_size):
        batch = samples_array[i:i+batch_size]
        batch_predictions = model.predict(batch, verbose=0)
        
        # Extract probabilities
        if batch_predictions.ndim > 1:
            batch_predictions = batch_predictions[:, 0]
        
        predictions.extend(batch_predictions)
    
    return predictions


class PredictionCache:
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, sample_hash):
        return self.cache.get(sample_hash)
    
    def add(self, sample_hash, prediction):
        # If cache is full, remove a random item
        if len(self.cache) >= self.max_size:
            key_to_remove = next(iter(self.cache))
            del self.cache[key_to_remove]
        
        self.cache[sample_hash] = prediction


def evaluate_discrimination_batch(model, sample_pairs, cache=None, threshold=0.05):
    if not sample_pairs:
        return [], []
    
    # Extract samples
    samples_a = [pair[0] for pair in sample_pairs]
    samples_b = [pair[1] for pair in sample_pairs]
    
    # Use cache if provided
    if cache:
        # Get hashes for all samples
        hashes_a = [get_sample_hash(sample) for sample in samples_a]
        hashes_b = [get_sample_hash(sample) for sample in samples_b]
        
        # Get predictions from cache or compute them
        preds_a = []
        samples_to_predict_a = []
        indices_to_predict_a = []
        
        for i, (sample, sample_hash) in enumerate(zip(samples_a, hashes_a)):
            cached_pred = cache.get(sample_hash)
            if cached_pred is not None:
                preds_a.append(cached_pred)
            else:
                samples_to_predict_a.append(sample)
                indices_to_predict_a.append(i)
        
        # Same for samples_b
        preds_b = []
        samples_to_predict_b = []
        indices_to_predict_b = []
        
        for i, (sample, sample_hash) in enumerate(zip(samples_b, hashes_b)):
            cached_pred = cache.get(sample_hash)
            if cached_pred is not None:
                preds_b.append(cached_pred)
            else:
                samples_to_predict_b.append(sample)
                indices_to_predict_b.append(i)
        
        # Predict missing values
        if samples_to_predict_a:
            new_preds_a = make_predictions_batch(model, samples_to_predict_a)
            # Update cache
            for i, pred in zip(indices_to_predict_a, new_preds_a):
                cache.add(hashes_a[i], pred)
                preds_a.append(pred)
        
        if samples_to_predict_b:
            new_preds_b = make_predictions_batch(model, samples_to_predict_b)
            # Update cache
            for i, pred in zip(indices_to_predict_b, new_preds_b):
                cache.add(hashes_b[i], pred)
                preds_b.append(pred)
    else:
        # Predict all samples
        preds_a = make_predictions_batch(model, samples_a)
        preds_b = make_predictions_batch(model, samples_b)
    
    # Calculate differences and determine discrimination
    is_discriminatory = []
    pred_diffs = []
    
    for pred_a, pred_b in zip(preds_a, preds_b):
        pred_diff = abs(pred_a - pred_b)
        pred_diffs.append(pred_diff)
        is_discriminatory.append(1 if pred_diff > threshold else 0)
    
    return is_discriminatory, pred_diffs


def calculate_idi_ratio(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000):
    """Calculate IDI ratio using random search with batched processing"""
    discrimination_count = 0
    discrimination_pairs = []
    prediction_diffs = []
    
    # Check if we need to adjust threshold for specific datasets
    threshold = 0.05
    dataset_name = None
    if 'race' in sensitive_columns and 'male' in sensitive_columns:
        dataset_name = "law_school"
        threshold = 0.05
        print(f"  Detected {dataset_name} dataset, using relaxed threshold of {threshold}")
    elif 'Black' in sensitive_columns and 'FemalePctDiv' in sensitive_columns:
        dataset_name = "communities_crime"
        threshold = 0.05
        print(f"  Detected {dataset_name} dataset, using relaxed threshold of {threshold}")
    
    # Keep track of unique discriminatory pairs
    unique_pairs = set()
    
    # Create prediction cache
    cache = PredictionCache()
    
    # Use batched processing
    batch_size = 50  # Adjust based on your system's capabilities
    
    for i in range(0, num_samples, batch_size):
        if i % 200 == 0:  # Progress indicator
            print(f"  Processing samples {i}-{min(i+batch_size, num_samples)}/{num_samples}...")
        
        # Generate batch of pairs
        current_batch_size = min(batch_size, num_samples - i)
        sample_pairs = []
        
        for _ in range(current_batch_size):
            try:
                sample_a, sample_b = generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns)
                sample_pairs.append((sample_a, sample_b))
            except Exception as e:
                print(f"Error generating sample pair: {str(e)}")
        
        # Evaluate batch
        if sample_pairs:
            is_discriminatory, pred_diffs = evaluate_discrimination_batch(model, sample_pairs, cache, threshold)
            
            for j, (is_disc, pred_diff) in enumerate(zip(is_discriminatory, pred_diffs)):
                prediction_diffs.append(pred_diff)
                
                if is_disc:
                    # Check if this is a unique pair
                    sample_a, sample_b = sample_pairs[j]
                    pair_hash = (get_sample_hash(sample_a), get_sample_hash(sample_b))
                    
                    if pair_hash not in unique_pairs:
                        unique_pairs.add(pair_hash)
                        discrimination_count += 1
                        discrimination_pairs.append((sample_a, sample_b))
    
    # Calculate IDI ratio
    total_generated = len(prediction_diffs)
    if total_generated == 0:
        print("Warning: No valid samples were generated!")
        return 0, [], []

    IDI_ratio = discrimination_count / total_generated
    
    print(f"  Found {discrimination_count} unique discriminatory instances out of {total_generated} evaluations (IDI Ratio: {IDI_ratio:.4f})")
    return IDI_ratio, discrimination_pairs, prediction_diffs

def genetic_algorithm_fairness_test(model, X_test, sensitive_columns, non_sensitive_columns, 
                                 population_size=50, generations=10, mutation_rate=0.2, 
                                 tournament_size=3, threshold=0.05):
    print("  Initializing population...")
    
    # Initialize metrics
    discrimination_pairs = []  # Will store actual pair objects
    prediction_diffs = []
    total_evaluations = 0
    
    # Use a set to track unique discriminatory pairs
    unique_pairs = set()  # Will store pair hashes
    
    # Create prediction cache
    cache = PredictionCache()

    # Check if we need to adjust threshold for specific datasets
    dataset_name = None
    if 'race' in sensitive_columns and 'male' in sensitive_columns:
        dataset_name = "law_school"
        threshold = 0.05
        print(f"  Detected {dataset_name} dataset, using relaxed threshold of {threshold}")
    elif 'Black' in sensitive_columns and 'FemalePctDiv' in sensitive_columns:
        dataset_name = "communities_crime"
        threshold = 0.05
        print(f"  Detected {dataset_name} dataset, using relaxed threshold of {threshold}")

    # Generate initial population with diversity
    population = []
    attempt_count = 0
    max_attempts = population_size * 3  # Allow 3 attempts per required population member
    
    # Track hashes for diversity
    population_hashes = set()
    
    while len(population) < population_size and attempt_count < max_attempts:
        attempt_count += 1
        try:
            sample_a, sample_b = generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns)
            pair_hash = (get_sample_hash(sample_a), get_sample_hash(sample_b))
            
            # Only add if not already in population (ensures diversity)
            if pair_hash not in population_hashes:
                population_hashes.add(pair_hash)
                population.append((sample_a, sample_b))
        except Exception as e:
            print(f"Error initializing population: {str(e)}")
            continue
    
    # Actual population size might be less than requested if errors occurred
    actual_pop_size = len(population)
    if actual_pop_size == 0:
        print("Failed to initialize population")
        return 0, [], []
    elif actual_pop_size < population_size:
        print(f"  Warning: Could only create {actual_pop_size}/{population_size} diverse population members")
    
    # Track best fitness over generations
    best_fitness_history = []
    best_discriminatory_count = 0
    
    # Evolve the population over generations
    for gen in range(generations):
        print(f"  Generation {gen+1}/{generations}...")
        
        # Evaluate current population in batch
        is_discriminatory, pred_diffs = evaluate_discrimination_batch(model, population, cache, threshold)
        total_evaluations += len(population)

        new_discriminatory_count = 0
        fitness_scores = []
        
        for i, (is_disc, pred_diff) in enumerate(zip(is_discriminatory, pred_diffs)):
            if is_disc:
                fitness = 10.0 + pred_diff  # Base points + bonus for larger differences
            else:
                fitness = pred_diff / threshold  # Proportional score based on how close to threshold
            
            fitness_scores.append(fitness)
            prediction_diffs.append(pred_diff)
            
            if is_disc:
                sample_a, sample_b = population[i]
                pair_hash = (get_sample_hash(sample_a), get_sample_hash(sample_b))
                
                # Only count and store unique pairs
                if pair_hash not in unique_pairs:
                    unique_pairs.add(pair_hash)
                    new_discriminatory_count += 1
                    discrimination_pairs.append((sample_a, sample_b))
        
        # Update best discriminatory count
        best_discriminatory_count = max(best_discriminatory_count, len(unique_pairs))
        
        # Keep track of best fitness
        best_fitness = max(fitness_scores) if fitness_scores else 0
        best_fitness_history.append(best_fitness)
        
        # Print progress info
        print(f"    Found {new_discriminatory_count} new unique discriminatory instances in this generation")
        print(f"    Total unique discriminatory instances so far: {len(unique_pairs)}")
        print(f"    Best fitness in this generation: {best_fitness:.4f}")
        print(f"    Average prediction difference: {np.mean(pred_diffs) if pred_diffs else 0:.4f}")
        
        # If last generation, stop here
        if gen == generations - 1:
            break
        
        # Check if we have valid fitness scores
        if not fitness_scores or max(fitness_scores) == 0:
            print("No valid fitness scores, stopping evolution")
            break
        
        # Implement elitism - keep top 10% of population
        elite_count = max(1, int(actual_pop_size * 0.1))  # At least 1 elite member
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        elite_population = [population[i] for i in elite_indices]
        
        # Tournament selection with increased tournament size
        parents = []
        for _ in range(actual_pop_size - elite_count):  # Need fewer parents due to elitism
            tournament_size_actual = min(tournament_size, len(population))
            if tournament_size_actual < 1:
                break
                
            tournament_indices = np.random.choice(range(len(population)), tournament_size_actual, replace=False)
            winner_idx = tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]
            parents.append(population[winner_idx])
        
        # Create new generation
        new_population = list(elite_population)  # Start with elite population
        
        while len(new_population) < actual_pop_size:
            if not parents:
                break
                
            # Randomly select parents
            parent1 = parents[np.random.randint(0, len(parents))]
            parent2 = parents[np.random.randint(0, len(parents))]
            
            # Create child by mixing features (crossover)
            child_a = parent1[0].copy()
            child_b = parent1[1].copy()
            
            # Enhanced crossover strategy
            # 1. Always crossover sensitive features (with 50% probability per feature)
            for col in sensitive_columns:
                if col in X_test.columns and np.random.random() < 0.5:
                    child_a[col] = parent2[0][col]
                    child_b[col] = parent2[1][col]
            
            # 2. Sometimes crossover non-sensitive features (with 30% probability per feature)
            for col in non_sensitive_columns:
                if col in X_test.columns and np.random.random() < 0.3:
                    child_a[col] = parent2[0][col]
                    child_b[col] = parent2[1][col]
            
            # Apply mutation with increased rate
            if np.random.random() < mutation_rate:
                for col in non_sensitive_columns:
                    if col in X_test.columns and np.issubdtype(X_test[col].dtype, np.number):
                        if np.random.random() < 0.25:  # 25% chance to mutate each feature
                            min_val = X_test[col].min()
                            max_val = X_test[col].max()
                            
                            perturbation = np.random.uniform(-0.2 * (max_val - min_val), 0.2 * (max_val - min_val))
                            
                            if np.issubdtype(X_test[col].dtype, np.integer):
                                child_a[col] = int(np.clip(child_a[col] + perturbation, min_val, max_val))
                                child_b[col] = int(np.clip(child_b[col] + perturbation, min_val, max_val))
                            else:
                                child_a[col] = np.clip(child_a[col] + perturbation, min_val, max_val)
                                child_b[col] = np.clip(child_b[col] + perturbation, min_val, max_val)
            
            # Check if this pair is already in the new population (avoid duplicates)
            pair_hash = (get_sample_hash(child_a), get_sample_hash(child_b))
            if not any(get_sample_hash(p[0]) == pair_hash[0] and get_sample_hash(p[1]) == pair_hash[1] for p in new_population):
                new_population.append((child_a, child_b))
        
        population = new_population
    
    # Calculate IDI ratio - use the number of unique pairs found
    unique_count = len(unique_pairs)
    IDI_ratio = unique_count / total_evaluations if total_evaluations > 0 else 0
    
    print(f"  Found {unique_count} unique discriminatory instances out of {total_evaluations} evaluations")
    print(f"  Final IDI Ratio: {IDI_ratio:.4f}")
    print(f"  Best fitness trend: {' -> '.join([f'{x:.2f}' for x in best_fitness_history])}")
    
    return IDI_ratio, discrimination_pairs, prediction_diffs


def run_method_test(model, X_test, sensitive_columns, non_sensitive_columns, method, run_idx, num_samples=1000):
    try:
        print(f"\nRun {run_idx} - {method}...")
        start_time = time.time()
        
        if method == 'baseline':
            idi_ratio, pairs, diffs = calculate_idi_ratio(
                model, X_test, sensitive_columns, non_sensitive_columns, num_samples
            )
        else:
            pop_size = 50
            generations = num_samples // pop_size
            idi_ratio, pairs, diffs = genetic_algorithm_fairness_test(
                model, X_test, sensitive_columns, non_sensitive_columns, pop_size, generations
            )
        
        elapsed_time = time.time() - start_time
        avg_diff = np.mean(diffs) if diffs else 0
        
        print(f"  {method.capitalize()}: IDI Ratio = {idi_ratio:.4f}, Time = {elapsed_time:.2f}s, "
              f"Found {len(pairs)} discriminatory instances")
        
        result = {'idi_ratio': idi_ratio, 'time': elapsed_time, 'avg_diff': avg_diff}
        return result
    
    except Exception as e:
        print(f"Error in run_method_test ({method}, run {run_idx}): {str(e)}")
        return {'idi_ratio': 0, 'time': 0, 'avg_diff': 0}


def compare_methods(X_test, model, sensitive_columns, non_sensitive_columns, runs=5, num_samples=1000, parallel=True):
    """
    Runs and compares the baseline random search and genetic algorithm approaches.
    """
    # Initialize results dictionaries
    baseline_results = {'idi_ratios': [], 'times': [], 'avg_pred_diffs': []}
    ga_results = {'idi_ratios': [], 'times': [], 'avg_pred_diffs': []}
    
    print(f"Running comparison over {runs} runs...")
    print(f"Baseline: {num_samples} samples per run")
    print(f"Genetic Algorithm: population=50, generations={num_samples//50} (approx. {num_samples} evaluations)")
    
    # Use parallel processing if requested and available
    if parallel and mp.cpu_count() > 1:
        n_jobs = max(1, min(runs, mp.cpu_count() - 1))
        print(f"Using {n_jobs} parallel processes for method comparison")
        
        # Run baseline tests in parallel
        baseline_job_args = [(model, X_test, sensitive_columns, non_sensitive_columns, 'baseline', i+1, num_samples) 
                            for i in range(runs)]
        baseline_job_results = Parallel(n_jobs=n_jobs)(delayed(run_method_test)(*args) for args in baseline_job_args)
        
        # Run GA tests in parallel
        ga_job_args = [(model, X_test, sensitive_columns, non_sensitive_columns, 'ga', i+1, num_samples) 
                      for i in range(runs)]
        ga_job_results = Parallel(n_jobs=n_jobs)(delayed(run_method_test)(*args) for args in ga_job_args)
        
        # Collect results
        for result in baseline_job_results:
            baseline_results['idi_ratios'].append(result['idi_ratio'])
            baseline_results['times'].append(result['time'])
            baseline_results['avg_pred_diffs'].append(result['avg_diff'])
            
        for result in ga_job_results:
            ga_results['idi_ratios'].append(result['idi_ratio'])
            ga_results['times'].append(result['time'])
            ga_results['avg_pred_diffs'].append(result['avg_diff'])
    else:
        # Sequential processing
        for run_idx in range(1, runs + 1):
            # Run baseline
            baseline_result = run_method_test(model, X_test, sensitive_columns, non_sensitive_columns, 
                                             'baseline', run_idx, num_samples)
            baseline_results['idi_ratios'].append(baseline_result['idi_ratio'])
            baseline_results['times'].append(baseline_result['time'])
            baseline_results['avg_pred_diffs'].append(baseline_result['avg_diff'])
            
            # Run GA
            ga_result = run_method_test(model, X_test, sensitive_columns, non_sensitive_columns, 
                                       'ga', run_idx, num_samples)
            ga_results['idi_ratios'].append(ga_result['idi_ratio'])
            ga_results['times'].append(ga_result['time'])
            ga_results['avg_pred_diffs'].append(ga_result['avg_diff'])
    
    return baseline_results, ga_results


def perform_statistical_analysis(baseline_results, ga_results):
    # Extract IDI ratios
    baseline_idi = np.array(baseline_results['idi_ratios'])
    ga_idi = np.array(ga_results['idi_ratios'])
    
    # Calculate basic statistics
    baseline_mean = np.mean(baseline_idi)
    baseline_median = np.median(baseline_idi)
    baseline_std = np.std(baseline_idi)
    ga_mean = np.mean(ga_idi)
    ga_median = np.median(ga_idi)
    ga_std = np.std(ga_idi)
    
    # Calculate percentiles for both methods
    baseline_percentiles = np.percentile(baseline_idi, [25, 50, 75])
    ga_percentiles = np.percentile(ga_idi, [25, 50, 75])
    
    # Calculate percentage improvements for each percentile
    percentile_improvements = {}
    percentile_names = ['25th', '50th (median)', '75th']
    
    for i, (baseline_perc, ga_perc) in enumerate(zip(baseline_percentiles, ga_percentiles)):
        if baseline_perc > 0:
            improvement = (ga_perc - baseline_perc) / baseline_perc * 100
        else:
            improvement = 0 if ga_perc == 0 else float('inf')
        percentile_improvements[percentile_names[i]] = improvement
    
    # Calculate mean percentage improvement
    if baseline_mean > 0:
        mean_pct_improvement = (ga_mean - baseline_mean) / baseline_mean * 100
    else:
        mean_pct_improvement = 0 if ga_mean == 0 else float('inf')
    
    # Perform Wilcoxon signed-rank test
    if len(baseline_idi) > 1:
        try:
            w_stat, p_value = stats.wilcoxon(ga_idi, baseline_idi)
        except Exception as e:
            print(f"Wilcoxon test failed: {str(e)}. Using t-test instead.")
            t_stat, p_value = stats.ttest_rel(ga_idi, baseline_idi)
            w_stat = t_stat
    else:
        w_stat, p_value = 0, 1.0
    
    # Calculate effect size (Cohen's d)
    if baseline_std > 0 or ga_std > 0:
        effect_size = (ga_mean - baseline_mean) / np.sqrt((baseline_std**2 + ga_std**2) / 2)
    else:
        effect_size = 0
    
    # Timing statistics
    baseline_time_mean = np.mean(baseline_results['times'])
    ga_time_mean = np.mean(ga_results['times'])
    
    # Prediction difference statistics
    baseline_diff_mean = np.mean(baseline_results['avg_pred_diffs'])
    ga_diff_mean = np.mean(ga_results['avg_pred_diffs'])
    
    # Store results
    stats_results = {
        'baseline_idi_mean': baseline_mean,
        'baseline_idi_median': baseline_median,
        'baseline_idi_std': baseline_std,
        'baseline_percentiles': baseline_percentiles,
        'ga_idi_mean': ga_mean,
        'ga_idi_median': ga_median,
        'ga_idi_std': ga_std,
        'ga_percentiles': ga_percentiles,
        'mean_improvement': mean_pct_improvement,
        'percentile_improvements': percentile_improvements,
        'wilcoxon_statistic': w_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'significant': p_value < 0.05,
        'baseline_time_mean': baseline_time_mean,
        'ga_time_mean': ga_time_mean,
        'baseline_diff_mean': baseline_diff_mean,
        'ga_diff_mean': ga_diff_mean
    }
    
    return stats_results


def visualize_comparison(baseline_results, ga_results, stats_results, dataset_name):
    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Check if we have valid results to visualize
    if not baseline_results['idi_ratios'] or not ga_results['idi_ratios']:
        print(f"Warning: Not enough data to generate visualizations for {dataset_name}")
        return
    
    # Create a DataFrame for plotting IDI ratios
    idi_data = pd.DataFrame({
        'Method': ['Baseline']*len(baseline_results['idi_ratios']) + ['Genetic Algorithm']*len(ga_results['idi_ratios']),
        'IDI Ratio': baseline_results['idi_ratios'] + ga_results['idi_ratios']
    })
    
    # Create a DataFrame for plotting times
    time_data = pd.DataFrame({
        'Method': ['Baseline']*len(baseline_results['times']) + ['Genetic Algorithm']*len(ga_results['times']),
        'Time (s)': baseline_results['times'] + ga_results['times']
    })
    
    # Figure 1: IDI Ratio Comparison (Box Plot + Bar Chart)
    plt.figure(figsize=(15, 8))
    
    # Box Plot
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Method', y='IDI Ratio', data=idi_data)
    plt.title('IDI Ratio Comparison')
    
    # Add jitter points for individual runs
    if len(idi_data) > 1:
        sns.stripplot(x='Method', y='IDI Ratio', data=idi_data, color='black', alpha=0.5, jitter=True)
    
    # Bar Chart with Multiple Statistics
    plt.subplot(1, 2, 2)
    methods = ['Baseline', 'Genetic Algorithm']
    x = np.arange(len(methods))
    width = 0.35
    
    # Plot mean values
    means = [stats_results['baseline_idi_mean'], stats_results['ga_idi_mean']]
    stds = [stats_results['baseline_idi_std'], stats_results['ga_idi_std']]
    
    bars = plt.bar(x, means, width, yerr=stds, capsize=10, alpha=0.7, label='Mean')
    
    # Plot median values
    medians = [stats_results['baseline_idi_median'], stats_results['ga_idi_median']]
    plt.scatter(x, medians, color='red', s=50, zorder=3, label='Median')
    
    # Add percentage improvement annotation
    for i, method in enumerate(methods):
        if i == 1:  # Only for GA
            plt.annotate(f"Mean: +{stats_results['mean_improvement']:.1f}%", 
                        xy=(i, means[i]), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=10, color='blue')
            
            # Add median improvement
            median_improvement = stats_results['percentile_improvements']['50th (median)']
            plt.annotate(f"Median: +{median_improvement:.1f}%", 
                        xy=(i, medians[i]), 
                        xytext=(0, 25), 
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=10, color='red')
    
    plt.ylabel('IDI Ratio')
    plt.title('Mean and Median IDI Ratio Comparison')
    plt.xticks(x, methods)
    plt.legend()
    
    # Add statistical significance annotation based on Wilcoxon test
    if stats_results['significant']:
        significance = f"Significant improvement (Wilcoxon p={stats_results['p_value']:.4f})"
    else:
        significance = f"No significant difference (Wilcoxon p={stats_results['p_value']:.4f})"
    
    plt.suptitle(f'IDI Ratio Comparison - {dataset_name}\n{significance}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    plt.savefig(f'results/{dataset_name}_idi_comparison.png', dpi=300)
    
    # Figure 2: Time Comparison
    plt.figure(figsize=(15, 6))
    
    # Box Plot for Time
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Method', y='Time (s)', data=time_data)
    plt.title('Execution Time Comparison')
    
    # Add jitter points for individual runs
    if len(time_data) > 1:  # Only add stripplot if we have multiple runs
        sns.stripplot(x='Method', y='Time (s)', data=time_data, color='black', alpha=0.5, jitter=True)
    
    # Bar Chart for Time
    plt.subplot(1, 2, 2)
    time_means = [stats_results['baseline_time_mean'], stats_results['ga_time_mean']]
    plt.bar(methods, time_means, alpha=0.7)
    plt.ylabel('Mean Execution Time (s)')
    plt.title('Mean Execution Time')
    
    # Add time difference percentage annotation
    if stats_results['baseline_time_mean'] > 0:
        time_pct_diff = (stats_results['ga_time_mean'] - stats_results['baseline_time_mean']) / stats_results['baseline_time_mean'] * 100
        label = f"+{time_pct_diff:.1f}%" if time_pct_diff > 0 else f"{time_pct_diff:.1f}%"
        
        plt.annotate(label, 
                    xy=(1, time_means[1]), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Execution Time Comparison - {dataset_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    plt.savefig(f'results/{dataset_name}_time_comparison.png', dpi=300)
    
    # Figure 3: Percentile Comparison Plot
    if len(baseline_results['idi_ratios']) > 1:  # Only create this plot if we have multiple runs
        plt.figure(figsize=(15, 8))
        
        # Percentile Comparison Plot
        percentiles = ['25th', '50th', '75th']
        baseline_perc = stats_results['baseline_percentiles']
        ga_perc = stats_results['ga_percentiles']
        
        x = np.arange(len(percentiles))
        width = 0.35
        
        plt.bar(x - width/2, baseline_perc, width, label='Baseline', alpha=0.7)
        plt.bar(x + width/2, ga_perc, width, label='Genetic Algorithm', alpha=0.7)
        
        plt.xlabel('Percentile')
        plt.ylabel('IDI Ratio')
        plt.title(f'Percentile Comparison - {dataset_name}')
        plt.xticks(x, percentiles)
        plt.legend()
        
        # Add improvement percentages
        for i, perc in enumerate(percentiles):
            improvement = stats_results['percentile_improvements'][perc if perc != '50th' else '50th (median)']
            plt.annotate(f"+{improvement:.1f}%", 
                        xy=(i + width/2, ga_perc[i]), 
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=10)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f'results/{dataset_name}_percentile_comparison.png', dpi=300)


# 14. Enhanced Save Results Function with Percentile Information
def save_results(baseline_results, ga_results, stats_results, dataset_name):
    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw results to CSV
    results_df = pd.DataFrame({
        'Run': list(range(1, len(baseline_results['idi_ratios']) + 1)),
        'Baseline_IDI': baseline_results['idi_ratios'],
        'GA_IDI': ga_results['idi_ratios'],
        'Baseline_Time': baseline_results['times'],
        'GA_Time': ga_results['times'],
        'Baseline_AvgDiff': baseline_results['avg_pred_diffs'],
        'GA_AvgDiff': ga_results['avg_pred_diffs']
    })
    results_df.to_csv(f'results/{dataset_name}_results_{timestamp}.csv', index=False)
    
    # Save statistical summary with enhanced metrics
    with open(f'results/{dataset_name}_stats_{timestamp}.txt', 'w') as f:
        f.write(f"Fairness Testing Statistical Analysis - {dataset_name}\n")
        f.write("="*80 + "\n\n")
        
        f.write("IDI Ratio Results:\n")
        f.write(f"Baseline Method Mean IDI Ratio: {stats_results['baseline_idi_mean']:.4f} ± {stats_results['baseline_idi_std']:.4f}\n")
        f.write(f"Baseline Method Median IDI Ratio: {stats_results['baseline_idi_median']:.4f}\n")
        f.write(f"Genetic Algorithm Mean IDI Ratio: {stats_results['ga_idi_mean']:.4f} ± {stats_results['ga_idi_std']:.4f}\n")
        f.write(f"Genetic Algorithm Median IDI Ratio: {stats_results['ga_idi_median']:.4f}\n\n")
        
        f.write("Percentile Analysis:\n")
        f.write("                    Baseline    GA       Improvement\n")
        percentiles = ['25th', '50th (median)', '75th']
        for i, perc in enumerate(percentiles):
            baseline_perc = stats_results['baseline_percentiles'][i]
            ga_perc = stats_results['ga_percentiles'][i]
            improvement = stats_results['percentile_improvements'][perc]
            f.write(f"{perc:20} {baseline_perc:.4f}      {ga_perc:.4f}    +{improvement:.2f}%\n")
        f.write("\n")
        
        f.write("Overall Improvement:\n")
        f.write(f"Mean Improvement: {stats_results['mean_improvement']:.2f}%\n\n")
        
        f.write("Time Performance:\n")
        f.write(f"Baseline Method Mean Time: {stats_results['baseline_time_mean']:.2f} seconds\n")
        f.write(f"Genetic Algorithm Mean Time: {stats_results['ga_time_mean']:.2f} seconds\n\n")
        
        f.write("Prediction Differences:\n")
        f.write(f"Baseline Mean Prediction Difference: {stats_results['baseline_diff_mean']:.4f}\n")
        f.write(f"Genetic Algorithm Mean Prediction Difference: {stats_results['ga_diff_mean']:.4f}\n\n")
        
        f.write("Statistical Tests:\n")
        f.write(f"Wilcoxon signed-rank test: W={stats_results['wilcoxon_statistic']:.4f}, p={stats_results['p_value']:.4f}\n")
        f.write(f"Effect size (Cohen's d): {stats_results['effect_size']:.4f}\n\n")
        
        if stats_results['significant']:
            f.write("Conclusion: The Genetic Algorithm significantly outperforms the baseline Random Search method.\n")
        else:
            f.write("Conclusion: No statistically significant difference between the methods.\n")


def process_dataset(dataset_config, runs=5, num_samples=1000):
    dataset_name = dataset_config['name']
    file_path = dataset_config['file_path']
    model_path = dataset_config['model_path']
    target_column = dataset_config['target_column']
    sensitive_columns = dataset_config['sensitive_columns']
    
    print(f"\n{'='*80}\nTesting dataset: {dataset_name}\n{'='*80}")
    
    try:
        # Load dataset and model
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, target_column)
        
        print(f"Loading model from {model_path}...")
        model = load_dnn_model(model_path)
        
        # Define non-sensitive columns
        non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]
        
        print(f"Sensitive columns: {sensitive_columns}")
        print(f"Non-sensitive columns: {non_sensitive_columns}")
        
        baseline_results, ga_results = compare_methods(
            X_test, model, sensitive_columns, non_sensitive_columns, 
            runs=runs, num_samples=num_samples
        )
        
        # Perform enhanced statistical analysis
        print("\nPerforming statistical analysis...")
        stats_results = perform_statistical_analysis(baseline_results, ga_results)
        
        # Visualize results with enhanced plots
        print("Creating visualizations...")
        visualize_comparison(baseline_results, ga_results, stats_results, dataset_name)
        
        # Save detailed results
        print("Saving results...")
        save_results(baseline_results, ga_results, stats_results, dataset_name)
        
        print(f"\nAnalysis for {dataset_name} complete. Results saved to 'results/' directory.")
        
        return {
            'dataset': dataset_name,
            'baseline_idi_mean': stats_results['baseline_idi_mean'],
            'baseline_idi_median': stats_results['baseline_idi_median'],
            'ga_idi_mean': stats_results['ga_idi_mean'],
            'ga_idi_median': stats_results['ga_idi_median'],
            'mean_improvement': stats_results['mean_improvement'],
            'median_improvement': stats_results['percentile_improvements']['50th (median)'],
            'p_value': stats_results['p_value'],
            'significant': stats_results['significant']
        }
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# 16. Main function with parallel dataset processing
def main():
    """
    Main function to run the fairness testing and comparison.
    """
    # Silence TensorFlow warnings
    silence_tensorflow_warnings()
    
    # Dataset configurations
    datasets = [
        {
            'name': 'adult',
            'file_path': 'dataset/processed_adult.csv',
            'model_path': 'DNN/model_processed_adult.h5',
            'target_column': 'Class-label',
            'sensitive_columns': ['gender', 'race', 'age']
        },
        {
            'name': 'compas',
            'file_path': 'dataset/processed_compas.csv',
            'model_path': 'DNN/model_processed_compas.h5',
            'target_column': 'Recidivism',
            'sensitive_columns': ['Race', 'Sex']
        },
        {
            'name': 'communities_crime',
            'file_path': 'dataset/processed_communities_crime.csv',
            'model_path': 'DNN/model_processed_communities_crime.h5',
            'target_column': 'class',
            'sensitive_columns': ['Black', 'FemalePctDiv']
        },
        {
            'name': 'dutch',
            'file_path': 'dataset/processed_dutch.csv',
            'model_path': 'DNN/model_processed_dutch.h5',
            'target_column': 'occupation',
            'sensitive_columns': ['age', 'sex']
        },
        {
            'name': 'credit',
            'file_path': 'dataset/processed_credit_with_numerical.csv',
            'model_path': 'DNN/model_processed_credit.h5',
            'target_column': 'class',
            'sensitive_columns': ['SEX', 'EDUCATION', 'MARRIAGE']
        },
        {
            'name': 'german',
            'file_path': 'dataset/processed_german.csv',
            'model_path': 'DNN/model_processed_german.h5',
            'target_column': 'CREDITRATING',
            'sensitive_columns': ['AgeInYears', 'PersonStatusSex']
        },
        {
            'name': 'kdd',
            'file_path': 'dataset/processed_kdd.csv',
            'model_path': 'DNN/model_processed_kdd.h5',
            'target_column': 'income',
            'sensitive_columns': ['sex', 'race']
        },
        {
            'name': 'law_school',
            'file_path': 'dataset/processed_law_school.csv',
            'model_path': 'DNN/model_processed_law_school.h5',
            'target_column': 'pass_bar',
            'sensitive_columns': ['race', 'male']
        }
    ]
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Settings for testing
    runs = 30 
    num_samples = 1000  
    
    use_parallel = mp.cpu_count() > 1
    if use_parallel:
        print(f"System has {mp.cpu_count()} CPU cores. Using parallel processing for datasets.")
        # Process datasets in parallel
        n_jobs = min(len(datasets), mp.cpu_count() - 1) 
        dataset_results = Parallel(n_jobs=n_jobs)(
            delayed(process_dataset)(dataset, runs, num_samples) for dataset in datasets
        )
        all_results = [r for r in dataset_results if r is not None]
    else:
        print("Running in sequential mode.")
        all_results = []
        for dataset in datasets:
            result = process_dataset(dataset, runs, num_samples)
            if result:
                all_results.append(result)
    
    if len(all_results) > 1:
        plt.figure(figsize=(15, 10))
        
        # Comparison by Mean
        plt.subplot(2, 1, 1)
        dataset_names = [r['dataset'] for r in all_results]
        baseline_values = [r['baseline_idi_mean'] for r in all_results]
        ga_values = [r['ga_idi_mean'] for r in all_results]
        
        x = np.arange(len(dataset_names))
        width = 0.35
        
        plt.bar(x - width/2, baseline_values, width, label='Baseline')
        plt.bar(x + width/2, ga_values, width, label='Genetic Algorithm')
        
        plt.xlabel('Dataset')
        plt.ylabel('Mean IDI Ratio')
        plt.title('Mean IDI Ratio Comparison Across Datasets')
        plt.xticks(x, dataset_names, rotation=45, ha='right')
        plt.legend()
        
        # Add improvement percentages
        for i, r in enumerate(all_results):
            if r['significant']:
                plt.annotate(f"+{r['mean_improvement']:.1f}%*", 
                            xy=(i + width/2, ga_values[i]), 
                            xytext=(0, 5),
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontweight='bold')
            else:
                plt.annotate(f"+{r['mean_improvement']:.1f}%", 
                            xy=(i + width/2, ga_values[i]), 
                            xytext=(0, 5),
                            textcoords='offset points',
                            ha='center', va='bottom')
        
        # Comparison by Median
        plt.subplot(2, 1, 2)
        baseline_medians = [r['baseline_idi_median'] for r in all_results]
        ga_medians = [r['ga_idi_median'] for r in all_results]
        
        plt.bar(x - width/2, baseline_medians, width, label='Baseline')
        plt.bar(x + width/2, ga_medians, width, label='Genetic Algorithm')
        
        plt.xlabel('Dataset')
        plt.ylabel('Median IDI Ratio')
        plt.title('Median IDI Ratio Comparison Across Datasets')
        plt.xticks(x, dataset_names, rotation=45, ha='right')
        plt.legend()
        
        # Add improvement percentages
        for i, r in enumerate(all_results):
            if r['significant']:
                plt.annotate(f"+{r['median_improvement']:.1f}%*", 
                            xy=(i + width/2, ga_medians[i]), 
                            xytext=(0, 5),
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontweight='bold')
            else:
                plt.annotate(f"+{r['median_improvement']:.1f}%", 
                            xy=(i + width/2, ga_medians[i]), 
                            xytext=(0, 5),
                            textcoords='offset points',
                            ha='center', va='bottom')
        
        plt.suptitle('Comparison Across All Datasets', fontsize=16)
        plt.tight_layout()
        plt.savefig('results/overall_comparison.png', dpi=300)
    
    # Save overall results with both mean and median metrics
    if all_results:  # Only try to save if we have results
        overall_df = pd.DataFrame(all_results)
        overall_df.to_csv('results/overall_results.csv', index=False)
        
        # Generate summary report for all datasets
        with open(f'results/summary_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
            f.write("Overall Fairness Testing Results Summary\n")
            f.write("="*80 + "\n\n")
            
            f.write("Comparison across all datasets:\n\n")
            f.write("Dataset             | Baseline (Mean/Median) | GA (Mean/Median) | Improvement (Mean/Median) | Significant\n")
            f.write("-"*100 + "\n")
            
            for r in all_results:
                dataset = r['dataset']
                baseline = f"{r['baseline_idi_mean']:.4f}/{r['baseline_idi_median']:.4f}"
                ga = f"{r['ga_idi_mean']:.4f}/{r['ga_idi_median']:.4f}"
                improvement = f"+{r['mean_improvement']:.2f}%/+{r['median_improvement']:.2f}%"
                sig = "Yes*" if r['significant'] else "No"
                
                f.write(f"{dataset:20} | {baseline:22} | {ga:17} | {improvement:25} | {sig}\n")
            
            f.write("-"*100 + "\n\n")
            f.write("* Statistically significant improvement (p < 0.05) using Wilcoxon signed-rank test\n\n")
            
            # Calculate overall averages
            mean_improvements = [r['mean_improvement'] for r in all_results]
            median_improvements = [r['median_improvement'] for r in all_results]
            significant_count = sum(1 for r in all_results if r['significant'])
            
            # Added check to prevent division by zero
            if all_results:  # Only compute if we have results
                f.write(f"Average Mean Improvement: {np.mean(mean_improvements):.2f}%\n")
                f.write(f"Average Median Improvement: {np.mean(median_improvements):.2f}%\n")
                f.write(f"Number of datasets with significant improvement: {significant_count}/{len(all_results)} ({significant_count/len(all_results)*100:.1f}%)\n\n")
                
                f.write("Conclusion: The Genetic Algorithm approach with unique pair tracking ")
                if significant_count > 0:
                    f.write("outperforms the baseline Random Search method ")
                    f.write(f"with an average improvement of {np.mean(mean_improvements):.2f}% in mean IDI ratio and ")
                    f.write(f"{np.mean(median_improvements):.2f}% in median IDI ratio. ")
                else:
                    f.write("shows some improvement over the baseline Random Search method, but ")
                    f.write("the differences were not statistically significant. ")
                f.write(f"Statistical significance was observed in {significant_count} out of {len(all_results)} datasets.\n")
            else:
                f.write("No valid results were obtained. Please check the error messages above.\n")
    else:
        print("\nNo valid results were obtained. Please check the error messages above.")
    
    print("\nAll analyses complete!")
    if all_results:
        print("Results saved to 'results/' directory.")
    else:
        print("No results were saved due to errors in processing.")


if __name__ == "__main__":
    main()