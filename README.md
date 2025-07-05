# AI Model Fairness Testing Tool

A comprehensive Python tool for detecting individual discrimination in deep neural networks using genetic algorithm optimization. This tool compares the effectiveness of genetic algorithms versus random search methods for identifying discriminatory model behavior.

## Overview

This tool implements and compares two approaches for detecting **Individual Discrimination** in AI models:

1. **Baseline Random Search**: Traditional random sampling approach
2. **Genetic Algorithm**: Evolutionary optimization to systematically find discriminatory instances

The tool calculates the **Individual Discrimination Index (IDI) ratio** - a metric that measures how frequently a model makes different predictions for similar individuals who differ primarily in sensitive attributes (e.g., race, gender, age).

## Research Context

Individual discrimination occurs when an AI model treats similar individuals differently based solely on their membership in a protected group. This tool helps researchers and practitioners:

- **Quantify discrimination** in trained neural networks
- **Compare detection methods** for finding biased predictions
- **Generate comprehensive reports** with statistical analysis and visualizations
- **Evaluate fairness** across multiple datasets and sensitive attributes

## Installation

### Prerequisites

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn scipy joblib
```

### System Requirements

- Python 3.7+
- TensorFlow 2.x
- Minimum 4GB RAM (8GB+ recommended for parallel processing)
- Multi-core CPU recommended for optimal performance

## Quick Start

### Basic Usage

```bash
python Fairness_testing_tool.py
```

This will automatically:
1. Process all 8 included datasets
2. Run both baseline and genetic algorithm approaches
3. Generate statistical comparisons
4. Create visualizations
5. Save results to the `results/` directory

### Configuration

The main configuration is in the `main()` function:

```python
runs = 30        # Number of independent runs per method
num_samples = 1000   # Number of sample pairs to evaluate per run
```

## Supported Datasets

The tool includes 8 preprocessed datasets with known fairness challenges:

| Dataset | Domain | Sensitive Attributes | Prediction Task |
|---------|--------|---------------------|-----------------|
| **Adult** | Census Income | Gender, Race, Age | Income Prediction |
| **COMPAS** | Criminal Justice | Race, Sex | Recidivism Risk |
| **Communities & Crime** | Crime Statistics | Race, Gender Diversity | Crime Rate |
| **Dutch Census** | Demographics | Age, Sex | Occupation Prediction |
| **German Credit** | Financial | Age, Personal Status | Credit Rating |
| **Default Credit** | Financial | Sex, Education, Marriage | Default Risk |
| **KDD Census** | Demographics | Sex, Race | Income Prediction |
| **Law School** | Education | Race, Gender | Bar Exam Pass Rate |

## Methodology

### Individual Discrimination Detection

The tool generates pairs of similar individuals that differ primarily in sensitive attributes:

1. **Sample Generation**: Creates pairs (A, B) where:
   - A and B are similar in non-sensitive features
   - A and B differ in sensitive attributes (e.g., race, gender)

2. **Discrimination Detection**: Measures prediction difference:
   - If |Prediction(A) - Prediction(B)| > threshold → Discriminatory
   - IDI Ratio = (Discriminatory Pairs) / (Total Pairs Evaluated)

### Genetic Algorithm Approach

The genetic algorithm optimizes sample pair generation through:

- **Population**: Set of sample pairs
- **Fitness Function**: Rewards larger prediction differences
- **Selection**: Tournament selection favoring high-fitness pairs
- **Crossover**: Combines features from parent pairs
- **Mutation**: Small perturbations to non-sensitive features
- **Elitism**: Preserves best-performing pairs across generations

### Statistical Analysis

For each dataset and method, the tool provides:

- **Descriptive Statistics**: Mean, median, standard deviation, percentiles
- **Significance Testing**: Wilcoxon signed-rank test
- **Effect Size**: Cohen's d
- **Performance Metrics**: Execution time, convergence analysis

## Results Interpretation

### IDI Ratio

- **Higher IDI Ratio** = More discrimination detected
- **Threshold**: Typically 0.05 (5% prediction difference)
- **Baseline vs. GA**: Genetic algorithm typically finds more discriminatory instances

### Statistical Significance

- **p < 0.05**: Statistically significant improvement
- **Effect Size**: Magnitude of improvement (Cohen's d)
- **Percentile Analysis**: Improvement across different performance levels

### Visualizations

The tool generates multiple plots:

1. **IDI Comparison**: Box plots and bar charts comparing methods
2. **Time Analysis**: Execution time comparisons
3. **Percentile Comparison**: Performance across quartiles
4. **Overall Summary**: Cross-dataset comparison

## Advanced Configuration

### Custom Datasets

To add your own dataset:

```python
{
    'name': 'your_dataset',
    'file_path': 'dataset/your_data.csv',
    'model_path': 'DNN/your_model.h5',
    'target_column': 'prediction_target',
    'sensitive_columns': ['sensitive_attr1', 'sensitive_attr2']
}
```

### Genetic Algorithm Parameters

Modify in `genetic_algorithm_fairness_test()`:

```python
population_size = 50      # Population size
generations = 20          # Number of generations
mutation_rate = 0.2       # Mutation probability
tournament_size = 3       # Tournament selection size
threshold = 0.05          # Discrimination threshold
```

### Performance Optimization

- **Parallel Processing**: Automatically uses multiple CPU cores
- **Prediction Caching**: Reduces redundant model predictions
- **Batch Processing**: Efficient batch predictions
- **Memory Management**: Configurable cache size

## Troubleshooting

### Common Issues

**TensorFlow Warnings**: The tool automatically suppresses TensorFlow logging. If you encounter issues:
```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

**Memory Issues**: Reduce batch size or population size:
```python
batch_size = 16          # Default: 32
population_size = 25     # Default: 50
```

**No Results Generated**: Check that:
- Dataset files exist in `dataset/` directory
- Model files exist in `DNN/` directory
- File paths match the configuration
- Models are compatible with your TensorFlow version

## Additional Resources

- [Fairness in Machine Learning](https://fairmlbook.org/)
- [AI Fairness 360 Toolkit](https://aif360.mybluemix.net/)
- [Google's ML Fairness Gym](https://github.com/google/ml-fairness-gym)
- [Microsoft's Fairlearn](https://fairlearn.org/)
