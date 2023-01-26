# Participatory Systems for Personalized Prediction


## Quickstart

### Setup

Install required packages `pip install -r requirements.txt`

### Train a Participatory System

#### Load a Dataset

```
data_file_processed = get_processed_data_file(settings['data_name'], rebalancing_type=settings['rebalancing_type'])
data = BinaryClassificationDataset.load(file=data_file_processed)
```

#### Train a Set of Classifiers

```
generic_model = train_model(X=data.training.X, G=None, y=data.training.y, settings=settings,
                                normalize_variables=False)
                                
candidate_models = []
    for encoding_type in ['onehot', 'intersectional']:
        for name_subset in powerset(data.group_attributes.names, min_size=1):
            curr_settings['encoding_type'] = encoding_type
            G_name_subset = G_train[list(name_subset)]
            curr_settings['training_groups'] = data.group_encoder.groups
            h = train_sklearn_linear_model(X_train, G_name_subset, y_train,
                                           method_name=curr_settings['method_name'],
                                           settings=curr_settings, normalize_variables=False)

            candidate_models.append(h)
```

#### Fit a Participatory System

```
p_seq = SequentialSystem(data, generic_model, assignment_metric=settings['assignment_metric'],
                         assignment_sample=settings['assignment_sample'])
p_seq.update_assignments(candidate_models)
```

### Reproduce Results

#### Dataset Processing
We provide raw data and processing code for:

- `apnea`


We provide processing code only for:

- `cshock_eicu`
- `cshock_eicu`
- `cshockR_mimic`
- `cshockR_mimic`
- `lungcancer`
- `saps`
- `support`

To process datasets, use the `create_dataset.py` script with dataset names separated with spaces.

`python3 create_datasets.py --data-names [DATA_NAME]`

where `DATA_NAME` is e.g., `apnea`

### Experimental Results

1. Train and benchmark models with a variety of parameters and benchmark criteria.

` python3 train_and_benchmark.py --data-name apnea --models sequential flat participatory_simple --table-type performance --assignment-metric auc`

2. Aggregate results across datasets into CSVs using `aggregate_results.py`. This will produce file similar to `aggregated_results.zip`

### Miscellaneous

#### Code Structure

```
├── data         # datasets and processing code       
├── psc          # source code                    
├── scripts      # scripts to run source code                                                       
└── results      # results files                                                    
```

### Debugging

Error: `ModuleNotFoundError: No module named 'psc'
Try running `export PYTHONPATH="${PYTHONPATH}:<path to repo>/psc/"`

