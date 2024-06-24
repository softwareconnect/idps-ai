# Train and Test Data
This repository contains various data files which are used for training and evaluating intrusion detection systems.
This README file provides an overview of the available data files in this repository and their respective formats, along with basic usage instructions.

## 1. NSL-KDD Dataset
### 1.1. Data Files

- **KDDTrain+.ARFF**: The full NSL-KDD train set with binary labels in ARFF format
- **KDDTrain+.TXT**: The full NSL-KDD train set including attack-type labels and difficulty level in CSV format
- **KDDTrain+_20Percent.ARFF**: A 20% subset of the KDDTrain+.arff file
- **KDDTrain+_20Percent.TXT**: A 20% subset of the KDDTrain+.txt file
- **KDDTest+.ARFF**: The full NSL-KDD test set with binary labels in ARFF format
- **KDDTest+.TXT**: The full NSL-KDD test set including attack-type labels and difficulty level in CSV format
- **KDDTest-21.ARFF**: A subset of the KDDTest+.arff file which does not include records with difficulty level of 21 out of 21
- **KDDTest-21.TXT**: A subset of the KDDTest+.txt file which does not include records with difficulty level of 21 out of 21

### 1.2. Usage

To use these data files, download the appropriate format based on your requirements. ARFF files can be used with tools like Weka, while TXT files (in CSV format) can be used with various data processing tools and programming languages like Python, R, etc.

### 1.3. License

This dataset is provided under the terms specified in the dataset source.

### 1.4. References

For more information about the NSL-KDD dataset, refer to the original paper or the [official website](https://www.unb.ca/cic/datasets/nsl.html).