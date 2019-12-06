# WEASEL Visualization

This is simply a small modification of the original [Project](https://github.com/patrickzib/SFA) used togenerate a CSV file that can then be processed by the visulalization tool [TECI](https://github.com/nicolaischneider/TSC-Visualization). Further Information about WEASEL can be found [here](https://github.com/patrickzib/SFA).

## Installation
Please refer to the original [repository](https://github.com/patrickzib/SFA).

## Usage
All datasets can be found under
>src
>   main
>       resources
>           datasets
>               univariate

New datasets need to be copied as a folder containing the *TRAIN* and *TEST* file into the *univariate* folder.

To generate a CSV file run `UCRClassificationTest.java`. The selected datasets can be established in `datasets`:
```
// The datasets to use
public static String[] datasets = new String[]{
    // ENTER THE NAME OF THE DATASET BELOW
    "CBF"
};
```
All generated CSV files can be found inside the WEASEL folder (root folder).
