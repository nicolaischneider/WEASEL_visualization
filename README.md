# WEASEL Visualization

This is a small modification of the original [Project](https://github.com/patrickzib/SFA) used to generate a CSV file. That file can then be processed by the visulalization tool [TECI](https://github.com/nicolaischneider/TSC-Visualization). Further information about WEASEL can be found [here](https://github.com/patrickzib/SFA).

## Installation
Please refer to the original [repository](https://github.com/patrickzib/SFA) for further information.

## Usage
All datasets can be found under
```
> src
    > main
        > resources
            > datasets
                > univariate
````
New datasets need to be pasted as a folder containing the *TRAIN* and *TEST* file into the *univariate* folder. E.g.
```
> ...
    > univariate
        > GunPoint
            > GunPoint_TEST
            > GunPoint_TRAIN
````

To generate a CSV file run `UCRClassificationTest.java`. The wished datasets can be established in `datasets`:
```Java
// The datasets to use:
public static String[] datasets = new String[]{
    // ENTER THE NAME OF THE DATASET BELOW
    "GunPoint","CBF","Beef"
};
```
In the example above WEASEL generates three CSV files for three different datasets (GunPoint, CBF and Beef). It is **crucial** that the entered names match the names of the dataset folders and the corresponding files. All generated CSV files can be found inside the WEASEL folder (root folder).

## Datasets
Datasets can be downloaded from [here](http://www.cs.ucr.edu/~eamonn/time_series_data/UCR_TS_Archive_2015.zip).
