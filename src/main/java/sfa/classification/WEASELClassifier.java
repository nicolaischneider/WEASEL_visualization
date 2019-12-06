// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import com.carrotsearch.hppc.cursors.LongIntCursor;
import de.bwaldvogel.liblinear.*;
import sfa.timeseries.TimeSeries;
import sfa.transformation.WEASEL;
import sfa.transformation.WEASEL.BagOfBigrams;
import sfa.transformation.WEASEL.Dictionary;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.atomic.AtomicInteger;
import java.io.*;

/**
 * The WEASEL (Word ExtrAction for time SEries cLassification) classifier as published in
 * <p>
 * Schäfer, P., Leser, U.: Fast and Accurate Time Series
 * Classification with WEASEL." CIKM 2017
 */
public class WEASELClassifier extends Classifier {

    // default training parameters
    public static int maxF = 6; // 6
    public static int minF = 4; // 4
    public static int maxS = 4; // 4

    public static SolverType solverType = SolverType.L2R_LR_DUAL;

    public static double p_value = 0.1;
    public static double bias = 1;
    public static double p = 0.1;
    public static int iterations = 5000;
    public static double c = 1;

    public static boolean lowerBounding = false;

    public static int MIN_WINDOW_LENGTH = 2;
    public static int MAX_WINDOW_LENGTH = 350;

    private String nameOfDataset;

    // the trained weasel
    WEASELModel model;

    public WEASELClassifier(String name) {
        super();
        nameOfDataset = name;
        System.out.println(nameOfDataset);
        Linear.resetRandom();
        Linear.disableDebugOutput();
    }

    public static class WEASELModel extends Model {

        public WEASELModel(){}

        public WEASELModel(
                boolean normed,
                int features,
                WEASEL model,
                de.bwaldvogel.liblinear.Model linearModel,
                int testing,
                int testSize,
                int training,
                int trainSize
        ) {
            super("WEASEL", testing, testSize, training, trainSize, normed, -1);
            this.features = features;
            this.weasel = model;
            this.linearModel = linearModel;
        }

        // the best number of Fourier values to be used
        public int features;

        // the trained WEASEL transformation
        public WEASEL weasel;

        // the trained liblinear classifier
        public de.bwaldvogel.liblinear.Model linearModel;
    }

    @Override
    public Score eval(
            final TimeSeries[] trainSamples, final TimeSeries[] testSamples) {
        long startTime = System.currentTimeMillis();

        Score score = fit(trainSamples);

        // training score
        if (DEBUG) {
            System.out.println(score.toString());
            outputResult(score.training, startTime, trainSamples.length);
        }

        // determine score
        int correctTesting = score(testSamples).correct.get();

        if (DEBUG) {
            System.out.println("WEASEL Testing:\t");
            outputResult(correctTesting, startTime, testSamples.length);
            System.out.println("");
        }

        return new Score(
                "WEASEL",
                correctTesting, testSamples.length,
                score.training, trainSamples.length,
                score.windowLength
        );
    }


    @Override
    public Score fit(final TimeSeries[] trainSamples) {
        // train the shotgun models for different window lengths
        this.model = fitWeasel(trainSamples);

        // return score
        return model.score;
    }


    @Override
    public Predictions score(final TimeSeries[] testSamples) {
        Double[] labels = predict(testSamples);
        return evalLabels(testSamples, labels);
    }

    public Double[] predict(TimeSeries[] samples) {

        // iterate each sample to classify
        final BagOfBigrams[] bagTest = new BagOfBigrams[samples.length];
        ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
            @Override
            public void run(int id, AtomicInteger processed) {
                for (int w = 0; w < model.weasel.windowLengths.length; w++) {
                    if (w % BLOCKS == id) {
                        int[][] wordsTest = model.weasel.createWords(samples, w);
                        BagOfBigrams[] bopForWindow = model.weasel.createBagOfPatterns(wordsTest, samples, w, model.features);
                        model.weasel.dict.filterChiSquared(bopForWindow);
                        mergeBobs(bagTest, bopForWindow);
                    }
                }
            }
        });

        FeatureNode[][] features = initLibLinear(bagTest, model.weasel.dict);
        Double[] labels = new Double[samples.length];

        ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
            @Override
            public void run(int id, AtomicInteger processed) {
                for (int ind = 0; ind < features.length; ind++) {
                    if (ind % BLOCKS == id) {
                        double label = Linear.predict(model.linearModel, features[ind]);
                        labels[ind] = label;
                    }
                }
            }
        });

        return labels;
    }

    public Predictions predictProbabilities(TimeSeries[] samples) {
        final Double[] labels = new Double[samples.length];
        final double[][] probabilities = new double[samples.length][];

        // iterate each sample to classify
        final BagOfBigrams[] bagTest = new BagOfBigrams[samples.length];
        ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
            @Override
            public void run(int id, AtomicInteger processed) {
                for (int w = 0; w < model.weasel.windowLengths.length; w++) {
                    if (w % BLOCKS == id) {
                        int[][] wordsTest = model.weasel.createWords(samples, w);
                        BagOfBigrams[] bopForWindow = model.weasel.createBagOfPatterns(wordsTest, samples, w, model.features);
                        model.weasel.dict.filterChiSquared(bopForWindow);
                        mergeBobs(bagTest, bopForWindow);
                    }
                }
            }
        });

        FeatureNode[][] features = initLibLinear(bagTest, model.weasel.dict);

        ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
            @Override
            public void run(int id, AtomicInteger processed) {
                for (int ind = 0; ind < features.length; ind++) {
                    if (ind % BLOCKS == id) {
                        probabilities[ind] = new double[model.linearModel.getNrClass()];
                        labels[ind] = Linear.predictProbability(model.linearModel, features[ind], probabilities[ind]);
                    }
                }
            }
        });

        return new Predictions(labels, probabilities, model.linearModel.getLabels());
    }

    public int[] getWindowLengths(final TimeSeries[] samples, boolean norm) {
        int min = norm && MIN_WINDOW_LENGTH<=2? Math.max(3,MIN_WINDOW_LENGTH) : MIN_WINDOW_LENGTH;
        int max = getMax(samples, MAX_WINDOW_LENGTH);

        int[] wLengths = new int[max - min + 1];
        int a = 0;
        for (int w = min; w <= max; w+=1, a++) {
            wLengths[a] = w;
        }
        return Arrays.copyOfRange(wLengths, 0, a);
    }

    protected WEASELModel fitWeasel(final TimeSeries[] samples) {
        try {
            int maxCorrect = -1;
            int bestF = -1;
            boolean bestNorm = false;

            optimize:
            for (final boolean mean : NORMALIZATION) {
                int[] windowLengths = getWindowLengths(samples, mean);
                WEASEL model = new WEASEL(maxF, maxS, windowLengths, mean, lowerBounding);
                final int[][][] words = model.createWords(samples);

                for (int f = minF; f <= maxF; f += 2) {
                    model.dict.reset();

                    final BagOfBigrams[] bop = new BagOfBigrams[samples.length];
                    final int ff = f;

                    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
                        @Override
                        public void run(int id, AtomicInteger processed) {
                            for (int w = 0; w < model.windowLengths.length; w++) {
                                if (w % BLOCKS == id) {
                                    BagOfBigrams[] bobForOneWindow = fitOneWindow(
                                            samples,
                                            model.windowLengths, mean,
                                            words[w], ff, w);
                                    mergeBobs(bop, bobForOneWindow);
                                }
                            }
                        }
                    });

                    // train liblinear
                    final Problem problem = initLibLinearProblem(bop, model.dict, bias);
                    System.out.println("Train Dict Size: " + model.dict.size() + " Memory: " + getUsedMemory() + " MB");
                    int correct = trainLibLinear(problem, solverType, c, iterations, p, folds);

                    if (correct > maxCorrect) {
                        maxCorrect = correct;
                        bestF = f;
                        bestNorm = mean;
                    }
                    if (correct == samples.length) {
                        break optimize;
                    }
                }
            }

            // obtain the final matrix
            int[] windowLengths = getWindowLengths(samples, bestNorm);
            WEASEL model = new WEASEL(maxF, maxS, windowLengths, bestNorm, lowerBounding);

            final BagOfBigrams[] bob = new BagOfBigrams[samples.length];
            final boolean mean = bestNorm;
            final int ff = bestF;
            final int[/* Fensterlänge */][/* Sample */][/* Offset*/] words = model.createWords(samples);

            ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
                @Override
                public void run(int id, AtomicInteger processed) {
                    for (int w = 0; w < model.windowLengths.length; w++) {
                        if (w % BLOCKS == id) {
                            BagOfBigrams[] bobForOneWindow = fitOneWindow(
                                    samples,
                                    model.windowLengths, mean,
                                    words[w], ff, w);
                            mergeBobs(bob, bobForOneWindow);
                        }
                    }
                }
            });


            // For Histogram Representation
            // generate the actual words from the int representation
            short[/* Fensterlänge */][/* Sample */][/* Offset*/][/* Wortlänge */] wordsSymbols = generateWordFromInt(words);

            // train liblinear
            Problem problem = initLibLinearProblem(bob, model.dict, bias);
            System.out.println("Final Dict Size: " + model.dict.size() + " Memory: " + getUsedMemory() + " MB");
            de.bwaldvogel.liblinear.Model linearModel = Linear.train(problem, new Parameter(solverType, c, iterations, p));

            // Feature[/* zeitreihe */][/* anzahl features */] features = problem.x;
            double[] weights = linearModel.getFeatureWeights();
            //System.out.println(Arrays.toString(weights));

            // CSV file
            String numOfSamples = Integer.toString(samples.length);
            int numOfWindowSizes = wordsSymbols.length;
            int wordLength = wordsSymbols[0][0][0].length;
            int numOfValuesWeights = 0;
            String[] labelOfSample = new String[samples.length];
            StringBuilder [] valuesOfSamples = new StringBuilder[samples.length];
            StringBuilder [] weightsOfSamples = new StringBuilder[samples.length];
            StringBuilder [][] wordList = new StringBuilder[samples.length][numOfWindowSizes];

            int numOfZero = 0;

            // iterate through all samples
            for (int sampleId = 0; sampleId < samples.length; sampleId++) { // Samples
                TimeSeries ts = samples[sampleId];

                // used datasets/variables
                double label = ts.getLabel();
                double[] plot = new double[ts.getLength()]; // weighs
                int[] count = new int[ts.getLength()]; // occurrence
                double[] tsValues = ts.getData(); // ALLE VALUES

                // initialize
                valuesOfSamples[sampleId] = new StringBuilder();
                weightsOfSamples[sampleId] = new StringBuilder();
                labelOfSample[sampleId] = Double.toString(label);

                for (int windowId = 0; windowId < words.length; windowId++) { // Windowlengths
                    int[] wordsForOneSample = words[windowId][sampleId];

                    BagOfBigrams sampleBob = bob[sampleId];

                    wordList[sampleId][windowId] = new StringBuilder();
                    wordList[sampleId][windowId].append(wordsForOneSample.length + ",");

                    for (int pos = 0; pos < wordsForOneSample.length; pos++) {  // Offsets
                        int word = wordsForOneSample[pos];

                        int index = 0;
                        if ((index = model.dict.getWordIndex(word, false)) > -1) { // words that were filtered are "-1"
                            double liblinearWeight = weights[index]; // LibLinear Gewicht

                            // get weight
                            int wordFrequency = sampleBob.bob.get(word);
                            double featureImportance = liblinearWeight;// * wordFrequency;

                            if (featureImportance == 0) {
                                System.out.println(liblinearWeight + " " + wordFrequency);
                                numOfZero++;
                            }

                            // add to counter for len of window
                            int len = windowLengths[windowId];
                            for (int posWinodow = pos; posWinodow < pos+len-1; posWinodow++) {
                                count[posWinodow]++;
                                plot[posWinodow] += featureImportance;
                            }
                        }

                        // add word
                        for (int wordPos = 0; wordPos < wordLength; wordPos++) {
                            wordList[sampleId][windowId].append(wordsSymbols[windowId][sampleId][pos][wordPos] + ",");
                        }

                        //count[pos]++;
                    }
                    wordList[sampleId][windowId].deleteCharAt(wordList[sampleId][windowId].length()-1);
                    wordList[sampleId][windowId].append("\n");
                }

                // add values and weights
                for (int i = 0; i < plot.length; i++) {

                    if (count[i] == 0) {
                        plot[i] = 0.0;
                        valuesOfSamples[sampleId].append(tsValues[i] + ",");
                        weightsOfSamples[sampleId].append(plot[i] + ",");
                        continue;
                    }
                    //plot[i] = (plot[i] / Double.valueOf(count[i])) * 100;
                    //plot[i]*=100;

                    // add to list
                    valuesOfSamples[sampleId].append(tsValues[i] + ",");
                    weightsOfSamples[sampleId].append(plot[i] + ",");
                }

                numOfValuesWeights = plot.length;

                valuesOfSamples[sampleId].deleteCharAt(valuesOfSamples[sampleId].length()-1);
                weightsOfSamples[sampleId].deleteCharAt(weightsOfSamples[sampleId].length()-1);

                valuesOfSamples[sampleId].append("\n");
                weightsOfSamples[sampleId].append("\n");
            }

            // Erstellen der CSV-Datei
            String fileName = nameOfDataset + "_WEASEL.csv";
            try (PrintWriter writer = new PrintWriter(new File(fileName))) {
                writer.write(nameOfDataset + "\n");
                writer.write(numOfSamples + "\n");                                              // LINE 1
                writer.write(Integer.toString(numOfValuesWeights) + "\n");                      // LINE 2
                writer.write(Integer.toString(numOfWindowSizes) + "\n");                        // LINE 3
                writer.write(Integer.toString(wordLength) + "\n");                              // LINE 4

                for (int sample = 0; sample < samples.length; sample++) {
                    writer.write(valuesOfSamples[sample].toString());                               // LINE 1 - SAMPLE
                    writer.write(weightsOfSamples[sample].toString());                              // LINE 2 - SAMPLE
                    writer.write(labelOfSample[sample] + "\n");                                  // LINE 3 - SAMPLE
                    for (int windowLength = 0; windowLength < numOfWindowSizes; windowLength++) {
                        writer.write(wordList[sample][windowLength].toString());                    // LINE X - WORDS OF WINDOW SIZE
                    }
                }

                System.out.println("created csv file named: "+fileName);
            } catch (FileNotFoundException e) {
                System.out.println(e.getMessage());
            }

            // short[] sfaWord = Words.toShortArray(word, maxF, (byte) Words.binlog(maxS));

            return new WEASELModel(
                    bestNorm,
                    bestF,
                    model,
                    linearModel,
                    0, // testing
                    1,
                    maxCorrect, // training
                    samples.length
            );

        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private short[][][][] generateWordFromInt(int[][][] words) {
        short[/* Fensterlänge */][/* Sample */][/* Offset*/][/* Wortlänge */] wordsSymbols = new short[words.length][][][];
        for (int i = 0; i < words.length; i++) {
            wordsSymbols[i] = new short[words[i].length][][];
            for (int j = 0; j < words[i].length; j++) {
                wordsSymbols[i][j] = new short[words[i][j].length][];
                for (int k = 0; k < words[i][j].length; k++) {
                    wordsSymbols[i][j][k] = Words.toShortArray(words[i][j][k], maxF, (byte) Words.binlog(maxS));
                }
            }
        }
        return wordsSymbols;
    }

    private BagOfBigrams[] fitOneWindow(
            TimeSeries[] samples,
            int[] windowLengths,
            boolean mean,
            int[][] word, int f,
            int w) {
        WEASEL modelForWindow = new WEASEL(f, maxS, windowLengths, mean, lowerBounding);
        BagOfBigrams[] bopForWindow = modelForWindow.createBagOfPatterns(word, samples, w, f);
        modelForWindow.trainChiSquared(bopForWindow, p_value);
        //modelForWindow.trainAnova(bopForWindow, p_value);
        return bopForWindow;
    }

    private synchronized void mergeBobs(
            BagOfBigrams[] bop,
            BagOfBigrams[] bopForWindow) {
        for (int i = 0; i < bop.length; i++) {
            if (bop[i]==null) {
                bop[i] = bopForWindow[i];
            }
            else {
                bop[i].bob.putAll(bopForWindow[i].bob);
            }
        }
    }

    protected static Problem initLibLinearProblem(
            final BagOfBigrams[] bob,
            final Dictionary dict,
            final double bias) {
        Linear.resetRandom();
        Linear.disableDebugOutput();

        Problem problem = new Problem();
        problem.bias = bias;
        problem.y = getLabels(bob);

        final FeatureNode[][] features = initLibLinear(bob, dict);

        problem.n = dict.size() + 1;
        problem.l = features.length;
        problem.x = features;
        return problem;
    }

    protected static FeatureNode[][] initLibLinear(
            final BagOfBigrams[] bob,
            final Dictionary dict) {

        FeatureNode[][] featuresTrain = new FeatureNode[bob.length][];
        for (int j = 0; j < bob.length; j++) {
            BagOfBigrams bop = bob[j];
            ArrayList<FeatureNode> features = new ArrayList<>(bop.bob.size());
            for (LongIntCursor word : bop.bob) {
                if (word.value > 0) {
                    features.add(new FeatureNode(dict.getWordIndex(word.key, true), word.value));
                }
            }
            FeatureNode[] featuresArray = features.toArray(new FeatureNode[]{});
            Arrays.parallelSort(featuresArray, new Comparator<FeatureNode>() {
                public int compare(FeatureNode o1, FeatureNode o2) {
                    return Integer.compare(o1.index, o2.index);
                }
            });
            featuresTrain[j] = featuresArray;
        }
        return featuresTrain;
    }

    protected static double[] getLabels(final BagOfBigrams[] bagOfPatternsTestSamples) {
        double[] labels = new double[bagOfPatternsTestSamples.length];
        for (int i = 0; i < bagOfPatternsTestSamples.length; i++) {
            labels[i] = Double.valueOf(bagOfPatternsTestSamples[i].label);
        }
        return labels;
    }

    public WEASELModel getModel() {
        return model;
    }

    public void setModel(WEASELModel model) {
        this.model = model;
    }
}
