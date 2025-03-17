package com.example.namecheck;

import org.springframework.stereotype.Service;
import smile.regression.LinearModel;
import smile.regression.OLS;
import smile.nlp.tokenizer.SimpleTokenizer;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.vector.DoubleVector;
import smile.data.vector.IntVector;
import smile.data.formula.Formula;
import java.util.*;
import java.util.stream.Collectors;
import org.apache.commons.math3.linear.*;

@Service

public class NameSimilarityService {
    private LinearModel model;
    private Map<String, Integer> vocabulary;
    private Map<String, Double> idfScores;
    public NameSimilarityService() {
        this.vocabulary = new HashMap<>();
        this.idfScores = new HashMap<>();
    }

    /**
     * Train the model using a dataset of name pairs and similarity scores (0-100%).
     */
    public String trainModel(List<String[]> namePairs, List<Double> similarityScores) {
        buildVocabulary(namePairs);
        List<double[]> featureList = new ArrayList<>();

        for (String[] pair : namePairs) {
            featureList.add(computeTFIDF(pair[0], pair[1]));
        }

        double[][] featureMatrix = featureList.toArray(new double[0][]);
        double[] labelArray = similarityScores.stream().mapToDouble(i -> i).toArray();

        // Convert featureMatrix and labelArray to DataFrame
        DataFrame df = DataFrame.of(
                DoubleVector.of("label", labelArray),
                DoubleVector.of("features", Arrays.stream(featureMatrix).mapToDouble(f -> f[1]).toArray())

        );

        // Fit the model using Formula
        model = OLS.fit(Formula.lhs("label"), df);
        return "Training complete!";
    }

    /**
     * Predict similarity between two names as a percentage (0-100%).
     */
    public double predictSimilarity(String name1, String name2) {
        if (model == null) {
            throw new IllegalStateException("Model is not trained.");
        }
        double[] features = computeTFIDF(name1, name2);
        double prediction = model.predict(features);
        return Math.max(0, Math.min(100, prediction)); // Ensure range 0-100%
    }

    private void buildVocabulary(List<String[]> namePairs) {
        Set<String> allWords = new HashSet<>();
        Map<String, Integer> documentFrequency = new HashMap<>();

        for (String[] pair : namePairs) {
            Set<String> uniqueWords = new HashSet<>();
            for (String name : pair) {
                String[] words = tokenize(name);
                allWords.addAll(Arrays.asList(words));
                uniqueWords.addAll(Arrays.asList(words));
            }
            for (String word : uniqueWords) {
                documentFrequency.put(word, documentFrequency.getOrDefault(word, 0) + 1);
            }
        }

        int totalDocuments = namePairs.size();
        vocabulary = allWords.stream()
                .collect(Collectors.toMap(word -> word, word -> vocabulary.size()));

        for (String word : vocabulary.keySet()) {
            int df = documentFrequency.getOrDefault(word, 1);
            idfScores.put(word, Math.log((double) totalDocuments / df));
        }
    }

    private double[] computeTFIDF(String name1, String name2) {
        double[] tfidf1 = computeTFIDFVector(name1);
        double[] tfidf2 = computeTFIDFVector(name2);
        double cosineSimilarity = cosineSimilarity(tfidf1, tfidf2);

        double[] featureVector = Arrays.copyOf(tfidf1, tfidf1.length + 1);
        featureVector[tfidf1.length] = cosineSimilarity;
        return featureVector;
    }

    private double[] computeTFIDFVector(String name) {
        String[] words = tokenize(name);
        Map<String, Integer> termFreq = new HashMap<>();

        for (String word : words) {
            termFreq.put(word, termFreq.getOrDefault(word, 0) + 1);
        }

        double[] vector = new double[vocabulary.size()];
        for (String word : termFreq.keySet()) {
            if (vocabulary.containsKey(word)) {
                int index = vocabulary.get(word);
                vector[index] = termFreq.get(word) * idfScores.getOrDefault(word, 0.0);
            }
        }
        return vector;
    }

    private double cosineSimilarity(double[] vec1, double[] vec2) {
        RealVector v1 = new ArrayRealVector(vec1);
        RealVector v2 = new ArrayRealVector(vec2);
        return v1.dotProduct(v2) / (v1.getNorm() * v2.getNorm());
    }

    private String[] tokenize(String text) {
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        return tokenizer.split(text.toLowerCase());
    }
}