package com.example.namecheck;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.Batchifier;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

@Service
public class NameSimilarityService {

    private static final Logger logger = LoggerFactory.getLogger(NameSimilarityService.class);
    private static final String TOKENIZER_NAME = "bert-base-uncased";  // Hugging Face tokenizer
    private static final Path MODEL_PATH = Paths.get("src/main/resources/paraphrase-MiniLM-L6-v2.onnx"); // ONNX Model Path
    private final HuggingFaceTokenizer tokenizer;

    public NameSimilarityService() {
        this.tokenizer = HuggingFaceTokenizer.newInstance(TOKENIZER_NAME);
    }

    /**
     * Tokenizes names using Hugging Face tokenizer and converts to float[][].
     */
    private float[][] tokenizeNames(String name1, String name2) {
        Encoding encoding1 = tokenizer.encode(name1);
        Encoding encoding2 = tokenizer.encode(name2);

        // Convert long[] to float[] for NDArray compatibility
        return new float[][]{
                convertToFloatArray(encoding1.getIds()),
                convertToFloatArray(encoding2.getIds())
        };
    }

    /**
     * Converts long[] to float[] (required for NDArray creation).
     */
    private float[] convertToFloatArray(long[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (float) input[i];
        }
        return output;
    }

    /**
     * Predicts similarity score using ONNX model.
     */
    public float predictSimilarity(String name1, String name2) {
        try {
            Criteria<float[][], float[]> criteria = Criteria.builder()
                    .setTypes(float[][].class, float[].class)
                    .optModelPath(MODEL_PATH)
                    .optEngine("OnnxRuntime")
                    .optTranslator(new NameSimilarityTranslator())
                    .build();

            try (ZooModel<float[][], float[]> model = ModelZoo.loadModel(criteria);
                 Predictor<float[][], float[]> predictor = model.newPredictor()) {

                float[][] tokenizedNames = tokenizeNames(name1, name2);
                float[] result = predictor.predict(tokenizedNames);
                return result[0];  // Similarity score
            }

        } catch (Exception e) {
            logger.error("Error during prediction: ", e);
            return -1.0f;
        }
    }

    /**
     * Trains a model using a dataset.
     */
    public String trainModel(MultipartFile file) {
        try {
            Criteria<float[][], float[]> criteria = Criteria.builder()
                    .setTypes(float[][].class, float[].class)
                    .optModelPath(MODEL_PATH)
                    .optEngine("OnnxRuntime")
                    .optTranslator(new NameSimilarityTranslator())
                    .build();

            try (ZooModel<float[][], float[]> model = ModelZoo.loadModel(criteria);
                 NDManager manager = NDManager.newBaseManager();
                 Reader reader = new InputStreamReader(file.getInputStream())) {

                List<float[][]> namePairs = new ArrayList<>();
                List<Float> similarityScores = new ArrayList<>();

                Iterable<CSVRecord> records = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader);
                for (CSVRecord record : records) {
                    String name1 = record.get("Name1");
                    String name2 = record.get("Name2");
                    float similarity = Float.parseFloat(record.get("Similarity"));

                    float[][] tokenizedNames = tokenizeNames(name1, name2);
                    namePairs.add(tokenizedNames);
                    similarityScores.add(similarity);
                }

                // Ensure data shapes match
                int batchSize = namePairs.size();
                int seqLength = namePairs.get(0)[0].length;

                NDArray data = manager.create(new Shape(batchSize, 2, seqLength));
                NDArray labels = manager.create(new Shape(batchSize, 1));

                for (int i = 0; i < batchSize; i++) {
                    data.set(new NDList(manager.create(namePairs.get(i))));
                    labels.set(i, similarityScores.get(i));
                }

                Dataset dataset = new ArrayDataset.Builder()
                        .setData(data)
                        .optLabels(labels)
                        .setSampling(batchSize, true)
                        .build();

                // Training configuration
                Loss loss = Loss.l2Loss();
                Tracker tracker = Tracker.fixed(0.001f);
                Adam optimizer = Adam.builder().optLearningRateTracker(tracker).build();
                TrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(optimizer);

                try (Trainer trainer = model.newTrainer(config)) {
                    trainer.initialize(new Shape(1, seqLength));

                    for (int epoch = 0; epoch < 10; epoch++) {
                        for (Batch batch : dataset.getData(manager)) {
                            EasyTrain.trainBatch(trainer, batch);
                            trainer.step();
                            batch.close();
                        }
                    }
                }

                return "Training complete! Model saved.";
            }
        } catch (Exception e) {
            logger.error("Error during training: ", e);
            return "Training failed: " + e.getMessage();
        }
    }

    /**
     * Translator for ONNX Similarity Model
     */
    private static class NameSimilarityTranslator implements Translator<float[][], float[]> {
        @Override
        public NDList processInput(TranslatorContext ctx, float[][] input) {
            return new NDList(ctx.getNDManager().create(input));
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            return list.singletonOrThrow().toFloatArray();
        }
    }
}
