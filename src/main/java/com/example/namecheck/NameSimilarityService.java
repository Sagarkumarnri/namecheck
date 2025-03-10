package com.example.namecheck;

import ai.djl.Application;
import ai.djl.ModelException;
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
import ai.djl.training.initializer.XavierInitializer;
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
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

@Service
public class NameSimilarityService {

    private static final Logger logger = LoggerFactory.getLogger(NameSimilarityService.class);

    // Load the ONNX model from the classpath
    private Path getModelPath() throws IOException {
        ClassPathResource resource = new ClassPathResource("paraphrase-MiniLM-L6-v2.onnx");
        Path tempFile = Files.createTempFile("model", ".onnx");
        try (InputStream inputStream = resource.getInputStream()) {
            Files.copy(inputStream, tempFile, StandardCopyOption.REPLACE_EXISTING);
        }
        return tempFile;
    }

    // Tokenize names using a pre-trained tokenizer
    private float[][] tokenizeNames(String name1, String name2) throws IOException, TranslateException, ModelException {
        // Load the tokenizer ONNX model
        Path tokenizerPath = Paths.get("src/main/resources/tokenizer.onnx"); // Path to the tokenizer ONNX model
        Criteria<String[], float[][]> criteria = Criteria.builder()
                .setTypes(String[].class, float[][].class)
                .optModelPath(tokenizerPath)
                .optEngine("OnnxRuntime")
                .optTranslator(new TokenizerTranslator())
                .build();

        try (ZooModel<String[], float[][]> tokenizerModel = ModelZoo.loadModel(criteria);
             Predictor<String[], float[][]> tokenizerPredictor = tokenizerModel.newPredictor()) {
            // Tokenize the input names
            return tokenizerPredictor.predict(new String[]{name1, name2});
        }
    }

    // Train the model using a dataset of name pairs and similarity scores
    public String trainModel(MultipartFile file) throws IOException, TranslateException, ModelException {
        Path modelPath = getModelPath();

        try (NDManager manager = NDManager.newBaseManager()) {
            // Load the ONNX model
            Criteria<float[][], float[]> criteria = Criteria.builder()
                    .setTypes(float[][].class, float[].class)
                    .optModelPath(modelPath)
                    .optEngine("OnnxRuntime")
                    .optTranslator(new NameSimilarityTranslator())
                    .build();

            try (ZooModel<float[][], float[]> model = ModelZoo.loadModel(criteria)) {
                // Prepare the dataset
                List<float[][]> namePairs = new ArrayList<>();
                List<Float> similarityScores = new ArrayList<>();

                Reader reader = new InputStreamReader(file.getInputStream());
                Iterable<CSVRecord> records = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader);
                for (CSVRecord record : records) {
                    String name1 = record.get("Name1");
                    String name2 = record.get("Name2");
                    float similarity = Float.parseFloat(record.get("Similarity"));

                    // Tokenize the names
                    float[][] tokenizedNames = tokenizeNames(name1, name2);
                    namePairs.add(tokenizedNames);
                    similarityScores.add(similarity);
                }

                // Convert the dataset to NDArrays
                NDArray[] data = new NDArray[namePairs.size()];
                NDArray[] labels = new NDArray[similarityScores.size()];
                for (int i = 0; i < namePairs.size(); i++) {
                    data[i] = manager.create(namePairs.get(i));
                    labels[i] = manager.create(new float[]{similarityScores.get(i)});
                }

                // Create the dataset
                Dataset dataset = new ArrayDataset.Builder()
                        .setData(data)
                        .optLabels(labels)
                        .setSampling(namePairs.size(), true)
                        .build();

                // Configure training
                Loss loss = Loss.l2Loss();
                Tracker tracker = Tracker.fixed(0.001f);
                Adam optimizer = Adam.builder().optLearningRateTracker(tracker).build();
                TrainingConfig config = new DefaultTrainingConfig(loss)
                        .optOptimizer(optimizer)
                        .optInitializer(new XavierInitializer(), "*");

                // Train the model
                Trainer trainer = model.newTrainer(config);
                trainer.initialize(new Shape(1, 6)); // Adjust shape based on tokenized input size

                for (int epoch = 0; epoch < 10; epoch++) { // Train for 10 epochs
                    for (Batch batch : dataset.getData(manager)) {
                        EasyTrain.trainBatch(trainer, batch);
                        trainer.step();
                        batch.close();
                    }
                }

                // Save the fine-tuned model
                model.save(modelPath.getParent(), "fine-tuned-name-similarity");
            }
            return "Training complete! Model saved.";
        } catch (Exception e) {
            logger.error("Error during training: ", e);
            return "Training failed: " + e.getMessage();
        }
    }

    // Predict similarity between two names
    public float predictSimilarity(String name1, String name2) throws IOException, TranslateException, ModelException {
        Path modelPath = getModelPath();

        // Load the ONNX model
        Criteria<float[][], float[]> criteria = Criteria.builder()
                .setTypes(float[][].class, float[].class)
                .optModelPath(modelPath)
                .optEngine("OnnxRuntime")
                .optTranslator(new NameSimilarityTranslator())
                .build();

        try (ZooModel<float[][], float[]> model = ModelZoo.loadModel(criteria);
             Predictor<float[][], float[]> predictor = model.newPredictor()) {
            // Tokenize the input names
            float[][] tokenizedNames = tokenizeNames(name1, name2);

            // Predict similarity
            float[] result = predictor.predict(tokenizedNames);
            return result[0]; // Return the similarity score
        }
    }

    // Custom translator for tokenizer
    private static class TokenizerTranslator implements Translator<String[], float[][]> {

        @Override
        public NDList processInput(TranslatorContext ctx, String[] input) {
            NDManager manager = ctx.getNDManager();
            NDArray array = manager.create(input);
            return new NDList(array);
        }

        @Override
        public float[][] processOutput(TranslatorContext ctx, NDList list) {
            NDArray output = list.singletonOrThrow();
            return new float[][]{output.toFloatArray()};
        }

    }

    // Custom translator for similarity model
    private static class NameSimilarityTranslator implements Translator<float[][], float[]> {

        @Override
        public NDList processInput(TranslatorContext ctx, float[][] input) {
            NDManager manager = ctx.getNDManager();
            NDArray array = manager.create(input);
            return new NDList(array);
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDArray output = list.singletonOrThrow();
            return output.toFloatArray();
        }

    }
}