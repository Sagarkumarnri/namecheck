package com.example.namecheck;

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
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.nio.file.*;
import java.util.*;

@Service
public class NameSimilarityService {
    private static final Logger logger = LoggerFactory.getLogger(NameSimilarityService.class);

    // 🔹 Load ONNX Model Path
    private Path getModelPath() throws IOException {
        return Paths.get("src/main/resources/trained_name_similarity.onnx");
    }

    // 🔹 Simple Tokenization: Convert Characters to Normalized Floats
    private float[] tokenize(String text) {
        float[] tokens = new float[128];
        char[] chars = text.toLowerCase().toCharArray();
        for (int i = 0; i < Math.min(chars.length, tokens.length); i++) {
            tokens[i] = chars[i] / 255.0f;
        }
        return tokens;
    }


    private float[][] processInput(String name1, String name2) {
        return new float[][]{tokenize(name1), tokenize(name2)};
    }


    public String trainModel(MultipartFile file) throws IOException, TranslateException, ModelException {
        Path modelPath = getModelPath();
        try (NDManager manager = NDManager.newBaseManager()) {


            List<float[][]> namePairs = new ArrayList<>();
            List<Float> similarityScores = new ArrayList<>();

            Reader reader = new InputStreamReader(file.getInputStream());
            Iterable<CSVRecord> records = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader);

            for (CSVRecord record : records) {
                String name1 = record.get("Name1");
                String name2 = record.get("Name2");
                float similarity = Float.parseFloat(record.get("Similarity"));

                float[][] input = processInput(name1, name2);
                namePairs.add(input);
                similarityScores.add(similarity);
            }

             NDArray[] data = new NDArray[namePairs.size()];
            NDArray[] labels = new NDArray[similarityScores.size()];

            for (int i = 0; i < namePairs.size(); i++) {
                data[i] = manager.create(namePairs.get(i));
                labels[i] = manager.create(new float[]{similarityScores.get(i)});
            }

            // Create dataset
            Dataset dataset = new ArrayDataset.Builder()
                    .setData(data)
                    .optLabels(labels)
                    .setSampling(namePairs.size(), true)
                    .build();

            // 2️⃣ **Configure Training**
            Loss loss = Loss.l2Loss();
            Tracker tracker = Tracker.fixed(0.001f);
            Adam optimizer = Adam.builder().optLearningRateTracker(tracker).build();
            TrainingConfig config = new DefaultTrainingConfig(loss)
                    .optOptimizer(optimizer)
                    .optInitializer(new XavierInitializer(), "*");


            Criteria<float[][], Float> criteria = Criteria.builder()
                    .setTypes(float[][].class, Float.class)
                    .optModelPath(modelPath)
                    .optEngine("OnnxRuntime")
                    .optTranslator(new NameSimilarityTranslator())
                    .build();

            try (ZooModel<float[][], Float> model = ModelZoo.loadModel(criteria);
                 Trainer trainer = model.newTrainer(config)) {

                trainer.initialize(new Shape(1, 128)); // Input shape

                 for (int epoch = 0; epoch < 10; epoch++) { // Train for 10 epochs
                    for (Batch batch : dataset.getData(manager)) {
                        EasyTrain.trainBatch(trainer, batch);
                        trainer.step();
                        batch.close();
                    }
                }


                model.save(modelPath.getParent(), "trained_name_similarity");

            }
        }
        return "Training Complete! Model Saved at: " + modelPath.toString();
    }


    public float predictSimilarity(String name1, String name2) throws IOException, TranslateException, ModelException {
        Path modelPath = getModelPath();
        try (NDManager manager = NDManager.newBaseManager()) {


            Criteria<float[][], Float> criteria = Criteria.builder()
                    .setTypes(float[][].class, Float.class)
                    .optModelPath(modelPath)
                    .optEngine("OnnxRuntime")
                    .optTranslator(new NameSimilarityTranslator())
                    .build();

            try (ZooModel<float[][], Float> model = ModelZoo.loadModel(criteria);
                 Predictor<float[][], Float> predictor = model.newPredictor()) {

                float[][] input = processInput(name1, name2);
                return predictor.predict(input);
            }
        }
    }


    private static class NameSimilarityTranslator implements Translator<float[][], Float> {
        @Override
        public NDList processInput(TranslatorContext ctx, float[][] input) {
            NDManager manager = ctx.getNDManager();
            return new NDList(manager.create(input));
        }

        @Override
        public Float processOutput(TranslatorContext ctx, NDList list) {
            return list.singletonOrThrow().getFloat();
        }
    }
}
