package com.example.namecheck;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDArrays;
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
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

@Service
public class NameSimilarityService {

    private static final Logger logger = LoggerFactory.getLogger(NameSimilarityService.class);


    private static final Path MODEL_PATH = Paths.get("src/main/resources/paraphrase-MiniLM-L6-v2.onnx");


    private float[][] tokenizeNames(String name1, String name2) throws IOException, TranslateException, ModelException {
        return new float[][]{tokenize(name1), tokenize(name2)};
    }

    private float[] tokenize(String name) {

        return new float[]{name.length()}; // Dummy tokenization (Replace with real tokenizer)
    }


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

                int batchSize = namePairs.size();
                List<NDArray> dataList = new ArrayList<>();
                List<NDArray> labelList = new ArrayList<>();

                for (int i = 0; i < batchSize; i++) {
                    dataList.add(manager.create(namePairs.get(i)));
                    labelList.add(manager.create(new float[]{similarityScores.get(i)}));
                }

                NDArray data = NDArrays.stack(new NDList(dataList));
                NDArray labels = NDArrays.stack(new NDList(labelList));

                Dataset dataset = new ArrayDataset.Builder()
                        .setData(data)
                        .optLabels(labels)
                        .setSampling(batchSize, true)
                        .build();

                Loss loss = Loss.l2Loss();
                Tracker tracker = Tracker.fixed(0.001f);
                Adam optimizer = Adam.builder().optLearningRateTracker(tracker).build();
                TrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(optimizer);

                try (Trainer trainer = model.newTrainer(config)) {
                    trainer.initialize(new Shape(1, namePairs.get(0)[0].length));

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
     * Predicts similarity between two names using the ONNX model.
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
                return result[0];
            }

        } catch (Exception e) {
            logger.error("Error during prediction: ", e);
            return -1.0f;
        }
    }


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
