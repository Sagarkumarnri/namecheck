package com.example.namecheck;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
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

    private Path getModelPath() throws IOException {
        ClassPathResource resource = new ClassPathResource("paraphrase-MiniLM-L6-v2.pt");
        Path tempFile = Files.createTempFile("model", ".onnx");
        try (InputStream inputStream = resource.getInputStream()) {
            Files.copy(inputStream, tempFile, StandardCopyOption.REPLACE_EXISTING);
        }
        return tempFile;
    }

    public String trainModel(MultipartFile file) throws IOException, TranslateException, ModelException {
        Path modelPath = getModelPath();
        try (NDManager manager = NDManager.newBaseManager()) {
            Criteria<String, Float> criteria = Criteria.builder()
                    .setTypes(String.class, Float.class)
                    .optModelPath(Paths.get("paraphrase-MiniLM-L6-v2.pt")) // Path to your `.pt` file
                    .optEngine("PyTorch")  // Ensure PyTorch engine is used
                    .build();

            try (ZooModel<String, Float> model = ModelZoo.loadModel(criteria)) {
                Block block = model.getBlock();
                SequentialBlock newBlock = new SequentialBlock()
                        .add(block)
                        .add(Linear.builder().setUnits(1).build());
                model.setBlock(newBlock);

                List<NDArray> namePairs = new ArrayList<>();
                List<NDArray> similarityScores = new ArrayList<>();
                Reader reader = new InputStreamReader(file.getInputStream());
                Iterable<CSVRecord> records = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader);
                for (CSVRecord record : records) {
                    float name1 = Float.parseFloat(record.get("Name1"));
                    float name2 = Float.parseFloat(record.get("Name2"));
                    float similarity = Float.parseFloat(record.get("Similarity"));
                    namePairs.add(manager.create(new float[]{name1, name2}));
                    similarityScores.add(manager.create(new float[]{similarity}));
                }

                Dataset dataset = new ArrayDataset.Builder()
                        .setData(namePairs.toArray(new NDArray[0]))
                        .optLabels(similarityScores.toArray(new NDArray[0]))
                        .setSampling(namePairs.size(), true)
                        .build();

                Loss loss = Loss.l2Loss();
                Tracker tracker = Tracker.fixed(0.001f);
                Adam optimizer = Adam.builder().optLearningRateTracker(tracker).build();
                TrainingConfig config = new DefaultTrainingConfig(loss)
                        .optOptimizer(optimizer)
                        .optInitializer(new XavierInitializer(), "*");
                Trainer trainer = model.newTrainer(config);
                trainer.initialize(new ai.djl.ndarray.types.Shape(1, 2));

                List<Batch> batches = new ArrayList<>();
                try (Batch batch = dataset.getData(manager).iterator().next()) {
                    batches.add(batch);
                    EasyTrain.trainBatch(trainer, batch);
                    trainer.step();
                }

                model.save(modelPath.getParent(), "fine-tuned-sentence-transformer");
            }
            return "Training complete! Model saved.";
        } catch (Exception e) {
            logger.error("Error during training: ", e);
            return "Training failed: " + e.getMessage();
        }
    }

    public float predictSimilarity(String name1, String name2) throws IOException, TranslateException, ModelException {
        Path modelPath = getModelPath();
        Criteria<String[], NDArray> criteria = Criteria.builder()
                .optApplication(Application.NLP.TEXT_EMBEDDING)
                .setTypes(String[].class, NDArray.class)
                .optModelPath(modelPath)
                .optTranslator(new NameSimilarityTranslator())
                .build();

        try (ZooModel<String[], NDArray> model = ModelZoo.loadModel(criteria);
             Predictor<String[], NDArray> predictor = model.newPredictor()) {
            NDArray result = predictor.predict(new String[]{name1, name2});
            return result.getFloat(0);
        }
    }

    private static class NameSimilarityTranslator implements Translator<String[], NDArray> {
        @Override
        public NDList processInput(TranslatorContext ctx, String[] input) {
            NDManager manager = ctx.getNDManager();
            return new NDList(manager.create(input));
        }

        @Override
        public NDArray processOutput(TranslatorContext ctx, NDList list) {
            return list.singletonOrThrow();
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }
}