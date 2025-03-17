package com.example.namecheck;

import com.example.namecheck.NameSimilarityService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.util.*;

@RestController
@RequestMapping("/api/names")
public class NameSimilarityController {

    @Autowired
    private NameSimilarityService nameSimilarityService;

    /**
     * Train model using a CSV file.
     * Expected CSV format: "Name1,Name2,Similarity(0-100)"
     */
    @PostMapping("/train")
    public ResponseEntity<String> trainModel(@RequestParam("file") MultipartFile file) {
        try {
            List<String[]> namePairs = new ArrayList<>();
            List<Double> similarityScores = new ArrayList<>();

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(file.getInputStream()))) {
                String line;
                boolean firstLine = true;
                while ((line = reader.readLine()) != null) {
                    if (firstLine) { firstLine = false; continue; } // Skip header
                    String[] values = line.split(",");
                    if (values.length == 3) {
                        namePairs.add(new String[]{values[0].trim(), values[1].trim()});
                        similarityScores.add(Double.parseDouble(values[2].trim()) / 100.0);
                    }
                }
            }

            String result = nameSimilarityService.trainModel(namePairs, similarityScores);
            return ResponseEntity.ok(result);

        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Training failed: " + e.getMessage());
        }
    }

    /**
     * Predict similarity percentage between two names.
     */
    @GetMapping("/predict")
    public ResponseEntity<Map<String, Object>> predictSimilarity(
            @RequestParam String name1,
            @RequestParam String name2) {
        try {
            double similarityScore = nameSimilarityService.predictSimilarity(name1, name2);

            Map<String, Object> response = new HashMap<>();
            response.put("name1", name1);
            response.put("name2", name2);
            response.put("similarity_percentage", similarityScore);

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        }
    }
}
