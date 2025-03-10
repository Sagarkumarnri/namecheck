package com.example.namecheck;


import ai.djl.ModelException;
import ai.djl.translate.TranslateException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("/api/names")
public class NameSimilarityController {

    @Autowired
    private NameSimilarityService nameSimilarityService;

    @PostMapping("/train")
    public ResponseEntity<String> trainModel(@RequestParam("file") MultipartFile file) {
        try {
            String result = nameSimilarityService.trainModel(file);
            return ResponseEntity.ok(result);
        } catch (IOException | TranslateException | ModelException e) {
            return ResponseEntity.status(500).body("Training failed: " + e.getMessage());
        }
    }

    @GetMapping("/similarity")
    public ResponseEntity<Float> predictSimilarity(@RequestParam("name1") String name1, @RequestParam("name2") String name2) {
        try {
            float similarity = nameSimilarityService.predictSimilarity(name1, name2);
            return ResponseEntity.ok(similarity);
        } catch (IOException | TranslateException | ModelException e) {
            return ResponseEntity.status(500).body(null);
        }
    }
}