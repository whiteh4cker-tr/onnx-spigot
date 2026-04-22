package tr.alperendemir.onnxSpigot.llm;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import tr.alperendemir.onnxSpigot.config.LLMConfig;

import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ONNXProvider implements LLMProvider {

    private final Logger logger;
    private final LLMConfig config;
    private final ExecutorService executor;
    private final AtomicBoolean loaded;
    private final Path modelsDirectory;

    private OrtEnvironment env;
    private OrtSession session;
    private BPETokenizer tokenizer;
    private JsonObject modelConfig;

    public ONNXProvider(LLMConfig config, Logger logger, Path pluginDataFolder) {
        this.config = config;
        this.logger = logger;
        this.modelsDirectory = pluginDataFolder.resolve("models");
        this.executor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "OnnxSpigot-LLM-Thread");
            t.setDaemon(true);
            return t;
        });
        this.loaded = new AtomicBoolean(false);

        File modelsDir = modelsDirectory.toFile();
        if (!modelsDir.exists()) {
            modelsDir.mkdirs();
            logger.info("Created models directory: " + modelsDirectory);
        }
    }

    @Override
    public CompletableFuture<Void> loadModel() {
        return CompletableFuture.runAsync(() -> {
            try {
                String modelName = config.getModelName();
                Path modelPath = modelsDirectory.resolve(modelName);
                File modelDir = modelPath.toFile();

                if (!modelDir.exists() || !modelDir.isDirectory()) {
                    logger.severe("Model not found: " + modelPath);
                    logger.severe("Expected folder structure: plugins/onnx-spigot/models/" + modelName + "/");
                    throw new RuntimeException("Model not found: " + modelName);
                }

                logger.info("Loading ONNX model from: " + modelPath);

                Path onnxModelPath = findOnnxModel(modelPath);
                if (onnxModelPath == null) {
                    throw new RuntimeException("No .onnx model file found in: " + modelPath);
                }

                loadModelConfig(modelPath);

                tokenizer = new BPETokenizer(modelPath);
                logger.info("Tokenizer loaded with vocab size: " + tokenizer.getVocabSize());

                env = OrtEnvironment.getEnvironment();
                OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
                opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
                opts.addCPU(true);

                session = env.createSession(onnxModelPath.toString(), opts);
                logger.info("ONNX session created successfully");

                Integer inferredHeadDim = null;
                for (String name : session.getInputNames()) {
                    if (inferredHeadDim == null && name.contains("past_key_values") && name.endsWith(".key")) {
                        try {
                            NodeInfo nodeInfo = session.getInputInfo().get(name);
                            if (nodeInfo != null && nodeInfo.getInfo() instanceof TensorInfo) {
                                TensorInfo tensorInfo = (TensorInfo) nodeInfo.getInfo();
                                long[] shape = tensorInfo.getShape();
                                if (shape.length == 4 && shape[3] > 0) {
                                    inferredHeadDim = (int) shape[3];
                                    logger.info("Inferred head_dim from model input shape: " + inferredHeadDim);
                                }
                            }
                        } catch (Exception ignored) {
                        }
                    }
                }

                if (inferredHeadDim != null && modelConfig != null) {
                    modelConfig.addProperty("_inferred_head_dim", inferredHeadDim);
                }

                loaded.set(true);
                logger.info("ONNX model loaded successfully: " + modelName);

            } catch (OrtException | IOException e) {
                logger.log(Level.SEVERE, "Failed to load ONNX model", e);
                throw new RuntimeException("Failed to load ONNX model", e);
            }
        }, executor);
    }

    private Path findOnnxModel(Path modelPath) {
        File[] files = modelPath.toFile().listFiles((dir, name) -> name.endsWith(".onnx"));
        if (files != null && files.length > 0) {
            for (File f : files) {
                if (f.getName().contains("decoder") || f.getName().equals("model.onnx")) {
                    return f.toPath();
                }
            }
            return files[0].toPath();
        }
        return null;
    }

    private void loadModelConfig(Path modelPath) throws IOException {
        Path configPath = modelPath.resolve("config.json");
        if (Files.exists(configPath)) {
            String content = Files.readString(configPath);
            Gson gson = new Gson();
            modelConfig = gson.fromJson(content, JsonObject.class);
        } else {
            modelConfig = new JsonObject();
        }
    }

    @Override
    public CompletableFuture<String> generateResponse(String prompt) {
        if (!isLoaded()) {
            return CompletableFuture.failedFuture(new IllegalStateException("Model not loaded"));
        }

        return CompletableFuture.supplyAsync(() -> {
            try {
                return generate(prompt);
            } catch (OrtException e) {
                logger.log(Level.SEVERE, "Failed to generate response", e);
                throw new RuntimeException("Failed to generate response", e);
            }
        }, executor);
    }

    private String generate(String prompt) throws OrtException {
        long[] inputIds = tokenizer.encode(prompt);
        int maxNewTokens = config.getMaxTokens();
        int eosTokenId = tokenizer.getEosTokenId();

        List<Long> generatedIds = new ArrayList<>();
        for (long id : inputIds) {
            generatedIds.add(id);
        }

        Set<String> inputNames = session.getInputNames();
        boolean requiresKVCache = inputNames.stream().anyMatch(name -> name.contains("past_key_values"));

        Map<String, OnnxTensor> pastKVCache = new HashMap<>();
        int numLayers = 0;
        int headDim = 128;
        int numKVHeads = 2;
        int numHeads = 16;

        if (modelConfig != null && modelConfig.has("num_hidden_layers")) {
            numLayers = modelConfig.get("num_hidden_layers").getAsInt();
        }

        if (modelConfig != null && modelConfig.has("num_key_value_heads")) {
            numKVHeads = modelConfig.get("num_key_value_heads").getAsInt();
        } else if (modelConfig != null && modelConfig.has("num_attention_heads")) {
            numKVHeads = modelConfig.get("num_attention_heads").getAsInt();
        }

        if (modelConfig != null && modelConfig.has("num_attention_heads")) {
            numHeads = modelConfig.get("num_attention_heads").getAsInt();
        }

        if (modelConfig != null && modelConfig.has("_inferred_head_dim")) {
            headDim = modelConfig.get("_inferred_head_dim").getAsInt();
        } else if (modelConfig != null && modelConfig.has("hidden_size")) {
            int hiddenSize = modelConfig.get("hidden_size").getAsInt();
            headDim = hiddenSize / numHeads;
        }

        for (int i = 0; i < maxNewTokens; i++) {
            long[] currentIds;
            long[] attentionMask;
            int pastSeqLen = 0;

            if (requiresKVCache && i > 0 && !pastKVCache.isEmpty()) {
                currentIds = new long[]{generatedIds.get(generatedIds.size() - 1)};
                pastSeqLen = generatedIds.size() - 1;
                attentionMask = new long[generatedIds.size()];
                Arrays.fill(attentionMask, 1L);
            } else {
                currentIds = generatedIds.stream().mapToLong(Long::longValue).toArray();
                attentionMask = new long[currentIds.length];
                Arrays.fill(attentionMask, 1L);
            }

            long[] inputShape = {1, currentIds.length};
            long[] maskShape = {1, attentionMask.length};
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(currentIds), inputShape);
            OnnxTensor attentionTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), maskShape);

            OnnxTensor positionIdsTensor = null;
            if (inputNames.contains("position_ids")) {
                long[] positionIds = new long[currentIds.length];
                for (int j = 0; j < positionIds.length; j++) {
                    positionIds[j] = pastSeqLen + j;
                }
                positionIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(positionIds), inputShape);
            }

            Map<String, OnnxTensor> inputs = new HashMap<>();
            if (inputNames.contains("input_ids")) {
                inputs.put("input_ids", inputTensor);
            }
            if (inputNames.contains("attention_mask")) {
                inputs.put("attention_mask", attentionTensor);
            }
            if (positionIdsTensor != null) {
                inputs.put("position_ids", positionIdsTensor);
            }

            if (requiresKVCache) {
                if (i == 0 || pastKVCache.isEmpty()) {
                    if (numLayers == 0) {
                        for (String inputName : inputNames) {
                            if (inputName.startsWith("past_key_values.")) {
                                try {
                                    String[] parts = inputName.split("\\.");
                                    if (parts.length >= 2) {
                                        int layerIdx = Integer.parseInt(parts[1]);
                                        numLayers = Math.max(numLayers, layerIdx + 1);
                                    }
                                } catch (NumberFormatException ignored) {
                                }
                            }
                        }
                    }

                    for (int layer = 0; layer < numLayers; layer++) {
                        long[] cacheShape = {1, numKVHeads, 0, headDim};
                        float[] emptyData = new float[0];

                        String keyName = "past_key_values." + layer + ".key";
                        String valueName = "past_key_values." + layer + ".value";

                        if (inputNames.contains(keyName)) {
                            OnnxTensor emptyKeyTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(emptyData), cacheShape);
                            inputs.put(keyName, emptyKeyTensor);
                        }
                        if (inputNames.contains(valueName)) {
                            OnnxTensor emptyValueTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(emptyData), cacheShape);
                            inputs.put(valueName, emptyValueTensor);
                        }
                    }
                } else {
                    inputs.putAll(pastKVCache);
                }
            }

            OrtSession.Result result = null;
            try {
                result = session.run(inputs);

                Optional<OnnxValue> logitsOptional = result.get("logits");
                OnnxValue logitsOutput;
                if (logitsOptional.isPresent()) {
                    logitsOutput = logitsOptional.get();
                } else if (result.size() > 0) {
                    logitsOutput = result.get(0);
                    if (logitsOutput == null) {
                        throw new OrtException("Could not find logits output");
                    }
                } else {
                    throw new OrtException("Could not find logits output");
                }

                float[][][] logits = (float[][][]) logitsOutput.getValue();
                float[] lastLogits = logits[0][logits[0].length - 1];

                float temperature = (float) config.getTemperature();
                if (temperature > 0) {
                    for (int j = 0; j < lastLogits.length; j++) {
                        lastLogits[j] /= temperature;
                    }
                }

                int nextToken = sampleToken(lastLogits, config.getTopP());
                if (nextToken == eosTokenId) {
                    break;
                }

                generatedIds.add((long) nextToken);

                if (requiresKVCache) {
                    for (OnnxTensor tensor : pastKVCache.values()) {
                        tensor.close();
                    }
                    pastKVCache.clear();

                    Set<String> outputNames = session.getOutputNames();
                    for (String outputName : outputNames) {
                        if (outputName.startsWith("present.") || outputName.contains("present_key_values")) {
                            Optional<OnnxValue> valueOpt = result.get(outputName);
                            if (valueOpt.isPresent() && valueOpt.get() instanceof OnnxTensor) {
                                String pastName = outputName
+                                        .replace("present.", "past_key_values.")
+                                        .replace("present_key_values.", "past_key_values.");
                                pastKVCache.put(pastName, (OnnxTensor) valueOpt.get());
                            }
                        }
                    }
                }

            } finally {
                inputTensor.close();
                attentionTensor.close();
                if (positionIdsTensor != null) {
                    positionIdsTensor.close();
                }

                if (requiresKVCache && (i == 0 || pastKVCache.isEmpty())) {
                    for (Map.Entry<String, OnnxTensor> entry : inputs.entrySet()) {
                        if (entry.getKey().startsWith("past_key_values.") && entry.getValue() != null) {
                            entry.getValue().close();
                        }
                    }
                }

                if (result != null && !requiresKVCache) {
                    result.close();
                }
            }
        }

        for (OnnxTensor tensor : pastKVCache.values()) {
            try {
                tensor.close();
            } catch (Exception ignored) {
            }
        }

        long[] newTokens = new long[generatedIds.size() - inputIds.length];
        for (int i = 0; i < newTokens.length; i++) {
            newTokens[i] = generatedIds.get(inputIds.length + i);
        }

        String response = tokenizer.decode(newTokens);
        return cleanResponse(response);
    }

    private int sampleToken(float[] logits, double topP) {
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (float logit : logits) {
            maxLogit = Math.max(maxLogit, logit);
        }

        float sum = 0;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - maxLogit);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }

        Integer[] indices = new Integer[probs.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, (a, b) -> Float.compare(probs[b], probs[a]));

        float cumSum = 0;
        int cutoff = indices.length;
        for (int i = 0; i < indices.length; i++) {
            cumSum += probs[indices[i]];
            if (cumSum >= topP) {
                cutoff = i + 1;
                break;
            }
        }

        float[] filteredProbs = new float[cutoff];
        float filteredSum = 0;
        for (int i = 0; i < cutoff; i++) {
            filteredProbs[i] = probs[indices[i]];
            filteredSum += filteredProbs[i];
        }

        float rand = (float) Math.random() * filteredSum;
        float cumulative = 0;
        for (int i = 0; i < cutoff; i++) {
            cumulative += filteredProbs[i];
            if (rand < cumulative) {
                return indices[i];
            }
        }

        return indices[0];
    }

    private String cleanResponse(String response) {
        response = response.trim();

        if (response.contains("<|")) {
            response = response.substring(0, response.indexOf("<|"));
        }

        if (response.length() > 500) {
            response = response.substring(0, 500) + "...";
        }

        return response;
    }

    @Override
    public boolean isLoaded() {
        return loaded.get() && session != null;
    }

    @Override
    public void unload() {
        loaded.set(false);

        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                logger.log(Level.WARNING, "Error closing ONNX session", e);
            }
            session = null;
        }

        tokenizer = null;
        logger.info("ONNX model unloaded");
    }

    @Override
    public String getName() {
        return "ONNX-Runtime";
    }

    public Path getModelsDirectory() {
        return modelsDirectory;
    }
}

