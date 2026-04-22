package tr.alperendemir.onnxSpigot.llm;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class BPETokenizer {

    private final Map<String, Integer> vocab;
    private final Map<Integer, String> reverseVocab;
    private final Map<String, Integer> merges;
    private final Map<Integer, Integer> byteEncoder;
    private final Map<Integer, Integer> byteDecoder;
    private final Pattern pattern;

    private int bosTokenId = -1;
    private int eosTokenId = -1;
    private int padTokenId = -1;

    public BPETokenizer(Path modelPath) throws IOException {
        this.vocab = new HashMap<>();
        this.reverseVocab = new HashMap<>();
        this.merges = new HashMap<>();
        this.byteEncoder = new HashMap<>();
        this.byteDecoder = new HashMap<>();

        this.pattern = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

        loadVocab(modelPath);
        loadMerges(modelPath);
        loadSpecialTokens(modelPath);
        initByteEncoder();
    }

    private void loadVocab(Path modelPath) throws IOException {
        Path vocabPath = modelPath.resolve("vocab.json");
        if (!Files.exists(vocabPath)) {
            Path tokenizerPath = modelPath.resolve("tokenizer.json");
            if (Files.exists(tokenizerPath)) {
                loadFromTokenizerJson(tokenizerPath);
                return;
            }
            throw new IOException("vocab.json or tokenizer.json not found in " + modelPath);
        }

        String content = Files.readString(vocabPath);
        Gson gson = new Gson();
        Map<String, Double> rawVocab = gson.fromJson(content, new TypeToken<Map<String, Double>>() {
        }.getType());

        for (Map.Entry<String, Double> entry : rawVocab.entrySet()) {
            int id = entry.getValue().intValue();
            vocab.put(entry.getKey(), id);
            reverseVocab.put(id, entry.getKey());
        }
    }

    private void loadFromTokenizerJson(Path tokenizerPath) throws IOException {
        String content = Files.readString(tokenizerPath);
        Gson gson = new Gson();
        JsonObject root = gson.fromJson(content, JsonObject.class);

        if (root.has("model") && root.getAsJsonObject("model").has("vocab")) {
            JsonObject vocabObj = root.getAsJsonObject("model").getAsJsonObject("vocab");
            for (String key : vocabObj.keySet()) {
                int id = vocabObj.get(key).getAsInt();
                vocab.put(key, id);
                reverseVocab.put(id, key);
            }
        }

        if (root.has("model") && root.getAsJsonObject("model").has("merges")) {
            var mergesArray = root.getAsJsonObject("model").getAsJsonArray("merges");
            int rank = 0;
            for (var merge : mergesArray) {
                merges.put(merge.getAsString(), rank++);
            }
        }
    }

    private void loadMerges(Path modelPath) throws IOException {
        Path mergesPath = modelPath.resolve("merges.txt");
        if (!Files.exists(mergesPath)) {
            return;
        }

        List<String> lines = Files.readAllLines(mergesPath);
        int rank = 0;
        for (String line : lines) {
            if (line.startsWith("#") || line.trim().isEmpty()) {
                continue;
            }
            merges.put(line.trim(), rank++);
        }
    }

    private void loadSpecialTokens(Path modelPath) throws IOException {
        Path configPath = modelPath.resolve("tokenizer_config.json");
        if (!Files.exists(configPath)) {
            eosTokenId = vocab.getOrDefault("<|endoftext|>", vocab.getOrDefault("</s>", -1));
            bosTokenId = vocab.getOrDefault("<|startoftext|>", vocab.getOrDefault("<s>", eosTokenId));
            padTokenId = vocab.getOrDefault("<pad>", eosTokenId);
            return;
        }

        String content = Files.readString(configPath);
        Gson gson = new Gson();
        JsonObject config = gson.fromJson(content, JsonObject.class);

        if (config.has("eos_token")) {
            String eosToken = extractTokenString(config.get("eos_token"));
            eosTokenId = vocab.getOrDefault(eosToken, -1);
        }
        if (config.has("bos_token")) {
            String bosToken = extractTokenString(config.get("bos_token"));
            bosTokenId = vocab.getOrDefault(bosToken, -1);
        }
        if (config.has("pad_token")) {
            String padToken = extractTokenString(config.get("pad_token"));
            padTokenId = vocab.getOrDefault(padToken, -1);
        }

        if (eosTokenId == -1) {
            eosTokenId = vocab.getOrDefault("<|endoftext|>", vocab.getOrDefault("</s>", 0));
        }
        if (bosTokenId == -1) {
            bosTokenId = eosTokenId;
        }
        if (padTokenId == -1) {
            padTokenId = eosTokenId;
        }
    }

    private String extractTokenString(com.google.gson.JsonElement element) {
        if (element.isJsonPrimitive()) {
            return element.getAsString();
        }
        if (element.isJsonObject()) {
            JsonObject obj = element.getAsJsonObject();
            if (obj.has("content")) {
                return obj.get("content").getAsString();
            }
        }
        return "";
    }

    private void initByteEncoder() {
        List<Integer> bs = new ArrayList<>();
        for (int i = '!'; i <= '~'; i++) {
            bs.add(i);
        }
        for (int i = '¡'; i <= '¬'; i++) {
            bs.add(i);
        }
        for (int i = '®'; i <= 'ÿ'; i++) {
            bs.add(i);
        }

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }

        for (int i = 0; i < bs.size(); i++) {
            byteEncoder.put(bs.get(i), cs.get(i));
            byteDecoder.put(cs.get(i), bs.get(i));
        }
    }

    public long[] encode(String text) {
        List<Integer> tokens = new ArrayList<>();

        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            String word = matcher.group();
            StringBuilder encoded = new StringBuilder();
            for (byte b : word.getBytes(java.nio.charset.StandardCharsets.UTF_8)) {
                int byteVal = b & 0xFF;
                Integer mapped = byteEncoder.get(byteVal);
                if (mapped != null) {
                    encoded.append((char) mapped.intValue());
                }
            }

            List<String> wordTokens = bpe(encoded.toString());
            for (String token : wordTokens) {
                Integer id = vocab.get(token);
                if (id != null) {
                    tokens.add(id);
                }
            }
        }

        return tokens.stream().mapToLong(Integer::longValue).toArray();
    }

    private List<String> bpe(String token) {
        if (token.isEmpty()) {
            return Collections.emptyList();
        }

        List<String> word = new ArrayList<>();
        for (char c : token.toCharArray()) {
            word.add(String.valueOf(c));
        }

        while (word.size() > 1) {
            int minRank = Integer.MAX_VALUE;
            int minIdx = -1;

            for (int i = 0; i < word.size() - 1; i++) {
                String pair = word.get(i) + " " + word.get(i + 1);
                Integer rank = merges.get(pair);
                if (rank != null && rank < minRank) {
                    minRank = rank;
                    minIdx = i;
                }
            }

            if (minIdx == -1) {
                break;
            }

            String merged = word.get(minIdx) + word.get(minIdx + 1);
            word.set(minIdx, merged);
            word.remove(minIdx + 1);
        }

        return word;
    }

    public String decode(long[] tokenIds) {
        StringBuilder text = new StringBuilder();

        for (long id : tokenIds) {
            String token = reverseVocab.get((int) id);
            if (token != null) {
                text.append(token);
            }
        }

        byte[] bytes = new byte[text.length()];
        int byteCount = 0;
        for (int i = 0; i < text.length(); i++) {
            int c = text.charAt(i);
            Integer decoded = byteDecoder.get(c);
            if (decoded != null) {
                bytes[byteCount++] = decoded.byteValue();
            }
        }

        return new String(bytes, 0, byteCount, java.nio.charset.StandardCharsets.UTF_8);
    }

    public int getEosTokenId() {
        return eosTokenId;
    }

    public int getVocabSize() {
        return vocab.size();
    }
}

