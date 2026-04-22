package tr.alperendemir.onnxSpigot.llm;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class UnigramTokenizer implements Tokenizer {

    private static final double NEG_INF = -1e30;
    private static final String WORD_START = "\u2581";

    private final Map<String, Integer> tokenToId = new HashMap<>();
    private final Map<Integer, String> idToToken = new HashMap<>();
    private final Map<String, Double> tokenScore = new HashMap<>();
    private final Map<Character, List<String>> candidatesByFirstChar = new HashMap<>();

    private int unkTokenId = -1;
    private int eosTokenId = -1;

    public UnigramTokenizer(Path tokenizerDirectory) throws IOException {
        loadTokenizerJson(tokenizerDirectory.resolve("tokenizer.json"));
        loadSpecialTokens(tokenizerDirectory);
        buildCandidateIndex();
    }

    private void loadTokenizerJson(Path tokenizerJsonPath) throws IOException {
        if (!Files.exists(tokenizerJsonPath)) {
            throw new IOException("tokenizer.json not found in " + tokenizerJsonPath.getParent());
        }

        String content = Files.readString(tokenizerJsonPath);
        JsonObject root = new Gson().fromJson(content, JsonObject.class);
        JsonObject model = root.getAsJsonObject("model");
        if (model == null) {
            throw new IOException("tokenizer.json missing model object");
        }

        JsonArray vocabArray = model.getAsJsonArray("vocab");
        if (vocabArray == null) {
            throw new IOException("tokenizer.json model.vocab is missing for unigram tokenizer");
        }

        int idx = 0;
        for (JsonElement element : vocabArray) {
            if (!element.isJsonArray()) {
                continue;
            }
            JsonArray pair = element.getAsJsonArray();
            if (pair.size() < 2) {
                continue;
            }

            String token = pair.get(0).getAsString();
            double score = pair.get(1).getAsDouble();
            tokenToId.put(token, idx);
            idToToken.put(idx, token);
            tokenScore.put(token, score);
            idx++;
        }

        if (model.has("unk_id") && !model.get("unk_id").isJsonNull()) {
            unkTokenId = model.get("unk_id").getAsInt();
        } else {
            unkTokenId = tokenToId.getOrDefault("<unk>", -1);
        }
    }

    private void loadSpecialTokens(Path tokenizerDirectory) throws IOException {
        Path configPath = tokenizerDirectory.resolve("tokenizer_config.json");
        if (!Files.exists(configPath)) {
            eosTokenId = fallbackEosTokenId();
            return;
        }

        String content = Files.readString(configPath);
        JsonObject config = new Gson().fromJson(content, JsonObject.class);

        if (config.has("eos_token")) {
            String eosToken = extractTokenString(config.get("eos_token"));
            eosTokenId = tokenToId.getOrDefault(eosToken, -1);
        }

        if (eosTokenId == -1) {
            eosTokenId = fallbackEosTokenId();
        }
    }

    private int fallbackEosTokenId() {
        return tokenToId.getOrDefault("<eos>", tokenToId.getOrDefault("</s>", tokenToId.getOrDefault("<|endoftext|>", -1)));
    }

    private String extractTokenString(JsonElement element) {
        if (element == null || element.isJsonNull()) {
            return "";
        }
        if (element.isJsonPrimitive()) {
            return element.getAsString();
        }
        if (element.isJsonObject()) {
            JsonObject obj = element.getAsJsonObject();
            if (obj.has("content") && obj.get("content").isJsonPrimitive()) {
                return obj.get("content").getAsString();
            }
        }
        return "";
    }

    private void buildCandidateIndex() {
        for (String token : tokenToId.keySet()) {
            if (token == null || token.isEmpty() || token.startsWith("<")) {
                continue;
            }
            char first = token.charAt(0);
            candidatesByFirstChar.computeIfAbsent(first, key -> new ArrayList<>()).add(token);
        }

        for (List<String> tokens : candidatesByFirstChar.values()) {
            tokens.sort((a, b) -> Integer.compare(b.length(), a.length()));
        }
    }

    @Override
    public long[] encode(String text) {
        if (text == null || text.isEmpty()) {
            return new long[0];
        }

        String normalized = WORD_START + text.replace("\n", "\n" + WORD_START).replace(" ", WORD_START);
        int n = normalized.length();

        double[] bestScore = new double[n + 1];
        int[] prevPos = new int[n + 1];
        int[] prevTokenId = new int[n + 1];

        for (int i = 0; i <= n; i++) {
            bestScore[i] = NEG_INF;
            prevPos[i] = -1;
            prevTokenId[i] = -1;
        }
        bestScore[0] = 0.0;

        for (int i = 0; i < n; i++) {
            if (bestScore[i] == NEG_INF) {
                continue;
            }

            char ch = normalized.charAt(i);
            List<String> candidates = candidatesByFirstChar.getOrDefault(ch, Collections.emptyList());
            for (String token : candidates) {
                if (!normalized.startsWith(token, i)) {
                    continue;
                }

                int next = i + token.length();
                int tokenId = tokenToId.get(token);
                double score = bestScore[i] + tokenScore.getOrDefault(token, -10.0);
                if (score > bestScore[next]) {
                    bestScore[next] = score;
                    prevPos[next] = i;
                    prevTokenId[next] = tokenId;
                }
            }

            if (unkTokenId >= 0) {
                int next = i + 1;
                if (next <= n) {
                    double score = bestScore[i] - 20.0;
                    if (score > bestScore[next]) {
                        bestScore[next] = score;
                        prevPos[next] = i;
                        prevTokenId[next] = unkTokenId;
                    }
                }
            }
        }

        if (prevPos[n] == -1) {
            return new long[0];
        }

        List<Long> ids = new ArrayList<>();
        int pos = n;
        while (pos > 0) {
            int tokenId = prevTokenId[pos];
            int from = prevPos[pos];
            if (tokenId < 0 || from < 0) {
                break;
            }
            ids.add((long) tokenId);
            pos = from;
        }

        Collections.reverse(ids);
        long[] result = new long[ids.size()];
        for (int i = 0; i < ids.size(); i++) {
            result[i] = ids.get(i);
        }
        return result;
    }

    @Override
    public String decode(long[] tokenIds) {
        StringBuilder out = new StringBuilder();
        for (long tokenId : tokenIds) {
            String token = idToToken.get((int) tokenId);
            if (token == null) {
                continue;
            }
            if (token.startsWith("<") && token.endsWith(">") && !token.toLowerCase(Locale.ROOT).startsWith("<0x")) {
                continue;
            }
            out.append(decodePiece(token));
        }

        String text = out.toString().replace(WORD_START, " ");
        return text.stripLeading();
    }

    private String decodePiece(String token) {
        if (token.length() == 6 && token.startsWith("<0x") && token.endsWith(">")) {
            try {
                int value = Integer.parseInt(token.substring(3, 5), 16);
                return String.valueOf((char) value);
            } catch (NumberFormatException ignored) {
            }
        }
        return token;
    }

    @Override
    public int getEosTokenId() {
        return eosTokenId;
    }

    @Override
    public int getVocabSize() {
        return tokenToId.size();
    }
}

