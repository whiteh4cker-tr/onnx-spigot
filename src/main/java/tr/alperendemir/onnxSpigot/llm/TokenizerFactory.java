package tr.alperendemir.onnxSpigot.llm;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public final class TokenizerFactory {

    private TokenizerFactory() {
    }

    public static Tokenizer create(Path tokenizerDirectory) throws IOException {
        Path tokenizerJson = tokenizerDirectory.resolve("tokenizer.json");
        if (Files.exists(tokenizerJson)) {
            String content = Files.readString(tokenizerJson);
            JsonObject root = new Gson().fromJson(content, JsonObject.class);
            JsonObject model = root != null ? root.getAsJsonObject("model") : null;
            String modelType = model != null && model.has("type") ? model.get("type").getAsString() : "";

            if ("Unigram".equalsIgnoreCase(modelType)) {
                return new UnigramTokenizer(tokenizerDirectory);
            }
            return new BPETokenizer(tokenizerDirectory);
        }

        if (Files.exists(tokenizerDirectory.resolve("vocab.json"))) {
            return new BPETokenizer(tokenizerDirectory);
        }

        throw new IOException("No supported tokenizer files found in " + tokenizerDirectory +
                ". Expected tokenizer.json or vocab.json.");
    }
}

