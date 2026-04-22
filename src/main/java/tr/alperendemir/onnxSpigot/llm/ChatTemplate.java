package tr.alperendemir.onnxSpigot.llm;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.logging.Logger;

public class ChatTemplate {

    private final String templateText;
    private final boolean gemmaStyle;
    private final List<String> stopSequences;

    private ChatTemplate(String templateText, boolean gemmaStyle, List<String> stopSequences) {
        this.templateText = templateText;
        this.gemmaStyle = gemmaStyle;
        this.stopSequences = List.copyOf(stopSequences);
    }

    public static ChatTemplate defaultTemplate() {
        return new ChatTemplate("", false, List.of("\nMessage:", "\nAssistant:"));
    }

    public static ChatTemplate load(Path modelPath, String configuredTokenizerPath, Logger logger) {
        Path tokenizerDir = resolveTokenizerDirectory(modelPath, configuredTokenizerPath);
        String chatTemplate = readChatTemplate(tokenizerDir);
        if (chatTemplate == null || chatTemplate.isBlank()) {
            return defaultTemplate();
        }

        String lower = chatTemplate.toLowerCase(Locale.ROOT);
        boolean gemma = lower.contains("<start_of_turn>") && lower.contains("<end_of_turn>");

        List<String> stops = new ArrayList<>();
        if (gemma) {
            stops.add("<end_of_turn>");
            stops.add("<start_of_turn>");
        }
        stops.add("\nMessage:");
        stops.add("\nAssistant:");

        return new ChatTemplate(chatTemplate, gemma, stops);
    }

    public String formatUserPrompt(String message) {
        if (gemmaStyle) {
            return "<bos><start_of_turn>user\n" + message + "<end_of_turn>\n<start_of_turn>model\n";
        }

        return "You are a helpful assistant running inside a Minecraft server. " +
                "Keep responses concise and relevant.\n" +
                "Message: " + message + "\nAssistant:";
    }

    public String trimAtStop(String response) {
        String out = response;
        int cutAt = -1;
        for (String stop : stopSequences) {
            int idx = out.indexOf(stop);
            if (idx >= 0 && (cutAt == -1 || idx < cutAt)) {
                cutAt = idx;
            }
        }

        if (cutAt >= 0) {
            out = out.substring(0, cutAt);
        }

        return out.trim();
    }

    public String getTemplateText() {
        return templateText;
    }

    public boolean isGemmaStyle() {
        return gemmaStyle;
    }

    private static Path resolveTokenizerDirectory(Path modelPath, String configuredTokenizerPath) {
        if (configuredTokenizerPath != null && !configuredTokenizerPath.isBlank()) {
            Path tokenizerPath = Paths.get(configuredTokenizerPath);
            if (!tokenizerPath.isAbsolute()) {
                tokenizerPath = modelPath.resolve(tokenizerPath).normalize();
            }
            return tokenizerPath;
        }
        return modelPath;
    }

    private static String readChatTemplate(Path tokenizerDir) {
        try {
            Path configPath = tokenizerDir.resolve("tokenizer_config.json");
            if (Files.exists(configPath)) {
                JsonObject root = new Gson().fromJson(Files.readString(configPath), JsonObject.class);
                if (root != null && root.has("chat_template") && !root.get("chat_template").isJsonNull()) {
                    return root.get("chat_template").getAsString();
                }
            }

            Path tokenizerJsonPath = tokenizerDir.resolve("tokenizer.json");
            if (Files.exists(tokenizerJsonPath)) {
                JsonObject root = new Gson().fromJson(Files.readString(tokenizerJsonPath), JsonObject.class);
                if (root != null && root.has("chat_template") && !root.get("chat_template").isJsonNull()) {
                    return root.get("chat_template").getAsString();
                }
            }
        } catch (IOException ignored) {
        }

        return "";
    }
}

