package tr.alperendemir.onnxSpigot.api;

import org.bukkit.entity.Player;
import tr.alperendemir.onnxSpigot.config.LLMConfig;
import tr.alperendemir.onnxSpigot.service.LLMService;

import java.util.concurrent.CompletableFuture;

public class OnnxChatApiImpl implements OnnxChatApi {

    private final LLMService llmService;
    private final LLMConfig config;

    public OnnxChatApiImpl(LLMService llmService, LLMConfig config) {
        this.llmService = llmService;
        this.config = config;
    }

    @Override
    public String getPermissionNode() {
        return config.getPermission();
    }

    @Override
    public boolean canUse(Player player) {
        return player.hasPermission(config.getPermission());
    }

    @Override
    public CompletableFuture<String> generate(String prompt) {
        return llmService.generateResponse(prompt);
    }

    @Override
    public CompletableFuture<String> chat(Player player, String message) {
        if (!canUse(player)) {
            return CompletableFuture.failedFuture(new SecurityException("Player lacks permission: " + config.getPermission()));
        }

        String prompt = "You are a helpful assistant running inside a Minecraft server. " +
                "Keep responses concise and relevant.\n" +
                "Player " + player.getName() + ": " + message + "\nAssistant:";

        return generate(prompt);
    }

    @Override
    public CompletableFuture<Void> loadModel() {
        return llmService.loadModel();
    }

    @Override
    public boolean isModelLoaded() {
        return llmService.isModelLoaded();
    }

    @Override
    public String getModelsDirectory() {
        return llmService.getPathInfo().modelsDirectory();
    }
}

