package tr.alperendemir.onnxSpigot.api;

import org.bukkit.entity.Player;

import java.util.concurrent.CompletableFuture;

public interface OnnxChatApi {

    String getPermissionNode();

    boolean canUse(Player player);

    CompletableFuture<String> generate(String prompt);

    CompletableFuture<String> chat(Player player, String message);

    CompletableFuture<Void> loadModel();

    boolean isModelLoaded();

    String getModelsDirectory();
}

