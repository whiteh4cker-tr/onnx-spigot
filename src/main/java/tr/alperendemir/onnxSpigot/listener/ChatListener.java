package tr.alperendemir.onnxSpigot.listener;

import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.player.AsyncPlayerChatEvent;
import tr.alperendemir.onnxSpigot.OnnxSpigot;
import tr.alperendemir.onnxSpigot.api.OnnxChatApi;
import tr.alperendemir.onnxSpigot.config.LLMConfig;

import java.util.concurrent.CompletionException;

public class ChatListener implements Listener {

    private final OnnxSpigot plugin;
    private final OnnxChatApi chatApi;
    private final LLMConfig config;

    public ChatListener(OnnxSpigot plugin, OnnxChatApi chatApi, LLMConfig config) {
        this.plugin = plugin;
        this.chatApi = chatApi;
        this.config = config;
    }

    @EventHandler(priority = EventPriority.NORMAL)
    public void onPlayerChat(AsyncPlayerChatEvent event) {
        String message = event.getMessage();
        String trigger = config.getChatTrigger();
        if (!message.startsWith(trigger)) {
            return;
        }

        event.setCancelled(true);
        Player player = event.getPlayer();

        if (!chatApi.canUse(player)) {
            sendOnMainThread(player, config.getNoPermissionMessage());
            return;
        }

        String prompt = message.substring(trigger.length()).trim();
        if (prompt.isEmpty()) {
            sendOnMainThread(player, config.getUsageMessage());
            return;
        }

        sendOnMainThread(player, config.getLoadingMessage());
        chatApi.chat(player, prompt)
                .thenAccept(response -> sendOnMainThread(player, config.getPrefix() + response))
                .exceptionally(error -> {
                    Throwable root = (error instanceof CompletionException && error.getCause() != null)
                            ? error.getCause()
                            : error;
                    if (root instanceof SecurityException) {
                        sendOnMainThread(player, config.getNoPermissionMessage());
                    } else if (root.getMessage() != null && root.getMessage().contains("Model not found")) {
                        sendOnMainThread(player, config.getModelNotFoundMessage());
                    } else {
                        plugin.getLogger().warning("Failed to generate chat response: " + root.getMessage());
                        sendOnMainThread(player, config.getErrorMessage());
                    }
                    return null;
                });
    }

    private void sendOnMainThread(Player player, String message) {
        plugin.getServer().getScheduler().runTask(plugin, () -> {
            if (player.isOnline()) {
                player.sendMessage(message);
            }
        });
    }
}

