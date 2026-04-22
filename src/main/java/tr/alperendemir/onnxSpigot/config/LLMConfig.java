package tr.alperendemir.onnxSpigot.config;

import org.bukkit.ChatColor;
import org.bukkit.configuration.file.FileConfiguration;

public class LLMConfig {

    private final String engine;
    private final String modelName;
    private final int maxTokens;
    private final double temperature;
    private final double topP;
    private final boolean preload;
    private final int timeout;
    private final int autoUnloadTimeout;

    private final String permission;
    private final String chatTrigger;

    private final String prefix;
    private final String loadingMessage;
    private final String notLoadedMessage;
    private final String errorMessage;
    private final String usageMessage;
    private final String modelNotFoundMessage;
    private final String noPermissionMessage;

    public LLMConfig(FileConfiguration config) {
        this.engine = config.getString("llm.engine", "onnx");
        this.modelName = config.getString("llm.model", "Qwen3-0.6B-ONNX");
        this.maxTokens = config.getInt("llm.max-tokens", 128);
        this.temperature = config.getDouble("llm.temperature", 0.7);
        this.topP = config.getDouble("llm.top-p", 0.9);
        this.preload = config.getBoolean("llm.preload", false);
        this.timeout = config.getInt("llm.timeout", 120);
        this.autoUnloadTimeout = config.getInt("llm.auto-unload-timeout", 15);

        this.permission = config.getString("chat.permission", "onnxspigot.llm.use");
        this.chatTrigger = config.getString("chat.trigger", "@llm");

        this.prefix = translateColors(config.getString("messages.prefix", "&b[LLM] &f"));
        this.loadingMessage = translateColors(config.getString("messages.loading", "&7Loading ONNX model, please wait..."));
        this.notLoadedMessage = translateColors(config.getString("messages.not-loaded", "&7Model is still loading, please wait..."));
        this.errorMessage = translateColors(config.getString("messages.error", "&cAn error occurred while processing your request."));
        this.usageMessage = translateColors(config.getString("messages.usage", "&eUsage: @llm <message>"));
        this.modelNotFoundMessage = translateColors(config.getString("messages.model-not-found", "&cModel not found. Place it under plugins/onnx-spigot/models/"));
        this.noPermissionMessage = translateColors(config.getString("messages.no-permission", "&cYou do not have permission to use LLM chat."));
    }

    private String translateColors(String message) {
        return ChatColor.translateAlternateColorCodes('&', message);
    }

    public String getEngine() {
        return engine;
    }

    public String getModelName() {
        return modelName;
    }

    public int getMaxTokens() {
        return maxTokens;
    }

    public double getTemperature() {
        return temperature;
    }

    public double getTopP() {
        return topP;
    }

    public boolean isPreload() {
        return preload;
    }

    public int getTimeout() {
        return timeout;
    }

    public int getAutoUnloadTimeout() {
        return autoUnloadTimeout;
    }

    public String getPermission() {
        return permission;
    }

    public String getChatTrigger() {
        return chatTrigger;
    }

    public String getPrefix() {
        return prefix;
    }

    public String getLoadingMessage() {
        return loadingMessage;
    }

    public String getNotLoadedMessage() {
        return notLoadedMessage;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public String getUsageMessage() {
        return usageMessage;
    }

    public String getModelNotFoundMessage() {
        return modelNotFoundMessage;
    }

    public String getNoPermissionMessage() {
        return noPermissionMessage;
    }
}

