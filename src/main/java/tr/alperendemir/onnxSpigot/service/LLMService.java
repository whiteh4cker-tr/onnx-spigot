package tr.alperendemir.onnxSpigot.service;

import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitTask;
import tr.alperendemir.onnxSpigot.config.LLMConfig;
import tr.alperendemir.onnxSpigot.llm.LLMProvider;
import tr.alperendemir.onnxSpigot.llm.ONNXProvider;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Level;

public class LLMService {

    private final JavaPlugin plugin;
    private final LLMConfig config;
    private final LLMProvider provider;
    private final AtomicLong lastUsedTime;

    private volatile CompletableFuture<Void> loadFuture;
    private BukkitTask autoUnloadTask;

    public LLMService(JavaPlugin plugin, LLMConfig config) {
        this.plugin = plugin;
        this.config = config;
        this.provider = new ONNXProvider(config, plugin.getLogger(), plugin.getDataFolder().toPath());
        this.lastUsedTime = new AtomicLong(System.currentTimeMillis());
    }

    public void initialize() {
        if (config.isPreload()) {
            loadModel();
        }
        startAutoUnloadChecker();
    }

    private void startAutoUnloadChecker() {
        int timeoutMinutes = config.getAutoUnloadTimeout();
        if (timeoutMinutes <= 0) {
            return;
        }

        long checkIntervalTicks = 20L * 60;
        autoUnloadTask = plugin.getServer().getScheduler().runTaskTimerAsynchronously(plugin, () -> {
            if (!provider.isLoaded()) {
                return;
            }

            long idleTime = System.currentTimeMillis() - lastUsedTime.get();
            long timeoutMillis = timeoutMinutes * 60L * 1000L;
            if (idleTime >= timeoutMillis) {
                plugin.getLogger().info("Auto-unloading ONNX model due to inactivity...");
                provider.unload();
            }
        }, checkIntervalTicks, checkIntervalTicks);
    }

    public synchronized CompletableFuture<Void> loadModel() {
        if (provider.isLoaded()) {
            touch();
            return CompletableFuture.completedFuture(null);
        }

        if (loadFuture != null && !loadFuture.isDone()) {
            return loadFuture;
        }

        loadFuture = provider.loadModel()
                .orTimeout(config.getTimeout(), TimeUnit.SECONDS)
                .whenComplete((unused, error) -> {
                    if (error != null) {
                        plugin.getLogger().log(Level.SEVERE, "Failed to load ONNX model", error);
                    } else {
                        touch();
                    }
                });

        return loadFuture;
    }

    public CompletableFuture<String> generateResponse(String prompt) {
        touch();
        return loadModel().thenCompose(unused -> provider.generateResponse(prompt))
                .orTimeout(config.getTimeout(), TimeUnit.SECONDS)
                .whenComplete((unused, error) -> touch());
    }

    public boolean isModelLoaded() {
        return provider.isLoaded();
    }

    public PathInfo getPathInfo() {
        if (provider instanceof ONNXProvider onnxProvider) {
            return new PathInfo(onnxProvider.getModelsDirectory().toString());
        }
        return new PathInfo(plugin.getDataFolder().toPath().resolve("models").toString());
    }

    public void shutdown() {
        if (autoUnloadTask != null) {
            autoUnloadTask.cancel();
        }
        provider.unload();
    }

    private void touch() {
        lastUsedTime.set(System.currentTimeMillis());
    }

    public record PathInfo(String modelsDirectory) {
    }
}

