package tr.alperendemir.onnxSpigot;

import org.bukkit.plugin.java.JavaPlugin;
import tr.alperendemir.onnxSpigot.api.OnnxChatApi;
import tr.alperendemir.onnxSpigot.api.OnnxChatApiImpl;
import tr.alperendemir.onnxSpigot.config.LLMConfig;
import tr.alperendemir.onnxSpigot.listener.ChatListener;
import tr.alperendemir.onnxSpigot.service.LLMService;

public final class OnnxSpigot extends JavaPlugin {

    private LLMService llmService;
    private OnnxChatApi chatApi;

    @Override
    public void onEnable() {
        saveDefaultConfig();

        LLMConfig llmConfig = new LLMConfig(getConfig());
        llmService = new LLMService(this, llmConfig);
        llmService.initialize();

        chatApi = new OnnxChatApiImpl(llmService, llmConfig);
        getServer().getServicesManager().register(OnnxChatApi.class, chatApi, this, org.bukkit.plugin.ServicePriority.Normal);

        getServer().getPluginManager().registerEvents(new ChatListener(this, chatApi, llmConfig), this);

        getLogger().info("ONNX inference plugin enabled. Model: " + llmConfig.getModelName());
    }

    @Override
    public void onDisable() {
        if (llmService != null) {
            llmService.shutdown();
        }
        getServer().getServicesManager().unregisterAll(this);
    }

    public OnnxChatApi getChatApi() {
        return chatApi;
    }
}
