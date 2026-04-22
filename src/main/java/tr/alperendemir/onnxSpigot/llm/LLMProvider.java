package tr.alperendemir.onnxSpigot.llm;

import java.util.concurrent.CompletableFuture;

public interface LLMProvider {

    CompletableFuture<Void> loadModel();

    CompletableFuture<String> generateResponse(String prompt);

    boolean isLoaded();

    void unload();

    String getName();
}

