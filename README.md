# onnx-spigot

A Spigot plugin that runs local ONNX LLM models, provides in-game chat via a trigger, and exposes a Bukkit service API for dependent plugins.

## Features

- Local ONNX inference from `plugins/onnx-spigot/models/`
- In-game chat trigger (default: `@llm <message>`)
- Single permission gate: `onnxspigot.llm.use`
- Public Bukkit API registered through `ServicesManager`

## Model Folder

Place your model files in:

`plugins/onnx-spigot/models/<model-name>/`

Required files:

- `*.onnx` (for example `model.onnx`)
- Tokenizer data:
  - `tokenizer.json`, or
  - `vocab.json`

Optional files:

- `merges.txt` (used for better BPE merges when `vocab.json` is used)
- `tokenizer_config.json` (special token IDs)
- `config.json` (model metadata)

Set `<model-name>` in `plugins/onnx-spigot/config.yml` under `llm.model`.

## Model Support

- Supported: Qwen 2 ONNX models
- Supported: Qwen 3 ONNX models
- Not supported: Gemma models

## Qwen3-0.6B-ONNX (Confirmed Working)

1. Download the model from: https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX
2. Create a folder at `plugins/onnx-spigot/models/Qwen3-0.6B-ONNX/`.
3. Copy the model files into that folder:
   - `*.onnx` (for example `model.onnx`)
   - `tokenizer.json` or `vocab.json`
   - Optional: `merges.txt`, `tokenizer_config.json`, `config.json`
4. Set this in `plugins/onnx-spigot/config.yml`:

   ```yaml
   llm:
     model: "Qwen3-0.6B-ONNX"
   ```

## API Usage (from another plugin)

```java
import org.bukkit.Bukkit;
import org.bukkit.plugin.RegisteredServiceProvider;
import tr.alperendemir.onnxSpigot.api.OnnxChatApi;

RegisteredServiceProvider<OnnxChatApi> registration =
        Bukkit.getServicesManager().getRegistration(OnnxChatApi.class);
if (registration != null) {
    OnnxChatApi api = registration.getProvider();
    api.generate("Explain redstone clocks in one paragraph.")
            .thenAccept(response -> Bukkit.getLogger().info(response));
}
```

## Build

```powershell
./gradlew.bat clean build
```
