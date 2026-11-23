using System.Text.Json;
using System.Text.Json.Serialization;

namespace Leonida.Core;

/// <summary>Where the launcher keeps its own files. Nothing goes into the checkout or the build's folders.</summary>
public static class AppPaths
{
    public static string Roaming { get; set; } =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Leonida", "Launcher");
    public static string Local { get; set; } =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Leonida", "Launcher");

    public static string Settings => Path.Combine(Roaming, "settings.json");
    public static string History => Path.Combine(Roaming, "history.json");
    public static string Logs => Path.Combine(Local, "logs");
    public static string Progress => Path.Combine(Local, "progress");
    public static string WebView => Path.Combine(Local, "webview");
}

/// <summary>How Play starts the port: gta6rt.exe --run, with the options runtime/src/main.cpp documents.</summary>
public sealed class PlayOptions
{
    public string? Overlay { get; set; }          // --overlay DIR (default <game>\overlay)
    public string? Mods { get; set; }             // --mods DIR    (default <folder of gta6rt.exe>\mods)
    public bool TraceD3D12 { get; set; }          // --trace-d3d12
    public string? TraceFile { get; set; }        // --trace FILE (must not exist yet)
    public bool VulkanBackend { get; set; } = true;   // LEONIDA_D3D12=vulkan
    public string? GameArguments { get; set; }    // after "--"
}

public sealed class Preset
{
    public string Name { get; set; } = "";
    public BuildOptions Options { get; set; } = new();
}

public sealed class LauncherSettings
{
    public int Version { get; set; } = 1;
    public string? RepoRoot { get; set; }
    public string? PythonOverride { get; set; }
    public BuildOptions Build { get; set; } = new();
    public PlayOptions Play { get; set; } = new();
    /// <summary>Extra environment for builds and Play (LLVM_MC, VULKAN_SDK, ...); an empty value is left out.</summary>
    public Dictionary<string, string> Environment { get; set; } = [];
    public List<Preset> Presets { get; set; } = [];
    public List<string> RecentOutputs { get; set; } = [];
    public string LastPage { get; set; } = "home";
    public string LastLeonidaSection { get; set; } = "build";
    public bool SidebarCollapsed { get; set; }
    public bool AutoScroll { get; set; } = true;
    public bool ConsoleExpanded { get; set; }
    public bool StopGracefully { get; set; } = true;
    public int StopGraceSeconds { get; set; } = 30;
    public bool NotifyWhenDone { get; set; } = true;
    public double? WindowLeft { get; set; }
    public double? WindowTop { get; set; }
    public double? WindowWidth { get; set; }
    public double? WindowHeight { get; set; }
    public bool WindowMaximized { get; set; }
    public HardwareOverride? Hardware { get; set; }

    // Launcher settings
    /// <summary>"system" (Windows' app theme), "light" or "dark".</summary>
    public string Theme { get; set; } = "system";
    /// <summary>A language code: "en", or a file in Languages (pl, ru, …).</summary>
    public string Language { get; set; } = "en";
    /// <summary>"last" (where you left off), "home", "leonida" or "caniru".</summary>
    public string StartPage { get; set; } = "last";
    public bool StartCollapsed { get; set; }
    /// <summary>Credits from the GitHub API (names, commit counts, profile pictures); off, the checkout's git history.</summary>
    public bool AllowGitHub { get; set; } = true;
    /// <summary>The Can I Run checker web page; off, the page shows only this PC's hardware.</summary>
    public bool AllowChecker { get; set; } = true;
    /// <summary>1.0.0 and 1.0.1 had one switch for both; a settings file with it off turns both off. Never written.</summary>
    [System.Text.Json.Serialization.JsonIgnore(Condition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull)]
    public bool? AllowOnline
    {
        get => null;
        set { if (value == false) { AllowGitHub = false; AllowChecker = false; } }
    }
    public bool SoundWhenDone { get; set; }
    /// <summary>A Windows notification when a build ends while the launcher is not in front.</summary>
    public bool ToastWhenDone { get; set; } = true;
    /// <summary>Closing the window during a build (or while the port runs) keeps it going from the tray.</summary>
    public bool CloseToTray { get; set; } = true;
    /// <summary>Set once the tray has explained itself, so the notice shows only the first time.</summary>
    public bool TrayExplained { get; set; }
    // Playtime: PLAY sessions of the port (dry runs don't count).
    public double PlaySeconds { get; set; }
    public int PlaySessions { get; set; }
    public DateTime? LastPlayed { get; set; }
    public double LastPlaySeconds { get; set; }
    public double ConsoleFontSize { get; set; } = 12.5;
    /// <summary>Build logs older than this many days are deleted when the launcher starts; 0 keeps them all.</summary>
    public int KeepLogsDays { get; set; } = 30;
    /// <summary>Picks the avatar's pattern and colours.</summary>
    public int AvatarSeed { get; set; } = Random.Shared.Next();
    /// <summary>The GPU tests page's test-program folder (the one with vk_compute_test.exe), when not found by itself.</summary>
    public string? GpuTestPrograms { get; set; }
    /// <summary>Show the picture the user chose (a square copy in the settings folder) instead of the random one.</summary>
    public bool AvatarCustom { get; set; }
}

/// <summary>What the user typed over the detected hardware on the Can I Run page (null: use the detection).</summary>
public sealed class HardwareOverride
{
    public string? Cpu { get; set; }
    public string? Gpu { get; set; }
    public double? RamGb { get; set; }
    public string? RamType { get; set; }
}

public static class Json
{
    public static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter() },
    };

    /// <summary>Reads a JSON file; a missing or damaged one gives a fresh object (a damaged one is kept aside).</summary>
    public static T Load<T>(string path) where T : new()
    {
        try
        {
            if (!File.Exists(path)) return new T();
            return JsonSerializer.Deserialize<T>(File.ReadAllText(path), Options) ?? new T();
        }
        catch (Exception e) when (e is JsonException or IOException or NotSupportedException)
        {
            try { File.Copy(path, path + ".damaged", true); } catch { }
            return new T();
        }
    }

    /// <summary>Writes beside the file and renames over it, so a crash mid-write never leaves half a file.</summary>
    public static void Save<T>(string path, T value)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var tmp = path + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(value, Options));
        File.Move(tmp, path, true);
    }
}

public sealed class SettingsStore
{
    public LauncherSettings Current { get; private set; } = Json.Load<LauncherSettings>(AppPaths.Settings);

    public void Save() => Json.Save(AppPaths.Settings, Current);

    /// <summary>Back to the defaults; the repo location is kept, it is not a preference.</summary>
    public void Reset()
    {
        var repo = Current.RepoRoot;
        Current = new LauncherSettings { RepoRoot = repo };
        Save();
    }

    public void RememberOutput(string? folder)
    {
        if (string.IsNullOrWhiteSpace(folder)) return;
        Current.RecentOutputs.RemoveAll(f => string.Equals(f, folder, StringComparison.OrdinalIgnoreCase));
        Current.RecentOutputs.Insert(0, folder);
        if (Current.RecentOutputs.Count > 8) Current.RecentOutputs.RemoveRange(8, Current.RecentOutputs.Count - 8);
    }
}

public sealed class HistoryEntry
{
    public string Id { get; set; } = "";
    public DateTime Started { get; set; }
    public DateTime? Ended { get; set; }
    public double Seconds { get; set; }
    public SessionOutcome Outcome { get; set; }
    public int? ExitCode { get; set; }
    public string CommandLine { get; set; } = "";
    public BuildOptions Options { get; set; } = new();
    public List<string> StagesDone { get; set; } = [];
    public string? FailedStage { get; set; }
    public int Warnings { get; set; }
    public int Errors { get; set; }
    public string? FirstError { get; set; }
    public int? Shaders { get; set; }
    public int? ShadersTranslated { get; set; }
    public int? ShadersValid { get; set; }
    public double? TranslateSeconds { get; set; }
    public long PeakMemoryBytes { get; set; }
    public string LogPath { get; set; } = "";
    public string? OutFolder { get; set; }
}

public sealed class HistoryStore
{
    const int Keep = 200;
    public List<HistoryEntry> Entries { get; } = Json.Load<List<HistoryEntry>>(AppPaths.History);

    public void Add(HistoryEntry e)
    {
        Entries.Insert(0, e);
        if (Entries.Count > Keep) Entries.RemoveRange(Keep, Entries.Count - Keep);
        Json.Save(AppPaths.History, Entries);
    }

    public void Clear()
    {
        Entries.Clear();
        Json.Save(AppPaths.History, Entries);
    }
}
