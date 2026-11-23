using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;
using Leonida.Core;

namespace Leonida.Launcher;

public abstract class Observable : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler? PropertyChanged;
    protected void Raise([CallerMemberName] string? name = null) => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    protected bool Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value)) return false;
        field = value;
        Raise(name);
        return true;
    }
}

/// <summary>What every page shares: settings, history, the checkout, Python, the probe's answers and the build
/// that is running (at most one).</summary>
public sealed class AppState : Observable
{
    public static AppState Instance { get; } = new();

    public SettingsStore Settings { get; } = new();
    public HistoryStore History { get; } = new();

    string? repo;
    PythonInfo? python;
    ProbeResult? probe;
    string? probeError;
    BuildSession? session;

    public string? Repo { get => repo; set { if (Set(ref repo, value)) Raise(nameof(HasRepo)); } }
    public bool HasRepo => Repo != null;
    public PythonInfo? Python { get => python; set => Set(ref python, value); }
    public ProbeResult? Probe { get => probe; set => Set(ref probe, value); }
    public string? ProbeError { get => probeError; set => Set(ref probeError, value); }
    public BuildSession? Session { get => session; set { if (Set(ref session, value)) Raise(nameof(IsBuilding)); } }
    public bool IsBuilding => Session?.IsRunning == true;

    public event Action? BuildFinished;
    public void NotifyBuildFinished() { Raise(nameof(IsBuilding)); BuildFinished?.Invoke(); }

    /// <summary>Finds the checkout and Python, then asks the probe; safe to call again after a settings change.</summary>
    public async Task InitializeAsync()
    {
        Settings.Current.Hardware = null;   // Can I Run compares only what it detects
        Repo = RepoLocator.Find(Settings.Current.RepoRoot);
        Python = await Task.Run(() => PythonLocator.Find(Settings.Current.PythonOverride));
        await RefreshProbeAsync();
    }

    public async Task RefreshProbeAsync(params string[] flags)
    {
        ProbeError = null;
        if (Repo == null) { ProbeError = "Choose the Leonida folder (the one with Build.bat) in Settings."; return; }
        if (Python == null) { ProbeError = "Python 3.12 or newer was not found. Install it from python.org and tick \"Add python.exe to PATH\"."; return; }
        try { Probe = await Core.Probe.RunAsync(Python, Repo, flags, BuildEnvironment()); }
        catch (Exception e) { ProbeError = e.Message; }
    }

    /// <summary>The environment variables set in Settings; an empty value leaves the variable as Windows has it.</summary>
    public Dictionary<string, string> BuildEnvironment() =>
        Settings.Current.Environment.Where(kv => !string.IsNullOrWhiteSpace(kv.Key) && !string.IsNullOrWhiteSpace(kv.Value))
                .ToDictionary(kv => kv.Key.Trim(), kv => kv.Value.Trim());

    /// <summary>Python's pipe encoding as the probe reported it (the ANSI code page on most PCs), else that code page.</summary>
    public Encoding OutputEncoding()
    {
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        var name = Probe?.StdoutEncoding;
        try
        {
            if (!string.IsNullOrEmpty(name))
                return name.Equals("utf-8", StringComparison.OrdinalIgnoreCase) ? new UTF8Encoding(false) : Encoding.GetEncoding(name);
        }
        catch (ArgumentException) { }
        return Encoding.GetEncoding(System.Globalization.CultureInfo.CurrentCulture.TextInfo.ANSICodePage);
    }

    public string OutFolder(BuildOptions o) =>
        !string.IsNullOrWhiteSpace(o.Out) ? Path.GetFullPath(o.Out) :
        Probe?.DefaultOut ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "Leonida");

    public string WorkFolder(BuildOptions o) =>
        !string.IsNullOrWhiteSpace(o.Work) ? Path.GetFullPath(o.Work) : Path.Combine(OutFolder(o), "work");

    public static void Open(string target)
    {
        try { Process.Start(new ProcessStartInfo(target) { UseShellExecute = true }); }
        catch (Exception e) { System.Windows.MessageBox.Show($"Could not open {target}:\n{e.Message}", "Leonida Launcher"); }
    }
}
