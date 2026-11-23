using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Leonida.Core;

namespace Leonida.Launcher.Views;

/// <summary>HISTORY: every build the launcher ran, how long it took, and its log.</summary>
public partial class HistoryView : UserControl
{
    readonly AppState state = AppState.Instance;

    public event Action<BuildOptions>? ReuseRequested;

    public HistoryView()
    {
        InitializeComponent();
        Builds.Instance.Changed += Show;
        Loaded += (_, _) => Show();
        Unloaded += (_, _) => Builds.Instance.Changed -= Show;
    }

    sealed class Row(HistoryEntry e)
    {
        public HistoryEntry Entry { get; } = e;
        public DateTime Started => Entry.Started;
        public string Outcome => Loc.T(Entry.Outcome.ToString());
        public string DurationText => Format.Duration(TimeSpan.FromSeconds(Entry.Seconds));
        public string StagesText => Entry.StagesDone.Count > 0 ? string.Join(", ", Entry.StagesDone) + (Entry.FailedStage != null ? Loc.F(" (failed in {0})", Entry.FailedStage) : "")
            : Entry.FailedStage != null ? Loc.F("failed in {0}", Entry.FailedStage) : "—";
        public string ShadersText => Entry.Shaders is int t ? Loc.F("{0} of {1}", (Entry.ShadersValid ?? 0).ToString("N0"), t.ToString("N0")) : "—";
        public int Warnings => Entry.Warnings;
        public int Errors => Entry.Errors;
        public string MemoryText => Entry.PeakMemoryBytes > 0 ? Format.Bytes(Entry.PeakMemoryBytes) : "—";
    }

    sealed record Bar(double Height, Brush Fill, string Label, string Tip);

    public void Show()
    {
        var entries = state.History.Entries;
        List.ItemsSource = entries.Select(e => new Row(e)).ToList();
        NoHistory.Visibility = entries.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
        var last = entries.Take(20).Reverse().ToList();
        var max = Math.Max(1, last.Select(e => e.Seconds).DefaultIfEmpty(1).Max());
        Bars.ItemsSource = last.Select(e => new Bar(Math.Max(3, 104 * e.Seconds / max),
            (Brush)FindResource(e.Outcome == SessionOutcome.Succeeded ? "Purple" : e.Outcome == SessionOutcome.Failed ? "Bad" : "Faint"),
            e.Started.ToString("dd.MM HH:mm"),
            $"{e.Started:yyyy-MM-dd HH:mm} · {Loc.T(e.Outcome.ToString())} · {Format.Duration(TimeSpan.FromSeconds(e.Seconds))}" +
            (e.TranslateSeconds is double ts ? $"\nShader translation: {Format.Duration(TimeSpan.FromSeconds(ts))}" : ""))).ToList();
        var ok = entries.Where(e => e.Outcome == SessionOutcome.Succeeded).ToList();
        ChartCaption.Text = Loc.T("How long builds took (last 20)") +
                            (ok.Count > 0 ? Loc.F("  ·  {0} succeeded, fastest {1}", ok.Count, Format.Duration(TimeSpan.FromSeconds(ok.Min(e => e.Seconds)))) : "");
    }

    void OnSelected(object sender, SelectionChangedEventArgs e)
    {
        if (List.SelectedItem is not Row r) { DetailCard.Visibility = Visibility.Collapsed; return; }
        var h = r.Entry;
        DetailCard.Visibility = Visibility.Visible;
        Detail.Text = $"{h.CommandLine}\nexit code {h.ExitCode?.ToString() ?? "—"} · {h.Warnings} warnings · {h.Errors} errors" +
                      (h.FirstError != null ? $"\nfirst error: {h.FirstError}" : "");
    }

    void OnLog(object sender, RoutedEventArgs e)
    {
        if (List.SelectedItem is Row r && File.Exists(r.Entry.LogPath)) AppState.Open(r.Entry.LogPath);
        else _ = Popup.Show("Log not found", "That log file is gone.");
    }

    void OnReuse(object sender, RoutedEventArgs e)
    {
        if (List.SelectedItem is Row r) ReuseRequested?.Invoke(r.Entry.Options.Clone());
    }

    async void OnClear(object sender, RoutedEventArgs e)
    {
        if (!await Popup.Ask("Clear history", "Clear the build history? The log files stay.", "Clear history", danger: true)) return;
        state.History.Clear();
        Show();
    }
}
