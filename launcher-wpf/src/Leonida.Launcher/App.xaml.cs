using System.IO;
using System.Windows;
using System.Windows.Threading;
using Leonida.Core;

namespace Leonida.Launcher;

public partial class App : Application
{
    Mutex? single;

    protected override void OnStartup(StartupEventArgs e)
    {
        // One launcher at a time: two would start two builds into the same work folder.
        single = new Mutex(true, @"Local\LeonidaLauncher", out var first);
        if (!first)
        {
            // Already open (maybe only in the tray): bring that one forward instead.
            try { using var show = EventWaitHandle.OpenExisting(ShowSignal); show.Set(); }
            catch (Exception) { MessageBox.Show(Loc.T("Leonida Launcher is already open."), "Leonida Launcher", MessageBoxButton.OK, MessageBoxImage.Information); }
            Shutdown();
            return;
        }
        ListenForShow();
        DispatcherUnhandledException += OnUnhandled;
        AppDomain.CurrentDomain.UnhandledException += (_, a) => Report(a.ExceptionObject as Exception);
        TaskScheduler.UnobservedTaskException += (_, a) => { Report(a.Exception); a.SetObserved(); };
        base.OnStartup(e);
        var settings = AppState.Instance.Settings.Current;
        Loc.Language = settings.Language;
        Theme.Apply(settings.Theme);
        DeleteOldLogs(settings.KeepLogsDays);
        new MainWindow().Show();
    }

    const string ShowSignal = @"Local\LeonidaLauncherShow";

    /// <summary>A second Leonida.exe signals this; the window comes back (from the tray too).</summary>
    void ListenForShow()
    {
        var signal = new EventWaitHandle(false, EventResetMode.AutoReset, ShowSignal);
        new Thread(() =>
        {
            while (signal.WaitOne())
                Dispatcher.BeginInvoke(() =>
                {
                    if (MainWindow is not Window w) return;
                    w.Show();
                    if (w.WindowState == WindowState.Minimized) w.WindowState = WindowState.Normal;
                    w.Activate();
                    Tray.Instance.Unpin();
                });
        }) { IsBackground = true, Name = "show signal" }.Start();
    }

    /// <summary>Build logs older than the setting (0: keep all). The launcher's own error log stays.</summary>
    static void DeleteOldLogs(int days)
    {
        if (days <= 0 || !Directory.Exists(AppPaths.Logs)) return;
        foreach (var f in Directory.EnumerateFiles(AppPaths.Logs, "*.log"))
        {
            try
            {
                if (Path.GetFileName(f) != "launcher-errors.log" && File.GetLastWriteTime(f) < DateTime.Now.AddDays(-days)) File.Delete(f);
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }

    void OnUnhandled(object sender, DispatcherUnhandledExceptionEventArgs e)
    {
        Report(e.Exception);
        _ = Popup.Show("Something went wrong", Loc.F("Something went wrong:\n\n{0}\n\nDetails are in {1}.", e.Exception.Message, CrashLog), problem: true);
        e.Handled = true;   // keep the launcher (and a running build) alive
    }

    static string CrashLog => Path.Combine(AppPaths.Logs, "launcher-errors.log");

    static void Report(Exception? ex)
    {
        if (ex == null) return;
        try
        {
            Directory.CreateDirectory(AppPaths.Logs);
            File.AppendAllText(CrashLog, $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {ex}\n\n");
        }
        catch (IOException) { }
    }

    protected override void OnExit(ExitEventArgs e)
    {
        single?.Dispose();
        base.OnExit(e);
    }
}
